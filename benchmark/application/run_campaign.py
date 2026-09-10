"""Application use case: run one complete benchmark campaign for a single
model - tuning (or a fixed config), the optional MoE offload sweep, the
final prefill/generation curves, and writing all campaign artifacts.

Extracted from run_bench.py's main() per-model loop (step 6 of the
refactor roadmap). This is the "use case" in hexagonal terms: it wires
together the domain/adapters already extracted in steps 1-5 and 7
(model resolution, campaign_store, docker_runner via run_one(), the MoE
bisection generator) into one coherent operation, so main() can shrink
to CLI parsing + wiring + calling this + exit-code logic.

Deliberately NOT split further into injected ports (e.g. an
EnvironmentInspector/CampaignStore interface) - run_bench.py remains the
only caller, so there is no second implementation to justify that
abstraction yet (see docs/architecture.md's "don't add ports
speculatively" note). If a second caller appears (e.g. a batch driver
across multiple models/hosts), function boundaries here are already
adapter-shaped and can be given a protocol at that point.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from adapters.outbound.campaign_store import (
    build_campaign_manifest,
    build_run_metadata,
    mean_ts,
    write_curve_summary,
    write_json,
    write_status_marker,
)
from application.auto_tune import kv_selection_tuning_log_entry, select_kv_config
from domain.kv_depth_planner import build_kv_depth_plan, build_kv_feasibility_manifest_payload


@dataclass(frozen=True)
class CampaignConfig:
    """Tuning-mode selection for one campaign - a narrower, non-argparse-
    coupled view of the CLI flags main() parses.

    quick=False (the default) always runs the full staged auto-tune
    (application/auto_tune.py) and, for a detected MoE model, the
    thorough --n-cpu-moe bisection (application/moe_sweep.py). quick=True
    skips tuning for a fixed config and, for MoE, runs the cheap 5-point
    sweep instead of the bisection. There is no third mode and no way to
    mix-and-match stages - see run_bench.py's module docstring.
    """
    quick: bool = False
    max_depth: int | None = None
    cooldown: int = 10
    force: bool = False


@dataclass
class CampaignOutcome:
    model_results_dir: Path
    summary_rows: int
    failed: bool
    partial: bool


def _config_payload(config) -> dict:
    payload = asdict(config)
    payload["gpu_layers"] = config.gpu_layers
    return payload


def run_model_campaign(
    *,
    image: str,
    gpu_gids: list[str],
    device: str,
    model: Path,
    results_root: Path,
    run_id: str,
    env: dict,
    config: CampaignConfig,
    gguf_metadata: dict,
    moe: dict | None,
    dense_block_count: int | None,
    model_size_bytes: int,
    depths: tuple[int, ...],
    max_ctx: int | None,
    progress,
    # Collaborators the caller has already imported/wired - passed in
    # rather than imported here, so this module stays free of run_bench's
    # BenchConfig/run_one/auto_tune/etc. import cycle.
    bench_config_cls,
    run_one,
    sweep_moe_offload_quick,
    sweep_moe_offload_thorough,
    preflight_dense_offload,
    sweep_dense_offload,
    model_slug: str,
    prefill_tokens: int,
    generation_tokens: int,
    repetitions: int,
    kv_depth_plan=None,
) -> CampaignOutcome:
    """Run tuning (or, in --quick mode, a fixed config), the MoE offload
    sweep for a detected MoE model (thorough bisection, or the cheap
    quick sweep in --quick mode), and the final prefill/generation
    curves for one model; write all campaign artifacts (curve_summary.csv,
    campaign_manifest.json, metadata.json, campaign.finished/partial/failed
    marker) to results_root/model_slug/run_id/.
    """
    model_results_dir = results_root / model_slug / run_id

    results: list = []
    tuning_log: list[dict] = []
    kv_feasibility_payload: dict | None = None

    # In the absence of calibration-grade placement/reserve evidence, every
    # feasibility result is deliberately unknown/runnable (see package
    # 01/02: kv_placement=None and reserve_policy=None until calibration
    # evidence lands). Only computed for the default flow - --quick never
    # consults the plan, so building/logging it there would be misleading.
    dense_offload_result = None
    dense_capacity_failed = False
    final_depths = depths
    selection = None
    if config.quick:
        bench_config = bench_config_cls(
            ubatch=2048, batch=2048, ctk="f16", ctv="f16", flash_attn="auto",
        ).validate()
        print(f"  --quick: using fixed config {bench_config.tag()}", flush=True)
    else:
        kv_depth_plan = kv_depth_plan or build_kv_depth_plan(
            requested_depths=depths,
            prefill_tokens=prefill_tokens,
            metadata=gguf_metadata,
            kv_offload_enabled=None,
            kv_placement=None,
            gpu_vram_bytes=env.get("gpu_vram_bytes"),
            runtime_scope=None,
            reserve_policy=None,
        )
        for dtype_plan in kv_depth_plan.dtype_plans:
            print(
                f"  KV plan {dtype_plan.ctk}: runnable={list(dtype_plan.runnable_depths)} "
                f"excluded={[item.context_size - prefill_tokens for item in dtype_plan.exclusions]}",
                flush=True,
            )
        kv_feasibility_payload = build_kv_feasibility_manifest_payload(
            kv_depth_plan,
            fixed_runtime_config={
                "batch": 2048, "ubatch": 2048, "flash_attn": "auto", "no_kv_offload": False,
            },
        )["kv_feasibility"]

        tuning_ncmoe = (moe or {}).get("block_count") or 0

        def probe_selected_dtype(candidate, depth):
            resolved_config = replace(
                candidate,
                n_cpu_moe=tuning_ncmoe,
                block_count=dense_block_count,
                n_cpu_layers=0,
            ).validate()
            return run_one(
                image=image, gpu_gids=gpu_gids, host_model_path=model,
                series="prefill", config=resolved_config, device=device,
                depths=(depth,), results_dir=model_results_dir, subdir="tuning",
                progress=progress,
            )

        selection = select_kv_config(plan=kv_depth_plan, probe=probe_selected_dtype)
        if selection is None:
            dense_capacity_failed = True
            bench_config = bench_config_cls().validate()
            print("  FAILED: no planned KV dtype/depth probe completed", flush=True)
        else:
            bench_config = replace(
                selection.config,
                n_cpu_moe=tuning_ncmoe,
                block_count=dense_block_count,
                n_cpu_layers=0,
            ).validate()
            final_depths = selection.runnable_depths
            tuning_log.append(kv_selection_tuning_log_entry(selection))
            print(
                f"  KV selection: depth={selection.target_depth} "
                f"winner={bench_config.ctk} runnable={list(final_depths)}",
                flush=True,
            )

    if not dense_capacity_failed and moe is None and dense_block_count:
        preflight_depth = max(final_depths)
        final_ngl = preflight_dense_offload(
            image=image, gpu_gids=gpu_gids, model=model, base_config=bench_config,
            device=device, depth=preflight_depth, block_count=dense_block_count,
            results_dir=model_results_dir, cooldown=config.cooldown,
            metadata=gguf_metadata, model_size_bytes=model_size_bytes,
            gpu_vram_bytes=env.get("gpu_vram_bytes") or 0, progress=progress,
        )
        if final_ngl is None:
            # The selected dtype may still have lower runnable depths. The full
            # dense sweep below retains those measurements rather than letting a
            # failed deepest preflight erase the entire campaign.
            print(
                f"  dense preflight: {bench_config.ctk} did not fit at depth={preflight_depth}; "
                "checking lower planned depths",
                flush=True,
            )
        else:
            bench_config = replace(
                bench_config, block_count=dense_block_count,
                n_cpu_layers=(dense_block_count + 1) - final_ngl,
            ).validate()

    moe_offload_result = None
    if moe is not None:
        block_count = moe["block_count"]
        if not block_count:
            print("  WARNING: MoE model but no *.block_count in GGUF metadata; "
                  "skipping --n-cpu-moe sweep", flush=True)
        else:
            thorough = not config.quick
            print(f"  MoE model detected: expert_count={moe['expert_count']} "
                  f"expert_used_count={moe['expert_used_count']} block_count={block_count}", flush=True)
            print(f"  sweeping --n-cpu-moe ({'thorough/binary-search' if thorough else 'quick/fixed-candidates'})", flush=True)
            sweep_fn = sweep_moe_offload_thorough if thorough else sweep_moe_offload_quick
            moe_offload_result = sweep_fn(
                image=image, gpu_gids=gpu_gids, model=model, base_config=bench_config,
                device=device, depths=final_depths, block_count=block_count,
                results_dir=model_results_dir, cooldown=config.cooldown, progress=progress,
            )
            moe_offload_result["expert_count"] = moe["expert_count"]
            moe_offload_result["expert_used_count"] = moe["expert_used_count"]
            moe_offload_result["block_count"] = block_count

    if moe is None and dense_block_count:
        print(
            f"  Dense model detected: block_count={dense_block_count}; "
            "discovering maximum --ngl and throughput at each depth",
            flush=True,
        )
        dense_sweep_result = sweep_dense_offload(
            image=image,
            gpu_gids=gpu_gids,
            model=model,
            base_config=bench_config,
            device=device,
            depths=final_depths,
            block_count=dense_block_count,
            results_dir=model_results_dir,
            cooldown=config.cooldown,
            metadata=gguf_metadata,
            model_size_bytes=model_size_bytes,
            gpu_vram_bytes=env.get("gpu_vram_bytes") or 0,
            quick=config.quick,
            progress=progress,
        )
        final_ngl = dense_sweep_result["final_ngl"]
        if dense_sweep_result["offload_needed"]:
            dense_offload_result = {
                key: value for key, value in dense_sweep_result.items()
                if key != "offload_needed"
            }
        if final_ngl is None:
            dense_capacity_failed = True
            bench_config = replace(
                bench_config,
                block_count=dense_block_count,
                n_cpu_layers=dense_block_count + 1,
            ).validate()
            if progress is not None:
                progress.prune(
                    2 * len(depths),
                    "final curves skipped because no dense --ngl fits",
                )
            print(
                "  FAILED: no --ngl fits at the deepest requested context; "
                "skipping final curves",
                flush=True,
            )
        else:
            max_gpu_layers = dense_block_count + 1
            bench_config = replace(
                bench_config,
                block_count=dense_block_count,
                n_cpu_layers=max_gpu_layers - final_ngl,
            ).validate()
            print(
                f"  final curves: fixed --ngl={bench_config.gpu_layers} "
                "(safe at the deepest probed context)",
                flush=True,
            )

    for series in (() if dense_capacity_failed else ("prefill", "generation")):
        print(f"  [{series} {bench_config.tag()}]", flush=True)
        result = run_one(
            image=image, gpu_gids=gpu_gids, host_model_path=model,
            series=series, config=bench_config, device=device,
            depths=final_depths, results_dir=model_results_dir, progress=progress,
        )
        results.append(result)
        print(f"    {result.status} (rc={result.return_code})", flush=True)
        time.sleep(config.cooldown)

    summary_path = model_results_dir / "curve_summary.csv"
    rows = write_curve_summary(results, summary_path)
    failed = [r for r in results if r.status == "failed"]
    campaign_failed = bool(failed) or dense_capacity_failed
    partial = [r for r in results if r.status == "partial"]
    mode = "quick" if config.quick else "full-sweep"
    completed_at = datetime.now(timezone.utc).isoformat()

    manifest = build_campaign_manifest(
        image=image,
        device=device,
        depths=depths,
        context_length=max_ctx,
        final_config=_config_payload(bench_config),
        tuning_log=tuning_log,
        moe_offload_curve=moe_offload_result,
        dense_offload_curve=dense_offload_result,
        repetitions=repetitions,
        prefill_tokens=prefill_tokens,
        generation_tokens=generation_tokens,
        mode=mode,
        completed_at=completed_at,
        summary_rows=rows,
        run_summaries=[
            {
                "series": r.series, "config": _config_payload(r.config), "status": r.status,
                "return_code": r.return_code, "depths_run": list(r.depths_run),
                "depths_skipped": list(r.depths_skipped), "stop_reason": r.stop_reason,
            }
            for r in results
        ],
        kv_feasibility=kv_feasibility_payload,
    )
    write_json(model_results_dir / "campaign_manifest.json", manifest)

    best_depth0_ts = None
    gen_rows = [r for r in results if r.series == "generation"]
    if gen_rows:
        ts = mean_ts(gen_rows[0].jsonl_path)
        best_depth0_ts = ts if ts >= 0 else None

    metadata = build_run_metadata(
        model_slug=model_slug,
        model_filename=model.name,
        model_architecture=gguf_metadata.get("general.architecture"),
        model_name=gguf_metadata.get("general.name"),
        model_context_length=max_ctx,
        model_moe=moe,
        run_id=run_id,
        run_completed_at=completed_at,
        mode=mode,
        environment=env,
        final_config=_config_payload(bench_config),
        depths_tested=depths,
        generation_tok_s_mean=best_depth0_ts,
        status="failed" if campaign_failed else ("partial" if partial else "finished"),
    )
    write_json(model_results_dir / "metadata.json", metadata)

    write_status_marker(model_results_dir, failed=campaign_failed, partial=bool(partial))

    print(f"  Curve summary: {summary_path} ({rows} rows)", flush=True)
    if failed:
        for r in failed:
            print(f"  FAILED: {r.model} {r.series} {r.config.tag()} (see {r.stderr_path})", flush=True)
    if partial:
        for r in partial:
            print(f"  PARTIAL: {r.model} {r.series} {r.config.tag()}: {r.stop_reason} "
                  f"(ran depths {list(r.depths_run)}, skipped {list(r.depths_skipped)})", flush=True)

    return CampaignOutcome(
        model_results_dir=model_results_dir, summary_rows=rows,
        failed=campaign_failed, partial=bool(partial),
    )
