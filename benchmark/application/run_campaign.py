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
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class CampaignConfig:
    """Tuning-mode selection and fixed-config overrides for one campaign -
    a narrower, non-argparse-coupled view of the CLI flags main() parses."""
    full_sweep: bool = False
    calibrate: bool = False
    sweep_moe_offload: bool = False
    sweep_moe_offload_thorough: bool = False
    max_depth: int | None = None
    ubatch: int | None = None
    batch: int | None = None
    ctk: str | None = None
    ctv: str | None = None
    flash_attn: str | None = None
    cooldown: int = 10
    force: bool = False


@dataclass
class CampaignOutcome:
    model_results_dir: Path
    summary_rows: int
    failed: bool
    partial: bool


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
    depths: tuple[int, ...],
    max_ctx: int | None,
    progress,
    # Collaborators the caller has already imported/wired - passed in
    # rather than imported here, so this module stays free of run_bench's
    # BenchConfig/run_one/auto_tune/etc. import cycle.
    bench_config_cls,
    run_one,
    auto_tune,
    pick_best_ubatch,
    sweep_moe_offload_quick,
    sweep_moe_offload_thorough,
    model_slug: str,
    ubatch_candidates: tuple[int, ...],
    prefill_tokens: int,
    generation_tokens: int,
    repetitions: int,
) -> CampaignOutcome:
    """Run tuning (or a fixed config), the optional MoE sweep, and the
    final prefill/generation curves for one model; write all campaign
    artifacts (curve_summary.csv, campaign_manifest.json, metadata.json,
    campaign.finished/partial/failed marker) to results_root/model_slug/run_id/.
    """
    model_results_dir = results_root / model_slug / run_id

    results: list = []
    tuning_log: list[dict] = []

    if config.full_sweep:
        tuning_ncmoe = (moe or {}).get("block_count") or 0
        if tuning_ncmoe:
            print(f"  MoE model: auto-tuning with --n-cpu-moe={tuning_ncmoe} "
                  "so KV/ubatch/batch probes do not OOM before the MoE curve runs", flush=True)
        bench_config = auto_tune(
            image=image, gpu_gids=gpu_gids, model=model, device=device,
            depths=depths, results_dir=model_results_dir, cooldown=config.cooldown,
            log=tuning_log, n_cpu_moe=tuning_ncmoe, progress=progress,
        )
        print(f"  full-sweep winner: {bench_config.tag()}", flush=True)
    elif config.calibrate:
        print("  calibration (legacy): sweeping ubatch candidates on prefill series", flush=True)
        for ubatch in ubatch_candidates:
            cfg = bench_config_cls(ubatch=ubatch)
            print(f"  [prefill calibration {cfg.tag()}]", flush=True)
            result = run_one(
                image=image, gpu_gids=gpu_gids, host_model_path=model,
                series="prefill", config=cfg, device=device,
                depths=depths, results_dir=model_results_dir, progress=progress,
            )
            results.append(result)
            print(f"    {result.status} (rc={result.return_code})", flush=True)
            time.sleep(config.cooldown)
        chosen_ubatch = pick_best_ubatch(model_results_dir, str(model), list(ubatch_candidates))
        bench_config = bench_config_cls(ubatch=chosen_ubatch)
        print(f"  calibration winner: ub={chosen_ubatch}", flush=True)
    else:
        bench_config = bench_config_cls(
            ubatch=config.ubatch or 2048,
            batch=config.batch or 2048,
            ctk=config.ctk or "f16",
            ctv=config.ctv or config.ctk or "f16",
            flash_attn=config.flash_attn or "auto",
        ).validate()
        print(f"  using fixed config: {bench_config.tag()}", flush=True)

    moe_offload_result = None
    # `--full-sweep` is the recommended/default tuning path, so a
    # detected MoE gets the cheap five-point curve automatically.
    # Thorough is explicit because it does more probes; it takes
    # precedence if both flags happen to be supplied.
    run_moe_sweep = moe is not None and (
        config.full_sweep or config.sweep_moe_offload or config.sweep_moe_offload_thorough
    )
    if moe is not None and run_moe_sweep:
        block_count = moe["block_count"]
        if not block_count:
            print("  WARNING: MoE model but no *.block_count in GGUF metadata; "
                  "skipping --n-cpu-moe sweep", flush=True)
        else:
            thorough = config.sweep_moe_offload_thorough
            print(f"  MoE model detected: expert_count={moe['expert_count']} "
                  f"expert_used_count={moe['expert_used_count']} block_count={block_count}", flush=True)
            print(f"  sweeping --n-cpu-moe ({'thorough/binary-search' if thorough else 'quick/fixed-candidates'})", flush=True)
            sweep_fn = sweep_moe_offload_thorough if thorough else sweep_moe_offload_quick
            moe_offload_result = sweep_fn(
                image=image, gpu_gids=gpu_gids, model=model, base_config=bench_config,
                device=device, depths=depths, block_count=block_count,
                results_dir=model_results_dir, cooldown=config.cooldown, progress=progress,
            )
            moe_offload_result["expert_count"] = moe["expert_count"]
            moe_offload_result["expert_used_count"] = moe["expert_used_count"]
            moe_offload_result["block_count"] = block_count
    elif moe is not None:
        print(f"  MoE model detected (expert_count={moe['expert_count']}) but "
              f"--sweep-moe-offload not given; skipping the --n-cpu-moe sweep. Every "
              f"expert stays resident in VRAM regardless of expert_used_count - see "
              f"README's MoE section.", flush=True)

    for series in ("prefill", "generation"):
        print(f"  [{series} {bench_config.tag()}]", flush=True)
        result = run_one(
            image=image, gpu_gids=gpu_gids, host_model_path=model,
            series=series, config=bench_config, device=device,
            depths=depths, results_dir=model_results_dir, progress=progress,
        )
        results.append(result)
        print(f"    {result.status} (rc={result.return_code})", flush=True)
        time.sleep(config.cooldown)

    summary_path = model_results_dir / "curve_summary.csv"
    rows = write_curve_summary(results, summary_path)
    failed = [r for r in results if r.status == "failed"]
    partial = [r for r in results if r.status == "partial"]
    mode = "full-sweep" if config.full_sweep else ("calibrate-legacy" if config.calibrate else "fixed")
    completed_at = datetime.now(timezone.utc).isoformat()

    manifest = build_campaign_manifest(
        image=image,
        device=device,
        depths=depths,
        context_length=max_ctx,
        final_config=asdict(bench_config),
        tuning_log=tuning_log,
        moe_offload_curve=moe_offload_result,
        repetitions=repetitions,
        prefill_tokens=prefill_tokens,
        generation_tokens=generation_tokens,
        mode=mode,
        completed_at=completed_at,
        summary_rows=rows,
        run_summaries=[
            {
                "series": r.series, "config": asdict(r.config), "status": r.status,
                "return_code": r.return_code, "depths_run": list(r.depths_run),
                "depths_skipped": list(r.depths_skipped), "stop_reason": r.stop_reason,
            }
            for r in results
        ],
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
        final_config=asdict(bench_config),
        depths_tested=depths,
        generation_tok_s_mean=best_depth0_ts,
        status="failed" if failed else ("partial" if partial else "finished"),
    )
    write_json(model_results_dir / "metadata.json", metadata)

    write_status_marker(model_results_dir, failed=bool(failed), partial=bool(partial))

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
        failed=bool(failed), partial=bool(partial),
    )
