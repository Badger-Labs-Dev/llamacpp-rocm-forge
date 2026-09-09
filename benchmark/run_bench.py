#!/usr/bin/env python3
"""Run a llama-bench parameter sweep against a Docker container on this host.

Standalone replacement for upstream's run_calibrated_campaign.py, which
assumes their Toolbx/Cockpit stack and SSH hosts. This drives `docker run` +
`llama-bench` directly and writes one JSONL file per (model, series, config)
combination plus a curve_summary.csv, mirroring the shape of upstream's
output closely enough to stay comparable.

Depths (llama-bench's "-d", how full the KV cache is before the timed run)
are derived per model from its GGUF *.context_length metadata (see
gguf_info.py): every entry in COMMON_CONTEXT_SIZES that fits under the
model's trained context length gets tested, plus depth 0. Testing past a
model's trained context produces throughput numbers but not meaningful
ones, since RoPE positions past that point were never seen in training.
Falls back to LEGACY_FIXED_DEPTHS if a model's context_length can't be read.

Tuning modes (pick one; --full-sweep is the general recommendation):

  (default)     Fixed config: ubatch=2048, batch=2048, ctk/ctv=f16, fa=auto.
  --ubatch N    Fixed config as above but with this ubatch.
  --calibrate   Legacy: sweep only UBATCH_CANDIDATES at depth 0, holding
                batch/KV-cache at their defaults (fa=auto). Kept for
                backward compatibility with earlier results directories.
  --full-sweep  Staged auto-tune across ubatch, batch, and KV cache dtype
                (f16/q8_0/q4_0) - see auto_tune(). Runs a handful of quick
                2-depth probes per stage rather than a full grid (which
                would be 4 ubatch x 4 batch x 3 KV = 48 combinations -
                infeasible to run at full depth x 3 repetitions). The
                final chosen config is then run across the model's full
                derived depth curve exactly once.

Flash attention is not swept: always passed as "auto" (llama-bench's -fa
auto|on|off, this build's own default), which lets llama.cpp decide per
model/backend at load time whether the fused kernel actually applies.
Forcing it "on" doesn't help on architectures where FA can't apply
anyway (KQ-bias models, some hybrid/SSM architectures, unsupported head
dims - these fall back to the ordinary attention path regardless of the
flag, silently), and for a few architectures (e.g. sparse-attention
models) FA isn't just a speed knob, disabling it changes output
correctness. llama.cpp's own "auto" already encodes this judgment call
better than we can by sweeping 0/1 ourselves.

llama.cpp requires flash attention ON whenever KV cache is quantized
(ctk/ctv != f16); BenchConfig.validate() enforces this rather than trying
invalid combinations - "auto" alone isn't a safe default there, since
auto isn't guaranteed to actually enable FA.

For MoE models (detected from GGUF *.expert_count metadata; a no-op for
dense models), --sweep-moe-offload additionally sweeps --n-cpu-moe
across the model's derived depths, answering "what's the minimum
--n-cpu-moe that fits at this context depth, and how much does
throughput drop as more gets offloaded to CPU". Every expert has to be
resident in VRAM regardless of how few are actually active per token
(the router can pick any of them per-token) - see moe_params() and
sweep_moe_offload_quick()/_thorough() for the actual mechanics.
--sweep-moe-offload-thorough binary-searches the exact fitting boundary
per depth instead of testing fixed evenly-spaced candidates.

Note: llama-bench has no flag for speculative decoding / multi-token
prediction (MTP) - that's a llama-server/llama-cli feature (-md draft
model), not something this benchmark tool measures.

Usage:
    ./run_bench.py --model /models/foo.gguf --full-sweep
"""

from __future__ import annotations

import argparse
import atexit
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from gguf_info import moe_params, read_gguf_metadata
from bench_app.adapters.outbound.campaign_store import (
    build_campaign_manifest,
    build_run_metadata,
    load_jsonl_rows,
    mean_ts,
    write_curve_summary,
    write_json,
    write_status_marker,
)
from bench_app.adapters.outbound.docker_llama_bench import (
    LlamaBenchProbe,
    docker_command as build_docker_command,
    llama_bench_command,
)
from bench_app.adapters.outbound.model_resolution import (
    COMMON_CONTEXT_SIZES,
    LEGACY_FIXED_DEPTHS,
    derive_depths_for_model as _derive_depths_for_model,
    model_key,
    model_slug,
    resolve_model_reference,
)
from bench_app.domain.moe_bisection import (
    extra_throughput_samples,
    resolve_boundary,
)
from bench_app.domain.planning import (
    campaign_budget,
    quick_moe_candidates,
    thorough_max_probes_per_depth,
)
import environment_info
import hf_models
from progress_tracker import ProgressTracker

UBATCH_CANDIDATES = (256, 512, 1024, 2048)
BATCH_CANDIDATES = (512, 1024, 2048, 4096)
KV_CACHE_TYPES = ("f16", "q8_0", "q4_0")
# KV cache dtypes that require flash attention (llama.cpp hard requirement -
# the non-fused attention path can't read a quantized KV cache at all, this
# isn't just a perf preference).
KV_TYPES_REQUIRING_FA = frozenset({"q8_0", "q4_0", "q5_0", "q5_1", "q4_1", "iq4_nl"})

PREFILL_TOKENS = 2048
GENERATION_TOKENS = 128
REPETITIONS = 3
GPU_LAYERS = 99
COOLDOWN_SECONDS = 10
DEPTH_TIMEOUT_SECONDS = 300  # per-depth llama-bench invocation; OOM can hang
                             # rather than error cleanly (see run_one), so a
                             # subprocess timeout is the only reliable guard
DEPTH_COOLDOWN_SECONDS = 2   # short pause between depths within one run_one()
                             # call, distinct from --cooldown between configs


@dataclass(frozen=True)
class BenchConfig:
    ubatch: int = 2048
    batch: int = 2048
    ctk: str = "f16"
    ctv: str = "f16"
    flash_attn: str = "auto"  # llama-bench's -fa: "auto" | "on" | "off".
        # "auto" lets llama.cpp decide per model/backend at load time
        # whether the fused kernel applies (some architectures - KQ-bias
        # models, certain hybrid/SSM models, unsupported head dims - fall
        # back to the ordinary path regardless of this flag, silently).
        # We stopped sweeping flash-attn on/off: forcing "on" doesn't help
        # on architectures where FA can't apply anyway, and "auto" is
        # already llama.cpp's own considered default as of this build.
    n_cpu_moe: int = 0  # llama-bench's -ncmoe/--n-cpu-moe: moves the MoE
        # feed-forward (expert) weights of the first N layers to CPU RAM,
        # keeping attention/shared weights on GPU. 0 (default) is a no-op
        # even for dense models - only meaningful for MoE models, where
        # every expert must otherwise be resident in VRAM regardless of
        # how few are active per token (see sweep_moe_offload() below).

    def tag(self) -> str:
        base = f"ub{self.ubatch}_b{self.batch}_kv{self.ctk}-{self.ctv}_fa{self.flash_attn}"
        return f"{base}_ncmoe{self.n_cpu_moe}" if self.n_cpu_moe else base

    def validate(self) -> "BenchConfig":
        """Force flash-attn on if the KV cache dtype requires it - "auto"
        is not guaranteed to enable FA, but a quantized KV cache can only
        be read via the fused kernel, so "auto" alone isn't safe here."""
        if (self.ctk in KV_TYPES_REQUIRING_FA or self.ctv in KV_TYPES_REQUIRING_FA) and self.flash_attn == "off":
            return replace(self, flash_attn="on")
        return self


@dataclass
class RunResult:
    model: str
    series: str  # "prefill" or "generation"
    config: BenchConfig
    status: str  # "ok" (all depths ran), "partial" (stopped early after a
                 # depth failed/timed out, but at least one depth succeeded),
                 # "failed" (no depth produced results)
    return_code: int
    jsonl_path: Path
    stderr_path: Path
    command: list[str]
    depths_run: tuple[int, ...] = ()
    depths_skipped: tuple[int, ...] = ()
    stop_reason: str | None = None  # e.g. "depth 65536 timed out after 300s"


# Names of docker containers currently running a benchmark, so they can be
# force-killed if this script is interrupted (Ctrl-C, SIGTERM, an unhandled
# exception). Without this, `docker run` surviving the parent script's death
# leaves the container - and the GPU memory it holds - running indefinitely;
# this has actually happened during development and is exactly the kind of
# thing that silently wastes VRAM until someone notices and runs `docker
# kill` by hand. `docker run --rm` alone does not protect against this: it
# only removes the container after IT exits, which doesn't happen just
# because the client/parent process died.
_ACTIVE_CONTAINERS: set[str] = set()


def _kill_active_containers() -> None:
    for name in list(_ACTIVE_CONTAINERS):
        subprocess.run(["docker", "kill", name], capture_output=True)
        _ACTIVE_CONTAINERS.discard(name)


def _install_cleanup_handlers() -> None:
    atexit.register(_kill_active_containers)

    def _handle_signal(signum, frame):
        _kill_active_containers()
        # Restore default handling and re-raise, so the process actually
        # exits with the conventional 128+signum code instead of hanging.
        signal.signal(signum, signal.SIG_DFL)
        signal.raise_signal(signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)


def derive_depths_for_model(model_path: Path) -> tuple[tuple[int, ...], int | None]:
    """Compatibility wrapper around the model-resolution adapter's depth
    derivation, pinned to this module's PREFILL_TOKENS."""
    return _derive_depths_for_model(model_path, prefill_tokens=PREFILL_TOKENS)


def probe_for(*, model_container_path: str, series: str, config: BenchConfig, device: str, depth: int) -> LlamaBenchProbe:
    """Build a LlamaBenchProbe from a BenchConfig - the single place that
    maps run_bench's configuration model onto the Docker adapter's probe
    contract, so build_llama_bench_command() and run_one() can't drift."""
    return LlamaBenchProbe(
        model_container_path=model_container_path,
        series=series,
        batch=config.batch,
        ubatch=config.ubatch,
        flash_attn=config.flash_attn,
        n_cpu_moe=config.n_cpu_moe,
        ctk=config.ctk,
        ctv=config.ctv,
        device=device,
        depth=depth,
        repetitions=REPETITIONS,
        gpu_layers=GPU_LAYERS,
        prefill_tokens=PREFILL_TOKENS,
        generation_tokens=GENERATION_TOKENS,
    )


def build_llama_bench_command(
    *,
    model_container_path: str,
    series: str,
    config: BenchConfig,
    device: str,
    depth: int,
) -> list[str]:
    """Compatibility wrapper around the Docker adapter's llama.cpp command."""
    return llama_bench_command(probe_for(
        model_container_path=model_container_path,
        series=series,
        config=config,
        device=device,
        depth=depth,
    ))


def run_one(
    *,
    image: str,
    gpu_gids: list[str],
    host_model_path: Path,
    series: str,
    config: BenchConfig,
    device: str,
    depths: tuple[int, ...],
    results_dir: Path,
    subdir: str | None = None,
    progress: ProgressTracker | None = None,
) -> RunResult:
    """Run llama-bench once per depth, ascending (shallowest first).

    Depths are run in separate `docker run` invocations rather than one
    llama-bench call with a combined "-d" list, specifically so an
    out-of-memory condition at a deep context doesn't take down the whole
    curve: llama-bench streams JSONL results per depth as it completes
    them (confirmed by direct testing - shallow depths finish and print
    before a later depth OOMs), but a large KV cache allocation can also
    just *hang* rather than error out cleanly, so a single multi-depth
    invocation has no way to bail out of one bad depth and keep going.
    Running depths ascending, one process per depth, with a timeout,
    means: smaller/valid depths always get recorded, and a failure or
    hang at some depth stops only the depths larger than it (they'd very
    likely fail too - KV cache need only grows with depth).
    """
    out_dir = results_dir / subdir if subdir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve symlinks before mounting: Hugging Face's local cache stores
    # models as snapshots/<hash>/model.gguf -> ../../blobs/<blob-hash>, so
    # mounting only the file's immediate parent directory (the snapshot
    # dir) would leave the symlink target outside the mount, unreadable
    # inside the container. Mounting the *real* file's parent directory
    # works for both plain files (no-op, same directory either way) and
    # HF-cache-style symlinks.
    real_model_path = host_model_path.resolve()
    container_model_path = f"/models/{real_model_path.name}"
    out_name = f"{model_key(str(host_model_path))}__{series}__{config.tag()}"
    jsonl_path = out_dir / f"{out_name}.jsonl"
    stderr_path = out_dir / f"{out_name}.stderr.log"

    sorted_depths = tuple(sorted(depths))
    depths_run: list[int] = []
    depths_skipped: list[int] = []
    stop_reason: str | None = None
    last_return_code = 0
    last_command: list[str] = []

    jsonl_f = jsonl_path.open("w", encoding="utf-8")
    stderr_f = stderr_path.open("w", encoding="utf-8")
    try:
        for i, depth in enumerate(sorted_depths):
            if stop_reason is not None:
                depths_skipped.append(depth)
                continue

            container_name = f"r9700-llm-bench-{uuid.uuid4().hex[:12]}"
            probe = probe_for(
                model_container_path=container_model_path,
                series=series,
                config=config,
                device=device,
                depth=depth,
            )
            docker_cmd = build_docker_command(
                image=image,
                container_name=container_name,
                gpu_gids=gpu_gids,
                model_directory=str(real_model_path.parent),
                probe=probe,
            )
            last_command = docker_cmd
            print(f"    $ {' '.join(docker_cmd)}", flush=True)
            if progress is not None:
                progress.before_probe(f"{series} depth={depth} {config.tag()}")

            stderr_f.write(f"\n===== depth={depth} =====\n")
            stderr_f.flush()

            # Track the container name so a signal handler or atexit hook
            # can `docker kill` it if this script gets interrupted mid-run
            # (see _ACTIVE_CONTAINERS above), and so a *timeout* here can
            # kill the specific container that hung rather than leaving it
            # running - subprocess.run(timeout=...) only kills the direct
            # child (docker CLI), not the container it started, so without
            # an explicit `docker kill` a timed-out depth would leak VRAM
            # exactly like an interrupted run would.
            _ACTIVE_CONTAINERS.add(container_name)
            probe_started_at = time.monotonic()
            try:
                proc = subprocess.run(
                    docker_cmd, stdout=jsonl_f, stderr=stderr_f,
                    timeout=DEPTH_TIMEOUT_SECONDS,
                )
                last_return_code = proc.returncode
                if proc.returncode != 0:
                    stop_reason = (
                        f"depth {depth} failed (rc={proc.returncode}); "
                        f"skipping larger depths"
                    )
                    print(f"    FAILED at depth={depth} (rc={proc.returncode}); "
                          f"skipping remaining {len(sorted_depths) - i - 1} larger depth(s)",
                          flush=True)
                    if progress is not None:
                        progress.prune(len(sorted_depths) - i - 1, "this series failed at a shallower depth")
                else:
                    depths_run.append(depth)
            except subprocess.TimeoutExpired:
                subprocess.run(["docker", "kill", container_name], capture_output=True)
                last_return_code = -1
                stop_reason = (
                    f"depth {depth} timed out after {DEPTH_TIMEOUT_SECONDS}s "
                    f"(likely OOM/thrashing rather than a clean error); "
                    f"skipping larger depths"
                )
                stderr_f.write(f"\n[TIMEOUT after {DEPTH_TIMEOUT_SECONDS}s - container killed]\n")
                stderr_f.flush()
                print(f"    TIMEOUT at depth={depth} after {DEPTH_TIMEOUT_SECONDS}s "
                      f"(container killed); skipping remaining "
                      f"{len(sorted_depths) - i - 1} larger depth(s)", flush=True)
                if progress is not None:
                    progress.prune(len(sorted_depths) - i - 1, "this series timed out at a shallower depth")
            finally:
                _ACTIVE_CONTAINERS.discard(container_name)
                if progress is not None:
                    progress.finish_probe(elapsed_seconds=time.monotonic() - probe_started_at)

            if stop_reason is None and i < len(sorted_depths) - 1:
                time.sleep(DEPTH_COOLDOWN_SECONDS)
    finally:
        jsonl_f.close()
        stderr_f.close()

    if not depths_run:
        status = "failed"
    elif stop_reason is not None:
        # At least one depth (possibly the last one) never produced
        # results - "ok" would wrongly imply the full curve is complete.
        status = "partial"
    else:
        status = "ok"

    return RunResult(
        model=str(host_model_path),
        series=series,
        config=config,
        status=status,
        return_code=last_return_code,
        jsonl_path=jsonl_path,
        stderr_path=stderr_path,
        command=last_command,
        depths_run=tuple(depths_run),
        depths_skipped=tuple(depths_skipped),
        stop_reason=stop_reason,
    )


def probe_depths(depths: tuple[int, ...]) -> tuple[int, ...]:
    """Reduce a full depth list to a cheap 2-point probe: shallowest + deepest.

    Used during auto-tuning stages, where we need *some* signal at both a
    cold cache and a full one (KV-cache-quantization benefits scale with
    depth, so testing depth 0 alone would underrate it) without paying for
    every depth in the model's full curve.
    """
    if len(depths) <= 2:
        return depths
    return (depths[0], depths[-1])


def probe_config(
    *,
    image: str,
    gpu_gids: list[str],
    model: Path,
    config: BenchConfig,
    device: str,
    depths: tuple[int, ...],
    results_dir: Path,
    progress: ProgressTracker | None = None,
) -> float:
    """Run a quick prefill-only probe of one config; return mean avg_ts."""
    result = run_one(
        image=image, gpu_gids=gpu_gids, host_model_path=model,
        series="prefill", config=config.validate(), device=device,
        depths=depths, results_dir=results_dir, subdir="tuning", progress=progress,
    )
    if result.status != "ok":
        return -1.0
    return mean_ts(result.jsonl_path)


def auto_tune(
    *,
    image: str,
    gpu_gids: list[str],
    model: Path,
    device: str,
    depths: tuple[int, ...],
    results_dir: Path,
    cooldown: int,
    log: list[dict],
    n_cpu_moe: int = 0,
    progress: ProgressTracker | None = None,
) -> BenchConfig:
    """Staged (coordinate-descent) auto-tune: KV cache dtype, then a
    ubatch x batch grid. Each stage probes at 2 depths (shallowest +
    deepest) rather than the full curve, and carries its winner into the
    next stage. For a MoE full-sweep, n_cpu_moe is conservatively set to
    block_count so the fixed-parameter stages can evaluate their intended
    knobs without an unrelated high-context MoE OOM; the separate MoE
    curve later measures every candidate with the winning base config.
    """
    probe_d = probe_depths(depths)
    print(f"  auto-tune: probing at depths {list(probe_d)}", flush=True)

    # Stage 1: KV cache dtype, flash-attn always "auto" (quantized KV
    # cache requires FA regardless, and validate() enforces that).
    print("  [stage 1/2] KV cache dtype", flush=True)
    kv_scores = {}
    for kv in KV_CACHE_TYPES:
        cfg = BenchConfig(ctk=kv, ctv=kv, n_cpu_moe=n_cpu_moe).validate()
        score = probe_config(image=image, gpu_gids=gpu_gids, model=model, config=cfg,
                             device=device, depths=probe_d, results_dir=results_dir, progress=progress)
        kv_scores[kv] = score
        print(f"    kv={kv}: mean_ts={score:.1f}", flush=True)
        time.sleep(cooldown)
    best_kv = max(kv_scores, key=lambda k: kv_scores[k])
    log.append({"stage": "kv_cache_dtype", "scores": kv_scores, "winner": best_kv})
    print(f"  stage 1 winner: kv={best_kv}", flush=True)

    # Stage 2: ubatch x batch grid, at the winning KV, depth 0 only
    # (batch/ubatch effects show up clearly even on a cold cache, and this
    # keeps the grid's 13 valid combinations cheap).
    print("  [stage 2/2] ubatch x batch grid (depth 0 only)", flush=True)
    grid_scores = {}
    grid_combo_lookup: dict[str, tuple[int, int]] = {}
    for ub in UBATCH_CANDIDATES:
        for b in BATCH_CANDIDATES:
            if ub > b:
                continue  # llama.cpp requires ubatch <= batch
            cfg = BenchConfig(ubatch=ub, batch=b, ctk=best_kv, ctv=best_kv,
                              n_cpu_moe=n_cpu_moe).validate()
            score = probe_config(image=image, gpu_gids=gpu_gids, model=model, config=cfg,
                                 device=device, depths=(0,), results_dir=results_dir, progress=progress)
            combo_key = f"ub{ub}_b{b}"
            grid_scores[combo_key] = score
            grid_combo_lookup[combo_key] = (ub, b)
            print(f"    ub={ub} b={b}: mean_ts={score:.1f}", flush=True)
            time.sleep(cooldown)
    best_combo = max(grid_scores, key=lambda k: grid_scores[k])
    best_ub, best_b = grid_combo_lookup[best_combo]
    log.append({"stage": "ubatch_batch_grid", "scores": grid_scores, "winner": best_combo})
    print(f"  stage 2 winner: ubatch={best_ub} batch={best_b}", flush=True)

    return BenchConfig(ubatch=best_ub, batch=best_b, ctk=best_kv, ctv=best_kv,
                       n_cpu_moe=n_cpu_moe).validate()


MOE_QUICK_CANDIDATE_COUNT = 5  # evenly spaced --n-cpu-moe candidates for quick mode
MOE_EXTRA_SAMPLE_COUNT = 3     # extra throughput samples above the found boundary,
                                # for the thorough mode's "how hard does tps dive" curve


def valid_batch_grid_count() -> int:
    return sum(ub <= batch for ub in UBATCH_CANDIDATES for batch in BATCH_CANDIDATES)


def planned_probe_count(args: argparse.Namespace, depths: tuple[int, ...], moe: dict | None) -> tuple[int, str]:
    """CLI adapter around the domain campaign-budget calculation."""
    should_sweep_moe = moe is not None and (
        args.full_sweep or args.sweep_moe_offload or args.sweep_moe_offload_thorough
    )
    budget = campaign_budget(
        depth_count=len(depths),
        full_sweep=args.full_sweep,
        calibrate=args.calibrate,
        valid_batch_pairs=valid_batch_grid_count(),
        kv_type_count=len(KV_CACHE_TYPES),
        tuning_depth_count=len(probe_depths(depths)),
        moe_block_count=(moe or {}).get("block_count") if should_sweep_moe else None,
        thorough_moe=args.sweep_moe_offload_thorough,
        quick_candidate_count=MOE_QUICK_CANDIDATE_COUNT,
        thorough_extra_sample_count=MOE_EXTRA_SAMPLE_COUNT,
    )
    return budget.total, budget.detail


def moe_offload_candidates(block_count: int, n: int = MOE_QUICK_CANDIDATE_COUNT) -> tuple[int, ...]:
    """Compatibility name for the domain's quick MoE candidate rule."""
    return quick_moe_candidates(block_count, n)


def probe_moe_offload(
    *,
    image: str,
    gpu_gids: list[str],
    model: Path,
    base_config: BenchConfig,
    n_cpu_moe: int,
    device: str,
    depth: int,
    results_dir: Path,
    progress: ProgressTracker | None = None,
) -> tuple[bool, float]:
    """Run one prefill probe at a specific (depth, n_cpu_moe); return
    (fits, mean_avg_ts). fits=False on any failure/timeout - VRAM
    exhaustion at a high depth can hang rather than error cleanly (see
    run_one's docstring), so this relies on the same per-depth timeout."""
    cfg = replace(base_config, n_cpu_moe=n_cpu_moe).validate()
    result = run_one(
        image=image, gpu_gids=gpu_gids, host_model_path=model,
        series="prefill", config=cfg, device=device,
        depths=(depth,), results_dir=results_dir, subdir="moe-tuning", progress=progress,
    )
    if result.status != "ok":
        return False, -1.0
    return True, mean_ts(result.jsonl_path)


def sweep_moe_offload_quick(
    *,
    image: str,
    gpu_gids: list[str],
    model: Path,
    base_config: BenchConfig,
    device: str,
    depths: tuple[int, ...],
    block_count: int,
    results_dir: Path,
    cooldown: int,
    progress: ProgressTracker | None = None,
) -> dict:
    """Quick mode: fixed evenly-spaced --n-cpu-moe candidates, tested at
    every depth. Simple and fast, but the boundary between fitting and
    not-fitting could fall between two candidates rather than exactly at
    one - see sweep_moe_offload_thorough() for the precise version."""
    candidates = moe_offload_candidates(block_count)
    by_depth = []
    for depth_index, depth in enumerate(depths):
        point_results = []
        min_ncmoe_that_fits = None
        for ncmoe in candidates:
            fits, ts = probe_moe_offload(
                image=image, gpu_gids=gpu_gids, model=model, base_config=base_config,
                n_cpu_moe=ncmoe, device=device, depth=depth, results_dir=results_dir,
                progress=progress,
            )
            point_results.append({
                "n_cpu_moe": ncmoe,
                "status": "ok" if fits else "failed",
                "avg_ts": ts if fits else None,
            })
            if fits and min_ncmoe_that_fits is None:
                min_ncmoe_that_fits = ncmoe
            print(f"    depth={depth} ncmoe={ncmoe}: "
                  f"{'ok, ts=' + format(ts, '.1f') if fits else 'FAILED'}", flush=True)
            time.sleep(cooldown)
        by_depth.append({
            "depth": depth,
            "min_ncmoe_that_fits": min_ncmoe_that_fits,
            "results": point_results,
        })
        if min_ncmoe_that_fits is None:
            print(f"    depth={depth}: nothing fit even at ncmoe={block_count} "
                  f"(fully offloaded); deeper depths won't fit either, stopping", flush=True)
            if progress is not None:
                progress.prune(
                    (len(depths) - depth_index - 1) * len(candidates),
                    "fully offloaded experts could not fit at this depth",
                )
            break

    return {
        "mode": "quick",
        "candidates_tested": list(candidates),
        "by_depth": by_depth,
    }


def sweep_moe_offload_thorough(
    *,
    image: str,
    gpu_gids: list[str],
    model: Path,
    base_config: BenchConfig,
    device: str,
    depths: tuple[int, ...],
    block_count: int,
    results_dir: Path,
    cooldown: int,
    progress: ProgressTracker | None = None,
) -> dict:
    """Thorough mode: binary search for the exact min_ncmoe_that_fits per
    depth, then a few extra throughput samples above that boundary for
    the "how hard does throughput dive as I offload more" curve.

    Relies on two monotonicity assumptions, both physically motivated
    rather than just convenient: (1) for a fixed depth, success in
    n_cpu_moe is monotonic - more layers offloaded to CPU only frees
    VRAM, never uses more, so once a value fits, every larger value
    fits too; (2) across depths, the fitting boundary is non-decreasing
    - a deeper context needs a larger KV cache, which only adds VRAM
    pressure, so a value that already failed at a shallower depth will
    also fail at any deeper one. (2) means each depth's search can start
    from the previous depth's boundary as a known-failing lower bound,
    instead of re-searching [0, block_count] from scratch - and once the
    boundary stabilizes across a run of depths (common - shallow depth
    increases often don't need more headroom), it costs just one probe
    per depth to confirm rather than a full search.
    """
    by_depth = []
    known_fail_floor = -1  # -1 means "no known-failing value yet" (0 might fit)
    per_depth_budget = thorough_max_probes_per_depth(block_count, MOE_EXTRA_SAMPLE_COUNT)

    for depth_index, depth in enumerate(depths):
        tested: dict[int, tuple[bool, float]] = {}

        def probe(ncmoe: int) -> bool:
            if ncmoe in tested:
                return tested[ncmoe][0]
            fits, ts = probe_moe_offload(
                image=image, gpu_gids=gpu_gids, model=model, base_config=base_config,
                n_cpu_moe=ncmoe, device=device, depth=depth, results_dir=results_dir,
                progress=progress,
            )
            tested[ncmoe] = (fits, ts)
            print(f"    depth={depth} ncmoe={ncmoe}: "
                  f"{'ok, ts=' + format(ts, '.1f') if fits else 'FAILED'}", flush=True)
            time.sleep(cooldown)
            return fits

        # Drive the pure boundary-search generator: it decides which
        # n_cpu_moe to try next, this loop performs the actual probe I/O.
        search = resolve_boundary(known_fail_floor=known_fail_floor, block_count=block_count)
        try:
            ncmoe = next(search)
            while True:
                ncmoe = search.send(probe(ncmoe))
        except StopIteration as stop:
            boundary, _search_tested = stop.value

        if boundary is None:
            by_depth.append({
                "depth": depth,
                "min_ncmoe_that_fits": None,
                "results": [{"n_cpu_moe": k, "status": "ok" if v[0] else "failed",
                              "avg_ts": v[1] if v[0] else None} for k, v in sorted(tested.items())],
            })
            print(f"    depth={depth}: nothing fits even at ncmoe={block_count} "
                  f"(fully offloaded); deeper depths won't fit either, stopping", flush=True)
            if progress is not None:
                progress.prune(
                    (per_depth_budget - len(tested))
                    + (len(depths) - depth_index - 1) * per_depth_budget,
                    "fully offloaded experts could not fit at this depth",
                )
            break

        # Extra throughput samples above the boundary, for the dive
        # curve - bisection alone only samples near the boundary, not
        # spread across the range above it.
        if boundary < block_count and MOE_EXTRA_SAMPLE_COUNT > 0:
            for ncmoe in extra_throughput_samples(
                boundary=boundary, block_count=block_count, sample_count=MOE_EXTRA_SAMPLE_COUNT,
            ):
                probe(ncmoe)

        known_fail_floor = boundary - 1
        by_depth.append({
            "depth": depth,
            "min_ncmoe_that_fits": boundary,
            "results": [{"n_cpu_moe": k, "status": "ok" if v[0] else "failed",
                          "avg_ts": v[1] if v[0] else None} for k, v in sorted(tested.items())],
        })
        if progress is not None:
            progress.prune(
                per_depth_budget - len(tested),
                "thorough search resolved this depth below its conservative bound",
            )

    return {
        "mode": "thorough",
        "by_depth": by_depth,
    }


def pick_best_ubatch(results_dir: Path, model: str, candidates: list[int]) -> int:
    """Legacy (--calibrate): choose the ubatch with the highest depth-0
    prefill throughput, holding batch/KV/FA at BenchConfig defaults."""
    best_ubatch = candidates[0]
    best_score = -1.0
    for ubatch in candidates:
        cfg = BenchConfig(ubatch=ubatch)
        path = results_dir / f"{model_key(model)}__prefill__{cfg.tag()}.jsonl"
        rows = [r for r in load_jsonl_rows(path) if r.get("n_depth") == 0]
        if not rows:
            continue
        ts_values = [r.get("avg_ts") or 0 for r in rows]
        score = sum(ts_values) / len(ts_values) if ts_values else 0
        if score > best_score:
            best_score = score
            best_ubatch = ubatch
    return best_ubatch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", default=[],
                         help="Host path to a .gguf file (repeatable). Also accepts Hugging "
                              "Face references: hf://org/repo/file.gguf, org/repo, "
                              "org/repo:QUANT (ollama-style tag), or org/repo/file.gguf - "
                              "resolved via the local HF cache, downloading if not already "
                              "present. org/repo with multiple .gguf files prompts an "
                              "interactive picker.")
    parser.add_argument("--image", default="r9700-llm-bench:rocm-7.2.4", help="Docker image to benchmark")
    parser.add_argument("--device", default="ROCm0", help="llama-bench -dev target (default: ROCm0, the R9700)")
    parser.add_argument("--gpu-gid", action="append", default=[],
                         help="Host GID(s) for /dev/dri and /dev/kfd access (repeatable); "
                              "defaults to `video` and `render` group GIDs on this host")
    parser.add_argument("--results-root", type=Path, default=Path("results"),
                         help="Root results directory; each model gets "
                              "<results-root>/<model-slug>/<run-id>/ (default: ./results)")
    parser.add_argument("--force", action="store_true",
                         help="Overwrite an existing run directory instead of refusing")
    parser.add_argument("--calibrate", action="store_true",
                         help="Legacy: sweep only ubatch at depth 0, other params fixed at defaults")
    parser.add_argument("--full-sweep", action="store_true",
                         help="Staged auto-tune across ubatch, batch, and KV cache dtype; "
                              "also runs the quick MoE offload sweep when GGUF metadata "
                              "identifies the model as MoE")
    parser.add_argument("--sweep-moe-offload", action="store_true",
                         help="MoE models only (auto-detected from GGUF metadata, no-op for dense "
                              "models): also sweep --n-cpu-moe across depths, evenly-spaced "
                              "candidates. Answers 'what's the minimum --n-cpu-moe that fits at "
                              "each context depth, and how much does throughput drop as I offload "
                              "more'. Runs after --full-sweep's ubatch/batch/KV stages (or after "
                              "the fixed/--calibrate config if --full-sweep isn't also given), "
                              "using that config as the base.")
    parser.add_argument("--sweep-moe-offload-thorough", action="store_true",
                         help="Like --sweep-moe-offload, but binary-searches for the exact "
                              "min_ncmoe_that_fits per depth instead of testing fixed evenly-spaced "
                              "candidates - more precise, similar or lower cost thanks to reusing "
                              "each depth's boundary as a starting point for the next.")
    parser.add_argument("--max-depth", type=int, default=None,
                         help="Cap derived depths at this value even if the model supports more "
                              "(useful for a quick smoke test on a long-context model)")
    parser.add_argument("--ubatch", type=int, default=None, help="Fixed ubatch (ignored with --calibrate/--full-sweep)")
    parser.add_argument("--batch", type=int, default=None, help="Fixed batch size (ignored with --full-sweep)")
    parser.add_argument("--ctk", default=None, help="Fixed KV cache key dtype, e.g. f16/q8_0/q4_0 (ignored with --full-sweep)")
    parser.add_argument("--ctv", default=None, help="Fixed KV cache value dtype (ignored with --full-sweep)")
    parser.add_argument("--flash-attn", choices=("auto", "on", "off"), default=None,
                         help="Fixed flash-attn mode (ignored with --full-sweep, which always "
                              "uses 'auto' - see BenchConfig.flash_attn)")
    parser.add_argument("--cooldown", type=int, default=COOLDOWN_SECONDS)
    return parser.parse_args()


def default_gpu_gids() -> list[str]:
    import grp
    gids = []
    for name in ("video", "render"):
        try:
            gids.append(str(grp.getgrnam(name).gr_gid))
        except KeyError:
            pass
    return gids


def main() -> None:
    _install_cleanup_handlers()
    args = parse_args()
    gpu_gids = args.gpu_gid or default_gpu_gids()
    if not gpu_gids:
        sys.exit("Could not resolve video/render group GIDs; pass --gpu-gid explicitly")

    if not args.model:
        sys.exit("At least one --model is required")

    # Keep the given path as-is (don't resolve symlinks here) - Hugging
    # Face cache entries are named model.gguf via a symlink to a
    # content-hash blob; resolving too early would make model_slug() use
    # the meaningless blob hash instead of the real model filename.
    # run_one() resolves symlinks separately, only for the Docker mount.
    models = []
    for m in args.model:
        try:
            models.append(resolve_model_reference(m))
        except hf_models.HfReferenceError as e:
            sys.exit(f"Error resolving {m!r}: {e}")
    for model in models:
        if not model.is_file():
            sys.exit(f"Model file not found: {model}")

    print("Querying environment (ROCm/llama.cpp/GPU versions)...", flush=True)
    env = environment_info.gather(args.image)
    run_id = environment_info.run_id(env)
    print(f"  {env}", flush=True)
    print(f"  run_id: {run_id}", flush=True)

    for model in models:
        slug = model_slug(model)
        model_results_dir = args.results_root / slug / run_id
        if model_results_dir.exists() and any(model_results_dir.iterdir()):
            if not args.force:
                sys.exit(
                    f"Refusing to overwrite existing run directory: {model_results_dir}\n"
                    f"(same model + ROCm + llama.cpp version already benchmarked here; "
                    f"pass --force to overwrite)"
                )
            for child in model_results_dir.rglob("*"):
                if child.is_file():
                    child.unlink()
        model_results_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n== {model.name} -> {model_results_dir} ==", flush=True)

        depths, max_ctx = derive_depths_for_model(model)
        if args.max_depth is not None:
            depths = tuple(d for d in depths if d <= args.max_depth) or (0,)
        if max_ctx is not None:
            print(f"  model context_length: {max_ctx}", flush=True)
            print(f"  derived depths (common context sizes up to {max_ctx}): {list(depths)}", flush=True)
        else:
            print(f"  WARNING: could not read context_length from GGUF metadata; "
                  f"falling back to fixed legacy depths: {list(depths)}", flush=True)

        try:
            gguf_metadata = read_gguf_metadata(model)
        except (OSError, ValueError):
            gguf_metadata = {}
        moe = moe_params(gguf_metadata)
        max_probes, progress_detail = planned_probe_count(args, depths, moe)
        progress = ProgressTracker(
            total_probes=max_probes,
            expected_gap_seconds=DEPTH_COOLDOWN_SECONDS,
        )
        progress.start(detail=progress_detail)

        results: list[RunResult] = []
        tuning_log: list[dict] = []

        if args.full_sweep:
            tuning_ncmoe = (moe or {}).get("block_count") or 0
            if tuning_ncmoe:
                print(f"  MoE model: auto-tuning with --n-cpu-moe={tuning_ncmoe} "
                      "so KV/ubatch/batch probes do not OOM before the MoE curve runs", flush=True)
            config = auto_tune(
                image=args.image, gpu_gids=gpu_gids, model=model, device=args.device,
                depths=depths, results_dir=model_results_dir, cooldown=args.cooldown,
                log=tuning_log, n_cpu_moe=tuning_ncmoe, progress=progress,
            )
            print(f"  full-sweep winner: {config.tag()}", flush=True)
        elif args.calibrate:
            print("  calibration (legacy): sweeping ubatch candidates on prefill series", flush=True)
            for ubatch in UBATCH_CANDIDATES:
                cfg = BenchConfig(ubatch=ubatch)
                print(f"  [prefill calibration {cfg.tag()}]", flush=True)
                result = run_one(
                    image=args.image, gpu_gids=gpu_gids, host_model_path=model,
                    series="prefill", config=cfg, device=args.device,
                    depths=depths, results_dir=model_results_dir, progress=progress,
                )
                results.append(result)
                print(f"    {result.status} (rc={result.return_code})", flush=True)
                time.sleep(args.cooldown)
            chosen_ubatch = pick_best_ubatch(model_results_dir, str(model), list(UBATCH_CANDIDATES))
            config = BenchConfig(ubatch=chosen_ubatch)
            print(f"  calibration winner: ub={chosen_ubatch}", flush=True)
        else:
            config = BenchConfig(
                ubatch=args.ubatch or 2048,
                batch=args.batch or 2048,
                ctk=args.ctk or "f16",
                ctv=args.ctv or args.ctk or "f16",
                flash_attn=args.flash_attn or "auto",
            ).validate()
            print(f"  using fixed config: {config.tag()}", flush=True)

        moe_offload_result = None
        # `--full-sweep` is the recommended/default tuning path, so a
        # detected MoE gets the cheap five-point curve automatically.
        # Thorough is explicit because it does more probes; it takes
        # precedence if both flags happen to be supplied.
        run_moe_sweep = moe is not None and (
            args.full_sweep or args.sweep_moe_offload or args.sweep_moe_offload_thorough
        )
        if moe is not None and run_moe_sweep:
            block_count = moe["block_count"]
            if not block_count:
                print("  WARNING: MoE model but no *.block_count in GGUF metadata; "
                      "skipping --n-cpu-moe sweep", flush=True)
            else:
                thorough = args.sweep_moe_offload_thorough
                print(f"  MoE model detected: expert_count={moe['expert_count']} "
                      f"expert_used_count={moe['expert_used_count']} block_count={block_count}", flush=True)
                print(f"  sweeping --n-cpu-moe ({'thorough/binary-search' if thorough else 'quick/fixed-candidates'})", flush=True)
                sweep_fn = sweep_moe_offload_thorough if thorough else sweep_moe_offload_quick
                moe_offload_result = sweep_fn(
                    image=args.image, gpu_gids=gpu_gids, model=model, base_config=config,
                    device=args.device, depths=depths, block_count=block_count,
                    results_dir=model_results_dir, cooldown=args.cooldown, progress=progress,
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
            print(f"  [{series} {config.tag()}]", flush=True)
            result = run_one(
                image=args.image, gpu_gids=gpu_gids, host_model_path=model,
                series=series, config=config, device=args.device,
                depths=depths, results_dir=model_results_dir, progress=progress,
            )
            results.append(result)
            print(f"    {result.status} (rc={result.return_code})", flush=True)
            time.sleep(args.cooldown)

        summary_path = model_results_dir / "curve_summary.csv"
        rows = write_curve_summary(results, summary_path)
        failed = [r for r in results if r.status == "failed"]
        partial = [r for r in results if r.status == "partial"]
        mode = "full-sweep" if args.full_sweep else ("calibrate-legacy" if args.calibrate else "fixed")
        completed_at = datetime.now(timezone.utc).isoformat()

        manifest = build_campaign_manifest(
            image=args.image,
            device=args.device,
            depths=depths,
            context_length=max_ctx,
            final_config=asdict(config),
            tuning_log=tuning_log,
            moe_offload_curve=moe_offload_result,
            repetitions=REPETITIONS,
            prefill_tokens=PREFILL_TOKENS,
            generation_tokens=GENERATION_TOKENS,
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
            model_slug=slug,
            model_filename=model.name,
            model_architecture=gguf_metadata.get("general.architecture"),
            model_name=gguf_metadata.get("general.name"),
            model_context_length=max_ctx,
            model_moe=moe,
            run_id=run_id,
            run_completed_at=completed_at,
            mode=mode,
            environment=env,
            final_config=asdict(config),
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

    any_failed = False
    any_partial = False
    for model in models:
        slug = model_slug(model)
        run_dir = args.results_root / slug / run_id
        if (run_dir / "campaign.failed").exists():
            any_failed = True
        elif (run_dir / "campaign.partial").exists():
            any_partial = True
    if any_failed:
        sys.exit(1)
    if any_partial:
        sys.exit(2)


if __name__ == "__main__":
    main()
