"""Application use case: sweep --n-cpu-moe across depths to find the MoE
offload boundary, either via evenly-spaced candidates (quick mode) or
binary search (thorough mode).

Extracted from run_bench.py's probe_moe_offload()/sweep_moe_offload_quick()/
sweep_moe_offload_thorough(). The thorough-mode boundary search itself is
pure domain logic living in domain.offload_bisection.resolve_boundary(); this
module drives that generator and performs the actual probe I/O via
application.run_curve.run_one().
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

from adapters.outbound.campaign_store import mean_ts
from application.run_curve import ProgressTracker, run_one
from domain.models import BenchConfig
from domain.offload_bisection import extra_throughput_samples, resolve_boundary
from domain.planning import quick_moe_candidates, thorough_max_probes_per_depth

MOE_QUICK_CANDIDATE_COUNT = 5  # evenly spaced --n-cpu-moe candidates for quick mode
MOE_EXTRA_SAMPLE_COUNT = 3     # extra throughput samples above the found boundary,
                                # for the thorough mode's "how hard does tps dive" curve


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
    candidates = quick_moe_candidates(block_count)
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
        search = resolve_boundary(known_fail_floor=known_fail_floor, max_offload=block_count)
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
                boundary=boundary, max_offload=block_count, sample_count=MOE_EXTRA_SAMPLE_COUNT,
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
