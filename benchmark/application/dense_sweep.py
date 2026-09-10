"""Application use case: discover dense-model GPU-layer capacity.

The search variable is ``n_cpu_layers = (block_count + 1) - ngl`` because
llama.cpp also counts the output layer. Increasing
it monotonically frees VRAM, so dense and MoE searches share the same pure
boundary generator while this module owns llama-bench probe I/O and translates
back to the user-facing ``--ngl`` value.
"""

from __future__ import annotations

import math
import time
from dataclasses import replace
from pathlib import Path

from adapters.outbound.campaign_store import mean_ts
from application.run_curve import PREFILL_TOKENS, ProgressTracker, run_one
from domain.models import BenchConfig
from domain.offload_bisection import resolve_boundary
from domain.planning import offload_boundary_max_probes
from domain.vram_estimate import estimate_full_offload_vram

DENSE_EXTRA_SAMPLE_COUNT = 3
DENSE_QUICK_EXTRA_SAMPLE_COUNT = 1


def _dense_extra_samples(
    *, boundary: int, max_offload: int, sample_count: int,
) -> tuple[int, ...]:
    """Spread exactly ``sample_count`` points through the slower CPU side.

    Include fully CPU-resident weights so even quick mode captures the two
    ends of the potentially nonlinear throughput curve.
    """
    remaining = max_offload - boundary
    if remaining <= 0 or sample_count <= 0:
        return ()
    return tuple(sorted({
        boundary + math.ceil(index * remaining / sample_count)
        for index in range(1, sample_count + 1)
    }))


def probe_dense_offload(
    *,
    image: str,
    gpu_gids: list[str],
    model: Path,
    base_config: BenchConfig,
    block_count: int,
    gpu_layers: int,
    device: str,
    depth: int,
    results_dir: Path,
    progress: ProgressTracker | None = None,
    subdir: str = "dense-tuning",
) -> tuple[bool, float]:
    """Probe one dense-model ``(depth, --ngl)`` point."""
    config = replace(
        base_config,
        block_count=block_count,
        n_cpu_layers=(block_count + 1) - gpu_layers,
    ).validate()
    result = run_one(
        image=image,
        gpu_gids=gpu_gids,
        host_model_path=model,
        series="prefill",
        config=config,
        device=device,
        depths=(depth,),
        results_dir=results_dir,
        subdir=f"{subdir}/depth-{depth}",
        progress=progress,
    )
    if result.status != "ok":
        return False, -1.0
    return True, mean_ts(result.jsonl_path)


def _estimate_record(
    *, metadata: dict, model_size_bytes: int, depth: int, gpu_vram_bytes: int,
) -> dict | None:
    if model_size_bytes <= 0 or gpu_vram_bytes <= 0:
        return None
    estimate = estimate_full_offload_vram(
        metadata=metadata,
        model_size_bytes=model_size_bytes,
        context_size=depth + PREFILL_TOKENS,
        gpu_vram_bytes=gpu_vram_bytes,
    )
    if estimate is None:
        return None
    return {
        "model_size_bytes": estimate.model_size_bytes,
        "kv_cache_bytes": estimate.kv_cache_bytes,
        "overhead_bytes": estimate.overhead_bytes,
        "total_bytes": estimate.total_bytes,
        "gpu_vram_bytes": estimate.gpu_vram_bytes,
        "full_offload_fits": estimate.fits,
    }


def _resolve_depth(
    *,
    image: str,
    gpu_gids: list[str],
    model: Path,
    base_config: BenchConfig,
    device: str,
    depth: int,
    block_count: int,
    results_dir: Path,
    cooldown: int,
    known_fail_floor: int,
    metadata: dict,
    model_size_bytes: int,
    gpu_vram_bytes: int,
    progress: ProgressTracker | None,
    subdir: str,
) -> tuple[dict, int | None, int]:
    max_gpu_layers = block_count + 1
    tested: dict[int, tuple[bool, float]] = {}
    estimate = _estimate_record(
        metadata=metadata,
        model_size_bytes=model_size_bytes,
        depth=depth,
        gpu_vram_bytes=gpu_vram_bytes,
    )
    if estimate is not None:
        predicted = "fits" if estimate["full_offload_fits"] else "does not fit"
        print(
            f"    depth={depth}: conservative estimate {predicted} "
            f"({estimate['total_bytes'] / 1024**3:.2f} GiB estimated / "
            f"{estimate['gpu_vram_bytes'] / 1024**3:.2f} GiB available)",
            flush=True,
        )

    def probe(n_cpu_layers: int) -> bool:
        if n_cpu_layers in tested:
            return tested[n_cpu_layers][0]
        gpu_layers = max_gpu_layers - n_cpu_layers
        fits, ts = probe_dense_offload(
            image=image,
            gpu_gids=gpu_gids,
            model=model,
            base_config=base_config,
            block_count=block_count,
            gpu_layers=gpu_layers,
            device=device,
            depth=depth,
            results_dir=results_dir,
            progress=progress,
            subdir=subdir,
        )
        tested[n_cpu_layers] = (fits, ts)
        detail = f"ok, ts={ts:.1f}" if fits else "FAILED"
        print(
            f"    depth={depth} ngl={gpu_layers} "
            f"(cpu_layers={n_cpu_layers}): {detail}",
            flush=True,
        )
        time.sleep(cooldown)
        return fits

    search = resolve_boundary(
        known_fail_floor=known_fail_floor,
        max_offload=max_gpu_layers,
    )
    try:
        candidate = next(search)
        while True:
            candidate = search.send(probe(candidate))
    except StopIteration as stop:
        boundary, _ = stop.value

    point = {
        "depth": depth,
        "max_ngl_that_fits": (
            max_gpu_layers - boundary if boundary is not None else None
        ),
        "estimate": estimate,
        "results": [
            {
                "n_cpu_layers": n_cpu_layers,
                "n_gpu_layers": max_gpu_layers - n_cpu_layers,
                "status": "ok" if fit_and_ts[0] else "failed",
                "avg_ts": fit_and_ts[1] if fit_and_ts[0] else None,
            }
            for n_cpu_layers, fit_and_ts in sorted(tested.items())
        ],
    }
    return point, boundary, len(tested)


def preflight_dense_offload(
    *,
    image: str,
    gpu_gids: list[str],
    model: Path,
    base_config: BenchConfig,
    device: str,
    depth: int,
    block_count: int,
    results_dir: Path,
    cooldown: int,
    metadata: dict,
    model_size_bytes: int,
    gpu_vram_bytes: int,
    progress: ProgressTracker | None = None,
) -> int | None:
    """Resolve a deepest-depth-safe ``--ngl`` before default auto-tuning."""
    point, _boundary, probes = _resolve_depth(
        image=image,
        gpu_gids=gpu_gids,
        model=model,
        base_config=base_config,
        device=device,
        depth=depth,
        block_count=block_count,
        results_dir=results_dir,
        cooldown=cooldown,
        known_fail_floor=-1,
        metadata=metadata,
        model_size_bytes=model_size_bytes,
        gpu_vram_bytes=gpu_vram_bytes,
        progress=progress,
        subdir="dense-preflight",
    )
    if progress is not None:
        budget = offload_boundary_max_probes(block_count + 1)
        progress.prune(
            budget - probes,
            "dense preflight resolved below its conservative bound",
        )
    return point["max_ngl_that_fits"]


def sweep_dense_offload(
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
    metadata: dict,
    model_size_bytes: int,
    gpu_vram_bytes: int,
    quick: bool = False,
    progress: ProgressTracker | None = None,
) -> dict:
    """Find every depth's exact maximum fitting ``--ngl`` and sample TPS."""
    sample_count = DENSE_QUICK_EXTRA_SAMPLE_COUNT if quick else DENSE_EXTRA_SAMPLE_COUNT
    per_depth_budget = offload_boundary_max_probes(block_count + 1) + sample_count
    known_fail_floor = -1
    by_depth = []
    probe_count = 0

    for depth_index, depth in enumerate(depths):
        point, boundary, probes = _resolve_depth(
            image=image,
            gpu_gids=gpu_gids,
            model=model,
            base_config=base_config,
            device=device,
            depth=depth,
            block_count=block_count,
            results_dir=results_dir,
            cooldown=cooldown,
            known_fail_floor=known_fail_floor,
            metadata=metadata,
            model_size_bytes=model_size_bytes,
            gpu_vram_bytes=gpu_vram_bytes,
            progress=progress,
            subdir="dense-tuning",
        )
        probe_count += probes
        by_depth.append(point)
        if boundary is None:
            print(
                f"    depth={depth}: no --ngl fits; KV/cache allocations exceed VRAM",
                flush=True,
            )
            for deeper_depth in depths[depth_index + 1:]:
                by_depth.append({
                    "depth": deeper_depth,
                    "max_ngl_that_fits": None,
                    "estimate": _estimate_record(
                        metadata=metadata,
                        model_size_bytes=model_size_bytes,
                        depth=deeper_depth,
                        gpu_vram_bytes=gpu_vram_bytes,
                    ),
                    "results": [],
                    "inferred_infeasible_from_depth": depth,
                })
            break
        known_fail_floor = boundary - 1

    max_gpu_layers = block_count + 1
    offload_needed = any(
        point["max_ngl_that_fits"] != max_gpu_layers for point in by_depth
    )
    if offload_needed and by_depth and by_depth[-1]["max_ngl_that_fits"] is not None:
        for point in by_depth:
            max_ngl = point["max_ngl_that_fits"]
            if max_ngl is None:
                continue
            boundary = max_gpu_layers - max_ngl
            tested_cpu_layers = {
                result["n_cpu_layers"] for result in point["results"]
            }
            for n_cpu_layers in _dense_extra_samples(
                boundary=boundary,
                max_offload=max_gpu_layers,
                sample_count=sample_count,
            ):
                if n_cpu_layers in tested_cpu_layers:
                    continue
                gpu_layers = max_gpu_layers - n_cpu_layers
                fits, ts = probe_dense_offload(
                    image=image,
                    gpu_gids=gpu_gids,
                    model=model,
                    base_config=base_config,
                    block_count=block_count,
                    gpu_layers=gpu_layers,
                    device=device,
                    depth=point["depth"],
                    results_dir=results_dir,
                    progress=progress,
                    subdir="dense-tuning",
                )
                probe_count += 1
                detail = f"ok, ts={ts:.1f}" if fits else "FAILED"
                print(
                    f"    depth={point['depth']} ngl={gpu_layers} "
                    f"(cpu_layers={n_cpu_layers}): {detail}",
                    flush=True,
                )
                point["results"].append({
                    "n_cpu_layers": n_cpu_layers,
                    "n_gpu_layers": gpu_layers,
                    "status": "ok" if fits else "failed",
                    "avg_ts": ts if fits else None,
                })
                time.sleep(cooldown)
            point["results"].sort(key=lambda result: result["n_cpu_layers"])

    if progress is not None:
        progress.prune(
            len(depths) * per_depth_budget - probe_count,
            "dense offload discovery resolved below its conservative bound",
        )

    deepest_ngl = next(
        (point["max_ngl_that_fits"] for point in reversed(by_depth)
         if point["max_ngl_that_fits"] is not None),
        None,
    )
    return {
        "mode": "quick" if quick else "thorough",
        "block_count": block_count,
        "max_gpu_layers": block_count + 1,
        "final_ngl": deepest_ngl,
        "offload_needed": offload_needed,
        "by_depth": by_depth,
    }
