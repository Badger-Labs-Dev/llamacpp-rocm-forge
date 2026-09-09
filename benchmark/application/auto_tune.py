"""Application use case: staged (coordinate-descent) auto-tune across KV
cache dtype and the ubatch/batch grid.

Extracted from run_bench.py's probe_depths()/probe_config()/auto_tune().
Depends on application.run_curve.run_one() for the actual probe
execution, and domain.models.BenchConfig for the configuration model.
"""

from __future__ import annotations

import time
from pathlib import Path

from adapters.outbound.campaign_store import mean_ts
from application.run_curve import ProgressTracker, run_one
from domain.models import BenchConfig

UBATCH_CANDIDATES = (256, 512, 1024, 2048)
BATCH_CANDIDATES = (512, 1024, 2048, 4096)
KV_CACHE_TYPES = ("f16", "q8_0", "q4_0")


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


def valid_batch_grid_count() -> int:
    return sum(ub <= batch for ub in UBATCH_CANDIDATES for batch in BATCH_CANDIDATES)
