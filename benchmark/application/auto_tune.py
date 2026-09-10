"""Application use case: auto-tune the KV-cache dtype at the fixed runtime
configuration.

Extracted from run_bench.py's probe_depths()/probe_config()/auto_tune().
Depends on application.run_curve.run_one() for the actual probe
execution, and domain.models.BenchConfig for the configuration model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from adapters.outbound.campaign_store import mean_ts
from application.run_curve import ProgressTracker, run_one
from domain.kv_depth_planner import KV_CACHE_TYPES, DtypeDepthPlan, KvDepthPlan
from domain.models import BenchConfig, RunResult


@dataclass(frozen=True)
class KvSelection:
    """The deepest successful cache dtype and its planned curve envelope."""

    config: BenchConfig
    target_depth: int
    runnable_depths: tuple[int, ...]
    probes: tuple[dict[str, object], ...]


def select_kv_config(
    *,
    plan: KvDepthPlan,
    probe: Callable[[BenchConfig, int], RunResult],
    mean_throughput: Callable[[Path], float] = mean_ts,
) -> KvSelection | None:
    """Select by depth coverage first, then measured throughput.

    For each requested runnable depth, highest first, probe canonical cache
    dtypes in q4 -> q8 -> f16 order. Only successful probes at the same depth
    compete on throughput; an unsuccessful result remains runtime evidence and
    does not modify the static plan.
    """
    dtype_by_name = {dtype_plan.ctk: dtype_plan for dtype_plan in plan.dtype_plans}
    all_probe_records: list[dict[str, object]] = []
    for depth in sorted({depth for item in plan.dtype_plans for depth in item.runnable_depths}, reverse=True):
        successes: list[tuple[float, BenchConfig, DtypeDepthPlan]] = []
        probe_records: list[dict[str, object]] = []
        for ctk in KV_CACHE_TYPES:
            dtype_plan = dtype_by_name.get(ctk)
            if dtype_plan is None or depth not in dtype_plan.runnable_depths:
                continue
            config = BenchConfig(batch=2048, ubatch=2048, ctk=ctk, ctv=ctk).validate()
            result = probe(config, depth)
            score = mean_throughput(result.jsonl_path) if result.status == "ok" else None
            record = {
                "depth": depth, "ctk": ctk, "ctv": ctk,
                "status": result.status, "avg_ts": score,
            }
            probe_records.append(record)
            all_probe_records.append(record)
            if score is not None and score >= 0:
                successes.append((score, config, dtype_plan))
        if successes:
            score, config, dtype_plan = max(successes, key=lambda candidate: candidate[0])
            return KvSelection(
                config=config,
                target_depth=depth,
                runnable_depths=dtype_plan.runnable_depths,
                probes=tuple(all_probe_records),
            )
    return None


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
    block_count: int | None = None,
    n_cpu_layers: int = 0,
    progress: ProgressTracker | None = None,
) -> BenchConfig:
    """Tune KV cache dtype at fixed batch=2048, ubatch=2048, and
    flash-attn=auto. Each dtype probes at the shallowest and deepest depth
    rather than the full curve. For a MoE full-sweep, n_cpu_moe is
    conservatively set to block_count so the KV probes can evaluate their
    intended knob without an unrelated high-context MoE OOM; the separate
    MoE curve later measures every candidate with the winning base config.
    """
    probe_d = probe_depths(depths)
    print(f"  auto-tune: probing at depths {list(probe_d)}", flush=True)

    # Stage 1: KV cache dtype, flash-attn always "auto" (quantized KV
    # cache requires FA regardless, and validate() enforces that).
    print("  [KV cache dtype] fixed batch=2048 ubatch=2048", flush=True)
    kv_scores = {}
    for kv in KV_CACHE_TYPES:
        cfg = BenchConfig(ubatch=2048, batch=2048, ctk=kv, ctv=kv, n_cpu_moe=n_cpu_moe,
                          block_count=block_count, n_cpu_layers=n_cpu_layers).validate()
        score = probe_config(image=image, gpu_gids=gpu_gids, model=model, config=cfg,
                             device=device, depths=probe_d, results_dir=results_dir, progress=progress)
        kv_scores[kv] = score
        print(f"    kv={kv}: mean_ts={score:.1f}", flush=True)
        time.sleep(cooldown)
    best_kv = max(kv_scores, key=lambda k: kv_scores[k])
    log.append({"stage": "kv_cache_dtype", "scores": kv_scores, "winner": best_kv})
    print(f"  KV cache dtype winner: kv={best_kv}", flush=True)

    return BenchConfig(ubatch=2048, batch=2048, ctk=best_kv, ctv=best_kv,
                       n_cpu_moe=n_cpu_moe, block_count=block_count,
                       n_cpu_layers=n_cpu_layers).validate()
