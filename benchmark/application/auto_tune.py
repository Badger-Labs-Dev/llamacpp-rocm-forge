"""Application use case: select the KV-cache dtype/depth to run final
curves at, using the pure per-dtype depth plan from
domain.kv_depth_planner.

Depends on application.run_curve.run_one() (via the injected ``probe``
callable) for the actual probe execution, and domain.models.BenchConfig
for the configuration model. Supersedes the old two-point auto_tune()
KV-dtype probe, which treated every non-f16-deepest failure as fatal
before q4/q8 got a chance (see KV-CACHE-REFACTOR-05).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from adapters.outbound.campaign_store import mean_ts
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
            # mean_throughput uses -1.0 as its own "no usable rows" sentinel
            # (see adapters.outbound.campaign_store.mean_ts) even when the
            # probe's own status is "ok" (e.g. an empty JSONL) - reject that
            # the same way as an unsuccessful probe rather than letting a
            # negative throughput win a comparison.
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
