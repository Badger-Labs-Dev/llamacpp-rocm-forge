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


def kv_selection_tuning_log_entry(selection: KvSelection) -> dict[str, object]:
    """Publish a ``select_kv_config`` result as a legacy-shaped tuning_log
    entry: ``{"stage", "scores", "winner"}``.

    The pre-KV-CACHE-REFACTOR viewer contract (application/viewer_dataset.py's
    _check_tuning_stage, viewer/src/domain/sensitivity.ts) only understands
    that shape - not KvSelection's own probes/target_depth record. Only the
    *final target depth's* probes are one comparable set (see
    select_kv_config's depth-then-throughput selection): scores from a
    shallower depth a dtype was never probed at would misrepresent that
    dtype as untested-and-worst rather than not-part-of-this-comparison, so
    probes at any other depth are intentionally excluded here. A probe that
    didn't succeed is scored with the same -1.0 "no usable rows" sentinel
    ``select_kv_config``/``mean_ts`` already use, never ``None`` - the
    viewer's sensitivity math requires every candidate to be a finite
    number and treats negative scores as simply non-comparable, not
    missing.
    """
    scores = {
        record["ctk"]: (
            record["avg_ts"] if record["status"] == "ok" and record["avg_ts"] is not None else -1.0
        )
        for record in selection.probes
        if record["depth"] == selection.target_depth
    }
    return {
        "stage": "kv_cache_dtype",
        "scores": scores,
        "winner": selection.config.ctk,
    }
