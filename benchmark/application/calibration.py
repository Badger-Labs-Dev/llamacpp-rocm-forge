"""Application use case: legacy --calibrate ubatch selection.

Extracted from run_bench.py's pick_best_ubatch(). Reads already-written
prefill JSONL rows for a set of ubatch candidates and picks the one with
the highest depth-0 throughput.
"""

from __future__ import annotations

from pathlib import Path

from adapters.outbound.campaign_store import load_jsonl_rows
from adapters.outbound.model_resolution import model_key
from domain.models import BenchConfig


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
