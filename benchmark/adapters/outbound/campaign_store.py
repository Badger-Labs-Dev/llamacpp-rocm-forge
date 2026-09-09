"""Filesystem persistence adapter: writes curve_summary.csv,
campaign_manifest.json, metadata.json, and campaign status markers for one
model's benchmark run.

Moved out of run_bench.py's main() and write_curve_summary() (behavior-
preserving extraction, see docs/architecture.md's migration rule) - the
manifest/metadata dict-building functions are pure and separated from the
actual file writes so they can be tested without touching a filesystem.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from adapters.outbound.model_resolution import model_key


def load_jsonl_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def mean_ts(jsonl_path: Path) -> float:
    rows = load_jsonl_rows(jsonl_path)
    values = [r.get("avg_ts") for r in rows if r.get("avg_ts") is not None]
    return sum(values) / len(values) if values else -1.0


def write_curve_summary(results: list, summary_path: Path) -> int:
    fieldnames = [
        "model", "series", "ubatch", "batch", "ctk", "ctv", "flash_attn", "n_cpu_moe",
        "n_depth", "n_prompt", "n_gen", "avg_ts", "avg_ns", "status",
    ]
    row_count = 0
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            rows = load_jsonl_rows(result.jsonl_path)
            for row in rows:
                # A row's own depth may have completed successfully even
                # when a *later*, deeper depth in the same series failed or
                # timed out (result.status == "partial" reflects the whole
                # series). Stamping every row with the series-level status
                # would wrongly mark a genuinely successful depth-0 sample
                # as "partial", hiding it from consumers (e.g. the viewer's
                # depth0_throughput()) that only trust "ok" rows. depths_run
                # only ever contains depths whose probe returned rc=0, so a
                # row's depth being in it is a reliable per-row signal.
                row_status = "ok" if row.get("n_depth") in result.depths_run else result.status
                writer.writerow({
                    "model": model_key(result.model),
                    "series": result.series,
                    "ubatch": result.config.ubatch,
                    "batch": result.config.batch,
                    "ctk": result.config.ctk,
                    "ctv": result.config.ctv,
                    "flash_attn": result.config.flash_attn,
                    "n_cpu_moe": result.config.n_cpu_moe,
                    "n_depth": row.get("n_depth"),
                    "n_prompt": row.get("n_prompt"),
                    "n_gen": row.get("n_gen"),
                    "avg_ts": row.get("avg_ts"),
                    "avg_ns": row.get("avg_ns"),
                    "status": row_status,
                })
                row_count += 1
    return row_count


def build_campaign_manifest(
    *,
    image: str,
    device: str,
    depths: tuple[int, ...],
    context_length: int | None,
    final_config: dict,
    tuning_log: list[dict],
    moe_offload_curve: dict | None,
    repetitions: int,
    prefill_tokens: int,
    generation_tokens: int,
    mode: str,
    completed_at: str,
    summary_rows: int,
    run_summaries: list[dict],
) -> dict:
    """Pure construction of campaign_manifest.json's contents.

    run_summaries is a list of already-built dicts (series/config/status/
    return_code/depths_run/depths_skipped/stop_reason) - keeping this
    module's only dependency on RunResult's *shape*, not the BenchConfig/
    dataclass types themselves.
    """
    return {
        "image": image,
        "device": device,
        "depths": list(depths),
        "context_length": context_length,
        "final_config": final_config,
        "tuning_log": tuning_log,
        "moe_offload_curve": moe_offload_curve,
        "repetitions": repetitions,
        "prefill_tokens": prefill_tokens,
        "generation_tokens": generation_tokens,
        "mode": mode,
        "completed_at": completed_at,
        "summary_rows": summary_rows,
        "runs": run_summaries,
    }


def build_run_metadata(
    *,
    model_slug: str,
    model_filename: str,
    model_architecture: str | None,
    model_name: str | None,
    model_context_length: int | None,
    model_moe: dict | None,
    run_id: str,
    run_completed_at: str,
    mode: str,
    environment: dict,
    final_config: dict,
    depths_tested: tuple[int, ...],
    generation_tok_s_mean: float | None,
    status: str,
) -> dict:
    """Pure construction of metadata.json's contents."""
    return {
        "model_slug": model_slug,
        "model_filename": model_filename,
        "model_architecture": model_architecture,
        "model_name": model_name,
        "model_context_length": model_context_length,
        "model_moe": model_moe,
        "run_id": run_id,
        "run_completed_at": run_completed_at,
        "mode": mode,
        "environment": environment,
        "final_config": final_config,
        "depths_tested": list(depths_tested),
        "generation_tok_s_mean": generation_tok_s_mean,
        "status": status,
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_status_marker(model_results_dir: Path, *, failed: bool, partial: bool) -> Path:
    marker = "campaign.failed" if failed else ("campaign.partial" if partial else "campaign.finished")
    marker_path = model_results_dir / marker
    marker_path.touch()
    return marker_path
