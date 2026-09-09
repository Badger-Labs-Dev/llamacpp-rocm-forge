#!/usr/bin/env python3
"""Aggregate every results/<model-slug>/<run-id>/ directory into one JSON
file the static viewer (viewer/) reads.

For each (model, run_id) pair, reads metadata.json (summary + environment),
campaign_manifest.json (tuning_log - the per-stage sensitivity data), and
curve_summary.csv (throughput vs depth for the winning config). Only
the default (auto-tune) mode has a tuning_log; --quick runs still
contribute their version-over-time datapoint but no sensitivity data.

Usage:
    ./generate_viewer_data.py [--results-root results] [--output ../viewer/public/results.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from application.viewer_dataset import SCHEMA_VERSION, validate_viewer_dataset

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = SCRIPT_DIR.parent / "results"
DEFAULT_OUTPUT = SCRIPT_DIR.parent / "viewer" / "public" / "results.json"


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def read_curve(path: Path) -> list[dict]:
    """Read curve_summary.csv into a list of rows with numeric fields
    coerced (they arrive as strings from csv.DictReader)."""
    if not path.is_file():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                avg_ts = float(row["avg_ts"]) if row["avg_ts"] else None
                # json.dumps emits the bare (non-JSON-spec) tokens NaN/
                # Infinity/-Infinity for these values; llama-bench never
                # legitimately produces them, so normalize to null rather
                # than publishing invalid JSON in results.json.
                if avg_ts is not None and not math.isfinite(avg_ts):
                    avg_ts = None
                rows.append({
                    "series": row["series"],
                    "n_depth": int(row["n_depth"]),
                    "avg_ts": avg_ts,
                    "status": row["status"],
                })
            except (KeyError, ValueError):
                continue
    return rows


def depth0_throughput(curve: list[dict], series: str) -> float | None:
    """Pull out the depth-0 avg_ts for a series - the cleanest single
    summary number for the version-over-time chart."""
    for row in curve:
        if row["series"] == series and row["n_depth"] == 0 and row["status"] == "ok":
            return row["avg_ts"]
    return None


def build_run_entry(model_dir: Path, run_dir: Path) -> dict | None:
    metadata = read_json(run_dir / "metadata.json")
    if metadata is None:
        return None  # incomplete/interrupted run, skip rather than error
    manifest = read_json(run_dir / "campaign_manifest.json") or {}
    curve = read_curve(run_dir / "curve_summary.csv")

    return {
        "run_id": metadata.get("run_id"),
        "run_completed_at": metadata.get("run_completed_at"),
        "status": metadata.get("status"),
        "mode": metadata.get("mode"),
        "environment": metadata.get("environment", {}),
        "final_config": metadata.get("final_config", {}),
        "depths_tested": metadata.get("depths_tested", []),
        "prefill_depth0_ts": depth0_throughput(curve, "prefill"),
        "generation_depth0_ts": depth0_throughput(curve, "generation"),
        "curve": curve,
        "tuning_log": manifest.get("tuning_log", []),
        "moe_offload_curve": manifest.get("moe_offload_curve"),
        "dense_offload_curve": manifest.get("dense_offload_curve"),
    }


def build_model_entry(model_dir: Path) -> dict | None:
    run_entries = []
    for run_dir in sorted(model_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        entry = build_run_entry(model_dir, run_dir)
        if entry is not None:
            run_entries.append(entry)
    if not run_entries:
        return None

    # Sort runs by completion time so the version-over-time chart has a
    # sensible default x-axis order even though run_id itself encodes
    # versions, not dates.
    run_entries.sort(key=lambda r: r.get("run_completed_at") or "")

    # Model-level metadata (name/architecture/context_length) is the same
    # across every run of the same model - pull it from any one run's
    # metadata.json rather than re-deriving it.
    sample_meta = read_json(model_dir / run_entries[-1]["run_id"] / "metadata.json") or {}

    return {
        "model_slug": model_dir.name,
        "model_filename": sample_meta.get("model_filename"),
        "model_architecture": sample_meta.get("model_architecture"),
        "model_name": sample_meta.get("model_name"),
        "model_context_length": sample_meta.get("model_context_length"),
        "runs": run_entries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.results_root.is_dir():
        raise SystemExit(f"Results root not found: {args.results_root}")

    models = []
    for model_dir in sorted(args.results_root.iterdir()):
        if not model_dir.is_dir():
            continue
        entry = build_model_entry(model_dir)
        if entry is not None:
            models.append(entry)

    output = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": models,
    }
    validate_viewer_dataset(output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({len(models)} model(s))")


if __name__ == "__main__":
    main()
