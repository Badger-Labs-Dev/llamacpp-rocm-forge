#!/usr/bin/env python3
"""Recommend llama-bench parameters (ubatch) from a run_bench.py results dir.

Reads curve_summary.csv (one row per model/series/ubatch/depth, see
run_bench.py) and prints, per model:

  - the "calibration winner" ubatch: whichever run_bench.py's --calibrate
    actually picked (highest depth-0 prefill throughput at calibration time)
  - the "mean-curve winner": ubatch with the best average prefill throughput
    across *all* tested depths, not just depth 0
  - the "worst-case winner": ubatch with the best throughput at the deepest
    tested context, i.e. the safest choice if you mostly run long contexts

These three usually agree. When they don't, calibration (depth-0-only) may
be picking a ubatch that looks good on a cold/short prompt but is not
actually the best choice once the KV cache is deep - this script exists to
catch that instead of blindly trusting the calibration pass.

Usage:
    ./recommend_settings.py results/20260908T144845Z
    ./recommend_settings.py results/20260908T144845Z/curve_summary.csv
    ./recommend_settings.py results/20260908T144845Z --json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    csv_path = path / "curve_summary.csv" if path.is_dir() else path
    if not csv_path.is_file():
        sys.exit(f"No curve_summary.csv found at {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["ubatch"] = int(row["ubatch"])
        row["n_depth"] = int(row["n_depth"])
        row["avg_ts"] = float(row["avg_ts"]) if row["avg_ts"] not in ("", None) else None
    return rows


def analyze_model(model: str, rows: list[dict]) -> dict:
    prefill = [r for r in rows if r["model"] == model and r["series"] == "prefill" and r["status"] == "ok"]
    generation = [r for r in rows if r["model"] == model and r["series"] == "generation" and r["status"] == "ok"]

    if not prefill:
        return {"model": model, "error": "no successful prefill rows"}

    ubatches = sorted({r["ubatch"] for r in prefill})
    depths = sorted({r["n_depth"] for r in prefill})
    max_depth = max(depths)
    min_depth = min(depths)

    by_ubatch = defaultdict(dict)  # ubatch -> {depth: ts}
    for r in prefill:
        by_ubatch[r["ubatch"]][r["n_depth"]] = r["avg_ts"]

    depth0_by_ubatch = {ub: vals.get(min_depth) for ub, vals in by_ubatch.items()}
    mean_by_ubatch = {ub: sum(vals.values()) / len(vals) for ub, vals in by_ubatch.items() if vals}
    worst_by_ubatch = {ub: vals.get(max_depth) for ub, vals in by_ubatch.items()}

    def best(d: dict) -> int | None:
        valid = {k: v for k, v in d.items() if v is not None}
        return max(valid, key=valid.get) if valid else None

    depth0_winner = best(depth0_by_ubatch)
    mean_winner = best(mean_by_ubatch)
    worst_winner = best(worst_by_ubatch)

    agree = len({depth0_winner, mean_winner, worst_winner}) == 1

    gen_summary = None
    if generation:
        gen_ubatches = sorted({r["ubatch"] for r in generation})
        gen_by_depth = {r["n_depth"]: r["avg_ts"] for r in generation}
        gen_summary = {
            "ubatch_tested": gen_ubatches,
            "depth0_ts": gen_by_depth.get(min(gen_by_depth)) if gen_by_depth else None,
            "worst_depth_ts": gen_by_depth.get(max(gen_by_depth)) if gen_by_depth else None,
        }

    return {
        "model": model,
        "ubatches_tested": ubatches,
        "depths_tested": depths,
        "depth0_winner": depth0_winner,
        "mean_curve_winner": mean_winner,
        "worst_case_winner": worst_winner,
        "winners_agree": agree,
        "recommended_ubatch": mean_winner,  # prefer the whole-curve view over depth-0-only
        "prefill_by_ubatch": {
            ub: {
                "depth0_ts": depth0_by_ubatch.get(ub),
                "mean_ts": mean_by_ubatch.get(ub),
                "worst_depth_ts": worst_by_ubatch.get(ub),
            }
            for ub in ubatches
        },
        "generation": gen_summary,
    }


def print_report(result: dict) -> None:
    model = result["model"]
    print(f"== {model} ==")
    if "error" in result:
        print(f"  {result['error']}")
        return

    print(f"  ubatches tested: {result['ubatches_tested']}")
    print(f"  depths tested:   {result['depths_tested']}")
    print()
    print(f"  {'ubatch':>8}  {'depth0 ts':>12}  {'mean ts':>12}  {'worst-depth ts':>15}")
    for ub, stats in result["prefill_by_ubatch"].items():
        d0 = f"{stats['depth0_ts']:.1f}" if stats["depth0_ts"] is not None else "n/a"
        mn = f"{stats['mean_ts']:.1f}" if stats["mean_ts"] is not None else "n/a"
        wc = f"{stats['worst_depth_ts']:.1f}" if stats["worst_depth_ts"] is not None else "n/a"
        marker = " *" if ub == result["recommended_ubatch"] else ""
        print(f"  {ub:>8}  {d0:>12}  {mn:>12}  {wc:>15}{marker}")

    print()
    print(f"  depth-0 (calibration-style) winner: ubatch={result['depth0_winner']}")
    print(f"  mean-across-curve winner:            ubatch={result['mean_curve_winner']}")
    print(f"  worst-case (deepest context) winner: ubatch={result['worst_case_winner']}")
    if result["winners_agree"]:
        print(f"  -> all three agree: recommend ubatch={result['recommended_ubatch']}")
    else:
        print(
            "  -> DISAGREEMENT: depth-0-only calibration may not reflect real "
            f"usage. Recommending mean-across-curve winner: ubatch={result['recommended_ubatch']}"
        )
        print(
            "     Consider re-running --calibrate with attention to your actual "
            "typical context depth, not just depth 0."
        )

    if result["generation"]:
        gen = result["generation"]
        print()
        print(f"  generation series (ubatch(es) tested: {gen['ubatch_tested']}):")
        if gen["depth0_ts"] is not None:
            print(f"    depth-0 tok/s:        {gen['depth0_ts']:.1f}")
        if gen["worst_depth_ts"] is not None:
            print(f"    deepest-depth tok/s:  {gen['worst_depth_ts']:.1f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_path", type=Path, help="Results dir or path to curve_summary.csv")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of a report")
    args = parser.parse_args()

    rows = load_rows(args.results_path)
    models = sorted({r["model"] for r in rows})
    results = [analyze_model(model, rows) for model in models]

    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        for result in results:
            print_report(result)


if __name__ == "__main__":
    main()
