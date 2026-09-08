#!/usr/bin/env python3
"""Run a llama-bench depth/ubatch sweep against a Docker container on this host.

Standalone replacement for upstream's run_calibrated_campaign.py, which
assumes their Toolbx/Cockpit stack and SSH hosts. This drives `docker run` +
`llama-bench` directly and writes one JSONL file per (model, series, ubatch)
combination plus a curve_summary.csv, mirroring the shape of upstream's
output closely enough to stay comparable.

Depths (llama-bench's "-d", how full the KV cache is before the timed run)
are derived per model from its GGUF *.context_length metadata (see
gguf_info.py): every entry in COMMON_CONTEXT_SIZES that fits under the
model's trained context length gets tested, plus depth 0. This replaces a
single fixed depth list applied to every model regardless of what it
actually supports - testing a model past its trained context produces
throughput numbers but not meaningful ones, since positions past that point
were never seen in training. If a model's context_length can't be read,
falls back to LEGACY_FIXED_DEPTHS (upstream's original fixed list, tuned for
Strix Halo's 128GB unified memory ceiling).

Other settings held fixed (do not vary within one comparison run), also
taken from amd-strix-halo-toolboxes' benchmark/toolbox_performance/
HOW_TO_RUN_BENCHMARKS_AGENTS.md:

    prefill tokens    2048
    generation tokens 128
    batch             2048
    repetitions       3
    cooldown          10s between test invocations
    GPU layers        99
    flash attention   enabled
    mmap              disabled
    KV cache quant    disabled

Usage:
    ./run_bench.py --model /models/foo.gguf --image r9700-llm-bench:rocm-7.2.4

Calibration (picking the best ubatch for a model) is a separate, explicit
first pass: run with --calibrate to sweep all ubatch candidates on the
prefill series only, then re-run without it (default) using the discovered
best ubatch for both prefill and generation series.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from gguf_info import context_length, read_gguf_metadata

# Common context-window sizes seen across model releases (powers of two, plus
# the odd-but-common 24576/49152 seen in some Qwen configs). Depths are
# derived from whichever of these fit under a model's trained
# *.context_length, rather than a single fixed list applied to every model
# regardless of what it actually supports - see gguf_info.py.
COMMON_CONTEXT_SIZES = (
    2048, 4096, 8192, 16384, 24576, 32768, 49152, 65536, 98304, 131072, 262144,
)
LEGACY_FIXED_DEPTHS = (0, 8192, 16384, 24576, 32768, 40960, 49152, 57344, 65536)

UBATCH_CANDIDATES = (256, 512, 1024, 2048)
PREFILL_TOKENS = 2048
GENERATION_TOKENS = 128
BATCH_SIZE = 2048
REPETITIONS = 3
GPU_LAYERS = 99
COOLDOWN_SECONDS = 10


@dataclass
class RunResult:
    model: str
    series: str  # "prefill" or "generation"
    ubatch: int
    status: str  # "ok", "failed", "skipped"
    return_code: int
    jsonl_path: Path
    stderr_path: Path
    command: list[str]


def model_key(model_path: str) -> str:
    return Path(model_path).name


def derive_depths_for_model(model_path: Path) -> tuple[tuple[int, ...], int | None]:
    """Pick depths to test for this model: common context sizes that fit
    under its trained context_length, converted to llama-bench "-d" values.

    llama-bench's -d is how much KV cache is already "full" before the timed
    prefill/generation runs, so a context size C is tested at depth
    C - PREFILL_TOKENS (the prefill run itself consumes PREFILL_TOKENS more).
    Depth 0 (a cold/empty cache) is always included regardless of the
    model's context length.

    Returns (depths, model_context_length). model_context_length is None if
    it couldn't be read from the GGUF (falls back to LEGACY_FIXED_DEPTHS).
    """
    try:
        metadata = read_gguf_metadata(model_path)
        max_ctx = context_length(metadata)
    except (OSError, ValueError):
        max_ctx = None

    if max_ctx is None:
        return LEGACY_FIXED_DEPTHS, None

    depths = {0}
    for ctx_size in COMMON_CONTEXT_SIZES:
        if ctx_size > max_ctx:
            break
        depth = max(0, ctx_size - PREFILL_TOKENS)
        depths.add(depth)
    return tuple(sorted(depths)), max_ctx


def docker_gpu_args(gpu_gids: list[str]) -> list[str]:
    args = ["--device", "/dev/dri", "--device", "/dev/kfd"]
    for gid in gpu_gids:
        args += ["--group-add", gid]
    args += ["--security-opt", "seccomp=unconfined", "--ipc=host"]
    return args


def build_llama_bench_command(
    *,
    model_container_path: str,
    series: str,
    ubatch: int,
    device: str,
    depths: tuple[int, ...],
) -> list[str]:
    cmd = [
        "llama-bench",
        "-m", model_container_path,
        "-o", "jsonl",
        "-oe", "jsonl",
        "-r", str(REPETITIONS),
        "-b", str(BATCH_SIZE),
        "-ub", str(ubatch),
        "-ngl", str(GPU_LAYERS),
        "-fa", "1",
        "-mmp", "0",
        "-ctk", "f16",
        "-ctv", "f16",
        "-dev", device,
        "-d", ",".join(str(d) for d in depths),
        "--progress",
    ]
    if series == "prefill":
        cmd += ["-p", str(PREFILL_TOKENS), "-n", "0"]
    else:
        cmd += ["-p", "0", "-n", str(GENERATION_TOKENS)]
    return cmd


def run_one(
    *,
    image: str,
    gpu_gids: list[str],
    host_model_path: Path,
    series: str,
    ubatch: int,
    device: str,
    depths: tuple[int, ...],
    results_dir: Path,
) -> RunResult:
    container_model_path = f"/models/{host_model_path.name}"
    out_name = f"{model_key(str(host_model_path))}__{series}__ub{ubatch}"
    jsonl_path = results_dir / f"{out_name}.jsonl"
    stderr_path = results_dir / f"{out_name}.stderr.log"

    bench_cmd = build_llama_bench_command(
        model_container_path=container_model_path,
        series=series,
        ubatch=ubatch,
        device=device,
        depths=depths,
    )

    docker_cmd = [
        "docker", "run", "--rm",
        *docker_gpu_args(gpu_gids),
        "-v", f"{host_model_path.parent}:/models:ro",
        image,
        *bench_cmd,
    ]

    print(f"  $ {' '.join(docker_cmd)}", flush=True)

    with jsonl_path.open("w", encoding="utf-8") as out_f, \
         stderr_path.open("w", encoding="utf-8") as err_f:
        proc = subprocess.run(docker_cmd, stdout=out_f, stderr=err_f)

    status = "ok" if proc.returncode == 0 and jsonl_path.stat().st_size > 0 else "failed"
    return RunResult(
        model=str(host_model_path),
        series=series,
        ubatch=ubatch,
        status=status,
        return_code=proc.returncode,
        jsonl_path=jsonl_path,
        stderr_path=stderr_path,
        command=docker_cmd,
    )


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


def pick_best_ubatch(results_dir: Path, model: str, candidates: list[int]) -> int:
    """Choose the ubatch with the highest mean prefill throughput at depth 0."""
    best_ubatch = candidates[0]
    best_ts = -1.0
    for ubatch in candidates:
        path = results_dir / f"{model_key(model)}__prefill__ub{ubatch}.jsonl"
        rows = [r for r in load_jsonl_rows(path) if r.get("n_depth") == 0]
        if not rows:
            continue
        ts_values = [r.get("avg_ts") or r.get("t/s") or 0 for r in rows]
        mean_ts = sum(ts_values) / len(ts_values) if ts_values else 0
        if mean_ts > best_ts:
            best_ts = mean_ts
            best_ubatch = ubatch
    return best_ubatch


def write_curve_summary(results: list[RunResult], summary_path: Path) -> int:
    fieldnames = [
        "model", "series", "ubatch", "n_depth", "n_prompt", "n_gen",
        "avg_ts", "avg_ns", "status",
    ]
    row_count = 0
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            rows = load_jsonl_rows(result.jsonl_path)
            for row in rows:
                writer.writerow({
                    "model": model_key(result.model),
                    "series": result.series,
                    "ubatch": result.ubatch,
                    "n_depth": row.get("n_depth"),
                    "n_prompt": row.get("n_prompt"),
                    "n_gen": row.get("n_gen"),
                    "avg_ts": row.get("avg_ts"),
                    "avg_ns": row.get("avg_ns"),
                    "status": result.status,
                })
                row_count += 1
    return row_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True, help="Host path to a .gguf file (repeatable)")
    parser.add_argument("--image", default="r9700-llm-bench:rocm-7.2.4", help="Docker image to benchmark")
    parser.add_argument("--device", default="ROCm0", help="llama-bench -dev target (default: ROCm0, the R9700)")
    parser.add_argument("--gpu-gid", action="append", default=[],
                         help="Host GID(s) for /dev/dri and /dev/kfd access (repeatable); "
                              "defaults to `video` and `render` group GIDs on this host")
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--calibrate", action="store_true",
                         help="Sweep all ubatch candidates on the prefill series only, to pick a winner")
    parser.add_argument("--max-depth", type=int, default=None,
                         help="Cap derived depths at this value even if the model supports more "
                              "(useful for a quick smoke test on a long-context model)")
    parser.add_argument("--ubatch", type=int, default=None,
                         help="Skip calibration; use this ubatch for both series")
    parser.add_argument("--cooldown", type=int, default=COOLDOWN_SECONDS)
    return parser.parse_args()


def default_gpu_gids() -> list[str]:
    import grp
    gids = []
    for name in ("video", "render"):
        try:
            gids.append(str(grp.getgrnam(name).gr_gid))
        except KeyError:
            pass
    return gids


def main() -> None:
    args = parse_args()
    gpu_gids = args.gpu_gid or default_gpu_gids()
    if not gpu_gids:
        sys.exit("Could not resolve video/render group GIDs; pass --gpu-gid explicitly")

    if args.results_dir.exists() and any(args.results_dir.iterdir()):
        sys.exit(f"Refusing to write into non-empty results dir: {args.results_dir}")
    args.results_dir.mkdir(parents=True, exist_ok=True)

    models = [Path(m).resolve() for m in args.model]
    for model in models:
        if not model.is_file():
            sys.exit(f"Model file not found: {model}")

    results: list[RunResult] = []
    depths_by_model: dict[str, tuple[int, ...]] = {}
    context_length_by_model: dict[str, int | None] = {}

    for model in models:
        print(f"== {model.name} ==", flush=True)

        depths, max_ctx = derive_depths_for_model(model)
        if args.max_depth is not None:
            depths = tuple(d for d in depths if d <= args.max_depth) or (0,)
        depths_by_model[model_key(str(model))] = depths
        context_length_by_model[model_key(str(model))] = max_ctx
        if max_ctx is not None:
            print(f"  model context_length: {max_ctx}", flush=True)
            print(f"  derived depths (common context sizes up to {max_ctx}): {list(depths)}", flush=True)
        else:
            print(f"  WARNING: could not read context_length from GGUF metadata; "
                  f"falling back to fixed legacy depths: {list(depths)}", flush=True)

        if args.calibrate:
            print("  calibration: sweeping ubatch candidates on prefill series", flush=True)
            for ubatch in UBATCH_CANDIDATES:
                print(f"  [prefill calibration ub={ubatch}]", flush=True)
                result = run_one(
                    image=args.image, gpu_gids=gpu_gids, host_model_path=model,
                    series="prefill", ubatch=ubatch, device=args.device,
                    depths=depths, results_dir=args.results_dir,
                )
                results.append(result)
                print(f"    {result.status} (rc={result.return_code})", flush=True)
                time.sleep(args.cooldown)
            chosen_ubatch = pick_best_ubatch(args.results_dir, str(model), list(UBATCH_CANDIDATES))
            print(f"  calibration winner: ub={chosen_ubatch}", flush=True)
        else:
            chosen_ubatch = args.ubatch or 2048
            print(f"  using ubatch={chosen_ubatch} (no calibration)", flush=True)
            print(f"  [prefill ub={chosen_ubatch}]", flush=True)
            result = run_one(
                image=args.image, gpu_gids=gpu_gids, host_model_path=model,
                series="prefill", ubatch=chosen_ubatch, device=args.device,
                depths=depths, results_dir=args.results_dir,
            )
            results.append(result)
            print(f"    {result.status} (rc={result.return_code})", flush=True)
            time.sleep(args.cooldown)

        print(f"  [generation ub={chosen_ubatch}]", flush=True)
        result = run_one(
            image=args.image, gpu_gids=gpu_gids, host_model_path=model,
            series="generation", ubatch=chosen_ubatch, device=args.device,
            depths=depths, results_dir=args.results_dir,
        )
        results.append(result)
        print(f"    {result.status} (rc={result.return_code})", flush=True)
        time.sleep(args.cooldown)

    summary_path = args.results_dir / "curve_summary.csv"
    rows = write_curve_summary(results, summary_path)
    failed = [r for r in results if r.status == "failed"]

    manifest = {
        "image": args.image,
        "device": args.device,
        "depths_by_model": {k: list(v) for k, v in depths_by_model.items()},
        "context_length_by_model": context_length_by_model,
        "repetitions": REPETITIONS,
        "batch_size": BATCH_SIZE,
        "prefill_tokens": PREFILL_TOKENS,
        "generation_tokens": GENERATION_TOKENS,
        "calibrated": args.calibrate,
        "summary_rows": rows,
        "runs": [
            {
                "model": model_key(r.model), "series": r.series, "ubatch": r.ubatch,
                "status": r.status, "return_code": r.return_code,
            }
            for r in results
        ],
    }
    (args.results_dir / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    marker = "campaign.failed" if failed else "campaign.finished"
    (args.results_dir / marker).touch()

    print(f"\nCurve summary: {summary_path} ({rows} rows)", flush=True)
    if failed:
        for r in failed:
            print(f"FAILED: {r.model} {r.series} ub={r.ubatch} (see {r.stderr_path})", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
