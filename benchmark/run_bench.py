#!/usr/bin/env python3
"""Run a llama-bench parameter sweep against a Docker container on this host.

Standalone replacement for upstream's run_calibrated_campaign.py, which
assumes their Toolbx/Cockpit stack and SSH hosts. This drives `docker run` +
`llama-bench` directly and writes one JSONL file per (model, series, config)
combination plus a curve_summary.csv, mirroring the shape of upstream's
output closely enough to stay comparable.

Depths (llama-bench's "-d", how full the KV cache is before the timed run)
are derived per model from its GGUF *.context_length metadata (see
gguf_info.py): every entry in COMMON_CONTEXT_SIZES that fits under the
model's trained context length gets tested, plus depth 0. Testing past a
model's trained context produces throughput numbers but not meaningful
ones, since RoPE positions past that point were never seen in training.
Falls back to LEGACY_FIXED_DEPTHS if a model's context_length can't be read.

Tuning modes (pick one; --full-sweep is the general recommendation):

  (default)     Fixed config: ubatch=2048, batch=2048, ctk/ctv=f16, fa=1.
  --ubatch N    Fixed config as above but with this ubatch.
  --calibrate   Legacy: sweep only UBATCH_CANDIDATES at depth 0, holding
                batch/KV-cache/flash-attn at their defaults. Kept for
                backward compatibility with earlier results directories.
  --full-sweep  Staged auto-tune across ubatch, batch, KV cache dtype
                (f16/q8_0/q4_0), and flash-attn on/off - see auto_tune().
                Runs a handful of quick 2-depth probes per stage rather
                than a full grid (which would be 4 ubatch x 4 batch x 3 KV
                x 2 FA = 96 combinations - infeasible to run at full depth
                x 3 repetitions). The final chosen config is then run
                across the model's full derived depth curve exactly once.

llama.cpp requires flash attention ON whenever KV cache is quantized
(ctk/ctv != f16); auto_tune() enforces this rather than trying invalid
combinations.

Note: llama-bench has no flag for speculative decoding / multi-token
prediction (MTP) - that's a llama-server/llama-cli feature (-md draft
model), not something this benchmark tool measures.

Usage:
    ./run_bench.py --model /models/foo.gguf --full-sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
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
BATCH_CANDIDATES = (512, 1024, 2048, 4096)
KV_CACHE_TYPES = ("f16", "q8_0", "q4_0")
FLASH_ATTN_CANDIDATES = (1, 0)
# KV cache dtypes that require flash attention (llama.cpp hard requirement).
KV_TYPES_REQUIRING_FA = frozenset({"q8_0", "q4_0", "q5_0", "q5_1", "q4_1", "iq4_nl"})

PREFILL_TOKENS = 2048
GENERATION_TOKENS = 128
REPETITIONS = 3
GPU_LAYERS = 99
COOLDOWN_SECONDS = 10


@dataclass(frozen=True)
class BenchConfig:
    ubatch: int = 2048
    batch: int = 2048
    ctk: str = "f16"
    ctv: str = "f16"
    flash_attn: int = 1

    def tag(self) -> str:
        return f"ub{self.ubatch}_b{self.batch}_kv{self.ctk}-{self.ctv}_fa{self.flash_attn}"

    def validate(self) -> "BenchConfig":
        """Force flash-attn on if the KV cache dtype requires it."""
        if (self.ctk in KV_TYPES_REQUIRING_FA or self.ctv in KV_TYPES_REQUIRING_FA) and not self.flash_attn:
            return replace(self, flash_attn=1)
        return self


@dataclass
class RunResult:
    model: str
    series: str  # "prefill" or "generation"
    config: BenchConfig
    status: str  # "ok", "failed"
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
    config: BenchConfig,
    device: str,
    depths: tuple[int, ...],
) -> list[str]:
    cmd = [
        "llama-bench",
        "-m", model_container_path,
        "-o", "jsonl",
        "-oe", "jsonl",
        "-r", str(REPETITIONS),
        "-b", str(config.batch),
        "-ub", str(config.ubatch),
        "-ngl", str(GPU_LAYERS),
        "-fa", str(config.flash_attn),
        "-mmp", "0",
        "-ctk", config.ctk,
        "-ctv", config.ctv,
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
    config: BenchConfig,
    device: str,
    depths: tuple[int, ...],
    results_dir: Path,
    subdir: str | None = None,
) -> RunResult:
    out_dir = results_dir / subdir if subdir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    container_model_path = f"/models/{host_model_path.name}"
    out_name = f"{model_key(str(host_model_path))}__{series}__{config.tag()}"
    jsonl_path = out_dir / f"{out_name}.jsonl"
    stderr_path = out_dir / f"{out_name}.stderr.log"

    bench_cmd = build_llama_bench_command(
        model_container_path=container_model_path,
        series=series,
        config=config,
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

    print(f"    $ {' '.join(docker_cmd)}", flush=True)

    with jsonl_path.open("w", encoding="utf-8") as out_f, \
         stderr_path.open("w", encoding="utf-8") as err_f:
        proc = subprocess.run(docker_cmd, stdout=out_f, stderr=err_f)

    status = "ok" if proc.returncode == 0 and jsonl_path.stat().st_size > 0 else "failed"
    return RunResult(
        model=str(host_model_path),
        series=series,
        config=config,
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


def mean_ts(jsonl_path: Path) -> float:
    rows = load_jsonl_rows(jsonl_path)
    values = [r.get("avg_ts") for r in rows if r.get("avg_ts") is not None]
    return sum(values) / len(values) if values else -1.0


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
) -> float:
    """Run a quick prefill-only probe of one config; return mean avg_ts."""
    result = run_one(
        image=image, gpu_gids=gpu_gids, host_model_path=model,
        series="prefill", config=config.validate(), device=device,
        depths=depths, results_dir=results_dir, subdir="tuning",
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
) -> BenchConfig:
    """Staged (coordinate-descent) auto-tune: flash-attn, then KV cache
    dtype, then a ubatch x batch grid. Each stage probes at 2 depths
    (shallowest + deepest) rather than the full curve, and carries its
    winner into the next stage. Full grid search (ubatch x batch x KV x FA)
    would be 4x4x3x2=96 configs at full depth x 3 reps - infeasible; this
    keeps total probe count in the dozens instead.
    """
    probe_d = probe_depths(depths)
    print(f"  auto-tune: probing at depths {list(probe_d)}", flush=True)

    # Stage 1: flash attention on/off, KV cache fixed at f16 (no FA requirement).
    print("  [stage 1/3] flash attention on/off", flush=True)
    fa_scores = {}
    for fa in FLASH_ATTN_CANDIDATES:
        cfg = BenchConfig(flash_attn=fa)
        score = probe_config(image=image, gpu_gids=gpu_gids, model=model, config=cfg,
                              device=device, depths=probe_d, results_dir=results_dir)
        fa_scores[fa] = score
        print(f"    fa={fa}: mean_ts={score:.1f}", flush=True)
        time.sleep(cooldown)
    best_fa = max(fa_scores, key=lambda k: fa_scores[k])
    log.append({"stage": "flash_attn", "scores": fa_scores, "winner": best_fa})
    print(f"  stage 1 winner: fa={best_fa}", flush=True)

    # Stage 2: KV cache dtype. Quantized KV requires FA on regardless of
    # stage 1's result, so probe all KV candidates with FA forced on for a
    # fair, valid comparison; only fall back to stage 1's FA winner at the
    # end if the KV winner turns out to be f16 (which has no FA requirement).
    print("  [stage 2/3] KV cache dtype (fa forced on for this stage)", flush=True)
    kv_scores = {}
    for kv in KV_CACHE_TYPES:
        cfg = BenchConfig(ctk=kv, ctv=kv, flash_attn=1)
        score = probe_config(image=image, gpu_gids=gpu_gids, model=model, config=cfg,
                              device=device, depths=probe_d, results_dir=results_dir)
        kv_scores[kv] = score
        print(f"    kv={kv}: mean_ts={score:.1f}", flush=True)
        time.sleep(cooldown)
    best_kv = max(kv_scores, key=lambda k: kv_scores[k])
    log.append({"stage": "kv_cache_dtype", "scores": kv_scores, "winner": best_kv})
    print(f"  stage 2 winner: kv={best_kv}", flush=True)

    final_fa = 1 if best_kv in KV_TYPES_REQUIRING_FA else best_fa

    # Stage 3: ubatch x batch grid, at the winning FA/KV, depth 0 only
    # (batch/ubatch effects show up clearly even on a cold cache, and this
    # keeps the grid's 16 combinations cheap).
    print("  [stage 3/3] ubatch x batch grid (depth 0 only)", flush=True)
    grid_scores = {}
    grid_combo_lookup: dict[str, tuple[int, int]] = {}
    for ub in UBATCH_CANDIDATES:
        for b in BATCH_CANDIDATES:
            if ub > b:
                continue  # llama.cpp requires ubatch <= batch
            cfg = BenchConfig(ubatch=ub, batch=b, ctk=best_kv, ctv=best_kv, flash_attn=final_fa)
            score = probe_config(image=image, gpu_gids=gpu_gids, model=model, config=cfg,
                                  device=device, depths=(0,), results_dir=results_dir)
            combo_key = f"ub{ub}_b{b}"
            grid_scores[combo_key] = score
            grid_combo_lookup[combo_key] = (ub, b)
            print(f"    ub={ub} b={b}: mean_ts={score:.1f}", flush=True)
            time.sleep(cooldown)
    best_combo = max(grid_scores, key=lambda k: grid_scores[k])
    best_ub, best_b = grid_combo_lookup[best_combo]
    log.append({"stage": "ubatch_batch_grid", "scores": grid_scores, "winner": best_combo})
    print(f"  stage 3 winner: ubatch={best_ub} batch={best_b}", flush=True)

    return BenchConfig(ubatch=best_ub, batch=best_b, ctk=best_kv, ctv=best_kv, flash_attn=final_fa).validate()


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


def write_curve_summary(results: list[RunResult], summary_path: Path) -> int:
    fieldnames = [
        "model", "series", "ubatch", "batch", "ctk", "ctv", "flash_attn",
        "n_depth", "n_prompt", "n_gen", "avg_ts", "avg_ns", "status",
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
                    "ubatch": result.config.ubatch,
                    "batch": result.config.batch,
                    "ctk": result.config.ctk,
                    "ctv": result.config.ctv,
                    "flash_attn": result.config.flash_attn,
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
                         help="Legacy: sweep only ubatch at depth 0, other params fixed at defaults")
    parser.add_argument("--full-sweep", action="store_true",
                         help="Staged auto-tune across ubatch, batch, KV cache dtype, and flash-attn")
    parser.add_argument("--max-depth", type=int, default=None,
                         help="Cap derived depths at this value even if the model supports more "
                              "(useful for a quick smoke test on a long-context model)")
    parser.add_argument("--ubatch", type=int, default=None, help="Fixed ubatch (ignored with --calibrate/--full-sweep)")
    parser.add_argument("--batch", type=int, default=None, help="Fixed batch size (ignored with --full-sweep)")
    parser.add_argument("--ctk", default=None, help="Fixed KV cache key dtype, e.g. f16/q8_0/q4_0 (ignored with --full-sweep)")
    parser.add_argument("--ctv", default=None, help="Fixed KV cache value dtype (ignored with --full-sweep)")
    parser.add_argument("--flash-attn", type=int, choices=(0, 1), default=None,
                         help="Fixed flash-attn on/off (ignored with --full-sweep)")
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
    final_config_by_model: dict[str, dict] = {}
    tuning_log_by_model: dict[str, list[dict]] = {}

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

        tuning_log: list[dict] = []

        if args.full_sweep:
            config = auto_tune(
                image=args.image, gpu_gids=gpu_gids, model=model, device=args.device,
                depths=depths, results_dir=args.results_dir, cooldown=args.cooldown,
                log=tuning_log,
            )
            print(f"  full-sweep winner: {config.tag()}", flush=True)
        elif args.calibrate:
            print("  calibration (legacy): sweeping ubatch candidates on prefill series", flush=True)
            for ubatch in UBATCH_CANDIDATES:
                cfg = BenchConfig(ubatch=ubatch)
                print(f"  [prefill calibration {cfg.tag()}]", flush=True)
                result = run_one(
                    image=args.image, gpu_gids=gpu_gids, host_model_path=model,
                    series="prefill", config=cfg, device=args.device,
                    depths=depths, results_dir=args.results_dir,
                )
                results.append(result)
                print(f"    {result.status} (rc={result.return_code})", flush=True)
                time.sleep(args.cooldown)
            chosen_ubatch = pick_best_ubatch(args.results_dir, str(model), list(UBATCH_CANDIDATES))
            config = BenchConfig(ubatch=chosen_ubatch)
            print(f"  calibration winner: ub={chosen_ubatch}", flush=True)
        else:
            config = BenchConfig(
                ubatch=args.ubatch or 2048,
                batch=args.batch or 2048,
                ctk=args.ctk or "f16",
                ctv=args.ctv or args.ctk or "f16",
                flash_attn=1 if args.flash_attn is None else args.flash_attn,
            ).validate()
            print(f"  using fixed config: {config.tag()}", flush=True)

        final_config_by_model[model_key(str(model))] = asdict(config)
        tuning_log_by_model[model_key(str(model))] = tuning_log

        for series in ("prefill", "generation"):
            print(f"  [{series} {config.tag()}]", flush=True)
            result = run_one(
                image=args.image, gpu_gids=gpu_gids, host_model_path=model,
                series=series, config=config, device=args.device,
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
        "final_config_by_model": final_config_by_model,
        "tuning_log_by_model": tuning_log_by_model,
        "repetitions": REPETITIONS,
        "prefill_tokens": PREFILL_TOKENS,
        "generation_tokens": GENERATION_TOKENS,
        "mode": "full-sweep" if args.full_sweep else ("calibrate-legacy" if args.calibrate else "fixed"),
        "summary_rows": rows,
        "runs": [
            {
                "model": model_key(r.model), "series": r.series, "config": asdict(r.config),
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
            print(f"FAILED: {r.model} {r.series} {r.config.tag()} (see {r.stderr_path})", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
