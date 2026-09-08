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

  (default)     Fixed config: ubatch=2048, batch=2048, ctk/ctv=f16, fa=auto.
  --ubatch N    Fixed config as above but with this ubatch.
  --calibrate   Legacy: sweep only UBATCH_CANDIDATES at depth 0, holding
                batch/KV-cache at their defaults (fa=auto). Kept for
                backward compatibility with earlier results directories.
  --full-sweep  Staged auto-tune across ubatch, batch, and KV cache dtype
                (f16/q8_0/q4_0) - see auto_tune(). Runs a handful of quick
                2-depth probes per stage rather than a full grid (which
                would be 4 ubatch x 4 batch x 3 KV = 48 combinations -
                infeasible to run at full depth x 3 repetitions). The
                final chosen config is then run across the model's full
                derived depth curve exactly once.

Flash attention is not swept: always passed as "auto" (llama-bench's -fa
auto|on|off, this build's own default), which lets llama.cpp decide per
model/backend at load time whether the fused kernel actually applies.
Forcing it "on" doesn't help on architectures where FA can't apply
anyway (KQ-bias models, some hybrid/SSM architectures, unsupported head
dims - these fall back to the ordinary attention path regardless of the
flag, silently), and for a few architectures (e.g. sparse-attention
models) FA isn't just a speed knob, disabling it changes output
correctness. llama.cpp's own "auto" already encodes this judgment call
better than we can by sweeping 0/1 ourselves.

llama.cpp requires flash attention ON whenever KV cache is quantized
(ctk/ctv != f16); BenchConfig.validate() enforces this rather than trying
invalid combinations - "auto" alone isn't a safe default there, since
auto isn't guaranteed to actually enable FA.

Note: llama-bench has no flag for speculative decoding / multi-token
prediction (MTP) - that's a llama-server/llama-cli feature (-md draft
model), not something this benchmark tool measures.

Usage:
    ./run_bench.py --model /models/foo.gguf --full-sweep
"""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import re
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from gguf_info import context_length, read_gguf_metadata
import environment_info
import hf_models

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
# KV cache dtypes that require flash attention (llama.cpp hard requirement -
# the non-fused attention path can't read a quantized KV cache at all, this
# isn't just a perf preference).
KV_TYPES_REQUIRING_FA = frozenset({"q8_0", "q4_0", "q5_0", "q5_1", "q4_1", "iq4_nl"})

PREFILL_TOKENS = 2048
GENERATION_TOKENS = 128
REPETITIONS = 3
GPU_LAYERS = 99
COOLDOWN_SECONDS = 10
DEPTH_TIMEOUT_SECONDS = 300  # per-depth llama-bench invocation; OOM can hang
                             # rather than error cleanly (see run_one), so a
                             # subprocess timeout is the only reliable guard
DEPTH_COOLDOWN_SECONDS = 2   # short pause between depths within one run_one()
                             # call, distinct from --cooldown between configs


@dataclass(frozen=True)
class BenchConfig:
    ubatch: int = 2048
    batch: int = 2048
    ctk: str = "f16"
    ctv: str = "f16"
    flash_attn: str = "auto"  # llama-bench's -fa: "auto" | "on" | "off".
        # "auto" lets llama.cpp decide per model/backend at load time
        # whether the fused kernel applies (some architectures - KQ-bias
        # models, certain hybrid/SSM models, unsupported head dims - fall
        # back to the ordinary path regardless of this flag, silently).
        # We stopped sweeping flash-attn on/off: forcing "on" doesn't help
        # on architectures where FA can't apply anyway, and "auto" is
        # already llama.cpp's own considered default as of this build.

    def tag(self) -> str:
        return f"ub{self.ubatch}_b{self.batch}_kv{self.ctk}-{self.ctv}_fa{self.flash_attn}"

    def validate(self) -> "BenchConfig":
        """Force flash-attn on if the KV cache dtype requires it - "auto"
        is not guaranteed to enable FA, but a quantized KV cache can only
        be read via the fused kernel, so "auto" alone isn't safe here."""
        if (self.ctk in KV_TYPES_REQUIRING_FA or self.ctv in KV_TYPES_REQUIRING_FA) and self.flash_attn == "off":
            return replace(self, flash_attn="on")
        return self


@dataclass
class RunResult:
    model: str
    series: str  # "prefill" or "generation"
    config: BenchConfig
    status: str  # "ok" (all depths ran), "partial" (stopped early after a
                 # depth failed/timed out, but at least one depth succeeded),
                 # "failed" (no depth produced results)
    return_code: int
    jsonl_path: Path
    stderr_path: Path
    command: list[str]
    depths_run: tuple[int, ...] = ()
    depths_skipped: tuple[int, ...] = ()
    stop_reason: str | None = None  # e.g. "depth 65536 timed out after 300s"


# Names of docker containers currently running a benchmark, so they can be
# force-killed if this script is interrupted (Ctrl-C, SIGTERM, an unhandled
# exception). Without this, `docker run` surviving the parent script's death
# leaves the container - and the GPU memory it holds - running indefinitely;
# this has actually happened during development and is exactly the kind of
# thing that silently wastes VRAM until someone notices and runs `docker
# kill` by hand. `docker run --rm` alone does not protect against this: it
# only removes the container after IT exits, which doesn't happen just
# because the client/parent process died.
_ACTIVE_CONTAINERS: set[str] = set()


def _kill_active_containers() -> None:
    for name in list(_ACTIVE_CONTAINERS):
        subprocess.run(["docker", "kill", name], capture_output=True)
        _ACTIVE_CONTAINERS.discard(name)


def _install_cleanup_handlers() -> None:
    atexit.register(_kill_active_containers)

    def _handle_signal(signum, frame):
        _kill_active_containers()
        # Restore default handling and re-raise, so the process actually
        # exits with the conventional 128+signum code instead of hanging.
        signal.signal(signum, signal.SIG_DFL)
        signal.raise_signal(signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)


def model_key(model_path: str) -> str:
    return Path(model_path).name


def model_slug(model_path: Path) -> str:
    """Filesystem-safe identity for a model: its filename without the
    .gguf extension, lowercased, non-alphanumerics collapsed to '-'."""
    stem = model_path.stem
    slug = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
    return slug or "model"


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
    depth: int,
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
        "-fa", config.flash_attn,
        "-mmp", "0",
        "-ctk", config.ctk,
        "-ctv", config.ctv,
        "-dev", device,
        "-d", str(depth),
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
    """Run llama-bench once per depth, ascending (shallowest first).

    Depths are run in separate `docker run` invocations rather than one
    llama-bench call with a combined "-d" list, specifically so an
    out-of-memory condition at a deep context doesn't take down the whole
    curve: llama-bench streams JSONL results per depth as it completes
    them (confirmed by direct testing - shallow depths finish and print
    before a later depth OOMs), but a large KV cache allocation can also
    just *hang* rather than error out cleanly, so a single multi-depth
    invocation has no way to bail out of one bad depth and keep going.
    Running depths ascending, one process per depth, with a timeout,
    means: smaller/valid depths always get recorded, and a failure or
    hang at some depth stops only the depths larger than it (they'd very
    likely fail too - KV cache need only grows with depth).
    """
    out_dir = results_dir / subdir if subdir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve symlinks before mounting: Hugging Face's local cache stores
    # models as snapshots/<hash>/model.gguf -> ../../blobs/<blob-hash>, so
    # mounting only the file's immediate parent directory (the snapshot
    # dir) would leave the symlink target outside the mount, unreadable
    # inside the container. Mounting the *real* file's parent directory
    # works for both plain files (no-op, same directory either way) and
    # HF-cache-style symlinks.
    real_model_path = host_model_path.resolve()
    container_model_path = f"/models/{real_model_path.name}"
    out_name = f"{model_key(str(host_model_path))}__{series}__{config.tag()}"
    jsonl_path = out_dir / f"{out_name}.jsonl"
    stderr_path = out_dir / f"{out_name}.stderr.log"

    sorted_depths = tuple(sorted(depths))
    depths_run: list[int] = []
    depths_skipped: list[int] = []
    stop_reason: str | None = None
    last_return_code = 0
    last_command: list[str] = []

    jsonl_f = jsonl_path.open("w", encoding="utf-8")
    stderr_f = stderr_path.open("w", encoding="utf-8")
    try:
        for i, depth in enumerate(sorted_depths):
            if stop_reason is not None:
                depths_skipped.append(depth)
                continue

            container_name = f"r9700-llm-bench-{uuid.uuid4().hex[:12]}"
            bench_cmd = build_llama_bench_command(
                model_container_path=container_model_path,
                series=series, config=config, device=device, depth=depth,
            )
            docker_cmd = [
                "docker", "run", "--rm",
                "--name", container_name,
                *docker_gpu_args(gpu_gids),
                "-v", f"{real_model_path.parent}:/models:ro",
                image,
                *bench_cmd,
            ]
            last_command = docker_cmd
            print(f"    $ {' '.join(docker_cmd)}", flush=True)

            stderr_f.write(f"\n===== depth={depth} =====\n")
            stderr_f.flush()

            # Track the container name so a signal handler or atexit hook
            # can `docker kill` it if this script gets interrupted mid-run
            # (see _ACTIVE_CONTAINERS above), and so a *timeout* here can
            # kill the specific container that hung rather than leaving it
            # running - subprocess.run(timeout=...) only kills the direct
            # child (docker CLI), not the container it started, so without
            # an explicit `docker kill` a timed-out depth would leak VRAM
            # exactly like an interrupted run would.
            _ACTIVE_CONTAINERS.add(container_name)
            try:
                proc = subprocess.run(
                    docker_cmd, stdout=jsonl_f, stderr=stderr_f,
                    timeout=DEPTH_TIMEOUT_SECONDS,
                )
                last_return_code = proc.returncode
                if proc.returncode != 0:
                    stop_reason = (
                        f"depth {depth} failed (rc={proc.returncode}); "
                        f"skipping larger depths"
                    )
                    print(f"    FAILED at depth={depth} (rc={proc.returncode}); "
                          f"skipping remaining {len(sorted_depths) - i - 1} larger depth(s)",
                          flush=True)
                else:
                    depths_run.append(depth)
            except subprocess.TimeoutExpired:
                subprocess.run(["docker", "kill", container_name], capture_output=True)
                last_return_code = -1
                stop_reason = (
                    f"depth {depth} timed out after {DEPTH_TIMEOUT_SECONDS}s "
                    f"(likely OOM/thrashing rather than a clean error); "
                    f"skipping larger depths"
                )
                stderr_f.write(f"\n[TIMEOUT after {DEPTH_TIMEOUT_SECONDS}s - container killed]\n")
                stderr_f.flush()
                print(f"    TIMEOUT at depth={depth} after {DEPTH_TIMEOUT_SECONDS}s "
                      f"(container killed); skipping remaining "
                      f"{len(sorted_depths) - i - 1} larger depth(s)", flush=True)
            finally:
                _ACTIVE_CONTAINERS.discard(container_name)

            if stop_reason is None and i < len(sorted_depths) - 1:
                time.sleep(DEPTH_COOLDOWN_SECONDS)
    finally:
        jsonl_f.close()
        stderr_f.close()

    if not depths_run:
        status = "failed"
    elif stop_reason is not None:
        # At least one depth (possibly the last one) never produced
        # results - "ok" would wrongly imply the full curve is complete.
        status = "partial"
    else:
        status = "ok"

    return RunResult(
        model=str(host_model_path),
        series=series,
        config=config,
        status=status,
        return_code=last_return_code,
        jsonl_path=jsonl_path,
        stderr_path=stderr_path,
        command=last_command,
        depths_run=tuple(depths_run),
        depths_skipped=tuple(depths_skipped),
        stop_reason=stop_reason,
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
    """Staged (coordinate-descent) auto-tune: KV cache dtype, then a
    ubatch x batch grid. Each stage probes at 2 depths (shallowest +
    deepest) rather than the full curve, and carries its winner into the
    next stage. Flash-attn is not swept - always "auto", letting
    llama.cpp decide per model/backend whether the fused kernel applies
    (see BenchConfig.flash_attn) - so this is a 2-stage search, not 3.
    Full grid search (ubatch x batch x KV) would be 4x4x3=48 configs at
    full depth x 3 reps - infeasible; this keeps total probe count in the
    dozens instead.
    """
    probe_d = probe_depths(depths)
    print(f"  auto-tune: probing at depths {list(probe_d)}", flush=True)

    # Stage 1: KV cache dtype, flash-attn always "auto" (quantized KV
    # cache requires FA regardless, and validate() enforces that).
    print("  [stage 1/2] KV cache dtype", flush=True)
    kv_scores = {}
    for kv in KV_CACHE_TYPES:
        cfg = BenchConfig(ctk=kv, ctv=kv).validate()
        score = probe_config(image=image, gpu_gids=gpu_gids, model=model, config=cfg,
                              device=device, depths=probe_d, results_dir=results_dir)
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
            cfg = BenchConfig(ubatch=ub, batch=b, ctk=best_kv, ctv=best_kv).validate()
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
    print(f"  stage 2 winner: ubatch={best_ub} batch={best_b}", flush=True)

    return BenchConfig(ubatch=best_ub, batch=best_b, ctk=best_kv, ctv=best_kv).validate()


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
    parser.add_argument("--model", action="append", default=[],
                         help="Host path to a .gguf file (repeatable). Also accepts Hugging "
                              "Face references: hf://org/repo/file.gguf, org/repo, "
                              "org/repo:QUANT (ollama-style tag), or org/repo/file.gguf - "
                              "resolved via the local HF cache, downloading if not already "
                              "present. org/repo with multiple .gguf files prompts an "
                              "interactive picker.")
    parser.add_argument("--image", default="r9700-llm-bench:rocm-7.2.4", help="Docker image to benchmark")
    parser.add_argument("--device", default="ROCm0", help="llama-bench -dev target (default: ROCm0, the R9700)")
    parser.add_argument("--gpu-gid", action="append", default=[],
                         help="Host GID(s) for /dev/dri and /dev/kfd access (repeatable); "
                              "defaults to `video` and `render` group GIDs on this host")
    parser.add_argument("--results-root", type=Path, default=Path("results"),
                         help="Root results directory; each model gets "
                              "<results-root>/<model-slug>/<run-id>/ (default: ./results)")
    parser.add_argument("--force", action="store_true",
                         help="Overwrite an existing run directory instead of refusing")
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
    parser.add_argument("--flash-attn", choices=("auto", "on", "off"), default=None,
                         help="Fixed flash-attn mode (ignored with --full-sweep, which always "
                              "uses 'auto' - see BenchConfig.flash_attn)")
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
    _install_cleanup_handlers()
    args = parse_args()
    gpu_gids = args.gpu_gid or default_gpu_gids()
    if not gpu_gids:
        sys.exit("Could not resolve video/render group GIDs; pass --gpu-gid explicitly")

    if not args.model:
        sys.exit("At least one --model is required")

    # Keep the given path as-is (don't resolve symlinks here) - Hugging
    # Face cache entries are named model.gguf via a symlink to a
    # content-hash blob; resolving too early would make model_slug() use
    # the meaningless blob hash instead of the real model filename.
    # run_one() resolves symlinks separately, only for the Docker mount.
    models = []
    for m in args.model:
        if hf_models.looks_like_hf_reference(m):
            try:
                resolved = hf_models.resolve_hf_reference(m)
            except hf_models.HfReferenceError as e:
                sys.exit(f"Error resolving {m!r}: {e}")
            models.append(Path(resolved))
        else:
            models.append(Path(m).expanduser().absolute())
    for model in models:
        if not model.is_file():
            sys.exit(f"Model file not found: {model}")

    print("Querying environment (ROCm/llama.cpp/GPU versions)...", flush=True)
    env = environment_info.gather(args.image)
    run_id = environment_info.run_id(env)
    print(f"  {env}", flush=True)
    print(f"  run_id: {run_id}", flush=True)

    for model in models:
        slug = model_slug(model)
        model_results_dir = args.results_root / slug / run_id
        if model_results_dir.exists() and any(model_results_dir.iterdir()):
            if not args.force:
                sys.exit(
                    f"Refusing to overwrite existing run directory: {model_results_dir}\n"
                    f"(same model + ROCm + llama.cpp version already benchmarked here; "
                    f"pass --force to overwrite)"
                )
            for child in model_results_dir.rglob("*"):
                if child.is_file():
                    child.unlink()
        model_results_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n== {model.name} -> {model_results_dir} ==", flush=True)

        depths, max_ctx = derive_depths_for_model(model)
        if args.max_depth is not None:
            depths = tuple(d for d in depths if d <= args.max_depth) or (0,)
        if max_ctx is not None:
            print(f"  model context_length: {max_ctx}", flush=True)
            print(f"  derived depths (common context sizes up to {max_ctx}): {list(depths)}", flush=True)
        else:
            print(f"  WARNING: could not read context_length from GGUF metadata; "
                  f"falling back to fixed legacy depths: {list(depths)}", flush=True)

        try:
            gguf_metadata = read_gguf_metadata(model)
        except (OSError, ValueError):
            gguf_metadata = {}

        results: list[RunResult] = []
        tuning_log: list[dict] = []

        if args.full_sweep:
            config = auto_tune(
                image=args.image, gpu_gids=gpu_gids, model=model, device=args.device,
                depths=depths, results_dir=model_results_dir, cooldown=args.cooldown,
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
                    depths=depths, results_dir=model_results_dir,
                )
                results.append(result)
                print(f"    {result.status} (rc={result.return_code})", flush=True)
                time.sleep(args.cooldown)
            chosen_ubatch = pick_best_ubatch(model_results_dir, str(model), list(UBATCH_CANDIDATES))
            config = BenchConfig(ubatch=chosen_ubatch)
            print(f"  calibration winner: ub={chosen_ubatch}", flush=True)
        else:
            config = BenchConfig(
                ubatch=args.ubatch or 2048,
                batch=args.batch or 2048,
                ctk=args.ctk or "f16",
                ctv=args.ctv or args.ctk or "f16",
                flash_attn=args.flash_attn or "auto",
            ).validate()
            print(f"  using fixed config: {config.tag()}", flush=True)

        for series in ("prefill", "generation"):
            print(f"  [{series} {config.tag()}]", flush=True)
            result = run_one(
                image=args.image, gpu_gids=gpu_gids, host_model_path=model,
                series=series, config=config, device=args.device,
                depths=depths, results_dir=model_results_dir,
            )
            results.append(result)
            print(f"    {result.status} (rc={result.return_code})", flush=True)
            time.sleep(args.cooldown)

        summary_path = model_results_dir / "curve_summary.csv"
        rows = write_curve_summary(results, summary_path)
        failed = [r for r in results if r.status == "failed"]
        partial = [r for r in results if r.status == "partial"]
        mode = "full-sweep" if args.full_sweep else ("calibrate-legacy" if args.calibrate else "fixed")
        completed_at = datetime.now(timezone.utc).isoformat()

        manifest = {
            "image": args.image,
            "device": args.device,
            "depths": list(depths),
            "context_length": max_ctx,
            "final_config": asdict(config),
            "tuning_log": tuning_log,
            "repetitions": REPETITIONS,
            "prefill_tokens": PREFILL_TOKENS,
            "generation_tokens": GENERATION_TOKENS,
            "mode": mode,
            "completed_at": completed_at,
            "summary_rows": rows,
            "runs": [
                {
                    "series": r.series, "config": asdict(r.config), "status": r.status,
                    "return_code": r.return_code, "depths_run": list(r.depths_run),
                    "depths_skipped": list(r.depths_skipped), "stop_reason": r.stop_reason,
                }
                for r in results
            ],
        }
        (model_results_dir / "campaign_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        best_depth0_ts = None
        gen_rows = [r for r in results if r.series == "generation"]
        if gen_rows:
            ts = mean_ts(gen_rows[0].jsonl_path)
            best_depth0_ts = ts if ts >= 0 else None

        metadata = {
            "model_slug": slug,
            "model_filename": model.name,
            "model_architecture": gguf_metadata.get("general.architecture"),
            "model_name": gguf_metadata.get("general.name"),
            "model_context_length": max_ctx,
            "run_id": run_id,
            "run_completed_at": completed_at,
            "mode": mode,
            "environment": env,
            "final_config": asdict(config),
            "depths_tested": list(depths),
            "generation_tok_s_mean": best_depth0_ts,
            "status": "failed" if failed else ("partial" if partial else "finished"),
        }
        (model_results_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        marker = "campaign.failed" if failed else ("campaign.partial" if partial else "campaign.finished")
        (model_results_dir / marker).touch()

        print(f"  Curve summary: {summary_path} ({rows} rows)", flush=True)
        if failed:
            for r in failed:
                print(f"  FAILED: {r.model} {r.series} {r.config.tag()} (see {r.stderr_path})", flush=True)
        if partial:
            for r in partial:
                print(f"  PARTIAL: {r.model} {r.series} {r.config.tag()}: {r.stop_reason} "
                      f"(ran depths {list(r.depths_run)}, skipped {list(r.depths_skipped)})", flush=True)

    any_failed = False
    any_partial = False
    for model in models:
        slug = model_slug(model)
        run_dir = args.results_root / slug / run_id
        if (run_dir / "campaign.failed").exists():
            any_failed = True
        elif (run_dir / "campaign.partial").exists():
            any_partial = True
    if any_failed:
        sys.exit(1)
    if any_partial:
        sys.exit(2)


if __name__ == "__main__":
    main()
