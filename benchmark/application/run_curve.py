"""Application use case: run one full depth-curve for a single (model,
series, config) through Docker, ascending depth order, with a per-depth
timeout and VRAM-leak protection.

Extracted from run_bench.py's run_one()/probe_for()/build_llama_bench_command()
(further step in the refactor roadmap, after run_campaign.py's per-model
loop extraction). This is the most behavior-critical code in the
application layer - a bug here leaks VRAM on real hardware - so any
change here should be validated with a supervised real-GPU smoke test in
addition to the mocked unit tests, same as the docker_runner.py adapter
extraction it builds on.
"""

from __future__ import annotations

import time
from pathlib import Path

from adapters.outbound.docker_llama_bench import (
    LlamaBenchProbe,
    docker_command as build_docker_command,
    llama_bench_command,
)
from adapters.outbound.docker_runner import new_container_name, run_probe
from adapters.outbound.model_resolution import model_key
from adapters.outbound.terminal_progress import TerminalProgressReporter as ProgressTracker
from domain.models import BenchConfig, RunResult

REPETITIONS = 3
GPU_LAYERS = 99
PREFILL_TOKENS = 2048
GENERATION_TOKENS = 128
DEPTH_TIMEOUT_SECONDS = 300  # per-depth llama-bench invocation; OOM can hang
                             # rather than error cleanly (see run_one), so a
                             # subprocess timeout is the only reliable guard
DEPTH_COOLDOWN_SECONDS = 2   # short pause between depths within one run_one()
                             # call, distinct from --cooldown between configs


def probe_for(*, model_container_path: str, series: str, config: BenchConfig, device: str, depth: int) -> LlamaBenchProbe:
    """Build a LlamaBenchProbe from a BenchConfig - the single place that
    maps run_bench's configuration model onto the Docker adapter's probe
    contract, so build_llama_bench_command() and run_one() can't drift."""
    return LlamaBenchProbe(
        model_container_path=model_container_path,
        series=series,
        batch=config.batch,
        ubatch=config.ubatch,
        flash_attn=config.flash_attn,
        n_cpu_moe=config.n_cpu_moe,
        ctk=config.ctk,
        ctv=config.ctv,
        device=device,
        depth=depth,
        repetitions=REPETITIONS,
        gpu_layers=GPU_LAYERS,
        prefill_tokens=PREFILL_TOKENS,
        generation_tokens=GENERATION_TOKENS,
    )


def build_llama_bench_command(
    *,
    model_container_path: str,
    series: str,
    config: BenchConfig,
    device: str,
    depth: int,
) -> list[str]:
    """Compatibility wrapper around the Docker adapter's llama.cpp command."""
    return llama_bench_command(probe_for(
        model_container_path=model_container_path,
        series=series,
        config=config,
        device=device,
        depth=depth,
    ))


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
    progress: ProgressTracker | None = None,
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

            container_name = new_container_name()
            probe = probe_for(
                model_container_path=container_model_path,
                series=series,
                config=config,
                device=device,
                depth=depth,
            )
            docker_cmd = build_docker_command(
                image=image,
                container_name=container_name,
                gpu_gids=gpu_gids,
                model_directory=str(real_model_path.parent),
                probe=probe,
            )
            last_command = docker_cmd
            print(f"    $ {' '.join(docker_cmd)}", flush=True)
            if progress is not None:
                progress.before_probe(f"{series} depth={depth} {config.tag()}")

            stderr_f.write(f"\n===== depth={depth} =====\n")
            stderr_f.flush()

            outcome = run_probe(
                docker_cmd, container_name=container_name,
                jsonl_file=jsonl_f, stderr_file=stderr_f,
                timeout_seconds=DEPTH_TIMEOUT_SECONDS,
            )
            if progress is not None:
                progress.finish_probe(elapsed_seconds=outcome.elapsed_seconds)

            last_return_code = outcome.returncode
            if outcome.timed_out:
                stop_reason = (
                    f"depth {depth} timed out after {DEPTH_TIMEOUT_SECONDS}s "
                    f"(likely OOM/thrashing rather than a clean error); "
                    f"skipping larger depths"
                )
                print(f"    TIMEOUT at depth={depth} after {DEPTH_TIMEOUT_SECONDS}s "
                      f"(container killed); skipping remaining "
                      f"{len(sorted_depths) - i - 1} larger depth(s)", flush=True)
                if progress is not None:
                    progress.prune(len(sorted_depths) - i - 1, "this series timed out at a shallower depth")
            elif outcome.returncode != 0:
                stop_reason = (
                    f"depth {depth} failed (rc={outcome.returncode}); "
                    f"skipping larger depths"
                )
                print(f"    FAILED at depth={depth} (rc={outcome.returncode}); "
                      f"skipping remaining {len(sorted_depths) - i - 1} larger depth(s)",
                      flush=True)
                if progress is not None:
                    progress.prune(len(sorted_depths) - i - 1, "this series failed at a shallower depth")
            else:
                depths_run.append(depth)

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
