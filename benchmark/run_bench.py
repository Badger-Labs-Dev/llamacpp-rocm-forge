#!/usr/bin/env python3
"""Run a llama-bench parameter sweep against a Docker container on this host.

Drives `docker run` + `llama-bench` directly, writing one JSONL file per
(model, series, config) plus a curve_summary.csv.

Depths (llama-bench's "-d") are derived per model from its GGUF
*.context_length metadata (see adapters/outbound/gguf_metadata.py):
every COMMON_CONTEXT_SIZES entry that fits under the trained context
length is tested, plus depth 0. Falls back to LEGACY_FIXED_DEPTHS if
context_length can't be read.

Model type (dense vs MoE) is auto-detected from GGUF *.expert_count -
there is nothing to configure. Two modes, no in-between:

  (default)  Staged auto-tune across ubatch, batch, and KV cache dtype
             (application/auto_tune.py). Dense models map the exact --ngl
             boundary per depth; detected MoE models also
             binary-searches the exact --n-cpu-moe boundary per depth
             (application/moe_sweep.py's thorough sweep). This is the
             slow, thorough, "give me the real numbers" mode.
  --quick    Skips tuning: fixed ubatch=2048, batch=2048, ctk/ctv=f16,
             flash-attn=auto. Dense models still map --ngl with fewer
             throughput samples. A detected MoE model sweeps five
             evenly-spaced --n-cpu-moe candidates instead of the exact
             boundary search. Fast path for "does this run at all,
             roughly how fast".

Flash attention is always "auto" in both modes, never swept -
llama.cpp's own default already picks the fused kernel when it applies,
and forcing it on/off doesn't help on architectures where it can't
apply anyway.

Usage:
    uv run run_bench.py --model hf://org/repo/model.gguf
    uv run run_bench.py --model hf://org/repo/model.gguf --quick
"""

from __future__ import annotations

import argparse
import atexit
import signal
import sys
from pathlib import Path

from adapters.outbound import huggingface_models as hf_models
from adapters.outbound import rocm_environment as environment_info
from adapters.outbound.docker_runner import kill_active_containers
from adapters.outbound.gguf_metadata import (
    model_block_count,
    moe_params,
    read_gguf_metadata,
    total_model_size_bytes,
)
from adapters.outbound.model_resolution import (
    derive_depths_for_model,
    model_slug,
    resolve_model_reference,
)
from adapters.outbound.terminal_progress import TerminalProgressReporter as ProgressTracker
from application.auto_tune import (
    auto_tune,
    probe_depths,
    valid_batch_grid_count,
)
from application.dense_sweep import (
    DENSE_QUICK_EXTRA_SAMPLE_COUNT,
    preflight_dense_offload,
    sweep_dense_offload,
)
from application.moe_sweep import (
    MOE_EXTRA_SAMPLE_COUNT,
    MOE_QUICK_CANDIDATE_COUNT,
    sweep_moe_offload_quick,
    sweep_moe_offload_thorough,
)
from application.run_curve import (
    DEPTH_COOLDOWN_SECONDS,
    GENERATION_TOKENS,
    PREFILL_TOKENS,
    REPETITIONS,
    run_one,
)
from application.run_campaign import CampaignConfig, run_model_campaign
from domain.models import BenchConfig
from domain.planning import campaign_budget
from domain.progress import ProbeProgress

KV_CACHE_TYPES = ("f16", "q8_0", "q4_0")
COOLDOWN_SECONDS = 10


# Names of docker containers currently running a benchmark, so they can be
# force-killed if this script is interrupted (Ctrl-C, SIGTERM, an unhandled
# exception). See adapters.outbound.docker_runner for the actual
# tracking/kill mechanics; this module just wires signal/atexit handlers to
# it.
def _install_cleanup_handlers() -> None:
    atexit.register(kill_active_containers)

    def _handle_signal(signum, frame):
        kill_active_containers()
        # Restore default handling and re-raise, so the process actually
        # exits with the conventional 128+signum code instead of hanging.
        signal.signal(signum, signal.SIG_DFL)
        signal.raise_signal(signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)


def planned_probe_count(
    args: argparse.Namespace,
    depths: tuple[int, ...],
    moe: dict | None,
    dense_block_count: int | None = None,
) -> tuple[int, str]:
    """CLI adapter around the domain campaign-budget calculation."""
    budget = campaign_budget(
        depth_count=len(depths),
        quick=args.quick,
        valid_batch_pairs=valid_batch_grid_count(),
        kv_type_count=len(KV_CACHE_TYPES),
        tuning_depth_count=len(probe_depths(depths)),
        moe_block_count=(moe or {}).get("block_count"),
        quick_candidate_count=MOE_QUICK_CANDIDATE_COUNT,
        thorough_extra_sample_count=MOE_EXTRA_SAMPLE_COUNT,
        dense_block_count=dense_block_count,
        dense_quick_extra_sample_count=DENSE_QUICK_EXTRA_SAMPLE_COUNT,
    )
    return budget.total, budget.detail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", default=[],
                         help="Host path to a .gguf file (repeatable). Also accepts Hugging "
                              "Face references: hf://org/repo/file.gguf, org/repo, "
                              "org/repo:QUANT (ollama-style tag), or org/repo/file.gguf - "
                              "resolved via the local HF cache, downloading if not already "
                              "present. org/repo with multiple .gguf files prompts an "
                              "interactive picker.")
    parser.add_argument("--quick", action="store_true",
                         help="Fast path: skip auto-tuning (fixed ubatch=2048, batch=2048, "
                              "ctk/ctv=f16, flash-attn=auto) and, for a detected MoE model, "
                              "sweep the quick 5-point --n-cpu-moe curve instead of the exact "
                              "boundary bisection")
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
    parser.add_argument("--max-depth", type=int, default=None,
                         help="Cap derived depths at this value even if the model supports more "
                              "(useful for a quick smoke test on a long-context model)")
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
        try:
            models.append(resolve_model_reference(m))
        except hf_models.HfReferenceError as e:
            sys.exit(f"Error resolving {m!r}: {e}")
    for model in models:
        if not model.is_file():
            sys.exit(f"Model file not found: {model}")

    print("Querying environment (ROCm/llama.cpp/GPU versions)...", flush=True)
    env = environment_info.gather(args.image)
    run_id = environment_info.run_id(env)
    print(f"  {env}", flush=True)
    print(f"  run_id: {run_id}", flush=True)

    any_failed = False
    any_partial = False
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

        depths, max_ctx = derive_depths_for_model(model, prefill_tokens=PREFILL_TOKENS)
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
        moe = moe_params(gguf_metadata)
        block_count = model_block_count(gguf_metadata)
        dense_block_count = block_count if moe is None else None
        if moe is not None:
            print(f"  MoE model detected: expert_count={moe['expert_count']} "
                  f"expert_used_count={moe['expert_used_count']}", flush=True)
        elif dense_block_count is None:
            print(
                "  WARNING: dense model has no readable *.block_count; "
                "falling back to llama.cpp's full-offload request without --ngl discovery",
                flush=True,
            )
        max_probes, progress_detail = planned_probe_count(
            args, depths, moe, dense_block_count,
        )
        progress = ProgressTracker(
            ProbeProgress(total_probes=max_probes, expected_gap_seconds=DEPTH_COOLDOWN_SECONDS),
        )
        progress.start(detail=progress_detail)

        outcome = run_model_campaign(
            image=args.image, gpu_gids=gpu_gids, device=args.device, model=model,
            results_root=args.results_root, run_id=run_id, env=env,
            config=CampaignConfig(
                quick=args.quick, max_depth=args.max_depth,
                cooldown=args.cooldown, force=args.force,
            ),
            gguf_metadata=gguf_metadata, moe=moe,
            dense_block_count=dense_block_count,
            model_size_bytes=total_model_size_bytes(model),
            depths=depths, max_ctx=max_ctx,
            progress=progress,
            bench_config_cls=BenchConfig, run_one=run_one, auto_tune=auto_tune,
            sweep_moe_offload_quick=sweep_moe_offload_quick,
            sweep_moe_offload_thorough=sweep_moe_offload_thorough,
            preflight_dense_offload=preflight_dense_offload,
            sweep_dense_offload=sweep_dense_offload,
            model_slug=slug,
            prefill_tokens=PREFILL_TOKENS, generation_tokens=GENERATION_TOKENS,
            repetitions=REPETITIONS,
        )
        if outcome.failed:
            any_failed = True
        elif outcome.partial:
            any_partial = True

    if any_failed:
        sys.exit(1)
    if any_partial:
        sys.exit(2)


if __name__ == "__main__":
    main()
