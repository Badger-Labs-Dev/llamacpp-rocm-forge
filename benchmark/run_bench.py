#!/usr/bin/env python3
"""Run a llama-bench parameter sweep against a Docker container on this host.

Drives `docker run` + `llama-bench` directly, writing one JSONL file per
(model, series, config) plus a curve_summary.csv.

Depths (llama-bench's "-d") are derived per model from its GGUF
*.context_length metadata (see adapters/outbound/gguf_metadata.py):
every COMMON_CONTEXT_SIZES entry that fits under the trained context
length is tested, plus depth 0. Falls back to LEGACY_FIXED_DEPTHS if
context_length can't be read.

Tuning modes (pick one; --full-sweep is the general recommendation):

  (default)     Fixed config: ubatch=2048, batch=2048, ctk/ctv=f16, fa=auto.
  --ubatch N    Fixed config as above but with this ubatch.
  --calibrate   Legacy: sweep only UBATCH_CANDIDATES at depth 0. Kept for
                backward compatibility with earlier results directories.
  --full-sweep  Staged auto-tune across ubatch, batch, and KV cache dtype
                (f16/q8_0/q4_0) - see application/auto_tune.py.

Flash attention is always "auto", not swept - llama.cpp's own default
already picks the fused kernel when it applies, and forcing it on/off
doesn't help on architectures where it can't apply anyway. Quantized KV
cache (ctk/ctv != f16) requires FA on regardless; BenchConfig.validate()
enforces that.

For MoE models (detected from GGUF *.expert_count; a no-op for dense
models), --sweep-moe-offload sweeps --n-cpu-moe across depths to find
the minimum offload that fits and how throughput drops as more gets
offloaded - see application/moe_sweep.py. --sweep-moe-offload-thorough
binary-searches the exact boundary per depth instead of using fixed
evenly-spaced candidates.

Usage:
    ./run_bench.py --model /models/foo.gguf --full-sweep
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
from adapters.outbound.gguf_metadata import moe_params, read_gguf_metadata
from adapters.outbound.model_resolution import (
    derive_depths_for_model,
    model_slug,
    resolve_model_reference,
)
from adapters.outbound.terminal_progress import TerminalProgressReporter as ProgressTracker
from application.auto_tune import (
    UBATCH_CANDIDATES,
    auto_tune,
    probe_depths,
    valid_batch_grid_count,
)
from application.calibration import pick_best_ubatch
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


def planned_probe_count(args: argparse.Namespace, depths: tuple[int, ...], moe: dict | None) -> tuple[int, str]:
    """CLI adapter around the domain campaign-budget calculation."""
    should_sweep_moe = moe is not None and (
        args.full_sweep or args.sweep_moe_offload or args.sweep_moe_offload_thorough
    )
    budget = campaign_budget(
        depth_count=len(depths),
        full_sweep=args.full_sweep,
        calibrate=args.calibrate,
        valid_batch_pairs=valid_batch_grid_count(),
        kv_type_count=len(KV_CACHE_TYPES),
        tuning_depth_count=len(probe_depths(depths)),
        moe_block_count=(moe or {}).get("block_count") if should_sweep_moe else None,
        thorough_moe=args.sweep_moe_offload_thorough,
        quick_candidate_count=MOE_QUICK_CANDIDATE_COUNT,
        thorough_extra_sample_count=MOE_EXTRA_SAMPLE_COUNT,
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
                         help="Staged auto-tune across ubatch, batch, and KV cache dtype; "
                              "also runs the quick MoE offload sweep when GGUF metadata "
                              "identifies the model as MoE")
    parser.add_argument("--sweep-moe-offload", action="store_true",
                         help="MoE models only (auto-detected from GGUF metadata, no-op for dense "
                              "models): also sweep --n-cpu-moe across depths, evenly-spaced "
                              "candidates. Answers 'what's the minimum --n-cpu-moe that fits at "
                              "each context depth, and how much does throughput drop as I offload "
                              "more'. Runs after --full-sweep's ubatch/batch/KV stages (or after "
                              "the fixed/--calibrate config if --full-sweep isn't also given), "
                              "using that config as the base.")
    parser.add_argument("--sweep-moe-offload-thorough", action="store_true",
                         help="Like --sweep-moe-offload, but binary-searches for the exact "
                              "min_ncmoe_that_fits per depth instead of testing fixed evenly-spaced "
                              "candidates - more precise, similar or lower cost thanks to reusing "
                              "each depth's boundary as a starting point for the next.")
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
        max_probes, progress_detail = planned_probe_count(args, depths, moe)
        progress = ProgressTracker(
            ProbeProgress(total_probes=max_probes, expected_gap_seconds=DEPTH_COOLDOWN_SECONDS),
        )
        progress.start(detail=progress_detail)

        outcome = run_model_campaign(
            image=args.image, gpu_gids=gpu_gids, device=args.device, model=model,
            results_root=args.results_root, run_id=run_id, env=env,
            config=CampaignConfig(
                full_sweep=args.full_sweep, calibrate=args.calibrate,
                sweep_moe_offload=args.sweep_moe_offload,
                sweep_moe_offload_thorough=args.sweep_moe_offload_thorough,
                max_depth=args.max_depth, ubatch=args.ubatch, batch=args.batch,
                ctk=args.ctk, ctv=args.ctv, flash_attn=args.flash_attn,
                cooldown=args.cooldown, force=args.force,
            ),
            gguf_metadata=gguf_metadata, moe=moe, depths=depths, max_ctx=max_ctx,
            progress=progress,
            bench_config_cls=BenchConfig, run_one=run_one, auto_tune=auto_tune,
            pick_best_ubatch=pick_best_ubatch,
            sweep_moe_offload_quick=sweep_moe_offload_quick,
            sweep_moe_offload_thorough=sweep_moe_offload_thorough,
            model_slug=slug, ubatch_candidates=UBATCH_CANDIDATES,
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
