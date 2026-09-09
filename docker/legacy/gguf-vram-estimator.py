#!/usr/bin/env python3
"""Print conservative GGUF full-offload VRAM estimates by context size."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmark"))

from adapters.outbound.gguf_metadata import (  # noqa: E402
    context_length,
    read_gguf_metadata,
    total_model_size_bytes,
)
from domain.vram_estimate import estimate_full_offload_vram  # noqa: E402

DEFAULT_CONTEXTS = (4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576)


def format_mem(size_bytes: int) -> str:
    mib = size_bytes / 1024**2
    return f"{mib:8.2f} MiB" if mib < 1024 else f"{mib / 1024:8.2f} GiB"


def run_estimator(path: Path, contexts: list[int], overhead_gib: float) -> None:
    metadata = read_gguf_metadata(path)
    model_size = total_model_size_bytes(path)
    trained_context = context_length(metadata)
    if trained_context:
        contexts = sorted({ctx for ctx in contexts if ctx <= trained_context} | {trained_context})
    else:
        contexts = sorted(set(contexts))

    print(f"\n--- Model '{metadata.get('general.name', 'N/A')}' ---")
    if trained_context:
        print(f"Max Context: {trained_context:,} tokens")
    print(f"Model Size: {format_mem(model_size).strip()} (from file size)")
    print(f"Incl. Overhead: {overhead_gib:.2f} GiB")
    print("\n--- Conservative full-offload estimate ---")
    print(f"{'Context Size':>15s} | {'Context Memory':>15s} | {'Est. Total VRAM':>15s}")
    print("-" * 51)

    for context in contexts:
        estimate = estimate_full_offload_vram(
            metadata=metadata,
            model_size_bytes=model_size,
            context_size=context,
            gpu_vram_bytes=sys.maxsize,
            overhead_bytes=int(overhead_gib * 1024**3),
        )
        if estimate is None:
            raise ValueError("GGUF lacks attention metadata required for a KV-cache estimate")
        print(
            f"{context:>15,} | {format_mem(estimate.kv_cache_bytes):>15s} | "
            f"{format_mem(estimate.total_bytes):>15s}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate model weights + conservative f16 KV cache + runtime overhead.",
    )
    parser.add_argument("gguf_file", type=Path, help="GGUF path (any shard of a split model)")
    parser.add_argument("-c", "--contexts", nargs="+", type=int, default=list(DEFAULT_CONTEXTS))
    parser.add_argument(
        "--overhead", type=float, default=2.0,
        help="GiB reserved for compute buffers and drivers (default: 2.0)",
    )
    args = parser.parse_args()
    try:
        run_estimator(args.gguf_file, args.contexts, args.overhead)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
