"""Model resolution adapter: turns a --model argument (host path or
Hugging Face reference) into a local GGUF file, and derives filesystem-safe
identity and benchmark depths from that file's GGUF metadata.
"""

from __future__ import annotations

import re
from pathlib import Path

from adapters.outbound.gguf_metadata import context_length, read_gguf_metadata
from adapters.outbound import huggingface_models as hf_models

# Common context-window sizes seen across model releases (powers of two, plus
# the odd-but-common 24576/49152 seen in some Qwen configs). Depths are
# derived from whichever of these fit under a model's trained
# *.context_length, rather than a single fixed list applied to every model
# regardless of what it actually supports - see gguf_metadata.py.
COMMON_CONTEXT_SIZES = (
    2048, 4096, 8192, 16384, 24576, 32768, 49152, 65536, 98304, 131072, 262144,
)
LEGACY_FIXED_DEPTHS = (0, 8192, 16384, 24576, 32768, 40960, 49152, 57344, 65536)


def resolve_model_reference(reference: str) -> Path:
    """Resolve a --model argument to a local file path: a Hugging Face
    reference (hf://, org/repo, org/repo:QUANT) via the HF cache, or a
    plain host path taken as-is.

    Raises hf_models.HfReferenceError if an HF reference can't be resolved.
    Does not resolve symlinks - HF cache entries are named model.gguf via
    a symlink to a content-hash blob, and resolving too early would make
    model_slug() use the meaningless blob hash instead of the real model
    filename. Docker-mount symlink resolution happens separately, only at
    the point of mounting.
    """
    if hf_models.looks_like_hf_reference(reference):
        return Path(hf_models.resolve_hf_reference(reference))
    return Path(reference).expanduser().absolute()


def model_key(model_path: str) -> str:
    return Path(model_path).name


def model_slug(model_path: Path) -> str:
    """Filesystem-safe identity for a model: its filename without the
    .gguf extension, lowercased, non-alphanumerics collapsed to '-'."""
    stem = model_path.stem
    slug = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
    return slug or "model"


def derive_depths_for_model(
    model_path: Path, *, prefill_tokens: int,
) -> tuple[tuple[int, ...], int | None]:
    """Pick depths to test for this model: common context sizes that fit
    under its trained context_length, converted to llama-bench "-d" values.

    llama-bench's -d is how much KV cache is already "full" before the timed
    prefill/generation runs, so a context size C is tested at depth
    C - prefill_tokens (the prefill run itself consumes prefill_tokens more).
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
        depth = max(0, ctx_size - prefill_tokens)
        depths.add(depth)
    return tuple(sorted(depths)), max_ctx
