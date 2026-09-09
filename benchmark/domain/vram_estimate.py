"""Pure VRAM estimates used to plan dense-model GPU-layer probes.

The estimate is advisory. A real llama-bench probe is always the authority
because backend work buffers and driver allocations vary by model/build.
"""

from __future__ import annotations

from dataclasses import dataclass

F16_BYTES = 2
DEFAULT_OVERHEAD_BYTES = 2 * 1024**3


@dataclass(frozen=True)
class VramEstimate:
    model_size_bytes: int
    kv_cache_bytes: int
    overhead_bytes: int
    total_bytes: int
    gpu_vram_bytes: int

    @property
    def fits(self) -> bool:
        return self.total_bytes <= self.gpu_vram_bytes


def _architecture(metadata: dict) -> str | None:
    value = metadata.get("general.architecture")
    return value if isinstance(value, str) and value else None


def _positive_int(metadata: dict, key: str) -> int | None:
    value = metadata.get(key)
    return value if isinstance(value, int) and value > 0 else None


def estimate_full_offload_vram(
    *,
    metadata: dict,
    model_size_bytes: int,
    context_size: int,
    gpu_vram_bytes: int,
    overhead_bytes: int = DEFAULT_OVERHEAD_BYTES,
) -> VramEstimate | None:
    """Estimate full-offload VRAM for an f16 KV cache.

    Returns None when the GGUF lacks enough attention metadata to calculate
    KV size. It deliberately budgets the full context for every layer—even
    when sliding-window metadata exists—so hybrid attention cannot make the
    estimate optimistic. The estimate intentionally does not make
    a fit/fail decision for the campaign; it only informs the operator before
    the real probe.
    """
    arch = _architecture(metadata)
    if arch is None:
        # Tests and callers may provide namespaced metadata without retaining
        # general.architecture. Infer only when there is exactly one block key.
        block_keys = [key for key in metadata if key.endswith(".block_count")]
        if len(block_keys) != 1:
            return None
        arch = block_keys[0].removesuffix(".block_count")

    block_count = _positive_int(metadata, f"{arch}.block_count")
    head_count = _positive_int(metadata, f"{arch}.attention.head_count")
    kv_head_count = _positive_int(metadata, f"{arch}.attention.head_count_kv")
    embedding_length = _positive_int(metadata, f"{arch}.embedding_length")
    if block_count is None or kv_head_count is None:
        return None

    key_length = _positive_int(metadata, f"{arch}.attention.key_length")
    value_length = _positive_int(metadata, f"{arch}.attention.value_length")
    if key_length is None or value_length is None:
        if head_count is None or embedding_length is None or embedding_length % head_count:
            return None
        head_dim = embedding_length // head_count
        key_length = key_length or head_dim
        value_length = value_length or head_dim

    bytes_per_token_per_layer = kv_head_count * (
        key_length * F16_BYTES + value_length * F16_BYTES
    )
    kv_cache_bytes = context_size * block_count * bytes_per_token_per_layer
    total = model_size_bytes + kv_cache_bytes + overhead_bytes
    return VramEstimate(
        model_size_bytes=model_size_bytes,
        kv_cache_bytes=kv_cache_bytes,
        overhead_bytes=overhead_bytes,
        total_bytes=total,
        gpu_vram_bytes=gpu_vram_bytes,
    )
