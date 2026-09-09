"""Read key metadata out of a GGUF file's header without any GGUF library.

GGUF stores model metadata as a flat key-value section at the start of the
file (before tensor data), so this only needs struct.unpack over the first
few KB - no llama.cpp Python bindings, no `gguf` pip package, no loading the
whole (possibly tens-of-GB) file.

Library-only: this has no CLI. The only consumer is
adapters/outbound/model_resolution.py (context_length, for deriving
benchmark depths) and run_bench.py (moe_params, for detecting/sizing a
MoE model's --n-cpu-moe sweep).
"""

from __future__ import annotations

import struct
from pathlib import Path

# GGUF value type codes -> (struct format char, size in bytes). Types 8
# (string) and 9 (array) are handled separately since they're variable-length.
_SCALAR_TYPES = {
    0: ("B", 1),   # uint8
    1: ("b", 1),   # int8
    2: ("H", 2),   # uint16
    3: ("h", 2),   # int16
    4: ("I", 4),   # uint32
    5: ("i", 4),   # int32
    6: ("f", 4),   # float32
    7: ("?", 1),   # bool
    10: ("Q", 8),  # uint64
    11: ("q", 8),  # int64
    12: ("d", 8),  # float64
}


def _read_str(f) -> str:
    (length,) = struct.unpack("<Q", f.read(8))
    return f.read(length).decode("utf-8", errors="replace")


def _read_value(f, value_type: int):
    if value_type == 8:  # string
        return _read_str(f)
    if value_type == 9:  # array
        elem_type, count = struct.unpack("<IQ", f.read(12))
        return [_read_value(f, elem_type) for _ in range(count)]
    if value_type not in _SCALAR_TYPES:
        raise ValueError(f"Unknown GGUF value type: {value_type}")
    fmt, size = _SCALAR_TYPES[value_type]
    (value,) = struct.unpack("<" + fmt, f.read(size))
    return value


def read_gguf_metadata(path: Path) -> dict:
    """Read every key/value pair from a GGUF file's metadata header."""
    with path.open("rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file (bad magic {magic!r})")
        (version,) = struct.unpack("<I", f.read(4))
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

        metadata: dict = {"gguf_version": version, "tensor_count": n_tensors}
        for _ in range(n_kv):
            key = _read_str(f)
            (value_type,) = struct.unpack("<I", f.read(4))
            value = _read_value(f, value_type)
            metadata[key] = value
    return metadata


def context_length(metadata: dict) -> int | None:
    """Find the model's trained max context length from its metadata.

    The key is namespaced by architecture, e.g. qwen2.context_length,
    llama.context_length - there's no fixed key name, so scan for the
    suffix instead of hardcoding every known architecture.
    """
    for key, value in metadata.items():
        if key.endswith(".context_length") and isinstance(value, int):
            return value
    return None


def moe_params(metadata: dict) -> dict | None:
    """Find MoE parameters from a model's metadata, if it's a MoE model.

    Like context_length(), the key is namespaced by architecture (e.g.
    qwen35moe.expert_count) - scan for the suffix. Presence of
    *.expert_count > 0 is what actually distinguishes a MoE model from a
    dense one; expert_used_count is the "used" part of names like "A3B"
    (active experts per token - the compute cost, not the memory cost:
    every expert still has to be resident in VRAM, since the router can
    route to any of them on any given token). block_count is the number
    of transformer layers, which is what --n-cpu-moe/-ncmoe actually
    counts - it moves the MoE feed-forward weights of the first N layers
    to CPU RAM, valid range 0..block_count.

    Returns None for dense models (no expert_count, or expert_count <= 1).
    """
    expert_count = None
    expert_used_count = None
    block_count = None
    for key, value in metadata.items():
        if key.endswith(".expert_count") and isinstance(value, int):
            expert_count = value
        elif key.endswith(".expert_used_count") and isinstance(value, int):
            expert_used_count = value
        elif key.endswith(".block_count") and isinstance(value, int):
            block_count = value

    if not expert_count or expert_count <= 1:
        return None

    return {
        "expert_count": expert_count,
        "expert_used_count": expert_used_count,
        "block_count": block_count,
    }
