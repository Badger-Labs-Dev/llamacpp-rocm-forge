#!/usr/bin/env python3
"""Read key metadata out of a GGUF file's header without any GGUF library.

GGUF stores model metadata as a flat key-value section at the start of the
file (before tensor data), so this only needs struct.unpack over the first
few KB - no llama.cpp Python bindings, no `gguf` pip package, no loading the
whole (possibly tens-of-GB) file.

Usage:
    ./gguf_info.py model.gguf
    ./gguf_info.py model.gguf --json
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
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


def summarize(metadata: dict) -> dict:
    return {
        "architecture": metadata.get("general.architecture"),
        "name": metadata.get("general.name"),
        "quantization_version": metadata.get("general.quantization_version"),
        "context_length": context_length(metadata),
        "tensor_count": metadata.get("tensor_count"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help="Path to a .gguf file")
    parser.add_argument("--json", action="store_true", help="Print full raw metadata as JSON")
    args = parser.parse_args()

    if not args.model.is_file():
        sys.exit(f"Not a file: {args.model}")

    metadata = read_gguf_metadata(args.model)

    if args.json:
        print(json.dumps(metadata, indent=2, sort_keys=True, default=str))
        return

    summary = summarize(metadata)
    print(f"file:              {args.model}")
    print(f"architecture:      {summary['architecture']}")
    print(f"name:              {summary['name']}")
    print(f"tensor_count:      {summary['tensor_count']}")
    if summary["context_length"] is not None:
        print(f"context_length:    {summary['context_length']}")
    else:
        print("context_length:    NOT FOUND (no '*.context_length' key in this file)")


if __name__ == "__main__":
    main()
