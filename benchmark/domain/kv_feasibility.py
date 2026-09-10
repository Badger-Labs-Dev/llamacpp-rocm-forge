"""Pure, conservative GPU KV-cache capacity assessment.

This module only performs exact accounting for the explicitly supported
conventional dense layout. It never reads a GGUF, inspects a GPU, or starts a
probe. Any unsupported layout or unvalidated reserve remains ``unknown`` so
callers retain the runtime feasibility probe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping


KvFeasibilityStatus = Literal["eligible", "excluded", "unknown"]
KvPlacement = Literal["gpu", "host", "unknown"]


@dataclass(frozen=True)
class KvRuntimeScope:
    """Identity of the measured configuration a reserve is valid for."""

    llama_cpp_commit: str
    gpu_identifier: str
    batch: int
    ubatch: int
    flash_attn: str
    kv_placement: KvPlacement


@dataclass(frozen=True)
class RuntimeReservePolicy:
    """A reserve whose scope has been externally validated.

    The assessor trusts a reserve only when it is positive, explicitly marked
    conservative, and declares support for the architecture being assessed.
    Evidence and scope validation belong outside this pure domain module.
    """

    bytes: int | None
    supported_architectures: frozenset[str]
    is_conservative: bool
    assumptions: tuple[str, ...]
    scope: KvRuntimeScope


@dataclass(frozen=True)
class KvFeasibility:
    status: KvFeasibilityStatus
    ctk: str
    ctv: str
    context_size: int
    kv_placement: KvPlacement
    kv_cache_bytes: int | None
    runtime_reserve_bytes: int | None
    gpu_vram_bytes: int | None
    required_gpu_bytes: int | None
    reason: str | None
    assumptions: tuple[str, ...]


_DTYPE_BYTES = {"f16": (1, 2), "q4_0": (32, 18), "q8_0": (32, 34)}
_SUPPORTED_ARCHITECTURES = frozenset({"llama"})
_REQUIRED_METADATA_SUFFIXES = (
    "block_count",
    "attention.head_count_kv",
    "attention.key_length",
    "attention.value_length",
)


def _positive_int(metadata: Mapping[str, object], key: str) -> int | None:
    value = metadata.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _unknown(*, ctk: str, ctv: str, context_size: int, gpu_vram_bytes: int | None,
             reason: str, assumptions: tuple[str, ...] = ()) -> KvFeasibility:
    return KvFeasibility(
        status="unknown", ctk=ctk, ctv=ctv, context_size=context_size,
        kv_placement="unknown", kv_cache_bytes=None, runtime_reserve_bytes=None,
        gpu_vram_bytes=gpu_vram_bytes, required_gpu_bytes=None, reason=reason,
        assumptions=assumptions,
    )


def _tensor_bytes(*, row_element_count: int, row_count: int, dtype: str) -> int | None:
    type_shape = _DTYPE_BYTES.get(dtype)
    if type_shape is None:
        return None
    block_elements, block_bytes = type_shape
    if row_element_count % block_elements:
        return None
    return row_count * (row_element_count // block_elements) * block_bytes


def _conventional_dense_kv_bytes(metadata: Mapping[str, object], *, architecture: str,
                                  context_size: int, ctk: str, ctv: str) -> tuple[int | None, str | None]:
    values: dict[str, int] = {}
    for suffix in _REQUIRED_METADATA_SUFFIXES:
        key = f"{architecture}.{suffix}"
        value = _positive_int(metadata, key)
        if value is None:
            return None, f"missing or invalid exact metadata: {key}"
        values[suffix] = value

    block_count = values["block_count"]
    kv_heads = values["attention.head_count_kv"]
    key_row_elements = kv_heads * values["attention.key_length"]
    value_row_elements = kv_heads * values["attention.value_length"]
    key_bytes = _tensor_bytes(
        row_element_count=key_row_elements, row_count=context_size, dtype=ctk,
    )
    value_bytes = _tensor_bytes(
        row_element_count=value_row_elements, row_count=context_size, dtype=ctv,
    )
    if key_bytes is None or value_bytes is None:
        return None, "unsupported cache dtype or tensor shape is not block-aligned"
    return block_count * (key_bytes + value_bytes), None


def assess_gpu_kv_feasibility(
    *,
    metadata: Mapping[str, object],
    context_size: int,
    ctk: str,
    ctv: str,
    kv_offload_enabled: bool | None,
    kv_placement: KvPlacement | None,
    gpu_vram_bytes: int | None,
    runtime_scope: KvRuntimeScope | None,
    reserve_policy: RuntimeReservePolicy | None,
) -> KvFeasibility:
    """Assess only exact GPU KV allocation against an externally validated reserve.

    The supported ``llama`` handler assumes every declared block has the same
    conventional K/V layout described by the four required metadata keys. Hybrid
    and unknown architectures return ``unknown``. Model weights are deliberately
    absent from this calculation.
    ``kv_offload_enabled`` alone is not placement evidence: the calibration
    shows llama.cpp can enable offload while selecting CPU cache buffers. A
    caller must supply an independently established ``kv_placement`` of
    ``"gpu"`` before this function can assess GPU capacity.
    """
    if not isinstance(context_size, int) or isinstance(context_size, bool) or context_size <= 0:
        return _unknown(
            ctk=ctk, ctv=ctv, context_size=context_size, gpu_vram_bytes=gpu_vram_bytes,
            reason="context_size must be a positive integer",
        )
    if kv_placement == "host":
        if kv_offload_enabled is True:
            return _unknown(
                ctk=ctk, ctv=ctv, context_size=context_size, gpu_vram_bytes=gpu_vram_bytes,
                reason="host KV placement conflicts with enabled KV offload",
            )
        return KvFeasibility(
            status="eligible", ctk=ctk, ctv=ctv, context_size=context_size,
            kv_placement="host", kv_cache_bytes=None, runtime_reserve_bytes=None,
            gpu_vram_bytes=gpu_vram_bytes, required_gpu_bytes=None,
            reason="KV cache is explicitly host-resident; host capacity is not assessed",
            assumptions=("explicit host KV placement",),
        )
    if kv_placement != "gpu":
        return _unknown(
            ctk=ctk, ctv=ctv, context_size=context_size, gpu_vram_bytes=gpu_vram_bytes,
            reason="GPU KV placement is unknown",
        )
    if kv_offload_enabled is not True:
        return _unknown(
            ctk=ctk, ctv=ctv, context_size=context_size, gpu_vram_bytes=gpu_vram_bytes,
            reason="GPU KV placement conflicts with disabled or unknown KV offload",
        )

    architecture = metadata.get("general.architecture")
    if not isinstance(architecture, str) or architecture not in _SUPPORTED_ARCHITECTURES:
        return _unknown(
            ctk=ctk, ctv=ctv, context_size=context_size, gpu_vram_bytes=gpu_vram_bytes,
            reason="unsupported architecture for exact GPU KV accounting",
        )
    assumptions = (
        "conventional dense attention in every block",
        *(f"metadata: {architecture}.{suffix}" for suffix in _REQUIRED_METADATA_SUFFIXES),
    )
    kv_cache_bytes, error = _conventional_dense_kv_bytes(
        metadata, architecture=architecture, context_size=context_size, ctk=ctk, ctv=ctv,
    )
    if error is not None:
        return _unknown(
            ctk=ctk, ctv=ctv, context_size=context_size, gpu_vram_bytes=gpu_vram_bytes,
            reason=error, assumptions=assumptions,
        )
    if (not isinstance(gpu_vram_bytes, int) or isinstance(gpu_vram_bytes, bool)
            or gpu_vram_bytes <= 0):
        return _unknown(
            ctk=ctk, ctv=ctv, context_size=context_size, gpu_vram_bytes=gpu_vram_bytes,
            reason="gpu_vram_bytes must be a positive integer for GPU KV assessment",
            assumptions=assumptions,
        )
    if (reserve_policy is None or not isinstance(reserve_policy.bytes, int)
            or isinstance(reserve_policy.bytes, bool) or reserve_policy.bytes <= 0
            or not reserve_policy.is_conservative
            or architecture not in reserve_policy.supported_architectures):
        return _unknown(
            ctk=ctk, ctv=ctv, context_size=context_size, gpu_vram_bytes=gpu_vram_bytes,
            reason="no conservative validated runtime reserve applies to this architecture",
            assumptions=assumptions,
        )
    if (runtime_scope is None or reserve_policy.scope != runtime_scope
            or runtime_scope.kv_placement != "gpu"):
        return _unknown(
            ctk=ctk, ctv=ctv, context_size=context_size, gpu_vram_bytes=gpu_vram_bytes,
            reason="runtime reserve scope does not exactly match GPU KV assessment",
            assumptions=assumptions,
        )

    assert isinstance(kv_cache_bytes, int)
    reserve_bytes = reserve_policy.bytes
    assert isinstance(reserve_bytes, int) and not isinstance(reserve_bytes, bool)
    required_gpu_bytes = kv_cache_bytes + reserve_bytes
    status: KvFeasibilityStatus = "excluded" if required_gpu_bytes > gpu_vram_bytes else "eligible"
    return KvFeasibility(
        status=status, ctk=ctk, ctv=ctv, context_size=context_size,
        kv_placement="gpu", kv_cache_bytes=kv_cache_bytes,
        runtime_reserve_bytes=reserve_bytes, gpu_vram_bytes=gpu_vram_bytes,
        required_gpu_bytes=required_gpu_bytes,
        reason=(
            "exact GPU KV cache plus validated runtime reserve exceeds GPU VRAM"
            if status == "excluded" else None
        ),
        assumptions=(*assumptions, *reserve_policy.assumptions),
    )
