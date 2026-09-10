"""Pure per-KV-dtype context-depth planning.

Static assessment can exclude only an exact GPU-resident allocation. Unknown
assessments stay runnable so the application layer retains the runtime probe.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from domain.kv_feasibility import (
    KvFeasibility,
    KvPlacement,
    KvRuntimeScope,
    RuntimeReservePolicy,
    assess_gpu_kv_feasibility,
)

KV_CACHE_TYPES = ("q4_0", "q8_0", "f16")


@dataclass(frozen=True)
class DtypeDepthPlan:
    """Eligibility decisions for one matching K/V cache dtype."""

    ctk: str
    ctv: str
    requested_depths: tuple[int, ...]
    runnable_depths: tuple[int, ...]
    eligible_depths: tuple[int, ...]
    unknown_depths: tuple[int, ...]
    exclusions: tuple[KvFeasibility, ...]
    assessments: tuple[KvFeasibility, ...]

    def to_payload(self) -> dict[str, object]:
        """Return a manifest-ready representation without doing I/O."""
        return {
            "ctk": self.ctk,
            "ctv": self.ctv,
            "requested_depths": list(self.requested_depths),
            "runnable_depths": list(self.runnable_depths),
            "eligible_depths": list(self.eligible_depths),
            "unknown_depths": list(self.unknown_depths),
            "exclusions": [asdict(result) for result in self.exclusions],
            "unknown": [
                asdict(result) for result in self.assessments if result.status == "unknown"
            ],
        }


@dataclass(frozen=True)
class KvDepthPlan:
    """Per-dtype static feasibility plan for an unchanged depth request."""

    requested_depths: tuple[int, ...]
    prefill_tokens: int
    dtype_plans: tuple[DtypeDepthPlan, ...]

    def to_payload(self) -> dict[str, object]:
        """Return a manifest-ready representation without doing I/O."""
        return {
            "requested_depths": list(self.requested_depths),
            "prefill_tokens": self.prefill_tokens,
            "dtype_plans": [dtype_plan.to_payload() for dtype_plan in self.dtype_plans],
        }


def _non_negative_ints(values: tuple[int, ...], *, field: str) -> tuple[int, ...]:
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
        raise ValueError(f"{field} must contain non-negative integers")
    return values


def build_kv_depth_plan(
    *,
    requested_depths: tuple[int, ...],
    prefill_tokens: int,
    metadata: Mapping[str, object],
    kv_offload_enabled: bool | None,
    kv_placement: KvPlacement | None,
    gpu_vram_bytes: int | None,
    runtime_scope: KvRuntimeScope | None,
    reserve_policy: RuntimeReservePolicy | None,
) -> KvDepthPlan:
    """Assess every requested depth independently for each canonical KV dtype.

    ``requested_depths`` are benchmark depths. The feasibility API receives the
    distinct llama.cpp allocation size ``depth + prefill_tokens`` exactly once.
    """
    requested_depths = _non_negative_ints(tuple(requested_depths), field="requested_depths")
    if (not isinstance(prefill_tokens, int) or isinstance(prefill_tokens, bool)
            or prefill_tokens < 0):
        raise ValueError("prefill_tokens must be a non-negative integer")

    dtype_plans: list[DtypeDepthPlan] = []
    for cache_type in KV_CACHE_TYPES:
        assessments = tuple(
            assess_gpu_kv_feasibility(
                metadata=metadata,
                context_size=depth + prefill_tokens,
                ctk=cache_type,
                ctv=cache_type,
                kv_offload_enabled=kv_offload_enabled,
                kv_placement=kv_placement,
                gpu_vram_bytes=gpu_vram_bytes,
                runtime_scope=runtime_scope,
                reserve_policy=reserve_policy,
            )
            for depth in requested_depths
        )
        eligible_depths = tuple(
            depth for depth, assessment in zip(requested_depths, assessments)
            if assessment.status == "eligible"
        )
        unknown_depths = tuple(
            depth for depth, assessment in zip(requested_depths, assessments)
            if assessment.status == "unknown"
        )
        dtype_plans.append(DtypeDepthPlan(
            ctk=cache_type,
            ctv=cache_type,
            requested_depths=requested_depths,
            runnable_depths=tuple(
                depth for depth, assessment in zip(requested_depths, assessments)
                if assessment.status != "excluded"
            ),
            eligible_depths=eligible_depths,
            unknown_depths=unknown_depths,
            exclusions=tuple(result for result in assessments if result.status == "excluded"),
            assessments=assessments,
        ))
    return KvDepthPlan(
        requested_depths=requested_depths,
        prefill_tokens=prefill_tokens,
        dtype_plans=tuple(dtype_plans),
    )


def deepest_runnable_depth(dtype_plan: DtypeDepthPlan) -> int | None:
    """Return the greatest requested depth that remains eligible or unknown."""
    return max(dtype_plan.runnable_depths, default=None)


def candidate_dtype_depth_targets(plan: KvDepthPlan) -> tuple[tuple[str, int], ...]:
    """Return execution targets in canonical dtype order, without ranking them."""
    return tuple(
        (dtype_plan.ctk, depth)
        for dtype_plan in plan.dtype_plans
        for depth in dtype_plan.runnable_depths
    )
