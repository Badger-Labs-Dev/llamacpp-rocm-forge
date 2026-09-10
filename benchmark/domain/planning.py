"""Pure planning rules for benchmark campaigns."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeBudget:
    total: int
    parts: tuple[tuple[str, int], ...]

    @property
    def detail(self) -> str:
        return ", ".join(f"{count} {name}" for name, count in self.parts)


def quick_moe_candidates(block_count: int, candidate_count: int = 5) -> tuple[int, ...]:
    if block_count <= 0:
        return (0,)
    if candidate_count <= 1 or block_count < candidate_count - 1:
        return tuple(range(block_count + 1))
    step = block_count / (candidate_count - 1)
    return tuple(sorted({round(index * step) for index in range(candidate_count)}))


def thorough_extra_candidates(
    boundary: int,
    block_count: int,
    sample_count: int = 3,
) -> tuple[int, ...]:
    remaining = block_count - boundary
    if remaining <= 0 or sample_count <= 0:
        return ()
    step = max(1, remaining // sample_count)
    return tuple(range(boundary + step, block_count, step))


def thorough_max_probes_per_depth(block_count: int, sample_count: int = 3) -> int:
    max_extra = max(
        (len(thorough_extra_candidates(boundary, block_count, sample_count))
         for boundary in range(block_count + 1)),
        default=0,
    )
    return offload_boundary_max_probes(block_count) + max_extra


def offload_boundary_max_probes(max_offload: int) -> int:
    """Conservative endpoint-check + bisection probe bound."""
    return 2 + math.ceil(math.log2(max_offload + 1))


def campaign_budget(
    *,
    depth_count: int,
    quick: bool,
    kv_type_count: int,
    tuning_depth_count: int,
    moe_block_count: int | None,
    quick_candidate_count: int = 5,
    thorough_extra_sample_count: int = 3,
    dense_block_count: int | None = None,
    dense_quick_extra_sample_count: int = 1,
) -> ProbeBudget:
    parts: list[tuple[str, int]] = []
    if not quick:
        parts.append(("tuning", kv_type_count * tuning_depth_count))

    if moe_block_count:
        count = (
            depth_count * len(quick_moe_candidates(moe_block_count, quick_candidate_count))
            if quick
            else depth_count * thorough_max_probes_per_depth(moe_block_count, thorough_extra_sample_count)
        )
        parts.append((("quick MoE" if quick else "thorough MoE"), count))

    if dense_block_count:
        max_gpu_layers = dense_block_count + 1
        sample_count = (
            dense_quick_extra_sample_count if quick else thorough_extra_sample_count
        )
        if not quick:
            parts.append((
                "dense preflight",
                offload_boundary_max_probes(max_gpu_layers),
            ))
        parts.append((
            "dense offload",
            depth_count * (offload_boundary_max_probes(max_gpu_layers) + sample_count),
        ))

    parts.append(("final curves", 2 * depth_count))
    return ProbeBudget(total=sum(count for _, count in parts), parts=tuple(parts))
