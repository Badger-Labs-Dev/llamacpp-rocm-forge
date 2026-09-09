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
    return 2 + math.ceil(math.log2(block_count + 1)) + max_extra


def campaign_budget(
    *,
    depth_count: int,
    quick: bool,
    valid_batch_pairs: int,
    kv_type_count: int,
    tuning_depth_count: int,
    moe_block_count: int | None,
    quick_candidate_count: int = 5,
    thorough_extra_sample_count: int = 3,
) -> ProbeBudget:
    parts: list[tuple[str, int]] = []
    if not quick:
        parts.append(("tuning", kv_type_count * tuning_depth_count + valid_batch_pairs))

    if moe_block_count:
        count = (
            depth_count * len(quick_moe_candidates(moe_block_count, quick_candidate_count))
            if quick
            else depth_count * thorough_max_probes_per_depth(moe_block_count, thorough_extra_sample_count)
        )
        parts.append((("quick MoE" if quick else "thorough MoE"), count))

    parts.append(("final curves", 2 * depth_count))
    return ProbeBudget(total=sum(count for _, count in parts), parts=tuple(parts))
