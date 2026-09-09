"""Pure campaign-progress domain model.

A probe is one Docker/llama-bench invocation. The model distinguishes the
campaign's conservative upper bound from work eliminated by evidence.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProgressSnapshot:
    completed: int
    pruned: int
    remaining: int
    current_total: int
    worst_case_percent: int
    current_plan_percent: int
    eta_seconds: int | None


@dataclass
class ProbeProgress:
    """Accumulate probe outcomes without knowing how progress is presented."""

    total_probes: int
    expected_gap_seconds: int = 2
    completed: int = 0
    pruned: int = 0
    durations: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.total_probes < 0:
            raise ValueError("total_probes must be non-negative")

    def record_completion(self, elapsed_seconds: float) -> ProgressSnapshot:
        self.completed += 1
        self.durations.append(max(0.0, elapsed_seconds))
        return self.snapshot()

    def record_pruning(self, count: int) -> ProgressSnapshot:
        if count > 0:
            available = max(0, self.total_probes - self.completed - self.pruned)
            self.pruned += min(count, available)
        return self.snapshot()

    def snapshot(self) -> ProgressSnapshot:
        current_total = self.total_probes - self.pruned
        remaining = max(0, current_total - self.completed)
        eta_seconds = None
        if self.durations and remaining:
            eta_seconds = round(
                remaining * (statistics.median(self.durations) + self.expected_gap_seconds)
            )
        return ProgressSnapshot(
            completed=self.completed,
            pruned=self.pruned,
            remaining=remaining,
            current_total=current_total,
            worst_case_percent=self._percentage(self.completed, self.total_probes),
            current_plan_percent=self._percentage(self.completed, current_total),
            eta_seconds=eta_seconds,
        )

    @staticmethod
    def _percentage(numerator: int, denominator: int) -> int:
        return 100 if denominator == 0 else round((numerator / denominator) * 100)
