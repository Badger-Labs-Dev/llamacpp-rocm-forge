"""Progress accounting for benchmark Docker probes.

The tracker owns one model campaign's probe budget. Callers report only that a
probe started, finished, or that a known group of future probes is impossible.
It keeps the conservative initial maximum separate from the smaller plan that
remains after pruning.
"""

from __future__ import annotations

import statistics
import sys
import time
from dataclasses import dataclass
from typing import Callable, TextIO


@dataclass(frozen=True)
class ProgressSnapshot:
    completed: int
    pruned: int
    remaining: int
    current_total: int
    worst_case_percent: int
    current_plan_percent: int
    eta_seconds: int | None


class ProgressTracker:
    """Track Docker-probe completion against a conservative campaign plan."""

    def __init__(
        self,
        *,
        total_probes: int,
        output: TextIO = sys.stdout,
        clock: Callable[[], float] = time.monotonic,
        expected_gap_seconds: int = 2,
    ) -> None:
        if total_probes < 0:
            raise ValueError("total_probes must be non-negative")
        self.total_probes = total_probes
        self.output = output
        self.clock = clock
        self.expected_gap_seconds = expected_gap_seconds
        self.completed = 0
        self.pruned = 0
        self._durations: list[float] = []
        self._started_at: float | None = None

    def start(self, *, detail: str) -> None:
        print(f"  Progress plan: up to {self.total_probes} Docker probes ({detail})", file=self.output, flush=True)

    def before_probe(self, label: str) -> None:
        snapshot = self.snapshot()
        message = (
            f"    [progress] {snapshot.completed} completed · {snapshot.pruned} pruned · "
            f"{snapshot.remaining} remaining "
            f"({snapshot.worst_case_percent}% worst-case, "
            f"{snapshot.current_plan_percent}% current plan)"
        )
        if snapshot.eta_seconds is not None:
            message += f" · ETA {self._format_duration(snapshot.eta_seconds)}"
        print(f"{message}\n    Next: {label}", file=self.output, flush=True)
        self._started_at = self.clock()

    def finish_probe(self, *, elapsed_seconds: float | None = None) -> None:
        if elapsed_seconds is None:
            if self._started_at is None:
                raise RuntimeError("finish_probe() called before before_probe()")
            elapsed_seconds = self.clock() - self._started_at
        self._started_at = None
        self.completed += 1
        self._durations.append(max(0.0, elapsed_seconds))

    def prune(self, count: int, reason: str) -> None:
        if count <= 0:
            return
        max_prunable = max(0, self.total_probes - self.completed - self.pruned)
        actual = min(count, max_prunable)
        if actual == 0:
            return
        self.pruned += actual
        snapshot = self.snapshot()
        print(
            f"    [progress] pruned {actual} future probe(s): {reason}; "
            f"{snapshot.remaining} remaining",
            file=self.output,
            flush=True,
        )

    def snapshot(self) -> ProgressSnapshot:
        current_total = self.total_probes - self.pruned
        remaining = max(0, current_total - self.completed)
        worst_case_percent = self._percentage(self.completed, self.total_probes)
        current_plan_percent = self._percentage(self.completed, current_total)
        eta_seconds = None
        if self._durations and remaining:
            typical_duration = statistics.median(self._durations)
            eta_seconds = round(remaining * (typical_duration + self.expected_gap_seconds))
        return ProgressSnapshot(
            completed=self.completed,
            pruned=self.pruned,
            remaining=remaining,
            current_total=current_total,
            worst_case_percent=worst_case_percent,
            current_plan_percent=current_plan_percent,
            eta_seconds=eta_seconds,
        )

    @staticmethod
    def _percentage(numerator: int, denominator: int) -> int:
        if denominator == 0:
            return 100
        return round((numerator / denominator) * 100)

    @staticmethod
    def _format_duration(seconds: int) -> str:
        minutes, seconds = divmod(seconds, 60)
        if minutes:
            return f"~{minutes}m {seconds:02d}s"
        return f"~{seconds}s"
