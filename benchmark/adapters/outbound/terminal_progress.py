"""Terminal adapter for the benchmark application's progress domain."""

from __future__ import annotations

import sys
import time
from typing import Callable, TextIO

from domain.progress import ProbeProgress, ProgressSnapshot


class TerminalProgressReporter:
    """Render campaign progress for an interactive CLI without owning its rules."""

    def __init__(
        self,
        progress: ProbeProgress,
        *,
        output: TextIO = sys.stdout,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.progress = progress
        self.output = output
        self.clock = clock
        self._started_at: float | None = None

    def start(self, *, detail: str) -> None:
        print(
            f"  Progress plan: up to {self.progress.total_probes} Docker probes ({detail})",
            file=self.output,
            flush=True,
        )

    def before_probe(self, label: str) -> None:
        snapshot = self.progress.snapshot()
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

    def finish_probe(self, *, elapsed_seconds: float | None = None) -> ProgressSnapshot:
        if elapsed_seconds is None:
            if self._started_at is None:
                raise RuntimeError("finish_probe() called before before_probe()")
            elapsed_seconds = self.clock() - self._started_at
        self._started_at = None
        return self.progress.record_completion(elapsed_seconds)

    def prune(self, count: int, reason: str) -> ProgressSnapshot:
        before = self.progress.pruned
        snapshot = self.progress.record_pruning(count)
        actual = self.progress.pruned - before
        if actual:
            print(
                f"    [progress] pruned {actual} future probe(s): {reason}; "
                f"{snapshot.remaining} remaining",
                file=self.output,
                flush=True,
            )
        return snapshot

    def snapshot(self) -> ProgressSnapshot:
        return self.progress.snapshot()

    @staticmethod
    def _format_duration(seconds: int) -> str:
        minutes, seconds = divmod(seconds, 60)
        return f"~{minutes}m {seconds:02d}s" if minutes else f"~{seconds}s"
