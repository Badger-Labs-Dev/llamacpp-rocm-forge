"""Compatibility import for the terminal progress adapter.

New application code belongs under ``bench_app``. This module keeps existing
CLI scripts and external imports working while the layout is migrated.

Preserved: the constructor signature (``total_probes``, ``expected_gap_seconds``,
``output``, ``clock``) and the ``start``/``before_probe``/``finish_probe``/
``prune``/``snapshot`` method names, plus (via the properties below) the
previous top-level attributes (``total_probes``, ``completed``, ``pruned``,
``expected_gap_seconds``) that lived directly on the old ``ProgressTracker``
before this became a thin wrapper around ``ProbeProgress``.

Not preserved: ``finish_probe()`` and ``prune()`` now return a
``ProgressSnapshot`` instead of ``None`` (extra information, not a breaking
removal for callers that ignored the return value, which every caller in
this repo does).
"""

from bench_app.adapters.outbound.terminal_progress import TerminalProgressReporter
from bench_app.domain.progress import ProbeProgress, ProgressSnapshot

__all__ = ["ProgressTracker", "ProgressSnapshot"]


class ProgressTracker(TerminalProgressReporter):
    """Backward-compatible terminal reporter constructor."""

    def __init__(self, *, total_probes: int, expected_gap_seconds: int = 2, **kwargs) -> None:
        super().__init__(
            ProbeProgress(total_probes=total_probes, expected_gap_seconds=expected_gap_seconds),
            **kwargs,
        )

    # The pre-refactor ProgressTracker stored these directly on self; they
    # now live on self.progress (a ProbeProgress). Exposed as read-only
    # properties so any code still reading tracker.completed etc. keeps
    # working rather than silently losing the attribute.
    @property
    def total_probes(self) -> int:
        return self.progress.total_probes

    @property
    def completed(self) -> int:
        return self.progress.completed

    @property
    def pruned(self) -> int:
        return self.progress.pruned

    @property
    def expected_gap_seconds(self) -> int:
        return self.progress.expected_gap_seconds
