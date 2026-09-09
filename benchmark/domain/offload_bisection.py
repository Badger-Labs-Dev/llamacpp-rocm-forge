"""Pure offload boundary search for a monotonic CPU-offload count.

Both supported knobs use the same domain rule: increasing the count frees
VRAM. For MoE the count is --n-cpu-moe; for dense models it is
``n_cpu_layers = max_gpu_layers - ngl``. This module decides which count to
try next without knowing how a probe is executed.

Two monotonicity assumptions justify the search shape: (1) for a fixed
depth, success in the CPU-offload count is monotonic—once a value
fits, every larger value fits too; (2) across depths, the fitting
boundary is non-decreasing - a value that failed at a shallower depth
also fails at any deeper one.

Usage: resolve_boundary() is a generator. The caller drives it by calling
next()/send(fits) in a loop, performing the actual probe I/O between
each step:

    gen = resolve_boundary(known_fail_floor=known_fail_floor, max_offload=max_offload)
    candidate = next(gen)
    try:
        while True:
            fits = do_the_real_probe(candidate)
            candidate = gen.send(fits)
    except StopIteration as stop:
        boundary, tested = stop.value
"""

from __future__ import annotations

from typing import Generator


def resolve_boundary(
    *, known_fail_floor: int, max_offload: int,
) -> Generator[int, bool, tuple[int | None, dict[int, bool]]]:
    """Yield the next CPU-offload candidate to probe; the caller sends
    back whether it fit. Returns (boundary, tested) via StopIteration.value
    once resolved: boundary is None if nothing fits even fully offloaded
    (max_offload), otherwise the smallest offload count that fits.
    """
    tested: dict[int, bool] = {}

    def record(candidate: int, fits: bool) -> None:
        tested[candidate] = fits

    # Fast path: does the previous depth's boundary still fit? If so it's
    # still optimal (monotonicity (2) rules out anything smaller working
    # now, and it demonstrably still works) - no search needed.
    lo = max(known_fail_floor, -1)
    starting_hint = lo + 1
    if starting_hint <= max_offload:
        fits = yield starting_hint
        record(starting_hint, fits)
        if fits:
            return starting_hint, tested
    else:
        fits = False  # starting_hint already exceeds max_offload

    # Binary search in (lo, max_offload]. Verify the top end first - if
    # even fully offloaded doesn't fit, this depth is infeasible outright,
    # regardless of the concrete offload knob.
    top_fits = yield max_offload
    record(max_offload, top_fits)
    if not top_fits:
        return None, tested

    hi = max_offload
    search_lo = max(lo, starting_hint)
    while hi - search_lo > 1:
        mid = (search_lo + hi) // 2
        mid_fits = yield mid
        record(mid, mid_fits)
        if mid_fits:
            hi = mid
        else:
            search_lo = mid
    return hi, tested


def extra_throughput_samples(
    *, boundary: int, max_offload: int, sample_count: int,
) -> tuple[int, ...]:
    """Additional CPU-offload candidates above a resolved boundary, for
    the "how hard does throughput dive as I offload more" curve - a thin
    re-export of the planning module's spacing rule so callers driving
    resolve_boundary() don't need a second import for the related step."""
    from domain.planning import thorough_extra_candidates
    return thorough_extra_candidates(boundary, max_offload, sample_count)
