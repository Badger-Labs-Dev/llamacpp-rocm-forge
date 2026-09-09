"""Pure MoE offload boundary search: decides which --n-cpu-moe value to
try next, given a per-depth history of prior probe results, without
knowing how a probe is actually executed.

Extracted from run_bench.sweep_moe_offload_thorough()'s binary-search
loop, per docs/architecture.md's migration rule: this policy was
identified as genuine domain logic (not I/O) still trapped in the CLI
script - see the architecture doc's note under the Python domain/
description.

Two monotonicity assumptions justify the search shape (see the original
docstring in run_bench.sweep_moe_offload_thorough, preserved there):
(1) for a fixed depth, success in n_cpu_moe is monotonic - once a value
fits, every larger value fits too; (2) across depths, the fitting
boundary is non-decreasing - a value that failed at a shallower depth
also fails at any deeper one.

Usage: resolve_boundary() is a generator. The caller drives it by calling
next()/send(fits) in a loop, performing the actual probe I/O between
each step:

    gen = resolve_boundary(known_fail_floor=known_fail_floor, block_count=block_count)
    ncmoe = next(gen)
    try:
        while True:
            fits = do_the_real_probe(ncmoe)
            ncmoe = gen.send(fits)
    except StopIteration as stop:
        boundary, tested = stop.value
"""

from __future__ import annotations

from typing import Generator


def resolve_boundary(
    *, known_fail_floor: int, block_count: int,
) -> Generator[int, bool, tuple[int | None, dict[int, bool]]]:
    """Yield the next --n-cpu-moe candidate to probe; the caller sends
    back whether it fit. Returns (boundary, tested) via StopIteration.value
    once resolved: boundary is None if nothing fits even fully offloaded
    (block_count), otherwise the smallest n_cpu_moe that fits.
    """
    tested: dict[int, bool] = {}

    def record(ncmoe: int, fits: bool) -> None:
        tested[ncmoe] = fits

    # Fast path: does the previous depth's boundary still fit? If so it's
    # still optimal (monotonicity (2) rules out anything smaller working
    # now, and it demonstrably still works) - no search needed.
    lo = max(known_fail_floor, -1)
    starting_hint = lo + 1
    if starting_hint <= block_count:
        fits = yield starting_hint
        record(starting_hint, fits)
        if fits:
            return starting_hint, tested
    else:
        fits = False  # starting_hint already exceeds block_count

    # Binary search in (lo, block_count]. Verify the top end first - if
    # even fully offloaded doesn't fit, this depth is infeasible outright,
    # regardless of n_cpu_moe.
    top_fits = yield block_count
    record(block_count, top_fits)
    if not top_fits:
        return None, tested

    hi = block_count
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
    *, boundary: int, block_count: int, sample_count: int,
) -> tuple[int, ...]:
    """Additional --n-cpu-moe candidates above a resolved boundary, for
    the "how hard does throughput dive as I offload more" curve - a thin
    re-export of the planning module's spacing rule so callers driving
    resolve_boundary() don't need a second import for the related step."""
    from bench_app.domain.planning import thorough_extra_candidates
    return thorough_extra_candidates(boundary, block_count, sample_count)
