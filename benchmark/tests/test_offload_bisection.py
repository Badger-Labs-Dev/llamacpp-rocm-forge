"""Tests for the pure MoE offload boundary-search domain logic."""

import unittest

from domain.offload_bisection import extra_throughput_samples, resolve_boundary


def drive(gen, fits_by_ncmoe: dict[int, bool]):
    """Run a resolve_boundary() generator to completion against a fixed
    fits-lookup, recording the probe order actually taken."""
    probed_order = []
    try:
        ncmoe = next(gen)
        while True:
            probed_order.append(ncmoe)
            ncmoe = gen.send(fits_by_ncmoe[ncmoe])
    except StopIteration as stop:
        boundary, tested = stop.value
    return boundary, tested, probed_order


class ResolveBoundaryTests(unittest.TestCase):
    def test_fast_path_reuses_previous_boundary_when_it_still_fits(self):
        # known_fail_floor=-1 means boundary 0 is the starting hint; if it
        # fits immediately, no binary search is needed at all.
        fits = {0: True}
        gen = resolve_boundary(known_fail_floor=-1, max_offload=40)
        boundary, tested, probed = drive(gen, fits)

        self.assertEqual(boundary, 0)
        self.assertEqual(probed, [0])
        self.assertEqual(tested, {0: True})

    def test_binary_search_finds_exact_boundary(self):
        # Boundary is 10: fits for ncmoe>=10, fails below.
        fits = {ncmoe: ncmoe >= 10 for ncmoe in range(41)}
        gen = resolve_boundary(known_fail_floor=-1, max_offload=40)
        boundary, tested, probed = drive(gen, fits)

        self.assertEqual(boundary, 10)
        # First probe is the fast-path hint (0), which fails here, then
        # verifies the top end (40), then bisects strictly between them.
        self.assertEqual(probed[0], 0)
        self.assertEqual(probed[1], 40)
        self.assertTrue(all(0 <= n <= 40 for n in probed))

    def test_nothing_fits_even_fully_offloaded_returns_none_boundary(self):
        fits = {ncmoe: False for ncmoe in range(41)}
        gen = resolve_boundary(known_fail_floor=-1, max_offload=40)
        boundary, tested, probed = drive(gen, fits)

        self.assertIsNone(boundary)
        self.assertEqual(probed, [0, 40])  # stops immediately after both fail

    def test_resuming_from_a_known_fail_floor_narrows_the_search(self):
        # Previous depth's boundary was 20 (known_fail_floor=19); this
        # depth's true boundary is 25, still within [20, block_count].
        fits = {ncmoe: ncmoe >= 25 for ncmoe in range(41)}
        gen = resolve_boundary(known_fail_floor=19, max_offload=40)
        boundary, tested, probed = drive(gen, fits)

        self.assertEqual(boundary, 25)
        self.assertTrue(all(n >= 20 for n in probed))


class ExtraThroughputSamplesTests(unittest.TestCase):
    def test_delegates_to_planning_module_spacing_rule(self):
        # Cross-check against the same values planning.thorough_extra_candidates
        # would produce directly, since this is a thin re-export.
        from domain.planning import thorough_extra_candidates
        self.assertEqual(
            extra_throughput_samples(boundary=10, max_offload=40, sample_count=3),
            thorough_extra_candidates(10, 40, 3),
        )


if __name__ == "__main__":
    unittest.main()
