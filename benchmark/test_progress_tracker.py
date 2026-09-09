import argparse
import io
import unittest

import run_bench
from progress_tracker import ProgressTracker
from run_bench import BenchConfig, planned_probe_count, sweep_moe_offload_quick


class ProgressTrackerTests(unittest.TestCase):
    def test_counts_completed_and_pruned_probes(self):
        clock = [0.0]
        output = io.StringIO()
        tracker = ProgressTracker(
            total_probes=10,
            output=output,
            clock=lambda: clock[0],
            expected_gap_seconds=2,
        )

        tracker.before_probe("MoE depth=8192 ncmoe=10")
        clock[0] = 10.0
        tracker.finish_probe()
        tracker.prune(3, "larger depths cannot fit")

        snapshot = tracker.snapshot()
        self.assertEqual(snapshot.completed, 1)
        self.assertEqual(snapshot.pruned, 3)
        self.assertEqual(snapshot.remaining, 6)
        self.assertEqual(snapshot.current_total, 7)
        self.assertEqual(snapshot.worst_case_percent, 10)
        self.assertEqual(snapshot.current_plan_percent, 14)
        self.assertEqual(snapshot.eta_seconds, 72)

    def test_does_not_prune_below_completed_work(self):
        output = io.StringIO()
        tracker = ProgressTracker(total_probes=2, output=output)
        tracker.finish_probe(elapsed_seconds=1)
        tracker.prune(99, "defensive clamp")

        snapshot = tracker.snapshot()
        self.assertEqual(snapshot.completed, 1)
        self.assertEqual(snapshot.pruned, 1)
        self.assertEqual(snapshot.remaining, 0)
        self.assertEqual(snapshot.current_total, 1)
    def test_quick_moe_sweep_prunes_unreachable_deeper_depths(self):
        tracker = ProgressTracker(total_probes=15, output=io.StringIO())
        original_probe = run_bench.probe_moe_offload

        def fake_probe(**kwargs):
            kwargs["progress"].finish_probe(elapsed_seconds=1)
            return kwargs["depth"] == 0, 100.0

        run_bench.probe_moe_offload = fake_probe
        try:
            result = sweep_moe_offload_quick(
                image="unused", gpu_gids=[], model=run_bench.Path("model.gguf"),
                base_config=BenchConfig(), device="ROCm0", depths=(0, 2048, 6144),
                block_count=4, results_dir=run_bench.Path("/tmp/unused"), cooldown=0,
                progress=tracker,
            )
        finally:
            run_bench.probe_moe_offload = original_probe

        self.assertEqual(len(result["by_depth"]), 2)
        snapshot = tracker.snapshot()
        self.assertEqual(snapshot.completed, 10)
        self.assertEqual(snapshot.pruned, 5)
        self.assertEqual(snapshot.remaining, 0)

    def test_plans_full_thorough_moe_sweep_as_a_conservative_upper_bound(self):
        args = argparse.Namespace(
            full_sweep=True,
            calibrate=False,
            sweep_moe_offload=False,
            sweep_moe_offload_thorough=True,
        )

        total, detail = planned_probe_count(args, depths=(0, 2048, 6144), moe={"block_count": 40})

        # 3 KV types × 2 probe depths + 13 valid ubatch/batch pairs,
        # then 3 depths × (2 endpoint checks + ceil(log2(41)) bisection
        # probes + up to 4 throughput samples), then 2 final series × 3 depths.
        self.assertEqual(total, 61)
        self.assertEqual(detail, "19 tuning, 36 thorough MoE, 6 final curves")


if __name__ == "__main__":
    unittest.main()
