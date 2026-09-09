import argparse
import io
import json
import tempfile
import unittest
from pathlib import Path

import run_bench
from adapters.outbound.campaign_store import write_curve_summary
from adapters.outbound.terminal_progress import TerminalProgressReporter as ProgressTracker
from domain.progress import ProbeProgress
from generate_viewer_data import depth0_throughput, read_curve
from run_bench import BenchConfig, RunResult, planned_probe_count, sweep_moe_offload_quick


class ProgressTrackerTests(unittest.TestCase):
    def test_counts_completed_and_pruned_probes(self):
        clock = [0.0]
        output = io.StringIO()
        tracker = ProgressTracker(
            ProbeProgress(total_probes=10, expected_gap_seconds=2),
            output=output,
            clock=lambda: clock[0],
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
        tracker = ProgressTracker(ProbeProgress(total_probes=2), output=output)
        tracker.finish_probe(elapsed_seconds=1)
        tracker.prune(99, "defensive clamp")

        snapshot = tracker.snapshot()
        self.assertEqual(snapshot.completed, 1)
        self.assertEqual(snapshot.pruned, 1)
        self.assertEqual(snapshot.remaining, 0)
        self.assertEqual(snapshot.current_total, 1)
    def test_quick_moe_sweep_prunes_unreachable_deeper_depths(self):
        tracker = ProgressTracker(ProbeProgress(total_probes=15), output=io.StringIO())
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

    def test_curve_summary_preserves_ok_status_for_a_depth_that_ran_before_a_later_failure(self):
        # Regression test: a "partial" series (deeper depth failed/timed out)
        # must not blank out the depth(s) that actually succeeded - see
        # write_curve_summary()'s row_status logic and depth0_throughput()'s
        # reliance on a per-row "ok" status.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            jsonl_path = tmp_path / "run.jsonl"
            jsonl_path.write_text(
                json.dumps({"n_depth": 0, "n_prompt": 2048, "n_gen": 0, "avg_ts": 123.4, "avg_ns": 1}) + "\n",
                encoding="utf-8",
            )
            result = RunResult(
                model="example.gguf", series="prefill", config=BenchConfig(),
                status="partial", return_code=1, jsonl_path=jsonl_path,
                stderr_path=tmp_path / "run.stderr.log", command=[],
                depths_run=(0,), depths_skipped=(8192,),
                stop_reason="depth 8192 timed out",
            )
            summary_path = tmp_path / "curve_summary.csv"
            row_count = write_curve_summary([result], summary_path)

            self.assertEqual(row_count, 1)
            curve = read_curve(summary_path)
            self.assertEqual(curve[0]["status"], "ok")
            self.assertEqual(depth0_throughput(curve, "prefill"), 123.4)

    def test_read_curve_normalizes_non_finite_avg_ts_to_null(self):
        # Regression test: json.dumps emits the bare (invalid-JSON) tokens
        # NaN/Infinity for non-finite floats; read_curve must normalize
        # these to None before they reach results.json.
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "curve_summary.csv"
            summary_path.write_text(
                "series,n_depth,avg_ts,status\nprefill,0,nan,ok\ngeneration,0,inf,ok\n",
                encoding="utf-8",
            )
            curve = read_curve(summary_path)
            self.assertIsNone(curve[0]["avg_ts"])
            self.assertIsNone(curve[1]["avg_ts"])
            # Python's json module accepts NaN/Infinity as a non-standard
            # extension on both dump and load, so round-tripping alone
            # wouldn't catch a regression here - assert the invalid tokens
            # never appear in the serialized text.
            serialized = json.dumps(curve)
            self.assertNotIn("NaN", serialized)
            self.assertNotIn("Infinity", serialized)


if __name__ == "__main__":
    unittest.main()
