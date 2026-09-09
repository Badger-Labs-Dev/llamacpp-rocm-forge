"""Characterization tests locking down run_bench.py's current behavior
before the hexagonal refactor's next steps (see docs/architecture.md's
migration rule: extract behind compatibility modules first, remove the
old path only once these keep passing).

subprocess.run is mocked throughout - these tests never invoke Docker.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run_bench
from adapters.outbound import docker_runner
from application import auto_tune as auto_tune_module
from application import moe_sweep as moe_sweep_module
from application import run_campaign as run_campaign_module
from application import run_curve as run_curve_module
from run_bench import BenchConfig, auto_tune, run_one


class FakeCompletedProcess:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode


def _write_jsonl_row(fileobj, n_depth: int, avg_ts: float) -> None:
    fileobj.write(json.dumps({"n_depth": n_depth, "avg_ts": avg_ts}) + "\n")


class RunOneCharacterizationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.results_dir = Path(self.tmp.name)
        # run_one() resolves host_model_path.resolve() and mounts its
        # parent - needs a real (if empty) file to exist.
        self.model_path = self.results_dir / "model.gguf"
        self.model_path.write_text("", encoding="utf-8")

    def test_all_depths_succeed_yields_ok_status_and_full_depths_run(self):
        def fake_run(cmd, stdout, stderr, timeout):
            _write_jsonl_row(stdout, n_depth=0, avg_ts=100.0)
            return FakeCompletedProcess(returncode=0)

        with mock.patch.object(docker_runner.subprocess, "run", side_effect=fake_run), \
             mock.patch.object(run_curve_module.time, "sleep"):
            result = run_one(
                image="unused", gpu_gids=[], host_model_path=self.model_path,
                series="prefill", config=BenchConfig(), device="ROCm0",
                depths=(0, 2048, 4096), results_dir=self.results_dir,
            )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.depths_run, (0, 2048, 4096))
        self.assertEqual(result.depths_skipped, ())
        self.assertEqual(result.return_code, 0)
        self.assertIsNone(result.stop_reason)

    def test_a_failing_depth_stops_and_skips_only_larger_depths(self):
        calls = []

        def fake_run(cmd, stdout, stderr, timeout):
            depth = int(cmd[cmd.index("-d") + 1])
            calls.append(depth)
            _write_jsonl_row(stdout, n_depth=depth, avg_ts=100.0)
            return FakeCompletedProcess(returncode=1 if depth == 2048 else 0)

        with mock.patch.object(docker_runner.subprocess, "run", side_effect=fake_run), \
             mock.patch.object(run_curve_module.time, "sleep"):
            result = run_one(
                image="unused", gpu_gids=[], host_model_path=self.model_path,
                series="prefill", config=BenchConfig(), device="ROCm0",
                depths=(0, 2048, 4096), results_dir=self.results_dir,
            )

        self.assertEqual(result.status, "partial")
        self.assertEqual(result.depths_run, (0,))
        self.assertEqual(result.depths_skipped, (4096,))
        self.assertEqual(calls, [0, 2048])  # depth 4096 never invoked at all
        self.assertIn("depth 2048 failed", result.stop_reason)

    def test_a_timeout_kills_the_container_and_skips_larger_depths(self):
        killed_containers = []

        def fake_run(cmd, stdout=None, stderr=None, timeout=None, capture_output=None):
            if cmd[:2] == ["docker", "kill"]:
                killed_containers.append(cmd[2])
                return FakeCompletedProcess(returncode=0)
            depth = int(cmd[cmd.index("-d") + 1])
            if depth == 0:
                _write_jsonl_row(stdout, n_depth=0, avg_ts=100.0)
                return FakeCompletedProcess(returncode=0)
            raise docker_runner.subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

        with mock.patch.object(docker_runner.subprocess, "run", side_effect=fake_run), \
             mock.patch.object(run_curve_module.time, "sleep"):
            result = run_one(
                image="unused", gpu_gids=[], host_model_path=self.model_path,
                series="prefill", config=BenchConfig(), device="ROCm0",
                depths=(0, 2048, 4096), results_dir=self.results_dir,
            )

        self.assertEqual(result.status, "partial")
        self.assertEqual(result.depths_run, (0,))
        self.assertEqual(result.depths_skipped, (4096,))
        self.assertEqual(len(killed_containers), 1)
        self.assertIn("timed out", result.stop_reason)
        # The container must not be left in the active-tracking set after
        # cleanup - a leak here would make a later interrupt kill a
        # long-gone container name instead of nothing.
        self.assertEqual(docker_runner._ACTIVE_CONTAINERS, set())

    def test_no_depth_succeeding_yields_failed_status(self):
        def fake_run(cmd, stdout, stderr, timeout):
            return FakeCompletedProcess(returncode=1)

        with mock.patch.object(docker_runner.subprocess, "run", side_effect=fake_run), \
             mock.patch.object(run_curve_module.time, "sleep"):
            result = run_one(
                image="unused", gpu_gids=[], host_model_path=self.model_path,
                series="prefill", config=BenchConfig(), device="ROCm0",
                depths=(0,), results_dir=self.results_dir,
            )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.depths_run, ())


class AutoTuneCharacterizationTests(unittest.TestCase):
    def test_stages_pick_the_highest_scoring_candidate_and_carry_it_forward(self):
        # probe_config() is the seam auto_tune() calls per candidate; fake
        # scores make kv=q8_0 win stage 1 and ub=512/b=1024 win stage 2,
        # then assert those propagate into the final BenchConfig and the
        # winner is recorded correctly per stage in the tuning log.
        def fake_probe_config(*, config, **kwargs):
            if config.ubatch == 2048 and config.batch == 2048 and config.n_cpu_moe == 0:
                # Stage 1 (KV sweep) holds ubatch/batch at BenchConfig
                # defaults; score by ctk so q8_0 wins.
                return {"f16": 100.0, "q8_0": 200.0, "q4_0": 50.0}[config.ctk]
            # Stage 2 (ubatch/batch grid): score so ub=512 b=1024 wins.
            return 300.0 if (config.ubatch, config.batch) == (512, 1024) else 100.0

        log: list[dict] = []
        with mock.patch.object(auto_tune_module, "probe_config", side_effect=fake_probe_config), \
             mock.patch.object(auto_tune_module.time, "sleep"):
            config = auto_tune(
                image="unused", gpu_gids=[], model=Path("model.gguf"), device="ROCm0",
                depths=(0, 4096), results_dir=Path("/tmp/unused"), cooldown=0, log=log,
            )

        self.assertEqual(config.ctk, "q8_0")
        self.assertEqual(config.ctv, "q8_0")
        self.assertEqual(config.ubatch, 512)
        self.assertEqual(config.batch, 1024)
        self.assertEqual(log[0]["stage"], "kv_cache_dtype")
        self.assertEqual(log[0]["winner"], "q8_0")
        self.assertEqual(log[1]["stage"], "ubatch_batch_grid")
        self.assertEqual(log[1]["winner"], "ub512_b1024")


class MainCharacterizationTests(unittest.TestCase):
    """End-to-end run of main() with a fixed config, mocking Docker,
    environment inspection, and argv - locks down the exact manifest/
    metadata.json shape written to disk so the filesystem-persistence
    extraction (adapters.outbound.campaign_store) can be graded
    against it."""

    def test_fixed_config_ok_run_completes_without_exit_and_writes_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_path = tmp_path / "model.gguf"
            model_path.write_text("", encoding="utf-8")
            results_root = tmp_path / "results"

            argv = [
                "run_bench.py", "--model", str(model_path),
                "--results-root", str(results_root), "--gpu-gid", "999",
                "--cooldown", "0", "--quick",
            ]

            def fake_run(cmd, stdout=None, stderr=None, timeout=None, capture_output=None):
                if stdout is not None:
                    _write_jsonl_row(stdout, n_depth=0, avg_ts=123.4)
                return FakeCompletedProcess(returncode=0)

            with mock.patch.object(run_bench.sys, "argv", argv), \
                 mock.patch.object(docker_runner.subprocess, "run", side_effect=fake_run), \
                 mock.patch.object(run_curve_module.time, "sleep"), \
                 mock.patch.object(run_campaign_module.time, "sleep"), \
                 mock.patch.object(
                     run_bench.environment_info, "gather",
                     return_value={"rocm_version": "7.2.4.1-1", "build_number": 9999},
                 ), \
                 mock.patch.object(run_bench, "read_gguf_metadata", return_value={}), \
                 mock.patch.object(run_bench, "moe_params", return_value=None):
                run_bench.main()  # must NOT raise SystemExit for a fully-ok run

            run_dir = results_root / "model" / "rocm7.2.4_llamacpp9999"
            metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            manifest = json.loads((run_dir / "campaign_manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(metadata["model_slug"], "model")
            self.assertEqual(metadata["status"], "finished")
            self.assertEqual(metadata["run_id"], "rocm7.2.4_llamacpp9999")
            self.assertAlmostEqual(metadata["generation_tok_s_mean"], 123.4)
            self.assertEqual(
                set(metadata.keys()),
                {
                    "model_slug", "model_filename", "model_architecture", "model_name",
                    "model_context_length", "model_moe", "run_id", "run_completed_at",
                    "mode", "environment", "final_config", "depths_tested",
                    "generation_tok_s_mean", "status",
                },
            )
            self.assertEqual(manifest["mode"], "quick")
            self.assertEqual(len(manifest["runs"]), 2)  # prefill + generation
            self.assertEqual(
                set(manifest.keys()),
                {
                    "image", "device", "depths", "context_length", "final_config",
                    "tuning_log", "moe_offload_curve", "repetitions", "prefill_tokens",
                    "generation_tokens", "mode", "completed_at", "summary_rows", "runs",
                },
            )
            self.assertTrue((run_dir / "campaign.finished").exists())
            self.assertTrue((run_dir / "curve_summary.csv").is_file())


class SweepMoeOffloadThoroughCharacterizationTests(unittest.TestCase):
    def test_full_sweep_output_shape_matches_expected_boundary_per_depth(self):
        # Boundary is 10 for depth=0 and stays 10 for depth=2048
        # (monotonicity across depths preserved by the fixed-fits table),
        # then extra throughput samples are recorded above the boundary.
        def fits(depth, ncmoe):
            return ncmoe >= 10

        def fake_probe_moe_offload(*, n_cpu_moe, depth, **kwargs):
            return fits(depth, n_cpu_moe), 100.0 + n_cpu_moe

        with mock.patch.object(moe_sweep_module, "probe_moe_offload", side_effect=fake_probe_moe_offload), \
             mock.patch.object(moe_sweep_module.time, "sleep"):
            result = run_bench.sweep_moe_offload_thorough(
                image="unused", gpu_gids=[], model=Path("model.gguf"),
                base_config=BenchConfig(), device="ROCm0", depths=(0, 2048),
                block_count=40, results_dir=Path("/tmp/unused"), cooldown=0,
            )

        self.assertEqual(result["mode"], "thorough")
        self.assertEqual(len(result["by_depth"]), 2)
        self.assertEqual(result["by_depth"][0]["depth"], 0)
        self.assertEqual(result["by_depth"][0]["min_ncmoe_that_fits"], 10)
        self.assertEqual(result["by_depth"][1]["depth"], 2048)
        self.assertEqual(result["by_depth"][1]["min_ncmoe_that_fits"], 10)
        for point in result["by_depth"]:
            for entry in point["results"]:
                self.assertIn(entry["status"], ("ok", "failed"))
                if entry["status"] == "ok":
                    self.assertIsNotNone(entry["avg_ts"])
                else:
                    self.assertIsNone(entry["avg_ts"])

    def test_nothing_fits_stops_early_and_reports_null_boundary(self):
        with mock.patch.object(moe_sweep_module, "probe_moe_offload", return_value=(False, -1.0)), \
             mock.patch.object(moe_sweep_module.time, "sleep"):
            result = run_bench.sweep_moe_offload_thorough(
                image="unused", gpu_gids=[], model=Path("model.gguf"),
                base_config=BenchConfig(), device="ROCm0", depths=(0, 2048, 4096),
                block_count=40, results_dir=Path("/tmp/unused"), cooldown=0,
            )

        self.assertEqual(len(result["by_depth"]), 1)  # stopped after depth 0
        self.assertIsNone(result["by_depth"][0]["min_ncmoe_that_fits"])


if __name__ == "__main__":
    unittest.main()
