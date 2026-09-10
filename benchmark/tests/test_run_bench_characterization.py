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
    def test_tunes_only_kv_dtype_at_the_fixed_runtime_configuration(self):
        # probe_config() is auto_tune()'s public collaborator seam. Its
        # captured calls specify the tuning contract without invoking Docker.
        calls = []

        def fake_probe_config(*, config, **kwargs):
            calls.append((config, kwargs["depths"]))
            return {"f16": 100.0, "q8_0": 200.0, "q4_0": 50.0}[config.ctk]

        log: list[dict] = []
        with mock.patch.object(auto_tune_module, "probe_config", side_effect=fake_probe_config), \
             mock.patch.object(auto_tune_module.time, "sleep"):
            config = auto_tune(
                image="unused", gpu_gids=[], model=Path("model.gguf"), device="ROCm0",
                depths=(0, 4096), results_dir=Path("/tmp/unused"), cooldown=0, log=log,
            )

        self.assertEqual(config.ctk, "q8_0")
        self.assertEqual(config.ctv, "q8_0")
        self.assertEqual((config.ubatch, config.batch), (2048, 2048))
        self.assertEqual(len(calls), 3)
        self.assertEqual({candidate.ctk for candidate, _ in calls}, {"f16", "q8_0", "q4_0"})
        self.assertTrue(all(
            (candidate.ubatch, candidate.batch, depths) == (2048, 2048, (0, 4096))
            for candidate, depths in calls
        ))
        self.assertEqual(log, [{
            "stage": "kv_cache_dtype",
            "scores": {"f16": 100.0, "q8_0": 200.0, "q4_0": 50.0},
            "winner": "q8_0",
        }])


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
                "--cooldown", "0", "--max-depth", "0", "--quick",
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
                     return_value={
                         "rocm_version": "10.0.0-4", "llama_cpp_identity": "v0.4.0",
                         "llama_cpp_commit": "5266f24da75dc449bd56cbed7addb9c8e4a6a73e",
                         "gpu_vram_bytes": 32 * 1024**3,
                     },
                 ), \
                 mock.patch.object(
                     run_bench, "read_gguf_metadata",
                     return_value={"general.architecture": "test", "test.block_count": 1},
                 ), \
                 mock.patch.object(run_bench, "moe_params", return_value=None):
                run_bench.main()  # must NOT raise SystemExit for a fully-ok run

            run_dir = results_root / "model" / "rocm10.0.0_llamacppv0.4.0"
            metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            manifest = json.loads((run_dir / "campaign_manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(metadata["model_slug"], "model")
            self.assertEqual(metadata["status"], "finished")
            self.assertEqual(metadata["run_id"], "rocm10.0.0_llamacppv0.4.0")
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
            self.assertEqual(manifest["final_config"]["gpu_layers"], 2)
            self.assertIsNone(manifest["dense_offload_curve"])
            self.assertEqual(len(manifest["runs"]), 2)  # prefill + generation
            self.assertEqual(
                set(manifest.keys()),
                {
                    "image", "device", "depths", "context_length", "final_config",
                    "tuning_log", "moe_offload_curve", "dense_offload_curve",
                    "repetitions", "prefill_tokens",
                    "generation_tokens", "mode", "completed_at", "summary_rows", "runs",
                },
            )
            self.assertTrue((run_dir / "campaign.finished").exists())
            self.assertTrue((run_dir / "curve_summary.csv").is_file())

    def test_infeasible_dense_campaign_is_failed_and_skips_final_curves(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model.gguf"
            model.write_bytes(b"x")
            (root / "results" / "model" / "test-run").mkdir(parents=True)
            run_one_mock = mock.Mock(side_effect=AssertionError("final curve must be skipped"))
            progress_mock = mock.Mock()
            outcome = run_campaign_module.run_model_campaign(
                image="unused", gpu_gids=[], device="ROCm0", model=model,
                results_root=root / "results", run_id="test-run", env={},
                config=run_campaign_module.CampaignConfig(quick=True, cooldown=0),
                gguf_metadata={"general.architecture": "test", "test.block_count": 1},
                moe=None, dense_block_count=1, model_size_bytes=1,
                depths=(0, 2048), max_ctx=4096, progress=progress_mock,
                bench_config_cls=BenchConfig, run_one=run_one_mock,
                auto_tune=mock.Mock(), sweep_moe_offload_quick=mock.Mock(),
                sweep_moe_offload_thorough=mock.Mock(),
                preflight_dense_offload=mock.Mock(return_value=None),
                sweep_dense_offload=mock.Mock(return_value={
                    "mode": "quick", "block_count": 1, "max_gpu_layers": 2,
                    "final_ngl": None, "offload_needed": True, "by_depth": [],
                }),
                model_slug="model", prefill_tokens=2048,
                generation_tokens=128, repetitions=3,
            )

            self.assertTrue(outcome.failed)
            self.assertEqual(outcome.summary_rows, 0)
            run_one_mock.assert_not_called()
            progress_mock.prune.assert_called_once_with(
                4, "final curves skipped because no dense --ngl fits",
            )
            run_dir = root / "results" / "model" / "test-run"
            self.assertTrue((run_dir / "campaign.failed").exists())
            metadata = json.loads((run_dir / "metadata.json").read_text())
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(metadata["final_config"]["gpu_layers"], 0)


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
