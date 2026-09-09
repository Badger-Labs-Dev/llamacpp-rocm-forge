import unittest

from adapters.outbound.docker_llama_bench import LlamaBenchProbe, docker_command
from adapters.outbound.model_resolution import model_slug
from application.viewer_dataset import SCHEMA_VERSION, validate_viewer_dataset
from domain.planning import campaign_budget, quick_moe_candidates
from domain.progress import ProbeProgress


class BenchmarkApplicationTests(unittest.TestCase):
    def test_budget_preserves_worst_case_and_work_parts(self):
        budget = campaign_budget(
            depth_count=3,
            full_sweep=True,
            calibrate=False,
            valid_batch_pairs=13,
            kv_type_count=3,
            tuning_depth_count=2,
            moe_block_count=40,
            thorough_moe=True,
        )
        self.assertEqual(budget.total, 61)
        self.assertEqual(budget.parts[0], ("tuning", 19))
        self.assertEqual(budget.parts[-1], ("final curves", 6))

    def test_progress_is_pure_and_pruning_changes_only_current_plan(self):
        progress = ProbeProgress(total_probes=10, expected_gap_seconds=2)
        progress.record_completion(10)
        snapshot = progress.record_pruning(3)
        self.assertEqual((snapshot.completed, snapshot.pruned, snapshot.remaining), (1, 3, 6))
        self.assertEqual((snapshot.worst_case_percent, snapshot.current_plan_percent), (10, 14))
        self.assertEqual(snapshot.eta_seconds, 72)

    def test_docker_adapter_renders_the_same_probe_contract(self):
        probe = LlamaBenchProbe(
            model_container_path="/models/model.gguf", series="prefill", batch=2048,
            ubatch=1024, flash_attn="auto", n_cpu_moe=7, ctk="f16", ctv="f16",
            device="ROCm0", depth=8192, repetitions=3, gpu_layers=99,
            prefill_tokens=2048, generation_tokens=128,
        )
        command = docker_command(
            image="test-image", container_name="test-container", gpu_gids=["123"],
            model_directory="/models-host", probe=probe,
        )
        self.assertIn("-ncmoe", command)
        self.assertEqual(command[command.index("-ncmoe") + 1], "7")
        self.assertIn("/models-host:/models:ro", command)

    def test_model_slug_collapses_non_alphanumerics_and_lowercases(self):
        from pathlib import Path
        self.assertEqual(model_slug(Path("Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")), "qwen3-6-35b-a3b-ud-q4-k-xl")
        self.assertEqual(model_slug(Path("plain.gguf")), "plain")
        self.assertEqual(model_slug(Path("---.gguf")), "model")  # all-non-alphanumeric stem falls back

    def test_viewer_contract_requires_version_and_required_run_fields(self):
        dataset = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": "2026-09-09T00:00:00+00:00",
            "models": [{
                "model_slug": "example",
                "runs": [{
                    "run_id": "test", "status": "finished", "mode": "fixed",
                    "environment": {}, "final_config": {}, "curve": [],
                    "tuning_log": [], "moe_offload_curve": None,
                }],
            }],
        }
        validate_viewer_dataset(dataset)

    def _valid_dataset(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at": "2026-09-09T00:00:00+00:00",
            "models": [{
                "model_slug": "example",
                "runs": [{
                    "run_id": "test", "status": "finished", "mode": "fixed",
                    "environment": {}, "final_config": {}, "curve": [],
                    "tuning_log": [], "moe_offload_curve": None,
                }],
            }],
        }

    def test_viewer_contract_rejects_wrong_schema_version(self):
        dataset = self._valid_dataset()
        dataset["schema_version"] = 2
        with self.assertRaises(ValueError):
            validate_viewer_dataset(dataset)

    def test_viewer_contract_rejects_missing_run_field(self):
        dataset = self._valid_dataset()
        del dataset["models"][0]["runs"][0]["curve"]
        with self.assertRaises(ValueError):
            validate_viewer_dataset(dataset)

    def test_viewer_contract_rejects_non_dict_runs(self):
        dataset = self._valid_dataset()
        dataset["models"][0]["runs"] = {}
        with self.assertRaises(ValueError):
            validate_viewer_dataset(dataset)

    def test_viewer_contract_rejects_empty_moe_offload_curve(self):
        dataset = self._valid_dataset()
        dataset["models"][0]["runs"][0]["moe_offload_curve"] = {}
        with self.assertRaises(ValueError):
            validate_viewer_dataset(dataset)

    def test_viewer_contract_rejects_tuning_stage_without_scores(self):
        dataset = self._valid_dataset()
        dataset["models"][0]["runs"][0]["tuning_log"] = [{}]
        with self.assertRaises(ValueError):
            validate_viewer_dataset(dataset)

    def test_viewer_contract_rejects_malformed_curve_row(self):
        dataset = self._valid_dataset()
        dataset["models"][0]["runs"][0]["curve"] = [{"series": "prefill"}]
        with self.assertRaises(ValueError):
            validate_viewer_dataset(dataset)

    def test_viewer_contract_accepts_populated_moe_offload_curve(self):
        dataset = self._valid_dataset()
        dataset["models"][0]["runs"][0]["moe_offload_curve"] = {
            "mode": "quick",
            "expert_count": 8,
            "expert_used_count": 2,
            "block_count": 40,
            "by_depth": [{
                "depth": 0,
                "min_ncmoe_that_fits": 0,
                "results": [{"n_cpu_moe": 0, "status": "ok", "avg_ts": 100.0}],
            }],
        }
        validate_viewer_dataset(dataset)


if __name__ == "__main__":
    unittest.main()
