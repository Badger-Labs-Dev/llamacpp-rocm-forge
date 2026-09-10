"""Behavior tests for dense-model -ngl capacity search."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from application import dense_sweep
from adapters.outbound.gguf_metadata import total_model_size_bytes
from domain.models import BenchConfig
from domain.vram_estimate import estimate_full_offload_vram


class VramEstimateTests(unittest.TestCase):
    def test_split_gguf_size_requires_and_sums_every_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "model-00001-of-00002.gguf"
            second = root / "model-00002-of-00002.gguf"
            first.write_bytes(b"123")
            second.write_bytes(b"4567")
            self.assertEqual(total_model_size_bytes(first), 7)
            second.unlink()
            with self.assertRaises(FileNotFoundError):
                total_model_size_bytes(first)

    def test_estimates_f16_kv_and_full_offload_fit_from_known_values(self):
        metadata = {
            "general.architecture": "test",
            "test.block_count": 4,
            "test.attention.head_count": 4,
            "test.attention.head_count_kv": 2,
            "test.embedding_length": 32,
        }

        estimate = estimate_full_offload_vram(
            metadata=metadata,
            model_size_bytes=1_000_000,
            context_size=100,
            gpu_vram_bytes=2_000_000,
            overhead_bytes=100_000,
        )

        # 4 layers * 100 tokens * 2 KV heads * (8 key + 8 value dims)
        # * 2 bytes per f16 element.
        self.assertEqual(estimate.kv_cache_bytes, 25_600)
        self.assertEqual(estimate.total_bytes, 1_125_600)
        self.assertTrue(estimate.fits)

    def test_missing_attention_metadata_returns_unknown_instead_of_guessing(self):
        estimate = estimate_full_offload_vram(
            metadata={"test.block_count": 4},
            model_size_bytes=1_000_000,
            context_size=100,
            gpu_vram_bytes=2_000_000,
        )
        self.assertIsNone(estimate)


class DenseSweepTests(unittest.TestCase):
    def test_full_offload_fit_skips_bisection_and_redundant_samples(self):
        probes: list[int] = []

        def fake_probe(**kwargs):
            ngl = kwargs["gpu_layers"]
            probes.append(ngl)
            return True, float(ngl)

        with mock.patch.object(dense_sweep, "probe_dense_offload", side_effect=fake_probe), \
             mock.patch.object(dense_sweep.time, "sleep"):
            curve = dense_sweep.sweep_dense_offload(
                image="unused", gpu_gids=[], model=Path("model.gguf"),
                base_config=BenchConfig(), device="ROCm0", depths=(0,),
                block_count=4, results_dir=Path("/tmp/unused"), cooldown=0,
                metadata={}, model_size_bytes=1, gpu_vram_bytes=1,
            )

        point = curve["by_depth"][0]
        self.assertEqual(point["max_ngl_that_fits"], 5)
        self.assertEqual(curve["final_ngl"], 5)
        self.assertEqual(probes[0], 5)
        self.assertEqual(len(probes), 1)
        self.assertFalse(curve["offload_needed"])

    def test_probe_artifacts_are_separated_by_depth(self):
        result = SimpleNamespace(status="ok", jsonl_path=Path("probe.jsonl"))
        with mock.patch.object(dense_sweep, "run_one", return_value=result) as run_one, \
             mock.patch.object(dense_sweep, "mean_ts", return_value=1.0):
            for depth in (0, 2048):
                dense_sweep.probe_dense_offload(
                    image="unused", gpu_gids=[], model=Path("model.gguf"),
                    base_config=BenchConfig(), block_count=4, gpu_layers=3,
                    device="ROCm0", depth=depth, results_dir=Path("/tmp/unused"),
                )

        self.assertEqual(
            [call.kwargs["subdir"] for call in run_one.call_args_list],
            ["dense-tuning/depth-0", "dense-tuning/depth-2048"],
        )

    def test_bisection_finds_max_ngl_that_fits_at_each_depth(self):
        # Expressed as minimum CPU-offloaded layers: 2 at depth 0, 3 deeper.
        min_cpu = {0: 2, 2048: 3}

        def fake_probe(**kwargs):
            max_gpu_layers = 5
            cpu_layers = max_gpu_layers - kwargs["gpu_layers"]
            fits = cpu_layers >= min_cpu[kwargs["depth"]]
            return fits, float(kwargs["gpu_layers"]) if fits else -1.0

        with mock.patch.object(dense_sweep, "probe_dense_offload", side_effect=fake_probe), \
             mock.patch.object(dense_sweep.time, "sleep"):
            curve = dense_sweep.sweep_dense_offload(
                image="unused", gpu_gids=[], model=Path("model.gguf"),
                base_config=BenchConfig(), device="ROCm0", depths=(0, 2048),
                block_count=4, results_dir=Path("/tmp/unused"), cooldown=0,
                metadata={}, model_size_bytes=1, gpu_vram_bytes=1,
            )

        self.assertEqual(
            [point["max_ngl_that_fits"] for point in curve["by_depth"]],
            [3, 2],
        )
        self.assertEqual(curve["final_ngl"], 2)

    def test_infeasible_depth_marks_every_deeper_depth_and_has_no_final_ngl(self):
        with mock.patch.object(
            dense_sweep, "probe_dense_offload", return_value=(False, -1.0),
        ), mock.patch.object(dense_sweep.time, "sleep"):
            curve = dense_sweep.sweep_dense_offload(
                image="unused", gpu_gids=[], model=Path("model.gguf"),
                base_config=BenchConfig(), device="ROCm0", depths=(0, 2048, 4096),
                block_count=4, results_dir=Path("/tmp/unused"), cooldown=0,
                metadata={}, model_size_bytes=1, gpu_vram_bytes=1,
            )

        self.assertIsNone(curve["final_ngl"])
        self.assertTrue(curve["offload_needed"])
        self.assertEqual(len(curve["by_depth"]), 3)
        self.assertTrue(all(
            point["max_ngl_that_fits"] is None for point in curve["by_depth"]
        ))
        self.assertEqual(curve["by_depth"][1]["results"], [])

    def test_failed_deepest_depth_keeps_lower_fitting_final_ngl(self):
        def fake_probe(**kwargs):
            return kwargs["depth"] == 0, 1.0

        with mock.patch.object(dense_sweep, "probe_dense_offload", side_effect=fake_probe), \
             mock.patch.object(dense_sweep.time, "sleep"):
            curve = dense_sweep.sweep_dense_offload(
                image="unused", gpu_gids=[], model=Path("model.gguf"),
                base_config=BenchConfig(), device="ROCm0", depths=(0, 2048),
                block_count=4, results_dir=Path("/tmp/unused"), cooldown=0,
                metadata={}, model_size_bytes=1, gpu_vram_bytes=1,
            )

        self.assertEqual(curve["final_ngl"], 5)
        self.assertEqual(curve["by_depth"][1]["max_ngl_that_fits"], None)

    def test_preflight_returns_safe_ngl_for_deepest_context_without_extra_samples(self):
        probes: list[int] = []

        def fake_probe(**kwargs):
            ngl = kwargs["gpu_layers"]
            probes.append(ngl)
            return ngl <= 2, 1.0

        with mock.patch.object(dense_sweep, "probe_dense_offload", side_effect=fake_probe), \
             mock.patch.object(dense_sweep.time, "sleep"):
            ngl = dense_sweep.preflight_dense_offload(
                image="unused", gpu_gids=[], model=Path("model.gguf"),
                base_config=BenchConfig(), device="ROCm0", depth=2048,
                block_count=4, results_dir=Path("/tmp/unused"), cooldown=0,
                metadata={}, model_size_bytes=1, gpu_vram_bytes=1,
            )

        self.assertEqual(ngl, 2)
        self.assertLessEqual(len(probes), 5)  # endpoint checks + bounded bisection

    def test_preflight_returns_none_when_even_ngl_zero_fails(self):
        with mock.patch.object(
            dense_sweep, "probe_dense_offload", return_value=(False, -1.0),
        ), mock.patch.object(dense_sweep.time, "sleep"):
            ngl = dense_sweep.preflight_dense_offload(
                image="unused", gpu_gids=[], model=Path("model.gguf"),
                base_config=BenchConfig(), device="ROCm0", depth=2048,
                block_count=4, results_dir=Path("/tmp/unused"), cooldown=0,
                metadata={}, model_size_bytes=1, gpu_vram_bytes=1,
            )
        self.assertIsNone(ngl)


if __name__ == "__main__":
    unittest.main()
