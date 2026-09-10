"""WP06 tests: tuning_log viewer compatibility, kv_feasibility manifest
payload, and finished/partial/failed status semantics in the presence of
static KV depth exclusions (see KV-CACHE-REFACTOR-06-ARTIFACTS-STATUS-DOCS.md).
"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from application import run_campaign as run_campaign_module
from application.auto_tune import KvSelection, kv_selection_tuning_log_entry
from application.viewer_dataset import SCHEMA_VERSION, validate_viewer_dataset
from domain.kv_depth_planner import build_kv_feasibility_manifest_payload, build_kv_depth_plan
from domain.kv_feasibility import KvRuntimeScope, RuntimeReservePolicy
from domain.models import BenchConfig, RunResult


class KvSelectionTuningLogEntryTests(unittest.TestCase):
    """The KV dtype/depth selection must still publish a legacy-shaped
    tuning_log entry ({"stage", "scores", "winner"}) so the existing
    viewer contract and sensitivity chart keep working - package 05's
    KvSelection alone is not manifest/viewer-ready."""

    def _selection(self, **overrides):
        defaults = dict(
            config=BenchConfig(batch=2048, ubatch=2048, ctk="q8_0", ctv="q8_0").validate(),
            target_depth=96_256,
            runnable_depths=(0, 96_256),
            probes=(
                {"depth": 96_256, "ctk": "q4_0", "ctv": "q4_0", "status": "partial", "avg_ts": None},
                {"depth": 96_256, "ctk": "q8_0", "ctv": "q8_0", "status": "ok", "avg_ts": 42.0},
            ),
        )
        defaults.update(overrides)
        return KvSelection(**defaults)

    def test_entry_shape_matches_legacy_kv_cache_dtype_stage(self):
        entry = kv_selection_tuning_log_entry(self._selection())

        self.assertEqual(entry["stage"], "kv_cache_dtype")
        self.assertEqual(entry["winner"], "q8_0")
        self.assertEqual(entry["scores"], {"q4_0": -1.0, "q8_0": 42.0})

    def test_unsuccessful_probes_use_the_negative_sentinel_not_none(self):
        # sensitivity.ts filters out negative values but requires every
        # candidate to be a finite number - never None/null.
        entry = kv_selection_tuning_log_entry(self._selection())
        self.assertTrue(all(isinstance(v, float) for v in entry["scores"].values()))

    def test_entry_satisfies_viewer_dataset_contract(self):
        entry = kv_selection_tuning_log_entry(self._selection())
        dataset = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": "2026-09-10T00:00:00+00:00",
            "models": [{
                "model_slug": "example",
                "runs": [{
                    "run_id": "test", "status": "finished", "mode": "full-sweep",
                    "run_completed_at": "2026-09-10T00:00:00+00:00",
                    "environment": {}, "final_config": {
                        "ubatch": 2048, "batch": 2048, "ctk": "q8_0", "ctv": "q8_0",
                        "flash_attn": "auto", "gpu_layers": 99, "block_count": None,
                        "n_cpu_layers": 0, "n_cpu_moe": 0,
                    }, "depths_tested": [0, 96_256],
                    "prefill_depth0_ts": None, "generation_depth0_ts": None, "curve": [],
                    "tuning_log": [entry],
                    "moe_offload_curve": None,
                }],
            }],
        }
        validate_viewer_dataset(dataset)  # must not raise

    def test_entry_is_json_serializable(self):
        entry = kv_selection_tuning_log_entry(self._selection())
        json.dumps(entry)  # must not raise


DENSE_METADATA = {
    "general.architecture": "llama",
    "llama.block_count": 2,
    "llama.attention.head_count_kv": 2,
    "llama.attention.key_length": 32,
    "llama.attention.value_length": 64,
}
SCOPE = KvRuntimeScope(
    llama_cpp_commit="test-commit", gpu_identifier="test-gpu",
    batch=2048, ubatch=2048, flash_attn="auto", kv_placement="gpu",
)
RESERVE = RuntimeReservePolicy(
    bytes=320, supported_architectures=frozenset({"llama"}),
    is_conservative=True, assumptions=("fixed benchmark configuration",), scope=SCOPE,
)


class KvFeasibilityManifestIntegrationTests(unittest.TestCase):
    """A manifest augmented with kv_feasibility must preserve the original
    requested depths and record exact static exclusions, independent of
    whatever the eventual runtime curve produces."""

    def test_manifest_payload_survives_a_json_roundtrip_with_exact_exclusions(self):
        plan = build_kv_depth_plan(
            requested_depths=(10, 20), prefill_tokens=2, metadata=DENSE_METADATA,
            kv_offload_enabled=True, kv_placement="gpu", gpu_vram_bytes=15_000,
            runtime_scope=SCOPE, reserve_policy=RESERVE,
        )
        payload = build_kv_feasibility_manifest_payload(
            plan,
            fixed_runtime_config={
                "batch": 2048, "ubatch": 2048, "flash_attn": "auto", "no_kv_offload": False,
            },
        )
        roundtripped = json.loads(json.dumps(payload))

        self.assertEqual(roundtripped["requested_depths"], [10, 20])
        f16 = roundtripped["kv_feasibility"]["per_dtype"]["f16"]
        self.assertEqual(f16["excluded_depths"], [20])
        self.assertEqual(f16["exclusions"][0]["required_gpu_bytes"], 17_216)
        # A static exclusion for f16 must never remove the depth from the
        # top-level requested_depths record, and must not silently apply
        # to a dtype (q4_0/q8_0) that never triggered it.
        q4 = roundtripped["kv_feasibility"]["per_dtype"]["q4_0"]
        self.assertEqual(q4["excluded_depths"], [])


class CampaignManifestKvFeasibilityTests(unittest.TestCase):
    """build_campaign_manifest must publish the static kv_feasibility plan
    as a new, additive field - never overwriting the existing depths/
    final_config/tuning_log fields other manifest consumers already rely
    on (see WP06's manifest-compatibility requirement)."""

    def test_manifest_includes_kv_feasibility_when_provided(self):
        from adapters.outbound.campaign_store import build_campaign_manifest

        plan = build_kv_depth_plan(
            requested_depths=(10, 20), prefill_tokens=2, metadata=DENSE_METADATA,
            kv_offload_enabled=True, kv_placement="gpu", gpu_vram_bytes=15_000,
            runtime_scope=SCOPE, reserve_policy=RESERVE,
        )
        kv_feasibility_payload = build_kv_feasibility_manifest_payload(
            plan,
            fixed_runtime_config={
                "batch": 2048, "ubatch": 2048, "flash_attn": "auto", "no_kv_offload": False,
            },
        )

        manifest = build_campaign_manifest(
            image="img", device="ROCm0", depths=(10, 20), context_length=None,
            final_config={}, tuning_log=[], moe_offload_curve=None,
            dense_offload_curve=None, repetitions=3, prefill_tokens=2,
            generation_tokens=128, mode="full-sweep", completed_at="now",
            summary_rows=0, run_summaries=[], kv_feasibility=kv_feasibility_payload,
        )

        self.assertEqual(manifest["depths"], [10, 20])  # original field untouched
        self.assertIn("kv_feasibility", manifest)
        self.assertEqual(manifest["kv_feasibility"], kv_feasibility_payload)

    def test_manifest_kv_feasibility_defaults_to_none_for_quick_mode(self):
        from adapters.outbound.campaign_store import build_campaign_manifest

        manifest = build_campaign_manifest(
            image="img", device="ROCm0", depths=(0,), context_length=None,
            final_config={}, tuning_log=[], moe_offload_curve=None,
            dense_offload_curve=None, repetitions=3, prefill_tokens=2,
            generation_tokens=128, mode="quick", completed_at="now",
            summary_rows=0, run_summaries=[],
        )

        self.assertIn("kv_feasibility", manifest)
        self.assertIsNone(manifest["kv_feasibility"])


class CampaignStatusPolicyWithStaticExclusionsTests(unittest.TestCase):
    """Integration slice through run_model_campaign(): a static KV depth
    exclusion recorded by the planner must never, by itself, turn a
    campaign partial or failed - only a real run_one()/dense_sweep
    failure/timeout does that (WP06 status policy table)."""

    def _run_result(self, *, series, depths_run, status="ok"):
        return RunResult(
            model="model.gguf", series=series, config=BenchConfig(batch=2048, ubatch=2048),
            status=status, return_code=0, jsonl_path=Path("/dev/null"),
            stderr_path=Path("/dev/null"), command=[], depths_run=depths_run,
        )

    def test_a_static_exclusion_with_all_runnable_depths_completing_is_finished(self):
        # f16 excludes depth 20 (metadata forces a small VRAM), but the
        # selected/probed dtype only ever receives its own runnable depths
        # (10,) - the exclusion must be visible in the manifest without
        # ever causing run_one/final-curve failures.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model.gguf"
            model.write_bytes(b"x")
            run_dir = root / "results" / "model" / "test-run"
            run_dir.mkdir(parents=True)

            kv_depth_plan = build_kv_depth_plan(
                requested_depths=(10, 20), prefill_tokens=2, metadata=DENSE_METADATA,
                kv_offload_enabled=True, kv_placement="gpu", gpu_vram_bytes=15_000,
                runtime_scope=SCOPE, reserve_policy=RESERVE,
            )
            selection = KvSelection(
                config=BenchConfig(batch=2048, ubatch=2048, ctk="f16", ctv="f16").validate(),
                target_depth=10,
                runnable_depths=(10,),
                probes=({"depth": 10, "ctk": "f16", "ctv": "f16", "status": "ok", "avg_ts": 1.0},),
            )
            run_one_mock = mock.Mock(side_effect=[
                self._run_result(series="prefill", depths_run=(10,)),
                self._run_result(series="generation", depths_run=(10,)),
            ])

            with mock.patch.object(run_campaign_module, "select_kv_config", return_value=selection), \
                 mock.patch.object(run_campaign_module.time, "sleep"):
                outcome = run_campaign_module.run_model_campaign(
                    image="unused", gpu_gids=[], device="ROCm0", model=model,
                    results_root=root / "results", run_id="test-run", env={"gpu_vram_bytes": 15_000},
                    config=run_campaign_module.CampaignConfig(quick=False, cooldown=0),
                    gguf_metadata=DENSE_METADATA, moe=None, dense_block_count=None,
                    model_size_bytes=1, depths=(10, 20), max_ctx=4096, progress=None,
                    bench_config_cls=BenchConfig, run_one=run_one_mock,
                    sweep_moe_offload_quick=mock.Mock(), sweep_moe_offload_thorough=mock.Mock(),
                    preflight_dense_offload=mock.Mock(), sweep_dense_offload=mock.Mock(),
                    model_slug="model", prefill_tokens=2, generation_tokens=128, repetitions=3,
                    kv_depth_plan=kv_depth_plan,
                )

            self.assertFalse(outcome.failed)
            self.assertFalse(outcome.partial)
            self.assertTrue((run_dir / "campaign.finished").exists())
            metadata = json.loads((run_dir / "metadata.json").read_text())
            self.assertEqual(metadata["status"], "finished")
            manifest = json.loads((run_dir / "campaign_manifest.json").read_text())
            f16_plan = manifest["kv_feasibility"]["per_dtype"]["f16"]
            self.assertEqual(f16_plan["excluded_depths"], [20])
            # The top-level "depths" field already carries the original
            # requested depths (see campaign_store.build_campaign_manifest)
            # - kv_feasibility must never truncate or contradict it, even
            # though f16 will never probe the excluded depth 20.
            self.assertEqual(manifest["depths"], [10, 20])


if __name__ == "__main__":
    unittest.main()
