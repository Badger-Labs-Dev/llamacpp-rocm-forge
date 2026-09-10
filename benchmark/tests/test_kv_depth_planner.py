import unittest
from typing import Any, cast

from domain.kv_depth_planner import (
    KV_CACHE_TYPES,
    KvDepthPlan,
    build_kv_depth_plan,
    build_kv_feasibility_manifest_payload,
    candidate_dtype_depth_targets,
    deepest_runnable_depth,
)
from domain.kv_feasibility import KvRuntimeScope, RuntimeReservePolicy


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


class KvDepthPlannerTests(unittest.TestCase):
    def plan(self, **overrides):
        inputs = dict(
            requested_depths=(10, 20), prefill_tokens=2, metadata=DENSE_METADATA,
            kv_offload_enabled=True, kv_placement="gpu", gpu_vram_bytes=16_000,
            runtime_scope=SCOPE, reserve_policy=RESERVE,
        )
        inputs.update(overrides)
        return build_kv_depth_plan(**inputs)

    def test_preserves_requested_depths_and_plans_each_dtype_independently(self):
        plan = self.plan(gpu_vram_bytes=15_000)

        self.assertEqual(plan.requested_depths, (10, 20))
        self.assertEqual(plan.prefill_tokens, 2)
        by_dtype = {dtype_plan.ctk: dtype_plan for dtype_plan in plan.dtype_plans}
        self.assertEqual(by_dtype["q4_0"].runnable_depths, (10, 20))
        self.assertEqual(by_dtype["q8_0"].runnable_depths, (10, 20))
        self.assertEqual(by_dtype["f16"].runnable_depths, (10,))
        self.assertEqual(by_dtype["f16"].eligible_depths, (10,))
        self.assertEqual(by_dtype["f16"].exclusions[0].context_size, 22)
        self.assertEqual(by_dtype["f16"].exclusions[0].required_gpu_bytes, 17_216)
        payload = plan.to_payload()
        dtype_payloads = cast(list[dict[str, Any]], payload["dtype_plans"])
        exclusions = cast(list[dict[str, Any]], dtype_payloads[-1]["exclusions"])
        exclusion = exclusions[0]
        self.assertEqual(exclusion["required_gpu_bytes"], 17_216)
        self.assertEqual(
            exclusion["reason"],
            "exact GPU KV cache plus validated runtime reserve exceeds GPU VRAM",
        )

    def test_unknown_remains_runnable_and_keeps_assessment_reason_in_payload(self):
        plan = self.plan(kv_placement=None)
        f16 = plan.dtype_plans[-1]

        self.assertEqual(f16.unknown_depths, (10, 20))
        self.assertEqual(f16.runnable_depths, (10, 20))
        self.assertEqual(f16.exclusions, ())
        self.assertEqual(
            plan.to_payload()["dtype_plans"][-1]["unknown"][0]["reason"],
            "GPU KV placement is unknown",
        )

    def test_fully_excluded_dtype_has_no_targets(self):
        plan = self.plan(gpu_vram_bytes=1)

        self.assertEqual(deepest_runnable_depth(plan.dtype_plans[0]), None)
        self.assertEqual(plan.dtype_plans[0].runnable_depths, ())
        self.assertEqual(
            candidate_dtype_depth_targets(plan),
            (),
        )

    def test_candidate_targets_are_dtype_first_coverage_order_without_selection_ranking(self):
        plan = self.plan(gpu_vram_bytes=15_000)

        self.assertEqual(KV_CACHE_TYPES, ("q4_0", "q8_0", "f16"))
        self.assertEqual(
            candidate_dtype_depth_targets(plan),
            (("q4_0", 10), ("q4_0", 20), ("q8_0", 10), ("q8_0", 20), ("f16", 10)),
        )

    def test_prefill_is_added_once_to_each_feasibility_context(self):
        plan = self.plan(requested_depths=(10,), prefill_tokens=3)

        self.assertEqual(plan.dtype_plans[-1].eligible_depths, (10,))
        self.assertEqual(plan.dtype_plans[-1].assessments[0].context_size, 13)

    def test_negative_depths_are_rejected_and_duplicate_requested_depths_are_preserved(self):
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            self.plan(requested_depths=(10, -1))

        zero_depth_plan = self.plan(requested_depths=(0,))
        self.assertEqual(zero_depth_plan.requested_depths, (0,))

        plan = self.plan(requested_depths=(10, 10))
        self.assertEqual(plan.requested_depths, (10, 10))
        self.assertEqual(plan.dtype_plans[0].runnable_depths, (10, 10))


class KvFeasibilityManifestPayloadTests(unittest.TestCase):
    """WP06: pure, manifest-ready serialization of a KvDepthPlan, following
    the shape documented in KV-CACHE-REFACTOR-06-ARTIFACTS-STATUS-DOCS.md."""

    def plan(self, **overrides):
        inputs = dict(
            requested_depths=(10, 20), prefill_tokens=2, metadata=DENSE_METADATA,
            kv_offload_enabled=True, kv_placement="gpu", gpu_vram_bytes=15_000,
            runtime_scope=SCOPE, reserve_policy=RESERVE,
        )
        inputs.update(overrides)
        return build_kv_depth_plan(**inputs)

    def test_payload_records_fixed_config_dtype_order_and_per_dtype_accounting(self):
        plan = self.plan()
        fixed_runtime_config = {
            "batch": 2048, "ubatch": 2048, "flash_attn": "auto", "no_kv_offload": False,
        }

        payload = build_kv_feasibility_manifest_payload(
            plan, fixed_runtime_config=fixed_runtime_config,
        )

        self.assertEqual(payload["requested_depths"], [10, 20])
        feasibility = payload["kv_feasibility"]
        self.assertEqual(feasibility["fixed_runtime_config"], fixed_runtime_config)
        self.assertEqual(feasibility["candidate_dtype_order"], ["q4_0", "q8_0", "f16"])
        f16 = feasibility["per_dtype"]["f16"]
        self.assertEqual(f16["runnable_depths"], [10])
        self.assertEqual(f16["eligible_depths"], [10])
        self.assertEqual(f16["unknown_depths"], [])
        self.assertEqual(f16["excluded_depths"], [20])
        self.assertEqual(len(f16["exclusions"]), 1)
        self.assertEqual(f16["exclusions"][0]["benchmark_depth"], 20)
        self.assertEqual(f16["exclusions"][0]["context_size"], 22)
        self.assertEqual(f16["exclusions"][0]["required_gpu_bytes"], 17_216)
        q4 = feasibility["per_dtype"]["q4_0"]
        self.assertEqual(q4["excluded_depths"], [])
        self.assertEqual(q4["exclusions"], [])

    def test_payload_is_json_serializable(self):
        import json

        payload = build_kv_feasibility_manifest_payload(
            self.plan(),
            fixed_runtime_config={
                "batch": 2048, "ubatch": 2048, "flash_attn": "auto", "no_kv_offload": False,
            },
        )
        json.dumps(payload)  # must not raise


if __name__ == "__main__":
    unittest.main()
