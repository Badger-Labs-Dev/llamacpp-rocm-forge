"""KV-CACHE-REFACTOR-07: static-planning integration fixture.

Exercises the real domain planner (domain.kv_depth_planner /
domain.kv_feasibility) together with the real application selection use
case (application.auto_tune.select_kv_config) as one pipeline, rather than
the hand-built DtypeDepthPlan fixtures the unit tests use. This is the
package's required "in-memory/small fixture with all three outcomes"
integration check (see KV-CACHE-REFACTOR-07-VERIFICATION.md):

* q4_0, deep depth      -> eligible (still probed, and wins selection)
* f16 (and q8_0), deep  -> statically excluded by exact GPU KV accounting
                           (zero Docker/probe invocations at that depth)
* unsupported architecture -> unknown/runnable for every dtype/depth (still
                           probed; never statically pruned)

No production code changes were needed for WP07: this file only adds
coverage that the existing WP02/04/05/06 pieces are wired the way WP07
requires when driven together instead of in isolation.
"""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from application.auto_tune import select_kv_config
from domain.kv_depth_planner import build_kv_depth_plan, build_kv_feasibility_manifest_payload
from domain.kv_feasibility import KvRuntimeScope, RuntimeReservePolicy


# Same conventional-dense metadata shape as test_kv_depth_planner.py /
# test_kv_feasibility.py: 2 blocks, 2 KV heads, key_length=32, value_length=64.
DENSE_METADATA = {
    "general.architecture": "llama",
    "llama.block_count": 2,
    "llama.attention.head_count_kv": 2,
    "llama.attention.key_length": 32,
    "llama.attention.value_length": 64,
}
UNSUPPORTED_METADATA = {
    "general.architecture": "qwen3.5moe",  # hybrid/unsupported: package 00's
    # locked decision #5 - the existing estimator must not be reused as a
    # hard bound, so this stays unknown, never excluded.
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
FIXED_RUNTIME_CONFIG = {
    "batch": 2048, "ubatch": 2048, "flash_attn": "auto", "no_kv_offload": False,
}

REQUESTED_DEPTHS = (1, 4096)
# Exact per-context-size GPU KV bytes for this fixture's dense metadata
# (2 blocks * (key_bytes + value_bytes) at context_size=4096), computed the
# same way domain.kv_feasibility._conventional_dense_kv_bytes does, so the
# expected numbers below are traceable rather than magic:
#   f16:  2 * ((2*32*4096*2) + (2*64*4096*2))                  = 3,145,728
#   q8_0: 2 * ((2*32/32*4096*34) + (2*64/32*4096*34))          = 1,671,168
#   q4_0: 2 * ((2*32/32*4096*18) + (2*64/32*4096*18))          =   884,736
_DEEP_F16_KV_BYTES = 3_145_728
_DEEP_Q8_KV_BYTES = 1_671_168
_DEEP_Q4_KV_BYTES = 884_736
GPU_VRAM_BYTES = 1_000_000  # fits q4_0 (+reserve), excludes q8_0 and f16 at depth=4096


def build_plan(*, metadata):
    return build_kv_depth_plan(
        requested_depths=REQUESTED_DEPTHS,
        prefill_tokens=0,
        metadata=metadata,
        kv_offload_enabled=True,
        kv_placement="gpu",
        gpu_vram_bytes=GPU_VRAM_BYTES,
        runtime_scope=SCOPE,
        reserve_policy=RESERVE,
    )


class StaticPlanningIntegrationFixtureTests(unittest.TestCase):
    """WP07's three-outcome fixture, run through the real planner."""

    def setUp(self):
        self.supported_plan = build_plan(metadata=DENSE_METADATA)
        self.unsupported_plan = build_plan(metadata=UNSUPPORTED_METADATA)
        self.by_dtype = {dp.ctk: dp for dp in self.supported_plan.dtype_plans}

    # -- exact byte accounting sanity (ties the fixture's hand-derived
    #    numbers to the real domain calculation, not just to each other) --

    def test_deep_depth_exact_byte_accounting_matches_hand_derivation(self):
        f16_exclusion = self.by_dtype["f16"].exclusions[0]
        q8_exclusion = self.by_dtype["q8_0"].exclusions[0]

        self.assertEqual(f16_exclusion.kv_cache_bytes, _DEEP_F16_KV_BYTES)
        self.assertEqual(f16_exclusion.required_gpu_bytes, _DEEP_F16_KV_BYTES + 320)
        self.assertEqual(q8_exclusion.kv_cache_bytes, _DEEP_Q8_KV_BYTES)
        self.assertEqual(q8_exclusion.required_gpu_bytes, _DEEP_Q8_KV_BYTES + 320)
        self.assertLess(_DEEP_Q4_KV_BYTES + 320, GPU_VRAM_BYTES)

    # -- outcome 1: q4_0 deep is eligible / runnable --

    def test_q4_0_deep_is_eligible(self):
        self.assertIn(4096, self.by_dtype["q4_0"].eligible_depths)
        self.assertIn(4096, self.by_dtype["q4_0"].runnable_depths)

    # -- outcome 2: f16 (and q8_0) deep is statically excluded, with exact
    #    bytes/reason/scope, only because the exact accounting proves it --

    def test_f16_and_q8_deep_are_statically_excluded_with_exact_accounting(self):
        for ctk in ("f16", "q8_0"):
            dtype_plan = self.by_dtype[ctk]
            self.assertNotIn(4096, dtype_plan.runnable_depths)
            self.assertEqual([e.context_size for e in dtype_plan.exclusions], [4096])
            exclusion = dtype_plan.exclusions[0]
            self.assertEqual(exclusion.status, "excluded")
            self.assertEqual(
                exclusion.reason,
                "exact GPU KV cache plus validated runtime reserve exceeds GPU VRAM",
            )
            self.assertIsNotNone(exclusion.kv_cache_bytes)
            self.assertIsNotNone(exclusion.required_gpu_bytes)

    # -- outcome 3: unsupported architecture is unknown/runnable everywhere,
    #    never statically pruned --

    def test_unsupported_architecture_is_unknown_and_runnable_at_every_depth(self):
        for dtype_plan in self.unsupported_plan.dtype_plans:
            self.assertEqual(dtype_plan.exclusions, ())
            self.assertEqual(set(dtype_plan.unknown_depths), set(REQUESTED_DEPTHS))
            self.assertEqual(set(dtype_plan.runnable_depths), set(REQUESTED_DEPTHS))

    # -- request ordering / original context limit preserved regardless of
    #    static exclusion --

    def test_requested_depths_and_ordering_survive_static_exclusion(self):
        self.assertEqual(self.supported_plan.requested_depths, REQUESTED_DEPTHS)
        for dtype_plan in self.supported_plan.dtype_plans:
            self.assertEqual(dtype_plan.requested_depths, REQUESTED_DEPTHS)
        # The deepest originally requested depth remains visible even for a
        # dtype where it was statically excluded from the runnable set.
        self.assertEqual(max(self.supported_plan.requested_depths), 4096)
        self.assertNotIn(4096, self.by_dtype["f16"].runnable_depths)

    # -- static accounting is represented verbatim in the manifest --

    def test_manifest_payload_carries_exact_accounting_verbatim(self):
        payload = build_kv_feasibility_manifest_payload(
            self.supported_plan, fixed_runtime_config=FIXED_RUNTIME_CONFIG,
        )["kv_feasibility"]

        self.assertEqual(payload["fixed_runtime_config"], FIXED_RUNTIME_CONFIG)
        self.assertEqual(payload["candidate_dtype_order"], ["q4_0", "q8_0", "f16"])
        f16_payload = payload["per_dtype"]["f16"]
        self.assertEqual(f16_payload["excluded_depths"], [4096])
        self.assertEqual(f16_payload["exclusions"][0]["kv_cache_bytes"], _DEEP_F16_KV_BYTES)
        self.assertEqual(
            f16_payload["exclusions"][0]["required_gpu_bytes"], _DEEP_F16_KV_BYTES + 320,
        )
        self.assertEqual(f16_payload["exclusions"][0]["benchmark_depth"], 4096)
        json.dumps(payload)  # manifest payload must stay JSON-serializable

    # -- end-to-end with real selection: excluded pairs cause zero probe
    #    (Docker) invocations; unknown pairs still cause a runtime probe --

    def test_excluded_pairs_are_never_probed_while_eligible_and_unknown_pairs_are(self):
        probe_calls: list[tuple[str, int]] = []

        def failing_at_deep_probe(config, depth):
            probe_calls.append((config.ctk, depth))
            # q4_0 fails at the deep depth too, so selection must also
            # *consider* q8_0/f16 at depth=4096 in order - proving any
            # invocation there is a deliberate skip, not a lucky early exit.
            status = "partial" if depth == 4096 and config.ctk == "q4_0" else "ok"
            return SimpleNamespace(status=status, jsonl_path=Path(f"/{config.ctk}-{depth}.jsonl"))

        selection = select_kv_config(
            plan=self.supported_plan,
            probe=failing_at_deep_probe,
            mean_throughput=lambda path: 10.0,
        )

        # f16/q8_0 were statically excluded at depth=4096: never probed there.
        self.assertNotIn(("q8_0", 4096), probe_calls)
        self.assertNotIn(("f16", 4096), probe_calls)
        # q4_0 was eligible at depth=4096 and was probed (and failed, by
        # fixture design), so selection descended to the shallow depth.
        self.assertIn(("q4_0", 4096), probe_calls)
        self.assertIsNotNone(selection)
        self.assertEqual(selection.target_depth, 1)

        probe_calls.clear()
        unknown_selection = select_kv_config(
            plan=self.unsupported_plan,
            probe=failing_at_deep_probe,
            mean_throughput=lambda path: 10.0,
        )

        # Nothing was statically excluded for the unsupported architecture:
        # every dtype was still probed at the deep depth.
        for ctk in ("q4_0", "q8_0", "f16"):
            self.assertIn((ctk, 4096), probe_calls)
        self.assertIsNotNone(unknown_selection)


if __name__ == "__main__":
    unittest.main()
