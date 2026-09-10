import unittest

from domain.kv_feasibility import (
    KvPlacement,
    KvRuntimeScope,
    RuntimeReservePolicy,
    assess_gpu_kv_feasibility,
)


DENSE_METADATA = {
    "general.architecture": "llama",
    "llama.block_count": 2,
    "llama.attention.head_count_kv": 2,
    "llama.attention.key_length": 32,
    "llama.attention.value_length": 64,
}

VALIDATED_SCOPE = KvRuntimeScope(
    llama_cpp_commit="test-commit",
    gpu_identifier="test-gpu",
    batch=2048,
    ubatch=2048,
    flash_attn="auto",
    kv_placement="gpu",
)

VALIDATED_RESERVE = RuntimeReservePolicy(
    bytes=320,
    supported_architectures=frozenset({"llama"}),
    is_conservative=True,
    assumptions=("fixed benchmark configuration",),
    scope=VALIDATED_SCOPE,
)


class KvFeasibilityTests(unittest.TestCase):
    def assess(self, *, metadata=DENSE_METADATA, context_size=10, ctk="f16", ctv="f16",
               kv_offload_enabled: bool | None = True,
               kv_placement: KvPlacement | None = "gpu",
               gpu_vram_bytes: int | None = 8_000,
               runtime_scope: KvRuntimeScope | None = VALIDATED_SCOPE,
               reserve_policy: RuntimeReservePolicy | None = VALIDATED_RESERVE):
        return assess_gpu_kv_feasibility(
            metadata=metadata,
            context_size=context_size,
            ctk=ctk,
            ctv=ctv,
            kv_offload_enabled=kv_offload_enabled,
            kv_placement=kv_placement,
            gpu_vram_bytes=gpu_vram_bytes,
            runtime_scope=runtime_scope,
            reserve_policy=reserve_policy,
        )

    def test_exact_f16_gpu_kv_exceeding_capacity_is_excluded_with_byte_accounting(self):
        result = self.assess(gpu_vram_bytes=7_999)

        self.assertEqual(result.status, "excluded")
        self.assertEqual(result.kv_placement, "gpu")
        self.assertEqual(result.kv_cache_bytes, 7_680)
        self.assertEqual(result.runtime_reserve_bytes, 320)
        self.assertEqual(result.required_gpu_bytes, 8_000)
        self.assertEqual(result.gpu_vram_bytes, 7_999)
        self.assertIsNotNone(result.reason)
        self.assertTrue(result.assumptions)

    def test_exact_gpu_kv_equal_to_capacity_is_eligible(self):
        result = self.assess(gpu_vram_bytes=8_000)

        self.assertEqual(result.status, "eligible")
        self.assertEqual(result.required_gpu_bytes, 8_000)

    def test_supported_cache_dtypes_use_distinct_exact_block_accounting(self):
        q4 = self.assess(ctk="q4_0", ctv="q4_0", gpu_vram_bytes=10_000)
        q8 = self.assess(ctk="q8_0", ctv="q8_0", gpu_vram_bytes=10_000)
        f16 = self.assess(ctk="f16", ctv="f16", gpu_vram_bytes=10_000)

        self.assertEqual(q4.kv_cache_bytes, 2_160)
        self.assertEqual(q8.kv_cache_bytes, 4_080)
        self.assertEqual(f16.kv_cache_bytes, 7_680)
        self.assertEqual((q4.status, q8.status, f16.status), ("eligible", "eligible", "eligible"))

    def test_explicit_host_kv_is_eligible_without_claiming_host_feasibility(self):
        result = self.assess(kv_offload_enabled=False, kv_placement="host", gpu_vram_bytes=None)

        self.assertEqual(result.status, "eligible")
        self.assertEqual(result.kv_placement, "host")
        self.assertIsNone(result.kv_cache_bytes)
        self.assertIsNone(result.required_gpu_bytes)
        self.assertIsNone(result.runtime_reserve_bytes)

    def test_enabled_kv_offload_without_placement_evidence_is_unknown(self):
        result = self.assess(kv_placement=None)

        self.assertEqual(result.status, "unknown")
        self.assertEqual(result.kv_placement, "unknown")
        self.assertIn("placement", result.reason)

    def test_quantized_row_width_that_is_not_block_aligned_is_unknown(self):
        metadata = DENSE_METADATA | {
            "llama.attention.head_count_kv": 1,
            "llama.attention.key_length": 16,
            "llama.attention.value_length": 16,
        }

        result = self.assess(metadata=metadata, context_size=2, ctk="q4_0", ctv="q4_0")

        self.assertEqual(result.status, "unknown")
        self.assertIn("block-aligned", result.reason)

    def test_missing_required_metadata_is_unknown_and_never_excluded(self):
        metadata = DENSE_METADATA.copy()
        del metadata["llama.attention.value_length"]

        result = self.assess(metadata=metadata)

        self.assertEqual(result.status, "unknown")
        self.assertIn("value_length", result.reason)
        self.assertIsNone(result.required_gpu_bytes)

    def test_hybrid_architecture_is_unknown_until_an_exact_handler_exists(self):
        metadata = {
            "general.architecture": "qwen35",
            "qwen35.block_count": 65,
            "qwen35.attention.head_count_kv": 4,
            "qwen35.attention.key_length": 256,
            "qwen35.attention.value_length": 256,
        }

        result = self.assess(metadata=metadata)

        self.assertEqual(result.status, "unknown")
        self.assertIn("unsupported architecture", result.reason)

    def test_reserve_scope_mismatch_is_unknown_not_excluded(self):
        different_scope = KvRuntimeScope(
            llama_cpp_commit="other-commit",
            gpu_identifier="test-gpu",
            batch=2048,
            ubatch=2048,
            flash_attn="auto",
            kv_placement="gpu",
        )

        result = self.assess(gpu_vram_bytes=1, runtime_scope=different_scope)

        self.assertEqual(result.status, "unknown")
        self.assertIn("scope", result.reason)

    def test_absent_zero_or_nonconservative_reserve_policy_is_unknown_not_excluded(self):
        absent = self.assess(reserve_policy=None, gpu_vram_bytes=1)
        zero = self.assess(
            reserve_policy=RuntimeReservePolicy(
                bytes=0,
                supported_architectures=frozenset({"llama"}),
                is_conservative=True,
                assumptions=("zero is not a reserve",),
                scope=VALIDATED_SCOPE,
            ),
            gpu_vram_bytes=1,
        )
        nonconservative = self.assess(
            reserve_policy=RuntimeReservePolicy(
                bytes=320,
                supported_architectures=frozenset({"llama"}),
                is_conservative=False,
                assumptions=("unvalidated",),
                scope=VALIDATED_SCOPE,
            ),
            gpu_vram_bytes=1,
        )

        self.assertEqual(absent.status, "unknown")
        self.assertEqual(zero.status, "unknown")
        self.assertEqual(nonconservative.status, "unknown")


if __name__ == "__main__":
    unittest.main()
