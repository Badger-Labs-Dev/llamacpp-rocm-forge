# KV feasibility domain API (package 02 handoff)

## Public imports

```python
from domain.kv_feasibility import (
    KvFeasibility,
    KvRuntimeScope,
    RuntimeReservePolicy,
    assess_gpu_kv_feasibility,
)
```

`assess_gpu_kv_feasibility()` is pure: it accepts parsed metadata and explicit
facts, and performs no file, Docker, GPU, or campaign-status operations.

## Supported accounting

The only exact handler is conventional dense `general.architecture="llama"`.
It requires these positive integer keys:

- `llama.block_count`
- `llama.attention.head_count_kv`
- `llama.attention.key_length`
- `llama.attention.value_length`

It calculates K and V separately for `f16`, `q8_0`, and `q4_0`; quantized
row widths must be divisible by their 32-element GGML block. Mixed K/V dtypes
are accounted independently. Hybrid architectures, including `qwen35`, and
missing or malformed metadata return `unknown`.

The package-01 calibration provides **no current hard-exclusion scope or
reserve policy**. Its current Qwen3.5/Qwen3.8 `-ngl 0` configuration must pass
`kv_placement=None` and `reserve_policy=None`, therefore remains `unknown` and
must retain its runtime probe.

## Reserve and runtime scope contract

An exclusion is possible only with all of:

1. explicit `kv_placement="gpu"` and `kv_offload_enabled=True`;
2. a positive, conservative `RuntimeReservePolicy` that declares the
   architecture; and
3. a `KvRuntimeScope` exactly equal to `RuntimeReservePolicy.scope`.

`KvRuntimeScope` binds the reserve to the llama.cpp commit, GPU identifier,
fixed `batch`/`ubatch`, flash-attention mode, and placement. A mismatch or
missing scope is `unknown`, never excluded. `KvFeasibility` reports exact KV,
reserve, required-GPU, and capacity byte fields only for a supported,
GPU-resident assessment. Equality fits; only `required_gpu_bytes >
gpu_vram_bytes` excludes.

## Test fixture values

`benchmark/tests/test_kv_feasibility.py` uses synthetic conventional metadata:
2 blocks, 2 KV heads, K width 32, V width 64, context 10, and a 320-byte
synthetic validated reserve. Expected KV bytes: q4_0 = 2,160; q8_0 = 4,080;
f16 = 7,680. Required f16 GPU bytes are 8,000, so 7,999 excludes and 8,000 is
eligible. These fixture values are API arithmetic tests, not production
calibration evidence.
