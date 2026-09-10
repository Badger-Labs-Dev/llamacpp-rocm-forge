# KV feasibility calibration (package 01)

**Decision:** package 02 must implement **unknown / no static GPU-KV pruning** for the current standard configuration (`-ngl 0`, `-nkvo 0`, `batch=2048`, `ubatch=2048`, `flash_attn=auto`) on Qwen3.5 and any architecture not separately calibrated. No production benchmark behavior changes are made by this package.

## Compatibility statement

```text
Hard exclusion supported for: none in this calibration.
Unknown/no-prune for: Qwen3.5/Qwen3.8 and all current campaign architectures
                     under -ngl 0, -nkvo 0, batch=2048, ubatch=2048,
                     flash-attn=auto.
Reserve: not established; no byte value may be used for pruning.
```

This is a conservative result, not an assertion that the target always fits. It preserves the runtime watchdog probe for uncertain allocation, allocator, workspace, and performance failures.

## Exact build and device

The locally tagged benchmark image was queried directly:

```text
image:  llamacpp-rocm-forge:rocm_10.0.0-llama_v0.4.0-gfx1201-bench
identity: v0.4.0
commit:   5266f24da75dc449bd56cbed7addb9c8e4a6a73e
ROCm:     10.0.0-4
GPU:      AMD Radeon AI PRO R9700 (gfx1201)
VRAM:     34,208,743,424 bytes (32,624 MiB reported by llama-bench)
```

The image identity and commit are in [image-identity.txt](kv-calibration/2026-09-10/image-identity.txt). The source checkout used for citations was detached at that exact commit, which is `v0.4.0`; the image Dockerfile records the resolved commit at [docker/Dockerfile.rocm-10.0.0.ubuntu26:115-142](../docker/Dockerfile.rocm-10.0.0.ubuntu26#L115-L142).

The live-GPU scripts and immutable raw records are under [docs/kv-calibration/](kv-calibration/). They are deliberately outside `results/`; no pre-existing result artifact was altered.

## Source evidence at `5266f24`

All locations below refer to the exact detached `v0.4.0` source tree used to build the image.

1. `tools/llama-bench/llama-bench.cpp:390-405` initializes `no_kv_offload` to `false`; `:837-843` parses `-nkvo`/`--no-kv-offload`; and `:1326-1340` maps it to `cparams.offload_kqv = !no_kv_offload`.
2. `src/llama-kv-cache.cpp:212-234` starts with a CPU buffer type and, when `offload` is true, selects `ggml_backend_dev_buffer_type(model.dev_layer(il))`. It then creates K and V tensors from per-layer `n_embd_k_gqa`, `n_embd_v_gqa`, and `kv_size`.
3. `src/llama-model.cpp:2503-2563` selects `llama_memory_hybrid` and passes both the configured cache types and `cparams.offload_kqv`. `src/llama-memory-hybrid.cpp:11-65` splits hybrid memory into attention KV and recurrent memory, passing the same offload flag to each. `src/llama-memory-recurrent.cpp:83-94` makes the same CPU-versus-`model.dev_layer(i)` device choice for recurrent state.
4. `src/llama-arch.cpp:1064-1086` classifies `LLM_ARCH_QWEN35` and `LLM_ARCH_QWEN35MOE` as hybrid. The Qwen3.8 probe shows 16 attention-cache layers plus 64 recurrent-state layers, so the former all-transformer-block estimate is not exact for this model.
5. `ggml/src/ggml.c:669-676,693-700,737-744` defines `f16`, `q4_0`, and `q8_0` type/block sizing. `ggml/src/ggml-common.h:194-199` defines q4_0 as 32 elements per 18-byte block; `:251-256` defines q8_0 as 32 elements per 34-byte block. The runtime tensor allocation additionally depends on each layer's K/V width and cache shape, not only a generic bytes-per-element multiplier.

### Answer to the placement question

> With `-ngl 0` and default `-nkvo 0`, does this image allocate the KV cache on GPU?

**No for the exercised image/configuration.** `-nkvo 0` does set `offload_kqv=true`, but the source chooses the cache device from `model.dev_layer(il)`. With `-ngl 0`, the live logs assign all model layers to CPU, and the default f16 probe reports `CPU KV buffer size = 40.50 MiB`, exactly as does the explicit host-KV control. Therefore `-ngl 0` is not evidence of GPU-resident KV for this v0.4.0 build. This corrects the expectation recorded in the work package; the live result controls.

## Live probe records

All probes use one repetition, fixed `batch=2048`, `ubatch=2048`, `flash-attn=auto`, `-ngl 0`, ROCm0, Docker device access, and a 300-second watchdog. Each row links its exact command, JSONL stdout, stderr allocation log, GPU before/after snapshots, and exit status.

| ID | Model / cache control | Outcome and key allocation evidence |
|---|---|---|
| A | TinyLlama 15M, f16, `-nkvo 0` | Success (0). [stderr](kv-calibration/2026-09-10/A-default-f16.stderr.log) reports all layers at CPU and `CPU KV buffer size = 40.50 MiB`; [JSONL](kv-calibration/2026-09-10/A-default-f16.jsonl) records `no_kv_offload:false`. |
| B | TinyLlama 15M, f16, `-nkvo 1` | Success (0). [stderr](kv-calibration/2026-09-10/B-host-f16.stderr.log) has the same CPU cache location and size; [JSONL](kv-calibration/2026-09-10/B-host-f16.jsonl) records `no_kv_offload:true`. This is the required contrast and demonstrates why the flag alone is insufficient to infer placement under `-ngl 0`. |
| C | TinyLlama 15M, q8_0, `-nkvo 0` | Expected configuration rejection (1), preserved at [stderr](kv-calibration/2026-09-10/C-default-q8_0.stderr.log): quantized V enables flash attention, then the 48-wide K head fails the 32-element q8_0 block divisibility requirement. |
| D | TinyLlama 15M, q4_0, `-nkvo 0` | Expected configuration rejection (1), preserved at [stderr](kv-calibration/2026-09-10/D-default-q4_0.stderr.log), for the same 48/32 divisibility reason. |
| E | Qwen3.8-27B Q6_K_M, f16, depth 2048, `-nkvo 0` | Success (0); [stderr](kv-calibration/2026-09-10/E-qwen3_8-f16-depth-2048.stderr.log) and [JSONL](kv-calibration/2026-09-10/E-qwen3_8-f16-depth-2048.jsonl) preserve the target-model allocation/output. |
| H | Qwen3.8-27B Q6_K_M, f16, depth 63488, `-nkvo 0` | Success (0), bracketing the stress depth with the adjacent planned campaign depth. [stderr](kv-calibration/2026-09-10/H-qwen3_8-f16-depth-63488.stderr.log) reports a `CPU KV buffer size = 4096.00 MiB`; [JSONL](kv-calibration/2026-09-10/H-qwen3_8-f16-depth-63488.jsonl) reports 216.38 tokens/s. |
| F | Qwen3.8-27B Q6_K_M, f16, depth 96256, `-nkvo 0` | Watchdog timeout (124), not OOM. [stderr](kv-calibration/2026-09-10/F-qwen3_8-f16-depth-96256.stderr.log) is preserved verbatim. No static conclusion is drawn from it. |
| G-q8 | Qwen3.8-27B Q6_K_M, q8_0, depth 2048, `-nkvo 0` | Success (0). [stderr](kv-calibration/2026-09-10/G-qwen3_8-q8_0-depth-2048.stderr.log) shows `CPU KV buffer size = 136.00 MiB` and CPU recurrent state. |
| G-q4 | Qwen3.8-27B Q6_K_M, q4_0, depth 2048, `-nkvo 0` | Success (0). [stderr](kv-calibration/2026-09-10/G-qwen3_8-q4_0-depth-2048.stderr.log) shows `CPU KV buffer size = 72.00 MiB`, plus `CPU RS buffer size = 149.62 MiB`. |

The executable command and all requested raw artifacts accompany each record (`*.command.txt`, `*.jsonl`, `*.stderr.log`, `*.gpu.csv`, and `*.exit-code`). The q4/q8 Qwen follow-up script exists because the small model's 48-wide head cannot use 32-element quantized-KV blocks; it is documented rather than hidden: [run-qwen-quantized-2026-09-10.sh](kv-calibration/run-qwen-quantized-2026-09-10.sh).

No `kvcal-*` containers remain ([remaining-containers.txt](kv-calibration/2026-09-10/remaining-containers.txt)). GPU VRAM returned from 2,577,350,656 bytes before the suite to 2,551,717,888 bytes after the primary suite; the snapshots are [baseline](kv-calibration/2026-09-10/baseline.gpu.csv) and [final](kv-calibration/2026-09-10/final.gpu.csv).

## Reserve decision

A scalar GPU reserve is **not established**. The target q4 probe reports a 3,398.59 MiB ROCm compute buffer for the fixed configuration, but treating that one observed workspace as an all-model reserve would be fabricated: it was measured on a particular hybrid model and allocation mode, and the target's K/V and recurrent state were CPU resident. The 96,256-depth timeout also cannot justify a reserve because it is a timeout, not a classified allocation failure.

This calibration invalidates any future reserve on a change to batch, ubatch, flash-attention behavior, llama.cpp/ROCm image, device, layer offload, model architecture, or cache placement. Package 02 must not turn the observed workspace number into a hard bound.

## Handoff to package 02

Pass these exact inputs:

- **Source:** `v0.4.0`, `5266f24da75dc449bd56cbed7addb9c8e4a6a73e`.
- **Device:** R9700/gfx1201, `34,208,743,424` VRAM bytes.
- **Reserve:** absent (`None`/unknown); no supported pruning scope.
- **Supported metadata observations:** target GGUF exposes `general.architecture=qwen35`, `qwen35.block_count=65`, `qwen35.context_length=262144`, `qwen35.attention.head_count=24`, `qwen35.attention.head_count_kv=4`, `qwen35.attention.key_length=256`, and `qwen35.attention.value_length=256`. The source uses per-layer `n_embd_k_gqa`, `n_embd_v_gqa`, `has_kv`, and hybrid recurrent-layer filtering; the listed GGUF keys do not by themselves establish the complete exact allocation model.
- **Required behavior:** return unknown/no-prune for Qwen3.5 under the current `-ngl 0` campaign; retain runtime probes and preserve timeout as a timeout.
