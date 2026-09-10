# Work package 01 — llama.cpp KV allocation calibration

## Objective

Produce the evidence required to decide whether hard GPU-KV exclusions can be enabled for the fixed campaign configuration. This package may change no production planning behavior; its primary deliverable is a reproducible evidence record and a clearly bounded support decision.

## Why this is required

The existing estimator is intentionally advisory and assumes f16 KV for every transformer block. That is not enough for hard pruning, particularly for Qwen3.5/hybrid architectures. A hard skip must be based on allocation behavior that is exact under stated assumptions, not on a conservative upper bound selected to match one timeout.

## Scope

- Verify llama.cpp v0.4.0 KV placement under both default and explicit offload controls.
- Identify the exact metadata and source-level inputs that determine actual KV allocation for a supported architecture.
- Measure or otherwise defensibly bound fixed-config runtime reserve on the target ROCm GPU.
- Record whether Qwen3.5 can receive hard-exclusion support now, or must remain `unknown` until more information is available.

## Immutable configuration under test

```text
llama.cpp: v0.4.0 / build commit baked into the benchmark image
backend: ROCm target image
GPU: target campaign GPU
batch=2048
ubatch=2048
flash_attn=auto
repetitions: minimal supervised feasibility probe, not final benchmark sampling
```

Do not vary batch or ubatch in this package. The point is to characterize the adopted standard configuration.

## Required source evidence

Use the exact upstream revision represented by the image, not current `master`.

Confirm and cite file/line locations for:

1. `llama-bench` CLI parsing for `--no-kv-offload` / `-nkvo` and its default.
2. Mapping from parsed parameter to `llama_context_params.offload_kqv`.
3. KV-cache buffer/device selection when offload is enabled.
4. Architecture-specific logic relevant to Qwen3.5/hybrid attention/recurrence layers.
5. Storage sizing rules for `f16`, `q8_0`, and `q4_0` KV tensors.

The source notes must explicitly answer:

> With `-ngl 0` and default `-nkvo 0`, does llama.cpp allocate the KV cache on GPU?

Expected answer from prior investigation: yes, subject to verifying the exact image build.

## Required live-GPU probes

Start with a small representative GGUF. Use Qwen3.8 only after the small model produces readable allocation logs.

For each run, preserve command, stdout/stderr, JSONL, GPU identity, and timestamp.

| Run | Required settings | Purpose |
|---|---|---|
| A | `-ngl 0`, default/explicit `-nkvo 0`, f16 | establish default GPU-KV behavior |
| B | `-ngl 0`, explicit `-nkvo 1`, f16 | contrast host-KV behavior; not a new campaign default |
| C | `-ngl 0`, `-nkvo 0`, q8_0 | confirm quantized K/V allocation/log path |
| D | `-ngl 0`, `-nkvo 0`, q4_0 | confirm quantized K/V allocation/log path |

Use enough logging to capture KV buffer size and backend/device. A test that only reports end-to-end success is insufficient.

For the target Qwen model, select several depths around the planned threshold. At least one must be comfortably successful and one must be expected to fail or stress allocation. Keep the watchdog enabled; record a timeout as a timeout, not as OOM.

## Runtime reserve methodology

The reserve represents non-KV GPU memory required by the fixed benchmark configuration: backend baseline, model-independent allocations, compute graph/workspace, and allocator headroom.

A candidate reserve is acceptable only if the evidence document states:

- the exact GPU VRAM value used;
- the exact fixed configuration and backend/build;
- how baseline/peak/residual allocation was observed or bounded;
- why the number is conservative across the supported models;
- what invalidates it (different batch, ubatch, flash-attention behavior, backend/build, or device);
- why it is not inferred solely from the 96,256-depth timeout.

If this cannot be supported, do **not** invent a reserve. The correct outcome is a feasibility implementation that returns `unknown` for the affected architecture/configuration.

## Deliverables

1. A committed Markdown evidence note, preferably `docs/kv-feasibility-calibration.md`, containing commands, source citations, observed allocation records, and support decision.
2. A small machine-readable fixture or test metadata fixture only if it can be checked into the repository without bundling large model artifacts.
3. An explicit compatibility statement, for example:

```text
Hard exclusion supported for: <architectures/configurations>
Unknown/no-prune for: <architectures/configurations>
Reserve: <bytes>, valid only for: <scope>
```

## Acceptance criteria

- [ ] Exact llama.cpp version/build is identified.
- [ ] Default `-nkvo` semantics are proven from source and live output.
- [ ] q4_0/q8_0/f16 are represented in evidence.
- [ ] Qwen3.5 hybrid handling is resolved, or explicitly marked unsupported for hard pruning.
- [ ] A reserve is justified with data, or the implementation plan explicitly uses `unknown` rather than a fabricated threshold.
- [ ] No production campaign behavior changes in this package unless separately approved.

## Handoff notes

Pass the evidence note, exact source revision, device VRAM bytes, reserve scope, and supported metadata keys to the agent implementing package 02. Do not pass an unexplained GiB estimate as a requirement.
