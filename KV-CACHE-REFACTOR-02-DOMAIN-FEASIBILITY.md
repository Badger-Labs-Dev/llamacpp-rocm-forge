# Work package 02 — pure KV feasibility domain model

## Objective

Create a pure, independently testable domain API that can distinguish:

- statically eligible GPU KV allocation;
- statically excluded GPU KV allocation;
- unknown/unprunable allocation.

It must not call Docker, read files, inspect GPUs, or decide campaign status.

## Prerequisite

Read `KV-CACHE-REFACTOR-01-CALIBRATION.md`. Only enable hard support for architectures/configurations whose exact accounting and reserve scope were established there. Unsupported or insufficient metadata must return `unknown`.

## Existing code to preserve

`benchmark/domain/vram_estimate.py:estimate_full_offload_vram()` remains an advisory full-model estimate. Do not modify it to make campaign fit/fail decisions. It includes model weights and intentionally conservative f16/all-block KV; it answers a different question.

## Proposed public model

Use explicit immutable dataclasses and string literals/enums consistent with repository style. Names may vary, but preserve these semantics:

```python
@dataclass(frozen=True)
class KvFeasibility:
    status: Literal["eligible", "excluded", "unknown"]
    ctk: str
    ctv: str
    context_size: int
    kv_placement: Literal["gpu", "host", "unknown"]
    kv_cache_bytes: int | None
    runtime_reserve_bytes: int | None
    gpu_vram_bytes: int | None
    required_gpu_bytes: int | None
    reason: str | None
    assumptions: tuple[str, ...]
```

A request type may be useful if it keeps the public function narrow and readable:

```python
assess_gpu_kv_feasibility(metadata, context_size, ctk, ctv,
                          kv_offload_enabled, gpu_vram_bytes,
                          reserve_policy) -> KvFeasibility
```

## Rules

1. `context_size` is the llama.cpp context allocation size. The caller—not this module—adds benchmark prefill tokens to benchmark depth.
2. If KV is explicitly host-resident, return `eligible` for GPU-KV capacity with `kv_placement="host"`; do not claim host memory feasibility.
3. If KV placement is unknown, return `unknown`.
4. Do not exclude on missing, malformed, or ambiguous GGUF metadata.
5. Exact K and V dtype accounting is required. Mixed K/V dtype support must either be exact or return `unknown`; do not silently treat mixed types as equal.
6. Every architecture handler must declare its assumptions and supported metadata keys.
7. Exclude only when all relevant inputs are known and:

```text
exact_gpu_kv_cache_bytes + validated_runtime_reserve_bytes > gpu_vram_bytes
```

8. Equality fits. Use integer bytes only; no floating-point GiB arithmetic in decision logic.
9. No model-weight bytes belong in this decision.
10. Do not encode a magic R9700-specific reserve without package 01 evidence and scope validation.

## Architecture support strategy

Start narrow. Use a registry/dispatcher keyed by `general.architecture` only if it avoids speculative generic behavior.

- A conventional dense-attention handler may be supported when metadata exactly establishes layer count, KV-head count, and K/V dimensions.
- Qwen3.5/hybrid must not be treated as conventional dense attention unless package 01 proves the exact active KV-layer/layout calculation.
- Unknown architecture or missing per-layer layout: `unknown`.

This is intentionally less ambitious than the advisory estimator. False positives are more harmful than missed static exclusions.

## Test-first slices

Create focused pure tests; observe each fail before implementation.

1. A known exact f16 allocation exceeding capacity returns `excluded` with every byte field populated.
2. Equality with capacity returns `eligible`.
3. q4_0, q8_0, and f16 produce their exact distinct bytes for a supported fixture.
4. Explicit host KV returns no GPU exclusion.
5. Missing metadata produces `unknown` with an explanatory reason.
6. Hybrid/unsupported metadata produces `unknown` until its exact handler exists.
7. A nonconservative/absent reserve policy produces `unknown`, not `excluded`.
8. Existing `estimate_full_offload_vram()` tests remain green unchanged.

Use small synthetic metadata dictionaries. Do not require an actual multi-GiB GGUF in unit tests.

## Files likely touched

- `benchmark/domain/vram_estimate.py` — only if colocating shared byte utilities is clearer.
- New `benchmark/domain/kv_feasibility.py` — preferred for the new API.
- `benchmark/tests/test_vram_estimate.py` and/or a new `test_kv_feasibility.py`.
- Package 01 evidence doc if the implementation surfaces a missing assumption.

## Acceptance criteria

- [ ] Pure API has no application/Docker/filesystem imports.
- [ ] Every hard exclusion contains exact byte accounting and a reason.
- [ ] Unknown input never prunes.
- [ ] No full-offload model-weight arithmetic is used for hard exclusion.
- [ ] q4_0 → q8_0 → f16 are all covered by tests where supported.
- [ ] Focused tests and the full unit suite pass.

## Handoff output

Document the finalized public dataclasses/functions, their import paths, supported architecture list, reserve-policy interface, and fixture values for package 04.
