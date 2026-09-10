# Work package 04 — pure per-dtype context-depth planner

## Objective

Turn pure per-context `KvFeasibility` results into a pure campaign plan that preserves the original depth request while deciding which depths each KV dtype may be asked to probe.

## Prerequisite

Package 02 must provide the feasibility API and documented `eligible` / `excluded` / `unknown` semantics.

## Core rule

Plan **per dtype**, never by globally truncating the campaign depths:

```text
q4_0: own eligible/excluded/unknown depth sets
q8_0: own eligible/excluded/unknown depth sets
f16:  own eligible/excluded/unknown depth sets
```

A depth that is statically excluded for f16 can remain planned for q8_0 or q4_0. `unknown` remains planned because the static model is not entitled to prune it.

## Proposed domain types

Names may differ, but preserve this information:

```python
@dataclass(frozen=True)
class DtypeDepthPlan:
    ctk: str
    ctv: str
    requested_depths: tuple[int, ...]
    runnable_depths: tuple[int, ...]       # eligible + unknown
    eligible_depths: tuple[int, ...]
    unknown_depths: tuple[int, ...]
    exclusions: tuple[KvFeasibility, ...]

@dataclass(frozen=True)
class KvDepthPlan:
    requested_depths: tuple[int, ...]
    prefill_tokens: int
    dtype_plans: tuple[DtypeDepthPlan, ...]
```

The planner receives benchmark depths but calls the feasibility API with:

```text
context_size = depth + prefill_tokens
```

It must preserve both values so artifacts do not confuse benchmark depth with allocated context size.

## KV candidate order

Declare one canonical candidate sequence used by planning/application code:

```python
KV_CACHE_TYPES = ("q4_0", "q8_0", "f16")
```

This is an execution order intended to obtain deep successful data early. It is not ranking policy.

## Required pure functions

At minimum:

1. Build a `KvDepthPlan` from requested depths, fixed prefill tokens, metadata, placement, GPU bytes, and reserve policy.
2. Return the deepest runnable depth for a dtype.
3. Return candidate dtype/depth targets in coverage-first order for the application layer.
4. Produce a serializable payload or a conversion function suitable for the manifest, without performing I/O.

Keep `run_campaign.py` free of byte arithmetic and list-filtering policy.

## Test-first cases

Use synthetic feasibility input/fixtures where possible.

1. f16 can exclude 96,256 while q8/q4 retain it; original requested depths remain unchanged.
2. Unknown f16 feasibility retains that depth in `runnable_depths`.
3. A dtype with every requested depth excluded has no runnable target and is not probed later.
4. Input depth ordering is preserved and duplicates/invalid values follow existing repository conventions.
5. `prefill_tokens` is added exactly once.
6. Planner output contains precise exclusion reasons/bytes passed through from domain feasibility.
7. Candidate order is q4 → q8 → f16, while selection data remains independent of iteration order.

## Qwen regression fixture

Only after package 01 and 02 establish valid Qwen3.5 support, add the requested regression fixture:

- R9700 reportable VRAM: `34,208,743,424` bytes.
- fixed campaign prompt/prefill handling included.
- f16 retains 63,488 and excludes 96,256, 129,024, and 260,096 under the finalized evidence-backed reserve.
- q8_0 and q4_0 prove their own thresholds using actual supported metadata.

Do not create this test by encoding the old all-block f16 estimator or an arbitrary reserve.

## Files likely touched

- New `benchmark/domain/kv_feasibility.py` or new `benchmark/domain/kv_depth_planner.py`.
- New tests such as `benchmark/tests/test_kv_depth_planner.py`.
- Central KV candidate definition location, if package 03/05 needs it.

## Acceptance criteria

- [ ] Planner is pure and has no Docker/application imports.
- [ ] Static exclusions are dtype-specific and visible in output.
- [ ] Unknown is runnable, never pruned.
- [ ] The original requested/trained contexts are preserved.
- [ ] Tests cover f16/q8/q4 distinction and exact prompt-depth arithmetic.

## Handoff output

Provide exact types/functions, serialized payload shape, canonical dtype order, and a short sample plan to the package 05 and 06 agents.
