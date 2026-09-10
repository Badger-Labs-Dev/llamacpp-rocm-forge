# Work package 06 — artifacts, campaign status, and documentation

## Objective

Make static planning omissions visible and distinguish them from runtime failures. Update public documentation so campaign status accurately reflects late dense-capacity discovery and dtype-specific depth coverage.

## Prerequisites

Coordinate with package 04 for plan payload shape and package 05 for runtime-result classification/selection output.

## Manifest requirements

Preserve existing fields for compatibility. Add new fields rather than overwriting context/depth data.

Recommended shape:

```json
{
  "requested_depths": [0, 2048, 30720, 47104, 63488, 96256],
  "kv_feasibility": {
    "fixed_runtime_config": {
      "batch": 2048,
      "ubatch": 2048,
      "flash_attn": "auto",
      "no_kv_offload": false
    },
    "candidate_dtype_order": ["q4_0", "q8_0", "f16"],
    "per_dtype": {
      "f16": {
        "runnable_depths": [0, 2048, 63488],
        "eligible_depths": [0, 2048, 63488],
        "unknown_depths": [],
        "excluded_depths": [96256],
        "exclusions": [
          {
            "benchmark_depth": 96256,
            "context_size": 982...,
            "kv_placement": "gpu",
            "kv_cache_bytes": 0,
            "runtime_reserve_bytes": 0,
            "gpu_vram_bytes": 0,
            "required_gpu_bytes": 0,
            "reason": "exact GPU KV allocation plus validated reserve exceeds reportable VRAM",
            "assumptions": []
          }
        ]
      }
    }
  }
}
```

The ellipsis is illustrative only; production JSON must use exact integers. Do not emit placeholder values.

Required properties:

- Keep original requested depths and trained context limit intact.
- Record benchmark depth and allocated context size separately.
- Record byte accounting for every static exclusion.
- Record `unknown` separately from eligible/excluded.
- Record fixed runtime config and KV placement policy used for planning.
- Record selected final configuration and selection reason.
- Do not record full-offload advisory totals as if they were KV-only hard facts.

## Status policy

Implement a policy table equivalent to:

| Scenario | Status |
|---|---|
| Every planned runnable final depth completes | `finished` |
| Larger requested depths statically excluded; all runnable work completes | `finished` |
| A runtime curve/probe stops after lower useful work completed | `partial` |
| No configuration completes required baseline work | `failed` |

Rules:

- Static exclusions alone never make a campaign `partial` or `failed`.
- `timed_out` and `allocation_failed` remain visible in run summaries.
- Do not let the old `dense_capacity_failed` boolean convert a campaign with successful lower planned depths into `failed`.
- Status markers (`campaign.finished`, `.partial`, `.failed`), `metadata.json`, manifest, and console summary must agree.

## Summary and viewer compatibility

Review `curve_summary.csv` and the viewer dataset contract before changing behavior.

- The summary should contain actual completed measurements only.
- Static exclusions must be visible in manifest/metadata and any viewer data path that represents omitted planned depths.
- Do not represent an excluded point as a zero-throughput measurement.
- Do not label q4/q8 deep curves as f16 curves.
- If the viewer has no schema for planned exclusions, add a versioned optional field rather than breaking existing consumers.

## Documentation updates

Update at least:

- `docs/benchmarking.md`
- `docs/vram-estimator.md`
- `docs/architecture.md` if it describes campaign phases or artifact contracts

Document:

1. Fixed `batch=2048`, `ubatch=2048`; no normal grid/ladder.
2. q4 → q8 → f16 probing order and coverage-first selection.
3. Difference between static `excluded`, static `unknown`, clean allocation failure, and watchdog timeout.
4. The static hard-bound formula and its narrow applicability.
5. Why full model + KV estimate remains advisory and is not used as an all-`--ngl` no-go test.
6. Why omitted static depths are not completed measurements.
7. Correct `finished` / `partial` / `failed` semantics.

## Test-first requirements

1. A manifest with static exclusions preserves original requested depths and contains exact exclusion accounting.
2. A static exclusion alone produces `finished` when all runnable curves complete.
3. A timeout after some successful planned work produces `partial` and records `timed_out`.
4. No baseline success produces `failed`.
5. Manifest, metadata, and status marker agree for each case.
6. Existing consumers/tests remain compatible with optional new fields.

## Acceptance criteria

- [ ] Static exclusions are observable and cannot be mistaken for successful points.
- [ ] Every output uses consistent status semantics.
- [ ] Existing result artifacts remain untouched.
- [ ] Documentation states the actual new behavior and assumptions.
- [ ] Focused artifact tests and full suite pass.

## Handoff output

Provide example generated manifest/metadata/status-marker artifacts from unit/integration fixtures and list any viewer-contract follow-up required for package 07.
