# Work package 05 — campaign orchestration: fixed config, dtype-aware planning

## Objective

Integrate the pure planner into the application flow while preserving the watchdog as the runtime authority. Remove the current behavior that lets an f16-only deepest runtime preflight make all later work fail before q4/q8 are meaningfully considered.

## Prerequisites

- Package 03 fixed batch/ubatch tuning behavior.
- Package 04 pure `KvDepthPlan` contract.
- Coordinate manifest payload naming with package 06.

## Existing problem points

- `benchmark/application/run_campaign.py` currently runs dense preflight at `depths[-1]` using default f16 before auto-tuning.
- `auto_tune()` historically probes f16/q8/q4 and treats every non-`ok` result as `-1.0`; ties select iteration order rather than measured success.
- `benchmark/application/dense_sweep.py` treats all probe failures identically and may mark a deep context as capacity failure even after lower-depth measurements succeeded.

## Required campaign flow

```text
read GGUF metadata and requested/trained depths
  ↓
create per-dtype static KV plan
  ↓
record static exclusions before Docker work
  ↓
for candidate target depth from deepest runnable downward:
  for dtype in q4_0 → q8_0 → f16:
    skip only statically excluded dtype/depth pairs
    probe fixed batch=2048, ubatch=2048
    classify runtime result precisely
  select successful dtype(s) at this same target depth
  choose highest throughput only among those successes
  stop once a final candidate is selected
  ↓
resolve final dense --ngl for selected dtype/depth envelope
  ↓
run final prefill and generation curves with fixed config
across selected dtype's runnable depths
```

A real preflight must never run a pair statically excluded by package 04. It may run `unknown` pairs.

## Selection policy

1. Coverage wins over throughput: descend through candidate depths until at least one dtype completes that depth.
2. Compare throughput only between dtypes that completed the same depth under the same fixed 2048/2048 configuration.
3. If q4 succeeds at a deeper context than f16, q4 is a valid deep-capacity configuration. Do not silently describe it as the f16 curve.
4. If no dtype succeeds at a candidate depth, descend to the next candidate depth that is runnable for at least one dtype.
5. If no required baseline depth succeeds for any dtype, campaign failure is valid.

Do not add a lower-ubatch retry path.

## Runtime result classification

Introduce or preserve structured classification sufficient to distinguish:

| Category | Meaning | Static conclusion |
|---|---|---|
| `ok` | command completed and emitted usable measurements | none needed |
| `allocation_failed` | clean llama.cpp context/buffer allocation error | runtime failure only |
| `timed_out` | watchdog killed command | runtime failure only |
| `configuration_error` | unsupported CLI/backend/model configuration | runtime failure only |
| `process_error` | other nonzero execution failure | runtime failure only |

A timeout must not mutate the static planner or justify excluding later runs.

## Dense `--ngl` behavior

The dense sweep can still find the best/safe `--ngl` using real probes, but it must be scoped to the selected dtype and planned depth envelope.

- Keep the output-layer convention: max GPU layers is `block_count + 1`.
- Keep `estimate_full_offload_vram()` messages advisory only.
- Do not let a no-`--ngl` result at an excluded/untested deeper depth force all final curves to be skipped.
- If the selected dtype has runnable lower depths after a higher runtime failure, run final curves for those lower planned depths and report a partial runtime result as appropriate.

## Test-first integration slices

Mock `run_one` at the application boundary, but use real domain planner outputs.

1. A statically excluded f16 96K depth causes no Docker/preflight invocation, while q4 at that depth is still invoked.
2. q4 timeout at the deepest depth does not mark q8/f16 statically excluded and does not fail a campaign before lower-depth selection.
3. q8 and f16 that both complete the same target depth select the higher measured throughput.
4. A q4-only deeper success selects q4 without pretending f16 reached that context.
5. All final configs use `batch=2048`, `ubatch=2048`.
6. No code retries at smaller batch/ubatch after a failure.
7. The legacy dense preflight only receives a planned runnable pair.

## Likely files

- `benchmark/application/auto_tune.py`
- `benchmark/application/run_campaign.py`
- `benchmark/application/dense_sweep.py`
- `benchmark/application/run_curve.py` and Docker adapter only if result classification needs a narrow parsing seam
- characterization/integration tests

## Acceptance criteria

- [ ] Static planning runs before any dense runtime preflight/auto-tune Docker probe.
- [ ] q4 → q8 → f16 is observable in candidate execution order.
- [ ] Selection is depth-first, then throughput.
- [ ] No batch/ubatch grid or ladder remains in normal flow.
- [ ] Timeouts/allocation errors remain runtime evidence, not static exclusions.
- [ ] Focused and full test suites pass.

## Handoff output

Give package 06 the final runtime classification values, selection-result payload, and the exact point where planned/runnable depths become `run_one()` input.
