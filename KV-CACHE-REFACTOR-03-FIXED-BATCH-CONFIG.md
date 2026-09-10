# Work package 03 — remove batch/ubatch tuning and standardize 2048/2048

## Objective

Remove the batch/ubatch grid from normal auto-tuning and make the campaign’s standard runtime configuration explicit and fixed:

```text
batch=2048
ubatch=2048
flash_attn=auto
```

This package intentionally does not add a fallback ladder or another automatic optimizer.

## Why

The benchmark’s primary output is throughput versus context depth, not a broad search of prompt scheduling parameters. With the normal prompt workload fixed at 2,048 tokens, the 13-point `ubatch × batch` grid adds time and complexity without directly answering the campaign question.

The old grid contained 13 valid combinations. Removing it leaves KV dtype probing as the only auto-tune stage.

## Existing implementation

`benchmark/application/auto_tune.py` currently contains:

- `UBATCH_CANDIDATES`;
- `BATCH_CANDIDATES`;
- `valid_batch_grid_count()`;
- a second `ubatch_batch_grid` tuning stage;
- grid-specific log and console output.

`BenchConfig` defaults already use `ubatch=2048` and `batch=2048` in `benchmark/domain/models.py`.

## Required changes

1. Remove grid constants and `valid_batch_grid_count()`.
2. Remove the complete second tuning stage and its sleeps/probes/log record.
3. Update `auto_tune()` documentation from two-stage coordinate descent to KV-dtype-only tuning.
4. Ensure every config built in remaining auto-tune code explicitly or reliably uses `batch=2048`, `ubatch=2048`.
   - Prefer a named constant only if it improves consistency without adding a new abstraction layer.
5. Update console text and progress calculations so they no longer promise/count grid probes.
6. Update campaign/tuning manifest expectations to have one stage, `kv_cache_dtype`.
7. Do not remove `BenchConfig.batch` or `BenchConfig.ubatch`; they remain part of the public run configuration and filenames.
8. Do not alter explicit user/CLI override behavior without separately locating and documenting it. This package only removes automatic sweeping.

## Test-first requirements

Update/add characterization tests around `auto_tune()`.

- `auto_tune()` invokes exactly one candidate config per KV dtype, with no grid configs.
- Every auto-tune config uses `ubatch=2048` and `batch=2048`.
- Tuning log contains `kv_cache_dtype` and no `ubatch_batch_grid` stage.
- The fixed winner retains 2048/2048.
- Progress/probe accounting no longer reserves 13 grid probes.
- Existing tests that imported removed constants/functions are replaced with behavioral assertions, not compatibility shims.

Run the focused test module, then the full benchmark test suite.

## Non-goals

- Do not select a smaller ubatch after failure.
- Do not retry failed probes at another batch size.
- Do not add a `--tune-batch` mode in this refactor.
- Do not change KV dtype ordering/selection rules; package 05 owns that coordinated behavior.

## Acceptance criteria

- [ ] No source references remain to `UBATCH_CANDIDATES`, `BATCH_CANDIDATES`, `valid_batch_grid_count`, or `ubatch_batch_grid`.
- [ ] Normal tuning runs no batch/ubatch grid probe.
- [ ] Remaining automatic configs use fixed 2048/2048.
- [ ] Tests prove behavior and pass.
- [ ] Existing result artifacts are untouched.

## Handoff output

State which tests were updated, whether any CLI override behavior exists, and the exact current auto-tune call contract for package 05.
