# Benchmarking strategy and recovery

`benchmark/run_bench.py` runs `llama-bench` in a short-lived Docker container for each probe. It records raw JSONL, a compact CSV curve, `metadata.json`, and `campaign_manifest.json` under `results/<model-slug>/<run-id>/`.

The run ID is version-based: `rocm<version>_llamacpp<build-number>`. It intentionally does not include a date. The date is stored inside the JSON metadata; the directory identity answers the question this project is meant to track: how did this model behave under this ROCm and llama.cpp build?

## What the default (auto-tune) mode tunes

By default the driver stages a search rather than testing exhaustively:

1. A static, per-KV-dtype context-depth feasibility plan (`benchmark/domain/kv_feasibility.py`, `kv_depth_planner.py`) built before any Docker probe runs. It can statically **exclude** a `(KV dtype, depth)` pair only from an exact, GPU-resident KV-cache byte calculation plus a validated runtime reserve; anything the static model cannot prove stays `unknown` and remains runnable. This never truncates the campaign's depth list globally - exclusion is always dtype-specific, so q4/q8 can still probe a depth f16 was excluded from.
2. Runtime KV-cache dtype/depth selection: candidate `(dtype, depth)` pairs are probed in `q4_0` -> `q8_0` -> `f16` order, deepest depth first, at the fixed `batch=2048, ubatch=2048` configuration - never the old ubatch/batch grid. Coverage wins over throughput: the search descends through depths until at least one dtype completes that depth, and only completions at the *same* depth compete on measured throughput.
3. The winning configuration over the full depth curve, for both prefill and generation.

There is no independent ubatch x batch grid anymore, and no batch/ubatch fallback ladder if a probe fails - see [KV-cache feasibility refactor](../KV-CACHE-REFACTOR-00-OVERVIEW.md) for the full design rationale. `campaign_manifest.json`'s `kv_feasibility` field (present whenever the default mode ran) records the fixed runtime config, the canonical `q4_0`/`q8_0`/`f16` probing order, and every dtype's `runnable_depths`/`eligible_depths`/`unknown_depths`/`excluded_depths` with exact byte accounting for each exclusion. `tuning_log`'s `kv_cache_dtype` stage still publishes the legacy `{stage, scores, winner}` shape the viewer's sensitivity chart expects, scored only from probes at the final selected target depth.

For a dense model, the driver performs a real full-offload probe at the deepest selected depth before tuning. If it fails, an exact bisection finds a safe `--ngl` for tuning. After tuning, a per-depth dense sweep finds each maximum fitting `--ngl`; additional lower values are sampled only when at least one depth needs CPU offload. The final ordinary depth curve uses the deepest-depth-safe `--ngl` at every depth, so depth remains the only changing variable.

For an MoE model, auto-tune uses fully CPU-offloaded experts while tuning and for its ordinary full-depth curve. This keeps an unrelated expert-weight allocation from making a deep KV-cache probe fail before the dedicated MoE sweep can map the actual context-versus-offload trade-off. See [MoE expert offload](moe-offload.md).

`--quick` skips auto-tuning: it uses fixed `ubatch=2048, batch=2048, ctk/ctv=f16, fa=auto`. It never consults the KV feasibility plan (`kv_feasibility` is `null` in a `--quick` run's manifest). Dense models still get an exact `--ngl` boundary per depth because fit correctness is not a tuning-quality option; if offload is needed, only one extra CPU-offload throughput point is sampled.

## Static exclusion vs. unknown vs. runtime failure

These are four distinct things and the manifest keeps them distinct:

- **Statically excluded** (`kv_feasibility.per_dtype.<dtype>.excluded_depths`/`exclusions`): an exact GPU-resident KV-cache byte calculation plus a validated runtime reserve proves the allocation cannot fit. No Docker probe ever runs for that `(dtype, depth)` pair. Only the supported dense/`llama` architecture handler with exact metadata and a scope-matched reserve can produce this; anything else stays `unknown`.
- **Unknown** (`kv_feasibility.per_dtype.<dtype>.unknown_depths`): the static model cannot prove or disprove feasibility (unsupported architecture, missing metadata, unvalidated reserve, or `-nkvo` placement uncertainty). Unknown depths remain runnable and are still probed at runtime - the static model is never entitled to prune them.
- **Runtime allocation failure / timeout / configuration error**: a real probe ran and did not succeed. This is evidence about *this run*, not a fact fed back into the static planner - a timeout at one depth/dtype never marks a different depth or dtype as statically excluded.
- **Omitted static depth vs. a completed measurement**: an excluded depth never appears in `curve_summary.csv` as a zero-throughput row, and the viewer must never render it as if it were a completed point at 0 tok/s. It is absent from the curve and present only in `kv_feasibility`'s exclusion record.

## Campaign status semantics

- **`finished`**: every planned runnable depth for the selected dtype completed. This includes campaigns with static exclusions - a larger depth being statically excluded for the winning dtype does not, by itself, make a campaign anything other than `finished` once every depth that *was* planned for that dtype completes.
- **`partial`**: a runtime probe/curve stopped after some lower depth's work already completed successfully.
- **`failed`**: no configuration completed the required baseline work (e.g. no KV dtype/depth probe ever succeeded, or the deepest requested dense `--ngl` search found nothing that fits).
- `campaign.finished`/`.partial`/`.failed` (the on-disk marker), `metadata.json`'s `status`, `campaign_manifest.json`, and the console summary always agree - they are all derived from the same `failed`/`partial` booleans in `run_model_campaign()`.

Flash attention stays at llama.cpp's `-fa auto` in both modes. That lets the backend select the fused path only when the model and kernel support it. Quantized KV cache types still force flash attention on when llama.cpp requires it.

## Context depths come from the GGUF

The driver reads `<architecture>.context_length` from the GGUF header and tests the common context sizes that fit under that trained limit, plus depth 0. A model should not be benchmarked beyond its trained context window just because llama.cpp accepts the number. If the context length is absent, the driver falls back to its legacy depth list and prints a warning. The chosen `depths` and `context_length` are saved in the manifest.

## OOM and timeout behavior

Large context sizes can fail without returning a normal out-of-memory error. On this hardware, a 35B model at its full 262K context consumed roughly 97% of VRAM and then stopped making progress.

The driver therefore tests depths in ascending order, one Docker invocation at a time. A failure or timeout stops only the larger depths; already-completed smaller depths remain in the result. The status is:

- `ok`: every requested depth completed.
- `partial`: a later depth failed or timed out after earlier depths completed.
- `failed`: the first depth failed.

Check `stop_reason`, `depths_run`, and `depths_skipped` in `campaign_manifest.json` when a run is partial. Use `--max-depth N` when the absolute model limit is outside the context range you care about.

## Progress and ETA

Before the first Docker call, the driver prints a conservative maximum number of Docker probes for the selected mode. It includes tuning, the dense or MoE offload sweep, and the final prefill and generation curves. Between calls it reports completed, pruned, and remaining probes, plus two percentages:

- **worst-case**: completed work divided by the original maximum;
- **current plan**: completed work divided by the probes still needed after exclusions.

A failed depth or an infeasible fully-offloaded MoE point prunes work that cannot produce a useful result. ETA begins after the first completed probe and uses the rolling median Docker-probe duration plus the standard between-depth pause. It is deliberately an estimate; a deep OOM timeout can make the next probe much slower than the earlier ones.

## Recovering from an interrupted run

Every Docker invocation receives a generated `llamacpp-rocm-forge-...` name. The driver tracks those names and kills them on Ctrl-C, SIGTERM, unhandled exceptions, and per-depth timeouts. `kill -9` cannot run cleanup handlers. If VRAM remains occupied after a forceful interruption, inspect and remove any remaining containers:

```bash
docker ps --filter name=llamacpp-rocm-forge- --format '{{.Names}}'
docker kill $(docker ps --filter name=llamacpp-rocm-forge- -q)
rocm-smi --showmeminfo vram
```
