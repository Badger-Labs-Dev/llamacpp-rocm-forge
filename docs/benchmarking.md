# Benchmarking strategy and recovery

`benchmark/run_bench.py` runs `llama-bench` in a short-lived Docker container for each probe. It records raw JSONL, a compact CSV curve, `metadata.json`, and `campaign_manifest.json` under `results/<model-slug>/<run-id>/`.

The run ID is version-based: `rocm<version>_llamacpp<build-number>`. It intentionally does not include a date. The date is stored inside the JSON metadata; the directory identity answers the question this project is meant to track: how did this model behave under this ROCm and llama.cpp build?

## What the default (auto-tune) mode tunes

By default the driver stages a search rather than testing exhaustively:

1. KV-cache dtype: `f16`, `q8_0`, and `q4_0`, at the shallowest and deepest selected depths.
2. A valid ubatch × batch grid, using the selected KV dtype at depth 0.
3. The winning configuration over the full depth curve, for both prefill and generation.

For a dense model, the driver performs a real full-offload probe at the deepest selected depth before tuning. If it fails, an exact bisection finds a safe `--ngl` for tuning. After tuning, a per-depth dense sweep finds each maximum fitting `--ngl`; additional lower values are sampled only when at least one depth needs CPU offload. The final ordinary depth curve uses the deepest-depth-safe `--ngl` at every depth, so depth remains the only changing variable.

A full independent grid would be 4 ubatch values × 4 batch values × 3 KV dtypes, before multiplying by depths and repetitions. The staged search is cheaper and records every tested score in `campaign_manifest.json`'s `tuning_log`, so a later viewer can show what mattered and how much.

Flash attention stays at llama.cpp's `-fa auto` in both modes. That lets the backend select the fused path only when the model and kernel support it. Quantized KV cache types still force flash attention on when llama.cpp requires it.

For an MoE model, auto-tune uses fully CPU-offloaded experts while tuning and for its ordinary full-depth curve. This keeps an unrelated expert-weight allocation from making a deep KV-cache probe fail before the dedicated MoE sweep can map the actual context-versus-offload trade-off. See [MoE expert offload](moe-offload.md).

`--quick` skips auto-tuning: it uses fixed `ubatch=2048, batch=2048, ctk/ctv=f16, fa=auto`. Dense models still get an exact `--ngl` boundary per depth because fit correctness is not a tuning-quality option; if offload is needed, only one extra CPU-offload throughput point is sampled.

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
