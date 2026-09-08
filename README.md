# r9700-llm-bench

llama.cpp + ROCm 7.2.4 benchmarking on an AMD Radeon AI PRO R9700 (gfx1201),
running on plain Docker (Ubuntu 24.04 base) instead of Toolbx/Podman.

Forked out of [amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes),
which targets a different GPU (Strix Halo APU, gfx1151) and a different host
setup (Fedora Toolbx + their own "AI Toolbox Cockpit" orchestration tooling
across their own SSH hosts). This repo strips that down to what actually
applies to our hardware and workflow.

## Contents

- `docker/` - the Dockerfile (Ubuntu 24.04 base, ROCm 7.2.4, gfx1201 target)
  plus its direct build dependencies (grammar patch, VRAM estimator script).
- `benchmark/` - upstream's benchmark orchestration scripts, kept for
  reference/adaptation. `UPSTREAM_RUNBOOK_REFERENCE.md` is their agent runbook;
  it assumes their SSH hosts, `llama-cockpit` CLI, and Toolbx - none of which
  we have. Treat it as a spec for the benchmark *protocol* (depths, batch
  sizes, repetitions, etc.), not a runnable procedure here.
- `docs/` - reference docs (VRAM estimation, local build notes) carried over
  as-is.

## Status

Working end-to-end. `benchmark/run_bench.py` drives `llama-bench` inside a
plain `docker run` container through the depth/ubatch sweep, writing one
JSONL per (model, series, ubatch), a `curve_summary.csv`, and a
`campaign_manifest.json` per results directory. Validated with a smoke test
(reduced depth set) against a small Qwen2.5-0.5B GGUF on the R9700.

## Results layout

```
results/
  <model-slug>/                        e.g. qwen2-5-0-5b-instruct-q4-k-m
    <run-id>/                          e.g. rocm7.2.4_llamacpp9974
      metadata.json                    compact, for a future results webpage
      campaign_manifest.json           full detail: config, tuning log, depths
      curve_summary.csv                one row per (series, depth)
      *.jsonl / *.stderr.log           raw llama-bench output per series
      tuning/                          --full-sweep probe artifacts, if used
```

`<model-slug>` is derived from the GGUF filename (lowercased,
non-alphanumerics collapsed to `-`) — one real model+quant keeps the same
slug across every run, so results for it live together regardless of when
they were produced.

`<run-id>` is `rocm<version>_llamacpp<build-number>` — no date/timestamp.
The intent is to track results **across llama.cpp/ROCm versions over time**
(for an eventual static webpage comparing how a model's numbers change as
those versions move), so the versions themselves are the identity, not the
clock. `run_bench.py` queries both directly from the container
(`benchmark/environment_info.py`: `dpkg-query` for rocm-core, `llama-cli
--version` for the build) rather than trusting the image tag, which could
drift out of sync with what's actually installed.

Re-running the same model against the same ROCm+llama.cpp versions collides
on the same `run-id` on purpose — `run_bench.py` refuses to overwrite an
existing run directory unless you pass `--force`. `--results-root` sets the
root (default `./results`); each `--model` gets its own
`<results-root>/<slug>/<run-id>/` even within one invocation.

`metadata.json` is the compact, purpose-built file for a future site
generator to read across many run directories without parsing every CSV:
model name/architecture/context-length, the full environment (ROCm version,
llama.cpp build, GPU name/VRAM, host kernel), the final chosen config, and a
summary throughput number. `campaign_manifest.json` keeps the full detail
(every stage's tuning scores, every run's status) for deeper inspection.

## Building the image

```bash
cd docker
docker build -f Dockerfile.rocm-7.2.4-rdma-fix.ubuntu -t r9700-llm-bench:rocm-7.2.4 .
```

## Running a benchmark

GPU device access needs the host's `video`/`render` group GIDs passed into
the container (the script resolves these automatically via `grp`, or pass
`--gpu-gid` explicitly). `-dev ROCm0` in the script targets the discrete
R9700 specifically — the host also exposes the Ryzen iGPU as `ROCm1`, and
`llama-bench --list-devices` (run via `docker run ... llama-bench
--list-devices`) shows both if you need to confirm device numbering.

Full auto-tuned sweep — recommended default. Stages through flash-attn
on/off, KV cache dtype (f16/q8_0/q4_0), then a ubatch × batch grid, each
stage probing at 2 depths only (shallowest + deepest of the model's derived
curve) to keep total runtime bounded; then runs the winning combination
across the model's full depth curve exactly once:

```bash
python3 benchmark/run_bench.py \
  --model ~/models/your-model.gguf \
  --full-sweep
```

Fixed config (fast, when you already know what you want):

```bash
python3 benchmark/run_bench.py \
  --model ~/models/your-model.gguf \
  --ubatch 1024 --batch 2048 --ctk q8_0 --ctv q8_0 --flash-attn 1
```

Legacy `--calibrate` (ubatch-only sweep, batch/KV/flash-attn held at
defaults) is kept for results directories produced before the full sweep
existed.

Multiple models in one campaign: repeat `--model` — each gets its own
`<results-root>/<slug>/<run-id>/` directory (see "Results layout" below).
Use `--max-depth N` to cap the depth sweep for a quick smoke test without
waiting through a model's full context range. Use `--force` to overwrite an
existing run directory (same model + same ROCm/llama.cpp versions).

`benchmark/UPSTREAM_RUNBOOK_REFERENCE.md`, `run_calibrated_campaign.py`,
`validate_campaign.py`, `generate_results_json.py`, and
`merge_curve_summary.py` are upstream's originals, kept for reference; they
depend on their Toolbx/Cockpit/SSH-host stack and don't run here as-is.

## What gets swept, and why not everything

`llama-bench` exposes more knobs than we tune. `--full-sweep` covers
ubatch, batch size, KV cache dtype, and flash-attention — the ones most
likely to move throughput meaningfully on a single GPU. Left out
deliberately:

- `-ngl` (GPU layers) — only matters when a model doesn't fully fit in
  VRAM; irrelevant at `-ngl 99` for models that do.
- `-sm` (split-mode), `-nkvo`/`-nopo`/`--no-host` (offload toggles) — only
  matter for multi-GPU or specific memory-pressure scenarios, not a
  single-R9700 setup.
- Speculative decoding / multi-token prediction (MTP) — `llama-bench` has
  no flag for this; it's a `llama-server`/`llama-cli` feature (`-md`, a
  draft model), not something this throughput benchmark measures.

A true grid over ubatch(4) × batch(4) × KV(3) × flash-attn(2) is 96
combinations — infeasible to run at full depth × 3 repetitions per
combination. `auto_tune()` in `run_bench.py` instead does staged
(coordinate-descent) tuning: pick the best flash-attn setting, then the
best KV cache dtype (with flash-attn forced on, since llama.cpp requires it
whenever the KV cache is quantized), then the best ubatch/batch pair —
carrying each stage's winner into the next rather than testing every
combination of everything. This isn't guaranteed to find the true global
optimum (coordinate descent can miss interactions between axes), but it's a
reasonable tradeoff against runtime, and each run's `campaign_manifest.json`
`tuning_log` records every stage's raw scores so you can see the tradeoffs
the auto-tuner made and second-guess them if something looks off.

## Depths are derived per model, not fixed

`run_bench.py` reads each model's trained max context length straight from
its GGUF metadata (`benchmark/gguf_info.py`, no extra dependencies — just
`struct.unpack` over the GGUF key-value header) and tests every entry in a
list of common context sizes (2048, 4096, 8192, ..., up to 262144) that fits
under that limit, plus depth 0. A model's rotary position embeddings never
saw positions past its trained context length, so testing beyond it
produces a throughput number but not a meaningful one — the earlier fixed
nine-depth list (0 through 65536 for every model) silently did this for any
model with a shorter trained context, which is why this changed.

Check what a model actually supports directly:

```bash
python3 benchmark/gguf_info.py ~/models/your-model.gguf
```

If a model's context length can't be read from its GGUF metadata,
`run_bench.py` falls back to the original fixed nine-depth list and prints a
warning. Each run's `campaign_manifest.json` records `context_length` and
`depths` (per-model now, since each model gets its own manifest — see
"Results layout" above).

## Reading results: which ubatch to use

`benchmark/recommend_settings.py` reads a `curve_summary.csv` (or a results
directory containing one) and recommends a ubatch per model:

```bash
python3 benchmark/recommend_settings.py results/qwen2-5-0-5b-instruct-q4-k-m/rocm7.2.4_llamacpp9974
```

It checks three things, since a naive "best depth-0 throughput" pick (what
`--calibrate` uses internally, matching upstream's convention) can
disagree with what's actually best across the full context range:

- **depth-0 winner** - fastest at a cold/short prompt (what `--calibrate` picks)
- **mean-curve winner** - fastest averaged across all tested depths
- **worst-case winner** - fastest at the deepest tested context

When these disagree, the script recommends the mean-curve winner and says so
explicitly, rather than silently trusting the depth-0-only calibration pass.
Add `--json` for machine-readable output.


