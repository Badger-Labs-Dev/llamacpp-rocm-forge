# r9700-llm-bench

llama.cpp + ROCm 7.2.4 benchmarking on an AMD Radeon AI PRO R9700 (gfx1201),
running on plain Docker (Ubuntu 24.04 base) instead of Toolbx/Podman.

Forked out of [amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes),
which targets a different GPU (Strix Halo APU, gfx1151) and a different host
setup (Fedora Toolbx + their own "AI Toolbox Cockpit" orchestration tooling
across their own SSH hosts). This repo strips that down to what actually
applies to our hardware and workflow.

## Prerequisites

- Docker, with the `r9700-llm-bench:rocm-7.2.4` image built (see "Building
  the image" below).
- [`uv`](https://docs.astral.sh/uv/) — manages this repo's Python
  environment so the benchmark scripts don't depend on whatever
  `python3`/pip packages happen to be on your `PATH`. Install once:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- Create the project's virtual environment (installs `huggingface_hub`,
  the only real dependency — everything else the scripts use is Python
  stdlib):
  ```bash
  uv sync
  ```
  This creates `.venv/` and `uv.lock` in the repo root. Re-run `uv sync`
  any time `pyproject.toml`'s dependencies change; you don't need to
  activate the venv yourself — every example below uses `uv run`, which
  finds and uses it automatically.

## Contents

- `docker/` - the Dockerfile (Ubuntu 24.04 base, ROCm 7.2.4, gfx1201 target)
  plus its direct build dependencies (grammar patch, VRAM estimator script).
- `benchmark/` - the benchmark driver (`run_bench.py`) and its helpers
  (`hf_models.py`, `gguf_info.py`, `environment_info.py`,
  `recommend_settings.py`, `generate_viewer_data.py`), plus upstream's
  original orchestration scripts kept for reference/adaptation.
  `UPSTREAM_RUNBOOK_REFERENCE.md` is their agent runbook; it assumes their
  SSH hosts, `llama-cockpit` CLI, and Toolbx - none of which we have. Treat
  it as a spec for the benchmark *protocol* (depths, batch sizes,
  repetitions, etc.), not a runnable procedure here.
- `docs/` - reference docs (VRAM estimation, local build notes) carried over
  as-is.
- `viewer/` - static React + TypeScript + Vite app that visualizes
  `results/` (see "Visualizing results" below).

## Status

Working end-to-end. `benchmark/run_bench.py` drives `llama-bench` inside a
plain `docker run` container through a parameter sweep, writing one JSONL
per (model, series, config), a `curve_summary.csv`, and a
`campaign_manifest.json`/`metadata.json` per run directory.

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
llama.cpp build, GPU name/VRAM, host kernel), when the run completed
(`run_completed_at`, UTC ISO 8601), the final chosen config, and a summary
throughput number. `campaign_manifest.json` keeps the full detail (every
stage's tuning scores, every run's status, the same completion timestamp as
`completed_at`) for deeper inspection.

## Pointing --model at your Hugging Face cache

`--model` accepts several kinds of Hugging Face reference directly, in
addition to plain filesystem paths — resolved via
`benchmark/hf_models.py`, which downloads through the standard Hugging
Face cache (`huggingface_hub.hf_hub_download`) if the file isn't already
present, or returns the cached path instantly if it is:

```bash
# Exact file, the same URI shown by a model card's "Download with hf CLI" button
uv run benchmark/run_bench.py \
  --model "hf://unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf" \
  --full-sweep

# Ollama-style tag, matching what the model card's "Use this model -> ollama"
# button shows (hf.co/org/repo:QUANT) - just drop the hf.co/ prefix
uv run benchmark/run_bench.py \
  --model "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL" \
  --full-sweep

# Bare repo, no filename: lists .gguf files and prompts you to pick one
# interactively (errors clearly instead of hanging if stdin isn't a TTY,
# e.g. in a script or cron job - pass an exact file or quant tag there)
uv run benchmark/run_bench.py --model "unsloth/Qwen3.6-35B-A3B-GGUF" --full-sweep
```

`benchmark/hf_models.py` also works standalone, e.g. to just see what's in
a repo without running a benchmark:

```bash
uv run benchmark/hf_models.py list unsloth/Qwen3.6-35B-A3B-GGUF
```

A plain filesystem path (`/home/...`, `./...`, `~/...`) is never
misdetected as an HF reference — only `hf://...` or bare `org/repo[...]`
forms with no leading slash trigger this path.

HF's local cache stores each downloaded file as `snapshots/<hash>/model.gguf`,
itself a symlink to `blobs/<content-hash>` (no `.gguf` extension) —
`run_bench.py` resolves that symlink before mounting the file into the
container, so the Docker mount always targets the real blob's parent
directory, not the symlink's. `model_slug()` still derives from the
*original* filename (e.g. `qwen3-6-35b-a3b-ud-q4-k-xl`), not the
meaningless blob hash, so results stay under a readable directory name
either way. llama.cpp identifies GGUF files by their magic bytes, not their
extension, so loading an extension-less blob directly works fine (verified
against a real 35B model in this cache).

Copying models into a separate directory (e.g. `~/models/`, or `hf
download ... --local-dir ...`) still works and is sometimes worth doing
deliberately — e.g. pointing at a NAS/shared fileserver, or keeping
benchmark inputs decoupled from whatever else on this machine touches the
shared HF cache — but there's no correctness or performance reason to
prefer it now that HF references and cache paths both work directly.

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
uv run benchmark/run_bench.py \
  --model ~/models/your-model.gguf \
  --full-sweep
```

Fixed config (fast, when you already know what you want):

```bash
uv run benchmark/run_bench.py \
  --model ~/models/your-model.gguf \
  --ubatch 1024 --batch 2048 --ctk q8_0 --ctv q8_0 --flash-attn 1
```

Legacy `--calibrate` (ubatch-only sweep, batch/KV/flash-attn held at
defaults) is kept for results directories produced before the full sweep
existed.

Multiple models in one campaign: repeat `--model` — each gets its own
`<results-root>/<slug>/<run-id>/` directory (see "Results layout" above).
Use `--max-depth N` to cap the depth sweep for a quick smoke test without
waiting through a model's full context range. Use `--force` to overwrite an
existing run directory (same model + same ROCm/llama.cpp versions).

`benchmark/UPSTREAM_RUNBOOK_REFERENCE.md`, `run_calibrated_campaign.py`,
`validate_campaign.py`, `generate_results_json.py`, and
`merge_curve_summary.py` are upstream's originals, kept for reference; they
depend on their Toolbx/Cockpit/SSH-host stack and don't run here as-is.
Don't confuse `generate_results_json.py` (upstream's original, unmodified)
with `generate_viewer_data.py` (ours, described below) - similar names,
different purpose and format.

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
uv run benchmark/gguf_info.py ~/models/your-model.gguf
```

If a model's context length can't be read from its GGUF metadata,
`run_bench.py` falls back to the original fixed nine-depth list and prints a
warning. Each run's `campaign_manifest.json` records `context_length` and
`depths` (per-model now, since each model gets its own manifest — see
"Results layout" above).

**Large-context models can legitimately OOM (or hang) at their deepest
tested depth.** A model with a 262144 trained context length will have
that depth included in its sweep — a KV cache that large for a 30B+ model
can exceed the R9700's 32GB VRAM outright. `run_bench.py` runs depths
**ascending, one `docker run` per depth**, specifically to handle this:
smaller depths always get recorded before a large one is even attempted,
and if a depth fails (non-zero exit) or hangs, only the *larger* depths in
that run are skipped — everything smaller is kept.

This ascending/skip behavior exists because direct testing showed
out-of-memory at a large depth doesn't always fail cleanly — it can
**hang** (observed: a 35B model at its full 262K context sat at ~97% VRAM
used and never returned, rather than erroring). A clean subprocess error
wouldn't need this; a hang needs an explicit timeout, so each per-depth
invocation gets `DEPTH_TIMEOUT_SECONDS` (300s default) and the container
is force-killed if it's hit.

A run where every depth ran is `"ok"`; one where a later depth
failed/timed out but earlier ones succeeded is `"partial"` (both in
`campaign_manifest.json`'s per-run `status` and as a `campaign.partial`
marker file, distinct from `campaign.failed`); one where the *first*
depth already failed is `"failed"`. Check `stop_reason` and
`depths_skipped` in `campaign_manifest.json`, or the relevant
`.stderr.log`, to see exactly what happened. If *every* depth in a run
fails (not just deep ones), that's a different problem — check for a
stray container still holding VRAM from an earlier interrupted run (see
"Recovering from an interrupted run" below) before assuming it's a
context-size issue. Use `--max-depth N` to cap the sweep below a model's
full context if you don't need numbers at its absolute limit, or just
accept the partial curve — you still get every depth up to the point it
broke.

## Reading results: which ubatch to use

`benchmark/recommend_settings.py` reads a `curve_summary.csv` (or a results
directory containing one) and recommends a ubatch per model:

```bash
uv run benchmark/recommend_settings.py results/qwen2-5-0-5b-instruct-q4-k-m/rocm7.2.4_llamacpp9974
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

## Visualizing results

`viewer/` is a static React + TypeScript + Vite app (charts via
[Recharts](https://recharts.org)) that reads `viewer/public/results.json`
- generated from `results/` by `benchmark/generate_viewer_data.py` - and
renders three views per model:

1. **Performance across versions** - depth-0 throughput plotted against
   every ROCm/llama.cpp version (`run_id`) the model has been benchmarked
   with, in completion order. This is the primary goal: seeing whether a
   ROCm or llama.cpp upgrade actually helped.
2. **Parameter sensitivity** - a tornado chart, for a selected run: one bar
   per swept parameter (flash-attn, KV cache dtype, ubatch×batch), sized by
   the throughput swing between that parameter's best and worst tested
   candidate. Answers "how much does performance actually depend on this
   setting" at a glance, sorted biggest-impact-first.
3. **Recommended settings** - the winning config for that run, each field
   annotated with its sensitivity ("matters a lot" / "worth checking" /
   "pick whatever's convenient") pulled from the same tornado data, so the
   recommendation says which choices are safe to ignore, not just which one
   won.

The sensitivity data comes from `--full-sweep`'s `tuning_log` (see "What
gets swept" above) - fixed-config and `--calibrate` runs only contribute a
version-over-time data point, no tornado chart, since they don't probe
alternatives. The tornado chart is explicitly labeled as "sensitivity along
the coordinate-descent search path, not a full independent grid" - each
stage's candidates were tested holding the *previous* stage's winner fixed,
so this isn't a guaranteed independent effect, just what showed up along
the path the auto-tuner actually searched.

Regenerate the data after new benchmark runs, then run the viewer locally:

```bash
uv run benchmark/generate_viewer_data.py   # writes viewer/public/results.json
cd viewer
npm install    # first time only
npm run dev
```

Build for GitHub Pages (project-site hosting at
`https://<user>.github.io/r9700-llm-bench/`, set via `base` in
`viewer/vite.config.ts`):

```bash
cd viewer && npm run build   # outputs viewer/dist/
```

`results.json` (the aggregated file) isn't committed (see
`viewer/.gitignore`) - it's generated output, regenerate it any time. The
raw `results/` directory it's built from **is** committed (as of this
writing there's no other way to get real benchmark data - which only this
GPU's host machine can produce - into a CI runner that has no GPU at all).
The GitHub Actions workflow below builds off whatever's committed to
`results/` on `main`; running a new benchmark locally means also
committing its new `results/<slug>/<run-id>/` directory and pushing, or
the live site won't reflect it.

## Recovering from an interrupted run

Each `docker run` this script launches gets a unique
`--name r9700-llm-bench-<random>` container name. Three things clean these
up: Ctrl-C/SIGTERM/an unhandled exception in `run_bench.py` kills any
containers it started before exiting (`_install_cleanup_handlers()`); a
per-depth `docker kill` fires if that depth's invocation exceeds
`DEPTH_TIMEOUT_SECONDS` (a hang, not just an error - see "Depths are
derived per model" above); and `--rm` removes each container once it
exits normally. A `kill -9` on the Python process bypasses the first
mechanism (SIGKILL can't be caught), so if a run ever gets forcefully
killed and you notice VRAM staying pinned afterward, clean up by hand:

```bash
docker ps --filter name=r9700-llm-bench- --format '{{.Names}}'
docker kill $(docker ps --filter name=r9700-llm-bench- -q)
rocm-smi --showmeminfo vram   # confirm VRAM freed
```

