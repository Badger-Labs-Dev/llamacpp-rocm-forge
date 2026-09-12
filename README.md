# llamacpp-rocm-forge

Build llama.cpp Docker images against any ROCm + llama.cpp version combination, and benchmark them. Currently targets the AMD Radeon AI PRO R9700 (`gfx1201`) and the Ryzen 5 9600X iGPU (`gfx1036`) on Ubuntu 26.04; started as a pure benchmarking tool, but the build side ([docs/building.md](docs/building.md)) turned out valuable on its own. It's now also how correctly-tagged `bench`/`server`/`light` images get produced for other uses (e.g. an always-on `llama-server` deployment elsewhere, see [homelab-llm-router](https://github.com/rayjanwilson/homelab-llm-router)).

The build adapts [AMD's official ROCm 10.0.0 install docs](https://rocm.docs.amd.com/en/latest/install/rocm.html?fam=radeon&w=compute&gpu=amd-radeon-ai-pro-r9700&gfx=gfx1201&os=ubuntu&ubuntu-ver=26.04&i=pkgman) (apt packages) and [llama.cpp's own official ROCm Dockerfile](https://github.com/ggml-org/llama.cpp/blob/master/.devops/rocm.Dockerfile) (build structure); see [docs/building.md](docs/building.md#where-this-build-comes-from) for specifics.

## Quick start

### 1. Install prerequisites

- Docker, with the local `llamacpp-rocm-forge:rocm_10.0.0-llama_v0.4.0-gfx1201-bench` image built. See [building the image](docs/building.md).
- [`uv`](https://docs.astral.sh/uv/), which keeps the Python dependency for Hugging Face model resolution in this repo instead of relying on whichever `python3` is on `PATH`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # once, if needed
uv sync                                            # creates .venv from pyproject.toml
```

Every command below uses `uv run`; activating `.venv` manually is unnecessary.

### 2. Run a model

Model type (dense vs MoE) is auto-detected. By default, the driver tunes only KV-cache dtype at the fixed `batch=2048`, `ubatch=2048`, `flash_attn=auto` configuration, maps the per-depth offload boundary (`--ngl` for dense models or `--n-cpu-moe` for MoE), then benchmarks one fixed winning configuration across the model's supported context depths:

```bash
uv run benchmark/run_bench.py \
  --model "hf://unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
```

`--model` accepts a normal GGUF path, the `hf://...` URI from Hugging Face's download button, an Ollama-style `org/repo:QUANT` reference, or a bare `org/repo` that opens an interactive GGUF picker. Details: [Hugging Face model references](docs/huggingface-models.md).

Add `--quick` for a fast pass: fixed config (no auto-tuning), an exact dense `--ngl` boundary with fewer throughput samples, and for MoE, five evenly-spaced `--n-cpu-moe` candidates instead of the exact boundary search:

```bash
uv run benchmark/run_bench.py \
  --model ~/models/your-model.gguf \
  --quick
```

Use `--max-depth N` to cap a long-context model for a smoke test. Repeat `--model` to benchmark several models in one invocation.

## MoE models

A name such as `35B-A3B` means about 3B parameters are active per token; it does not mean the model occupies 3B parameters' worth of VRAM. For a detected mixture-of-experts GGUF, the driver automatically also sweeps `--n-cpu-moe` - the exact boundary bisection by default, or the quick 5-point curve with `--quick`.

See [MoE expert offload](docs/moe-offload.md) for the memory model, quick-versus-thorough trade-off, and how to read the results.

## Dense models

For each context depth, the driver first tries full GPU-layer offload. If that real probe fails, it binary-searches `--ngl` for the maximum fitting value. When any depth needs CPU offload, it also samples lower `--ngl` values to measure the nonlinear throughput cost. The ordinary prefill/generation depth curve then holds one `--ngl` fixed: the value safe at the deepest requested context.

## Results and viewer

Each model keeps results together, with a version-based run ID beneath it:

```text
results/
  <model-slug>/
    rocm<version>_llamacpp<build-number>/
      metadata.json
      campaign_manifest.json
      curve_summary.csv
      *.jsonl / *.stderr.log
      tuning/                 # fixed-config KV-cache dtype probes, when present
      moe-tuning/             # MoE probes, when present
```

Generate the viewer data and start the local React app:

```bash
uv run benchmark/generate_viewer_data.py
cd viewer
npm install                   # first run only
npm run dev
```

The raw `results/` directory is committed because CI has no GPU to reproduce it. `viewer/public/results.json` is generated and ignored. The [results and viewer guide](docs/results-and-viewer.md) covers the schema, charts, GitHub Pages build, and how partial runs are represented.

## Documentation

- [Building the Docker image](docs/building.md)
- [Hugging Face model references and cache behavior](docs/huggingface-models.md)
- [Benchmarking strategy, context depths, and recovery](docs/benchmarking.md)
- [MoE expert offload](docs/moe-offload.md)
- [Results and viewer](docs/results-and-viewer.md)
- [Application architecture](docs/architecture.md)
- [VRAM estimator](docs/vram-estimator.md)

## Scope

This repo was seeded from [amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes), which targets a Strix Halo APU, Fedora Toolbx, and a multi-host orchestration workflow. Its retained files under `benchmark/` are protocol references only; they are not part of this runnable R9700 path. The viewer generator for this repo is `benchmark/generate_viewer_data.py`.

The supported local hardware profiles are the R9700 (`gfx1201`, HIP UMA off) and Ryzen 5 9600X iGPU (`gfx1036`, HIP UMA on). Bake exposes them as named targets such as `bench-gfx1036`; the retained Makefile uses `GFX_TARGETS` and enables UMA only for its `HIP_UMA_TARGETS`. The Dockerfile also retains an explicit `GFX_TARGET=all` fat-image path. See [building the image](docs/building.md#building-for-the-local-gpus).
