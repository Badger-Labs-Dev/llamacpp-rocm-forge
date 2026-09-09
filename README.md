# r9700-llm-bench

Benchmark GGUF models with llama.cpp and ROCm 7.2.4 on an AMD Radeon AI PRO R9700 (`gfx1201`). The benchmark runs in plain Docker on Ubuntu 24.04. It is deliberately narrower than the Strix Halo project it began from: one discrete R9700, one local Docker workflow, and results that can drive a static recommendation site.

## Quick start

### 1. Install prerequisites

- Docker, with the local `r9700-llm-bench:rocm-7.2.4` image built. See [building the image](docs/building.md).
- [`uv`](https://docs.astral.sh/uv/), which keeps the Python dependency for Hugging Face model resolution in this repo instead of relying on whichever `python3` is on `PATH`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # once, if needed
uv sync                                            # creates .venv from pyproject.toml
```

Every command below uses `uv run`; activating `.venv` manually is unnecessary.

### 2. Run a model

The recommended path is a full sweep. It tunes KV-cache dtype and the ubatch/batch pair, then benchmarks the selected configuration across the model's supported context depths.

```bash
uv run benchmark/run_bench.py \
  --model "hf://unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf" \
  --full-sweep
```

`--model` accepts a normal GGUF path, the `hf://...` URI from Hugging Face's download button, an Ollama-style `org/repo:QUANT` reference, or a bare `org/repo` that opens an interactive GGUF picker. Details: [Hugging Face model references](docs/huggingface-models.md).

For a fixed configuration, skip tuning:

```bash
uv run benchmark/run_bench.py \
  --model ~/models/your-model.gguf \
  --ubatch 1024 --batch 2048 --ctk q8_0 --ctv q8_0
```

Use `--max-depth N` to cap a long-context model for a smoke test. Repeat `--model` to benchmark several models in one invocation.

## MoE models

A name such as `35B-A3B` means about 3B parameters are active per token; it does not mean the model occupies 3B parameters' worth of VRAM. For a detected mixture-of-experts GGUF, `--full-sweep` automatically runs a quick `--n-cpu-moe` curve alongside the ordinary sweep.

Use the thorough mode when the exact offload boundary matters:

```bash
uv run benchmark/run_bench.py \
  --model "hf://org/repo/model.gguf" \
  --full-sweep --sweep-moe-offload-thorough
```

See [MoE expert offload](docs/moe-offload.md) for the memory model, quick-versus-thorough trade-off, and how to read the results.

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
      tuning/                 # full-sweep probes, when present
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
