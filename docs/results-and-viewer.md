# Results and viewer

Every benchmark writes under `results/<model-slug>/<run-id>/`. The model slug comes from the GGUF filename; the run ID is `rocm<version>_llamacpp<build-number>`. Version is the directory identity because the project compares the same model over ROCm and llama.cpp changes. Completion time is stored as UTC ISO 8601 data inside the result files.

## Files in one run

| File or directory | Purpose |
| --- | --- |
| `metadata.json` | Compact model, environment, final configuration, and completion summary for aggregation. |
| `campaign_manifest.json` | Full configuration, tuning log, per-series status, stopped/skipped depths, and dense/MoE offload curves where applicable. |
| `curve_summary.csv` | Prefill and generation throughput for the ordinary final depth curve. |
| `*.jsonl` / `*.stderr.log` | Raw llama-bench output and diagnostics. |
| `tuning/` | Full-sweep KV and ubatch/batch probes. |
| `moe-tuning/` | MoE offload probes. |
| `dense-preflight/` | Default-mode deepest-depth probes used to select a safe `--ngl` for auto-tuning. |
| `dense-tuning/` | Per-depth dense `--ngl` boundary and throughput probes. |

The driver refuses to overwrite a directory with the same model and version identity. Pass `--force` only when replacing that run is intentional.

## Generate viewer data

The static React viewer does not read every run directory in the browser. Generate one compact data file after adding benchmark results:

```bash
uv run benchmark/generate_viewer_data.py
```

The generator reads `results/` and writes `viewer/public/results.json`. The file is ignored by Git because CI can regenerate it from the committed raw results.

## Run or build the viewer

```bash
cd viewer
npm install             # first time only
npm run dev
```

For GitHub Pages:

```bash
cd viewer
npm run build
```

`vite.config.ts` sets the project-site base path to `/r9700-llm-bench/`. The GitHub Actions deployment workflow runs `benchmark/generate_viewer_data.py` before the Vite build.

## What the viewer shows

1. **Performance across versions** compares depth-0 throughput across completed ROCm/llama.cpp runs for one model.
2. **Parameter sensitivity** turns a full sweep's staged `tuning_log` into a tornado chart. It reports the variation observed along the coordinate-descent path, not a fully independent parameter grid.
3. **Recommended settings** shows the ordinary final configuration with the sensitivity notes that explain which choices moved performance enough to care about.
4. **MoE expert offload** appears only when `moe_offload_curve` exists. It plots throughput by context depth for each sampled `-ncmoe` value and lists the smallest recorded value that fits at each depth.
5. **Dense layer offload** appears only when `dense_offload_curve` exists. It plots prefill throughput against `--ngl` for each context depth and lists the exact maximum fitting value.

`--quick` runs still supply version-over-time data, but do not have enough alternatives to produce a sensitivity chart.

## Publishing benchmark data

The raw `results/` directory is committed because only the R9700 host can produce the measurements and GitHub Actions has no GPU. To update the public site, commit a completed `results/<model-slug>/<run-id>/` directory with the source changes. The Pages workflow rebuilds the aggregate JSON and frontend from that input.
