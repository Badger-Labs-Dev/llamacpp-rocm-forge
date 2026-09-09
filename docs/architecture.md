# Architecture

This repository contains two applications that share a versioned data contract rather than source-code imports.

```text
benchmark application (Python) → viewer dataset v1 → viewer application (TypeScript)
```

## Benchmark application

The Python application lives directly under `benchmark/` (a single `benchmark/bench_app/` package existed briefly during the migration and has been collapsed - `benchmark/` only ever contains this one application, so the extra nesting level was redundant).

- `domain/` contains deterministic benchmark rules and the core configuration model: `BenchConfig`/`RunResult` (`models.py`), probe budgets and offload candidate selection (`planning.py`), progress accounting (`progress.py`), conservative VRAM math (`vram_estimate.py`), and the shared monotonic CPU-offload boundary search (`offload_bisection.py`).
- `application/` holds the use cases: `run_curve.py` (one depth curve through Docker), `auto_tune.py` (staged KV-cache/ubatch/batch search), `moe_sweep.py` (MoE sweeps), `dense_sweep.py` (estimate-first `--ngl` discovery and throughput samples), `run_campaign.py` (top-level orchestration and artifact writing), and `viewer_dataset.py` (the published viewer contract). These functions do not take speculative `Protocol` ports because `run_bench.py` remains their only caller.
- `adapters/outbound/` owns Docker command rendering (`docker_llama_bench.py`), Docker container execution with VRAM-leak protection (`docker_runner.py`), terminal output (`terminal_progress.py`), model resolution - HF references and GGUF-derived depths (`model_resolution.py`), GGUF header parsing (`gguf_metadata.py`), Hugging Face Hub resolution/download (`huggingface_models.py`), ROCm/environment inspection (`rocm_environment.py`), and campaign persistence - CSV/manifest/metadata writers (`campaign_store.py`).
- `run_bench.py` owns CLI parsing, per-model directory/depth setup, use-case wiring, and exit-code logic. The CLI has two modes: default performs full tuning and offload discovery; `--quick` uses the fixed fast path. `generate_viewer_data.py` is the other inbound CLI adapter.

A **probe** is one Docker `llama-bench` invocation. A **campaign** is all probes for one model/version run. A **curve** is the set of measurements across depths for one configuration and series.

## Viewer application

The viewer remains a static Vite application under `viewer/`.

- `src/domain/` contains framework-free contract types and calculation rules.
- `src/application/` selects models/runs and declares the `ResultsCatalog` port.
- `src/adapters/outbound/` supplies the HTTP implementation that reads `results.json`.
- `App.tsx` and the Recharts components are inbound presentation adapters.

React and Recharts do not own benchmark interpretation. They render application and domain values.

## Shared contract

`contracts/viewer-dataset.schema.json` describes the static data boundary. Python publishes schema version `1`; the viewer rejects unknown versions before attempting to render.

The contract is deliberately file-based. Benchmarking requires the R9700 host, while GitHub Pages only needs the committed result files and a static frontend build. A server API would add operational weight without improving that workflow.

## Migration rule

Refactors preserve the existing CLI paths and on-disk results layout. New domain or application behavior is extracted behind compatibility modules first; an adapter is removed only after the same commands, manifests, viewer data, and tests continue to work.
