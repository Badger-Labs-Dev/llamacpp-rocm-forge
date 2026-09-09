# Architecture

This repository contains two applications that share a versioned data contract rather than source-code imports.

```text
benchmark application (Python) → viewer dataset v1 → viewer application (TypeScript)
```

## Benchmark application

The Python application lives under `benchmark/bench_app/`.

- `domain/` contains deterministic benchmark rules: probe budgets and MoE candidate selection (`planning.py`), progress accounting (`progress.py`), and the MoE offload boundary bisection *search* (`moe_bisection.py` - a pure generator that yields the next `--n-cpu-moe` candidate to probe and receives a fit/fail result; `run_bench.py`'s `sweep_moe_offload_thorough` drives it and performs the actual probe I/O).
- `application/` defines the published viewer-dataset contract (`viewer_dataset.py`) and the campaign use case (`run_campaign.py` - runs one model's tuning/fixed-config, optional MoE sweep, and final curves, then writes all campaign artifacts). `run_campaign.run_model_campaign()` takes its collaborators (BenchConfig, run_one, auto_tune, etc.) as parameters rather than importing them, since `run_bench.py` is still its only caller; no `Protocol`/port exists for them yet - add one only when a second caller actually needs to swap an implementation (see the migration rule below).
- `adapters/outbound/` owns Docker command rendering (`docker_llama_bench.py`), Docker container execution with VRAM-leak protection (`docker_runner.py`), terminal output (`terminal_progress.py`), model resolution - HF references and GGUF-derived depths (`model_resolution.py`), ROCm/environment inspection (`rocm_environment.py`), and campaign persistence - CSV/manifest/metadata writers (`campaign_store.py`).
- `run_bench.py`, `generate_viewer_data.py`, and the other executable scripts are inbound CLI adapters. Existing commands remain stable during the migration. `benchmark/environment_info.py` and `benchmark/progress_tracker.py` are thin compatibility shims re-exporting the adapters above, so external imports and the standalone `environment_info.py` CLI keep working unchanged.

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
