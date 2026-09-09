# Architecture

This repository contains two applications that share a versioned data contract rather than source-code imports.

```text
benchmark application (Python) → viewer dataset v1 → viewer application (TypeScript)
```

## Benchmark application

The Python application lives directly under `benchmark/` (a single `benchmark/bench_app/` package existed briefly during the migration and has been collapsed - `benchmark/` only ever contains this one application, so the extra nesting level was redundant).

- `domain/` contains deterministic benchmark rules and the core configuration model: `BenchConfig`/`RunResult` (`models.py`), probe budgets and MoE candidate selection (`planning.py`), progress accounting (`progress.py`), and the MoE offload boundary bisection *search* (`moe_bisection.py` - a pure generator that yields the next `--n-cpu-moe` candidate to probe and receives a fit/fail result; `application/moe_sweep.py`'s `sweep_moe_offload_thorough` drives it and performs the actual probe I/O).
- `application/` holds the use cases: `run_curve.py` (run one full depth-curve for a (model, series, config) through Docker - the most behavior-critical code in the application layer, since a bug here leaks VRAM on real hardware), `auto_tune.py` (the staged KV-cache/ubatch/batch search, default mode), `moe_sweep.py` (quick and thorough `--n-cpu-moe` sweeps), `run_campaign.py` (the top-level per-model campaign: auto-tune-or-`--quick`-fixed-config, MoE sweep for a detected MoE model, final curves, writing all campaign artifacts), and the published viewer-dataset contract (`viewer_dataset.py`). None of these take a `Protocol`/port for their collaborators - `run_bench.py` is still the only caller of any of them, so there's no second implementation yet to justify that abstraction (see the migration rule below).
- `adapters/outbound/` owns Docker command rendering (`docker_llama_bench.py`), Docker container execution with VRAM-leak protection (`docker_runner.py`), terminal output (`terminal_progress.py`), model resolution - HF references and GGUF-derived depths (`model_resolution.py`), GGUF header parsing (`gguf_metadata.py`), Hugging Face Hub resolution/download (`huggingface_models.py`), ROCm/environment inspection (`rocm_environment.py`), and campaign persistence - CSV/manifest/metadata writers (`campaign_store.py`).
- `run_bench.py` is now CLI parsing (`argparse`), per-model directory/depth setup, wiring the use cases above together, and exit-code logic - the docstring, constants, and `main()` are the only things left that are genuinely CLI-specific. The CLI itself is deliberately two modes with no tuning flags: model type (dense/MoE) is auto-detected, the default runs the full auto-tune plus (for MoE) the thorough bisection, and `--quick` runs a fixed config plus (for MoE) the quick 5-point sweep - there is no third mode and no per-parameter override flags (`--ubatch`, `--ctk`, etc. were removed; see git history for the earlier legacy `--calibrate` mode, also removed). `generate_viewer_data.py` is the other inbound CLI adapter at the top level of `benchmark/`, alongside `tests/`.

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
