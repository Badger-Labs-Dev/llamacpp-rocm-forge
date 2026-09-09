# Architecture

This repository contains two applications that share a versioned data contract rather than source-code imports.

```text
benchmark application (Python) → viewer dataset v1 → viewer application (TypeScript)
```

## Benchmark application

The Python application lives under `benchmark/bench_app/`.

- `domain/` contains deterministic benchmark rules: probe budgets and MoE candidate selection (`planning.py`) and progress accounting (`progress.py`). The MoE offload bisection *search loop* (deciding which `n_cpu_moe` value to probe next and walking the results) still lives in `run_bench.py`'s `sweep_moe_offload_thorough`; only the per-depth probe-count budget it consumes (`thorough_max_probes_per_depth`) has been extracted so far.
- `application/` defines the published viewer-dataset contract (`viewer_dataset.py`). No application ports exist yet - add one only when a use case actually needs to depend on an interface rather than a concrete adapter (see the migration rule below).
- `adapters/outbound/` owns Docker command rendering (`docker_llama_bench.py`) and terminal output (`terminal_progress.py`). Hugging Face, filesystem, and ROCm integrations remain behind the existing CLI compatibility modules while they are migrated incrementally.
- `run_bench.py`, `generate_viewer_data.py`, and the other executable scripts are inbound CLI adapters. Existing commands remain stable during the migration.

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
