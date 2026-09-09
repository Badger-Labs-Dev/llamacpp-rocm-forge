# Benchmarking glossary

- **Campaign**: all benchmark work for one model and one version-based run ID.
- **Probe**: one Docker `llama-bench` invocation at one configuration, series, and depth.
- **Curve**: probe results across selected depths for one configuration and series.
- **Configuration**: batch, ubatch, KV-cache dtype, flash-attention mode, and CPU MoE offload count.
- **MoE boundary**: the smallest `n_cpu_moe` value that fits at one context depth.
- **Viewer dataset**: the versioned static JSON contract the Python benchmark application publishes for the TypeScript viewer application.
- **llama.cpp identity**: what a benchmark image's llama.cpp build is called in run IDs — a release tag (`v0.4.0`) if the image was built with `LLAMA_CPP_REF` set to a tag, otherwise a short commit SHA (`master` and other branch/commit builds). Read from files baked into the image (`/app/.llama-cpp-identity`), not parsed from `--version` output. See `docs/building.md`.
