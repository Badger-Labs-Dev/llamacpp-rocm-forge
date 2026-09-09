# Benchmarking glossary

- **Campaign**: all benchmark work for one model and one version-based run ID.
- **Probe**: one Docker `llama-bench` invocation at one configuration, series, and depth.
- **Curve**: probe results across selected depths for one configuration and series.
- **Configuration**: batch, ubatch, KV-cache dtype, flash-attention mode, and CPU MoE offload count.
- **MoE boundary**: the smallest `n_cpu_moe` value that fits at one context depth.
- **Viewer dataset**: the versioned static JSON contract the Python benchmark application publishes for the TypeScript viewer application.
