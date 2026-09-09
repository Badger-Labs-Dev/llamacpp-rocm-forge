# VRAM estimator

`docker/gguf-vram-estimator.py` estimates model weights, KV-cache memory, and a configurable runtime overhead for one GGUF model. It reads the GGUF header directly and supports multi-part GGUFs when every shard is present beside the selected file.

The estimate is useful for planning context sizes before an expensive benchmark. Treat it as a planning value, not a promise: driver allocations, backend behavior, quantization, and other GPU processes can move the real limit.

## Run it

```bash
python3 docker/gguf-vram-estimator.py \
  ~/models/your-model.gguf \
  --contexts 4096 32768 131072 262144
```

The default context list spans 4K through 1M tokens. Use `--contexts` to select the sizes you actually care about.

```bash
python3 docker/gguf-vram-estimator.py \
  ~/models/your-model.gguf \
  --contexts 8192 32768 65536 \
  --overhead 3.0
```

`--overhead` is GiB reserved for compute buffers, drivers, and similar runtime costs. Its default is 2.0 GiB.

## What it calculates

The script reads the architecture, transformer block count, KV head count, key/value dimensions, and trained context length. It conservatively budgets the full requested context for every layer, including sliding-window/hybrid models. It adds:

1. the on-disk GGUF size, including all shards when applicable;
2. the calculated KV cache for each requested context; and
3. the configured overhead.

If a requested context exceeds the GGUF's trained context length, the script omits that request and includes the trained maximum instead.

## Relationship to the benchmark

The benchmark uses the same pure estimate before dense probes, but never treats it as proof. `benchmark/run_bench.py` performs a real full-offload probe and binary-searches `--ngl` after a failure. See [benchmarking strategy and recovery](benchmarking.md).

For MoE models, the estimator remains a broad memory estimate. `--n-cpu-moe` changes where expert weights reside, so use the dedicated [MoE offload sweep](moe-offload.md) to find the real context-versus-offload boundary.
