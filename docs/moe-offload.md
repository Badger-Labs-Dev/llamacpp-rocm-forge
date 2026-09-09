# MoE expert offload

Mixture-of-experts models need a separate benchmark dimension. A name such as `35B-A3B` says that about 3B parameters are active for one token. It describes compute, not the memory required to make every possible expert immediately available.

The router can select any expert on the next token. With a normal GPU-offload run (`-ngl 99`), the model's expert weights sit in VRAM with the rest of the GPU-resident layers. That is why a 35B-A3B model can use model-weight VRAM closer to a 35B dense model than a 3B dense model.

llama.cpp's `--n-cpu-moe N` / `-ncmoe N` moves MoE feed-forward weights from the first *N* transformer layers to CPU RAM while keeping attention and shared weights on the GPU. It creates a direct trade-off:

- More CPU offload frees VRAM and permits more KV cache for context.
- More CPU offload adds CPU/PCIe work and reduces throughput.

## Detection

`run_bench.py` reads three namespaced GGUF metadata keys:

- `<architecture>.expert_count`
- `<architecture>.expert_used_count`
- `<architecture>.block_count`

`expert_count > 1` identifies an MoE. `block_count` is the upper bound for `-ncmoe`: a 40-layer model accepts values from `0` to `40`. Dense models lack `expert_count`, so the MoE stage is skipped.

## Which command to use

Model type is auto-detected - there is nothing to configure. By default, a detected MoE model automatically gets the thorough exact-boundary search:

```bash
uv run benchmark/run_bench.py \
  --model "hf://org/repo/model.gguf"
```

`--quick` runs the quick fixed-candidates curve instead, alongside the fixed (untuned) base config:

```bash
uv run benchmark/run_bench.py \
  --model "hf://org/repo/model.gguf" \
  --quick
```

## Quick mode

Quick mode tests five evenly spaced `-ncmoe` values from `0` through the model's `block_count` at each selected context depth. A 40-layer model produces `0`, `10`, `20`, `30`, and `40`.

It is the right default when the goal is to see the shape of the throughput trade-off quickly. Its reported minimum is the first **tested** fitting value. If `20` fails and `30` fits, the actual boundary lies somewhere in that interval.

## Thorough mode: bisection

Thorough mode finds the exact first fitting `-ncmoe` value with binary search, then adds several samples above it to show the throughput decline.

At one fixed context depth, fitting is treated as monotonic: once `-ncmoe N` fits, a larger value can only free more GPU memory. Start with a failed lower bound and a fitting upper bound, test the midpoint, and keep the half containing the transition. A 40-layer model needs at most about six boundary probes instead of one probe per possible layer count.

Context produces a second monotonic relationship. As depth grows, the KV cache takes more VRAM, so the fitting boundary can stay unchanged or move toward more CPU offload; it cannot move lower. The next depth begins from the previous depth's boundary rather than starting from zero every time.

The output is a `moe_offload_curve` in `campaign_manifest.json`. It contains a row per depth, its `min_ncmoe_that_fits`, and each measured `(n_cpu_moe, avg_ts)` point. The viewer turns that into a table and chart. There is deliberately no one global `-ncmoe` recommendation: choose the row for the context you need.
