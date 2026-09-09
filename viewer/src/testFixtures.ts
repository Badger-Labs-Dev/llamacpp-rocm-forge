import type { ResultsFile, RunEntry } from "./domain/results";

export function completeRun(overrides: Partial<RunEntry> = {}): RunEntry {
  return {
    run_id: "rocm-7.2-llama-1",
    run_completed_at: "2026-09-09T00:00:00+00:00",
    status: "finished",
    mode: "full-sweep",
    environment: {},
    final_config: {
      ubatch: 1024,
      batch: 2048,
      ctk: "q8_0",
      ctv: "q8_0",
      flash_attn: "on",
      gpu_layers: 41,
      block_count: 40,
      n_cpu_layers: 0,
      n_cpu_moe: 0,
    },
    depths_tested: [0, 4096],
    prefill_depth0_ts: 1000,
    generation_depth0_ts: 100,
    curve: [
      { series: "prefill", n_depth: 0, avg_ts: 1000, status: "ok" },
      { series: "generation", n_depth: 0, avg_ts: 100, status: "ok" },
      { series: "prefill", n_depth: 4096, avg_ts: 900, status: "ok" },
      { series: "generation", n_depth: 4096, avg_ts: 90, status: "ok" },
    ],
    tuning_log: [{ stage: "kv_cache_dtype", scores: { f16: 90, q8_0: 100 }, winner: "q8_0" }],
    moe_offload_curve: {
      mode: "thorough",
      expert_count: 64,
      expert_used_count: 8,
      block_count: 40,
      by_depth: [{
        depth: 0,
        min_ncmoe_that_fits: 0,
        results: [
          { n_cpu_moe: 0, status: "ok", avg_ts: 1000 },
          { n_cpu_moe: 1, status: "failed", avg_ts: null },
        ],
      }],
    },
    dense_offload_curve: {
      mode: "thorough",
      block_count: 40,
      max_gpu_layers: 41,
      final_ngl: 39,
      by_depth: [{
        depth: 4096,
        max_ngl_that_fits: 39,
        results: [
          { n_cpu_layers: 2, n_gpu_layers: 39, status: "ok", avg_ts: 900 },
          { n_cpu_layers: 1, n_gpu_layers: 40, status: "failed", avg_ts: null },
        ],
      }],
    },
    ...overrides,
  };
}

export function completeDataset(): ResultsFile {
  return {
    schema_version: 1,
    generated_at: "2026-09-09T00:00:00+00:00",
    models: [{
      model_slug: "example",
      model_filename: "example.gguf",
      model_architecture: "example",
      model_name: "Example",
      model_context_length: 131072,
      runs: [completeRun()],
    }],
  };
}