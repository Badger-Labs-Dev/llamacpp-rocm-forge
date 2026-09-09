export interface CurveRow {
  series: "prefill" | "generation";
  n_depth: number;
  avg_ts: number | null;
  status: string;
}

export interface TuningStage {
  stage: "flash_attn" | "kv_cache_dtype" | "ubatch_batch_grid" | string;
  scores: Record<string, number>;
  winner: string | number;
}

export interface FinalConfig {
  ubatch: number;
  batch: number;
  ctk: string;
  ctv: string;
  flash_attn: string; // "auto" | "on" | "off" - always "auto" as of the
    // change that stopped sweeping this (llama.cpp's own default, lets
    // it decide per model/backend whether the fused kernel applies)
  n_cpu_moe: number; // llama-bench's -ncmoe. A dense or fixed run normally
    // stays at 0. A full MoE sweep tunes its shared settings with all experts
    // offloaded so deep probes remain viable; MoeOffloadCurve supplies the
    // context-specific minimum rather than a single global recommendation.
}

export interface MoeOffloadPoint {
  n_cpu_moe: number;
  status: "ok" | "failed";
  avg_ts: number | null;
}

export interface MoeOffloadDepth {
  depth: number;
  min_ncmoe_that_fits: number | null; // null = nothing fit, even fully offloaded
  results: MoeOffloadPoint[];
}

export interface MoeOffloadCurve {
  mode: "quick" | "thorough";
  candidates_tested?: number[]; // quick mode only
  expert_count: number;
  expert_used_count: number | null;
  block_count: number;
  by_depth: MoeOffloadDepth[];
}

export interface Environment {
  rocm_version?: string;
  build_number?: number;
  build_commit?: string;
  gpu_name?: string;
  gpu_vram_gib?: number;
  host_kernel?: string;
  image?: string;
}

export interface RunEntry {
  run_id: string;
  run_completed_at: string;
  status: string;
  mode: string;
  environment: Environment;
  final_config: FinalConfig;
  depths_tested: number[];
  prefill_depth0_ts: number | null;
  generation_depth0_ts: number | null;
  curve: CurveRow[];
  tuning_log: TuningStage[];
  moe_offload_curve: MoeOffloadCurve | null;
}

export interface ModelEntry {
  model_slug: string;
  model_filename: string | null;
  model_architecture: string | null;
  model_name: string | null;
  model_context_length: number | null;
  runs: RunEntry[];
}

export interface ResultsFile {
  generated_at: string;
  models: ModelEntry[];
}
