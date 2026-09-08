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
  flash_attn: number;
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
