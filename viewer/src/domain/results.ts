// Viewer-domain representation of contracts/viewer-dataset.schema.json.
// Keep this framework-free: React and Recharts are adapters.

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
  flash_attn: string;
  gpu_layers?: number;
  block_count?: number | null;
  n_cpu_layers?: number;
  n_cpu_moe: number;
}

export interface MoeOffloadPoint {
  n_cpu_moe: number;
  status: "ok" | "failed";
  avg_ts: number | null;
}

export interface MoeOffloadDepth {
  depth: number;
  min_ncmoe_that_fits: number | null;
  results: MoeOffloadPoint[];
}

export interface MoeOffloadCurve {
  mode: "quick" | "thorough";
  candidates_tested?: number[];
  expert_count: number;
  expert_used_count: number | null;
  block_count: number;
  by_depth: MoeOffloadDepth[];
}

export interface DenseOffloadPoint {
  n_cpu_layers: number;
  n_gpu_layers: number;
  status: "ok" | "failed";
  avg_ts: number | null;
}

export interface DenseOffloadDepth {
  depth: number;
  max_ngl_that_fits: number | null;
  results: DenseOffloadPoint[];
}

export interface DenseOffloadCurve {
  mode: "quick" | "thorough";
  block_count: number;
  max_gpu_layers: number;
  final_ngl: number | null;
  by_depth: DenseOffloadDepth[];
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
  run_completed_at: string | null;
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
  dense_offload_curve?: DenseOffloadCurve | null;
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
  schema_version: 1;
  generated_at: string;
  models: ModelEntry[];
}
