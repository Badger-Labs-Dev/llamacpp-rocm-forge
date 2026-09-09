import type { ResultsCatalog } from "../../application/ports";
import type {
  CurveRow,
  DenseOffloadCurve,
  FinalConfig,
  ModelEntry,
  MoeOffloadCurve,
  ResultsFile,
  RunEntry,
  TuningStage,
} from "../../domain/results";

export class HttpResultsCatalog implements ResultsCatalog {
  private readonly url: string;

  constructor(url: string) {
    this.url = url;
  }

  async load(): Promise<ResultsFile> {
    const response = await fetch(this.url, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return parseResultsFile(await response.json());
  }
}

class ContractError extends Error {
  constructor(where: string) {
    super(`results.json does not match viewer dataset schema v1: ${where}`);
  }
}

function require_(condition: boolean, where: string): asserts condition {
  if (!condition) throw new ContractError(where);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function record(value: unknown, where: string): Record<string, unknown> {
  require_(isRecord(value), `${where} must be an object`);
  return value;
}

function integer(value: unknown, where: string): void {
  require_(Number.isInteger(value), `${where} must be an integer`);
}

function numberOrNull(value: unknown, where: string): void {
  require_(value === null || (typeof value === "number" && Number.isFinite(value)), `${where} must be a number or null`);
}

function integerOrNull(value: unknown, where: string): void {
  require_(value === null || Number.isInteger(value), `${where} must be an integer or null`);
}

function stringOrNull(value: unknown, where: string): void {
  require_(value === null || typeof value === "string", `${where} must be a string or null`);
}

function assertFinalConfig(value: unknown, where: string): asserts value is FinalConfig {
  const config = record(value, where);
  for (const key of ["ubatch", "batch", "gpu_layers", "n_cpu_layers", "n_cpu_moe"]) {
    integer(config[key], `${where}.${key}`);
  }
  integerOrNull(config.block_count, `${where}.block_count`);
  for (const key of ["ctk", "ctv", "flash_attn"]) {
    require_(typeof config[key] === "string", `${where}.${key} must be a string`);
  }
}

function assertCurveRow(value: unknown, where: string): asserts value is CurveRow {
  const row = record(value, where);
  require_(row.series === "prefill" || row.series === "generation", `${where}.series must be 'prefill' or 'generation'`);
  integer(row.n_depth, `${where}.n_depth`);
  numberOrNull(row.avg_ts, `${where}.avg_ts`);
  require_(row.status === "ok" || row.status === "partial" || row.status === "failed", `${where}.status must be 'ok', 'partial', or 'failed'`);
}

function assertTuningStage(value: unknown, where: string): asserts value is TuningStage {
  const stage = record(value, where);
  require_(typeof stage.stage === "string", `${where}.stage must be a string`);
  const scores = record(stage.scores, `${where}.scores`);
  require_(Object.keys(scores).length > 0, `${where}.scores must be non-empty`);
  require_(Object.values(scores).every((score) => typeof score === "number" && Number.isFinite(score)), `${where}.scores values must be numbers`);
  require_(typeof stage.winner === "string" || (typeof stage.winner === "number" && Number.isFinite(stage.winner)), `${where}.winner must be a string or number`);
}

function assertMoeOffloadCurve(value: unknown, where: string): asserts value is MoeOffloadCurve | null {
  if (value === null) return;
  const curve = record(value, where);
  require_(curve.mode === "quick" || curve.mode === "thorough", `${where}.mode must be 'quick' or 'thorough'`);
  integer(curve.expert_count, `${where}.expert_count`);
  integerOrNull(curve.expert_used_count, `${where}.expert_used_count`);
  integer(curve.block_count, `${where}.block_count`);
  if ("candidates_tested" in curve) {
    require_(Array.isArray(curve.candidates_tested), `${where}.candidates_tested must be an array`);
    curve.candidates_tested.forEach((candidate, index) => integer(candidate, `${where}.candidates_tested[${index}]`));
  }
  require_(Array.isArray(curve.by_depth), `${where}.by_depth must be an array`);
  curve.by_depth.forEach((entry, depthIndex) => {
    const depthWhere = `${where}.by_depth[${depthIndex}]`;
    const depth = record(entry, depthWhere);
    integer(depth.depth, `${depthWhere}.depth`);
    integerOrNull(depth.min_ncmoe_that_fits, `${depthWhere}.min_ncmoe_that_fits`);
    require_(Array.isArray(depth.results), `${depthWhere}.results must be an array`);
    depth.results.forEach((entryPoint, pointIndex) => {
      const pointWhere = `${depthWhere}.results[${pointIndex}]`;
      const point = record(entryPoint, pointWhere);
      integer(point.n_cpu_moe, `${pointWhere}.n_cpu_moe`);
      require_(point.status === "ok" || point.status === "failed", `${pointWhere}.status must be 'ok' or 'failed'`);
      numberOrNull(point.avg_ts, `${pointWhere}.avg_ts`);
    });
  });
}

function assertDenseOffloadCurve(value: unknown, where: string): asserts value is DenseOffloadCurve | null {
  if (value === null) return;
  const curve = record(value, where);
  require_(curve.mode === "quick" || curve.mode === "thorough", `${where}.mode must be 'quick' or 'thorough'`);
  integer(curve.block_count, `${where}.block_count`);
  integer(curve.max_gpu_layers, `${where}.max_gpu_layers`);
  integerOrNull(curve.final_ngl, `${where}.final_ngl`);
  require_(Array.isArray(curve.by_depth), `${where}.by_depth must be an array`);
  curve.by_depth.forEach((entry, depthIndex) => {
    const depthWhere = `${where}.by_depth[${depthIndex}]`;
    const depth = record(entry, depthWhere);
    integer(depth.depth, `${depthWhere}.depth`);
    integerOrNull(depth.max_ngl_that_fits, `${depthWhere}.max_ngl_that_fits`);
    if ("estimate" in depth) require_(depth.estimate === null || isRecord(depth.estimate), `${depthWhere}.estimate must be an object or null`);
    require_(Array.isArray(depth.results), `${depthWhere}.results must be an array`);
    depth.results.forEach((entryPoint, pointIndex) => {
      const pointWhere = `${depthWhere}.results[${pointIndex}]`;
      const point = record(entryPoint, pointWhere);
      integer(point.n_cpu_layers, `${pointWhere}.n_cpu_layers`);
      integer(point.n_gpu_layers, `${pointWhere}.n_gpu_layers`);
      require_(point.status === "ok" || point.status === "failed", `${pointWhere}.status must be 'ok' or 'failed'`);
      numberOrNull(point.avg_ts, `${pointWhere}.avg_ts`);
    });
  });
}

function assertRun(value: unknown, where: string): asserts value is RunEntry {
  const run = record(value, where);
  require_(typeof run.run_id === "string", `${where}.run_id must be a string`);
  stringOrNull(run.run_completed_at, `${where}.run_completed_at`);
  require_(run.status === "finished" || run.status === "partial" || run.status === "failed", `${where}.status must be 'finished', 'partial', or 'failed'`);
  require_(typeof run.mode === "string", `${where}.mode must be a string`);
  record(run.environment, `${where}.environment`);
  assertFinalConfig(run.final_config, `${where}.final_config`);
  require_(Array.isArray(run.depths_tested), `${where}.depths_tested must be an array`);
  run.depths_tested.forEach((depth, index) => integer(depth, `${where}.depths_tested[${index}]`));
  numberOrNull(run.prefill_depth0_ts, `${where}.prefill_depth0_ts`);
  numberOrNull(run.generation_depth0_ts, `${where}.generation_depth0_ts`);
  require_(Array.isArray(run.curve), `${where}.curve must be an array`);
  run.curve.forEach((row, index) => assertCurveRow(row, `${where}.curve[${index}]`));
  require_(Array.isArray(run.tuning_log), `${where}.tuning_log must be an array`);
  run.tuning_log.forEach((stage, index) => assertTuningStage(stage, `${where}.tuning_log[${index}]`));
  assertMoeOffloadCurve(run.moe_offload_curve, `${where}.moe_offload_curve`);
  if ("dense_offload_curve" in run) assertDenseOffloadCurve(run.dense_offload_curve, `${where}.dense_offload_curve`);
}

function assertModel(value: unknown, where: string): asserts value is ModelEntry {
  const model = record(value, where);
  require_(typeof model.model_slug === "string", `${where}.model_slug must be a string`);
  for (const key of ["model_filename", "model_architecture", "model_name"]) {
    if (key in model) stringOrNull(model[key], `${where}.${key}`);
  }
  if ("model_context_length" in model) integerOrNull(model.model_context_length, `${where}.model_context_length`);
  require_(Array.isArray(model.runs), `${where}.runs must be an array`);
  model.runs.forEach((run, index) => assertRun(run, `${where}.runs[${index}]`));
}

export function parseResultsFile(value: unknown): ResultsFile {
  const dataset = record(value, "root value");
  require_(dataset.schema_version === 1, "unsupported schema_version; expected 1");
  require_(typeof dataset.generated_at === "string" && !Number.isNaN(Date.parse(dataset.generated_at)), "generated_at must be a date-time string");
  require_(Array.isArray(dataset.models), "models must be an array");
  dataset.models.forEach((model, index) => assertModel(model, `models[${index}]`));
  return dataset as unknown as ResultsFile;
}
