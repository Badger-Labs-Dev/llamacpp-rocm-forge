import type { ResultsCatalog } from "../../application/ports";
import type {
  CurveRow,
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
    const data: unknown = await response.json();
    assertResultsFile(data);
    return data;
  }
}

class ContractError extends Error {
  constructor(where: string) {
    super(`results.json does not match viewer dataset schema v1: ${where}`);
  }
}

function require_(condition: boolean, where: string): void {
  if (!condition) throw new ContractError(where);
}

function record(value: unknown, where: string): Record<string, unknown> {
  require_(isRecord(value), `${where} must be an object`);
  return value as Record<string, unknown>;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

/** Deep contract check mirroring bench_app.application.viewer_dataset on
 * the Python side: schema-valid-but-unrenderable payloads (e.g. an empty
 * {} moe_offload_curve, or a tuning_log entry with no scores) must fail
 * here rather than reach chart code as an unchecked `as ResultsFile`. */
function assertResultsFile(value: unknown): asserts value is ResultsFile {
  const dataset = record(value, "root value");
  require_(dataset.schema_version === 1, "schema_version must be 1");
  require_(typeof dataset.generated_at === "string", "generated_at must be a string");
  require_(Array.isArray(dataset.models), "models must be an array");
  (dataset.models as unknown[]).forEach((model, i) => assertModel(model, `models[${i}]`));
}

function assertModel(value: unknown, where: string): asserts value is ModelEntry {
  const model = record(value, where);
  require_(typeof model.model_slug === "string", `${where}.model_slug must be a string`);
  require_(Array.isArray(model.runs), `${where}.runs must be an array`);
  (model.runs as unknown[]).forEach((run, i) => assertRun(run, `${where}.runs[${i}]`));
}

function assertRun(value: unknown, where: string): asserts value is RunEntry {
  const run = record(value, where);
  const required = ["run_id", "status", "mode", "environment", "final_config", "curve", "tuning_log", "moe_offload_curve"];
  for (const key of required) {
    require_(key in run, `${where} is missing '${key}'`);
  }
  require_(typeof run.run_id === "string", `${where}.run_id must be a string`);
  require_(typeof run.status === "string", `${where}.status must be a string`);
  require_(typeof run.mode === "string", `${where}.mode must be a string`);
  record(run.environment, `${where}.environment`);
  record(run.final_config, `${where}.final_config`);
  require_(Array.isArray(run.curve), `${where}.curve must be an array`);
  (run.curve as unknown[]).forEach((row, i) => assertCurveRow(row, `${where}.curve[${i}]`));
  require_(Array.isArray(run.tuning_log), `${where}.tuning_log must be an array`);
  (run.tuning_log as unknown[]).forEach((stage, i) => assertTuningStage(stage, `${where}.tuning_log[${i}]`));
  assertMoeOffloadCurve(run.moe_offload_curve, `${where}.moe_offload_curve`);
}

function assertCurveRow(value: unknown, where: string): asserts value is CurveRow {
  const row = record(value, where);
  require_(typeof row.series === "string", `${where}.series must be a string`);
  require_(typeof row.n_depth === "number", `${where}.n_depth must be a number`);
  require_(row.avg_ts === null || typeof row.avg_ts === "number", `${where}.avg_ts must be a number or null`);
  require_(typeof row.status === "string", `${where}.status must be a string`);
}

function assertTuningStage(value: unknown, where: string): asserts value is TuningStage {
  const stage = record(value, where);
  require_(typeof stage.stage === "string", `${where}.stage must be a string`);
  const scores = record(stage.scores, `${where}.scores`);
  require_(Object.keys(scores).length > 0, `${where}.scores must be non-empty`);
  require_(Object.values(scores).every((v) => typeof v === "number"), `${where}.scores values must be numbers`);
  require_("winner" in stage, `${where} is missing 'winner'`);
}

function assertMoeOffloadCurve(value: unknown, where: string): asserts value is MoeOffloadCurve | null {
  if (value === null) return;
  const curve = record(value, where);
  require_(curve.mode === "quick" || curve.mode === "thorough", `${where}.mode must be 'quick' or 'thorough'`);
  require_(typeof curve.block_count === "number", `${where}.block_count must be a number`);
  require_(Array.isArray(curve.by_depth), `${where}.by_depth must be an array`);
  (curve.by_depth as unknown[]).forEach((point, i) => {
    const depthPoint = record(point, `${where}.by_depth[${i}]`);
    require_(typeof depthPoint.depth === "number", `${where}.by_depth[${i}].depth must be a number`);
    require_(Array.isArray(depthPoint.results), `${where}.by_depth[${i}].results must be an array`);
  });
}
