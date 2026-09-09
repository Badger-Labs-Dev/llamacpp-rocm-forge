import { describe, expect, it } from "vitest";
import { completeDataset } from "../../testFixtures";
import { parseResultsFile } from "./httpResultsCatalog";

function payload(): Record<string, unknown> {
  return structuredClone(completeDataset()) as unknown as Record<string, unknown>;
}

function run(data: Record<string, unknown>): Record<string, unknown> {
  const model = (data.models as unknown[])[0] as Record<string, unknown>;
  return (model.runs as Record<string, unknown>[])[0];
}

describe("parseResultsFile", () => {
  it("accepts a complete dataset", () => {
    expect(parseResultsFile(payload())).toEqual(completeDataset());
  });

  it("rejects malformed nested dense results", () => {
    const data = payload();
    const dense = run(data).dense_offload_curve as Record<string, unknown>;
    ((dense.by_depth as unknown[])[0] as Record<string, unknown>).results = [{ n_gpu_layers: "39" }];
    expect(() => parseResultsFile(data)).toThrow(/dense_offload_curve.*n_cpu_layers/);
  });

  it("requires nullable nested fields to be present", () => {
    const data = payload();
    const dense = run(data).dense_offload_curve as Record<string, unknown>;
    delete dense.final_ngl;
    expect(() => parseResultsFile(data)).toThrow(/dense_offload_curve.*final_ngl/);
  });

  it("rejects malformed nested MoE results", () => {
    const data = payload();
    const moe = run(data).moe_offload_curve as Record<string, unknown>;
    ((moe.by_depth as unknown[])[0] as Record<string, unknown>).results = [{ n_cpu_moe: 0, status: "ok", avg_ts: "fast" }];
    expect(() => parseResultsFile(data)).toThrow(/moe_offload_curve.*avg_ts/);
  });

  it("rejects empty final config", () => {
    const data = payload();
    run(data).final_config = {};
    expect(() => parseResultsFile(data)).toThrow(/final_config.*ubatch/);
  });

  it("rejects malformed final config values", () => {
    const data = payload();
    (run(data).final_config as Record<string, unknown>).gpu_layers = null;
    expect(() => parseResultsFile(data)).toThrow(/final_config.*gpu_layers/);
  });

  it("rejects unknown schema versions explicitly", () => {
    const data = payload();
    data.schema_version = 2;
    expect(() => parseResultsFile(data)).toThrow(/unsupported schema_version; expected 1/);
  });

  it("rejects invalid run and point statuses", () => {
    const data = payload();
    run(data).status = "complete";
    expect(() => parseResultsFile(data)).toThrow(/status must be 'finished', 'partial', or 'failed'/);

    const other = payload();
    ((run(other).curve as unknown[])[0] as Record<string, unknown>).status = "unknown";
    expect(() => parseResultsFile(other)).toThrow(/curve\[0\].status/);
  });

  it("rejects arrays where records are required", () => {
    const data = payload();
    run(data).environment = [];
    expect(() => parseResultsFile(data)).toThrow(/environment must be an object/);
  });
});