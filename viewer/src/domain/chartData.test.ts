import { describe, expect, it } from "vitest";
import { depthCurveData, moeChartData, versionChartData } from "./chartData";
import { completeRun } from "../testFixtures";

describe("depthCurveData", () => {
  it("keeps measured, failed, and missing series distinct", () => {
    const rows = depthCurveData([
      { series: "prefill", n_depth: 0, avg_ts: 10, status: "ok" },
      { series: "generation", n_depth: 0, avg_ts: null, status: "failed" },
      { series: "prefill", n_depth: 4096, avg_ts: null, status: "partial" },
    ], [0, 4096, 8192]);
    expect(rows).toEqual([
      { depth: 0, prefill: { state: "measured", value: 10 }, generation: { state: "failed", value: null } },
      { depth: 4096, prefill: { state: "failed", value: null }, generation: { state: "missing", value: null } },
      { depth: 8192, prefill: { state: "missing", value: null }, generation: { state: "missing", value: null } },
    ]);
  });
});

describe("versionChartData", () => {
  it("preserves gaps and a single run's measurements", () => {
    const rows = versionChartData([
      completeRun({ run_id: "good" }),
      completeRun({ run_id: "failed", status: "failed", prefill_depth0_ts: 2, generation_depth0_ts: 1 }),
      completeRun({ run_id: "missing", prefill_depth0_ts: null, generation_depth0_ts: null }),
    ]);
    expect(rows[0].prefill).toBe(1000);
    expect(rows[1]).toMatchObject({ prefill: null, measuredPrefill: 2, state: "failed" });
    expect(rows[2]).toMatchObject({ prefill: null, state: "missing" });
  });
});

describe("moeChartData", () => {
  it("distinguishes tested success, tested failure, and untested candidates", () => {
    const curve = completeRun().moe_offload_curve!;
    curve.by_depth.push({ depth: 4096, min_ncmoe_that_fits: 1, results: [{ n_cpu_moe: 1, status: "ok", avg_ts: 80 }] });
    const { candidates, rows } = moeChartData(curve);
    expect(candidates).toEqual([0, 1]);
    expect(rows[1].points[0]).toEqual({ state: "untested", value: null });
    expect(rows[0].points[1]).toEqual({ state: "failed", value: null });
  });
});
