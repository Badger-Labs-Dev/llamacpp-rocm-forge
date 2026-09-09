import type { CurveRow, DenseOffloadCurve, MoeOffloadCurve, RunEntry } from "./results";

export type MeasurementState = "measured" | "failed" | "missing" | "untested";

export interface Measurement {
  state: MeasurementState;
  value: number | null;
}

export interface DepthCurveDatum {
  depth: number;
  prefill: Measurement;
  generation: Measurement;
}

const missing = (): Measurement => ({ state: "missing", value: null });

export function depthCurveData(curve: CurveRow[], depths: number[] = []): DepthCurveDatum[] {
  const rows = new Map<number, DepthCurveDatum>();
  for (const depth of depths) {
    rows.set(depth, { depth, prefill: missing(), generation: missing() });
  }
  for (const point of curve) {
    const row = rows.get(point.n_depth) ?? {
      depth: point.n_depth,
      prefill: missing(),
      generation: missing(),
    };
    row[point.series] = point.status === "ok"
      ? (point.avg_ts === null ? missing() : { state: "measured", value: point.avg_ts })
      : { state: "failed", value: null };
    rows.set(point.n_depth, row);
  }
  return [...rows.values()].sort((left, right) => left.depth - right.depth);
}

export interface VersionDatum {
  runId: string;
  status: RunEntry["status"];
  state: "measured" | "failed" | "missing";
  prefill: number | null;
  generation: number | null;
  measuredPrefill: number | null;
  measuredGeneration: number | null;
}

export function versionChartData(runs: RunEntry[]): VersionDatum[] {
  return runs.map((run) => {
    const hasMeasurement = run.prefill_depth0_ts !== null || run.generation_depth0_ts !== null;
    const state = run.status === "failed" ? "failed" : (hasMeasurement ? "measured" : "missing");
    return {
      runId: run.run_id,
      status: run.status,
      state,
      prefill: state === "measured" ? run.prefill_depth0_ts : null,
      generation: state === "measured" ? run.generation_depth0_ts : null,
      measuredPrefill: run.prefill_depth0_ts,
      measuredGeneration: run.generation_depth0_ts,
    };
  });
}

export interface OffloadChartDatum {
  depth: number;
  points: Record<number, Measurement>;
}

function offloadMeasurement(result: { status: "ok" | "failed"; avg_ts: number | null } | undefined): Measurement {
  if (!result) return { state: "untested", value: null };
  if (result.status === "failed") return { state: "failed", value: null };
  return result.avg_ts === null ? missing() : { state: "measured", value: result.avg_ts };
}

export function moeChartData(curve: MoeOffloadCurve): { candidates: number[]; rows: OffloadChartDatum[] } {
  const candidates = [...new Set([
    ...(curve.candidates_tested ?? []),
    ...curve.by_depth.flatMap((depth) => depth.results.map((point) => point.n_cpu_moe)),
  ])].sort((left, right) => left - right);
  const rows = curve.by_depth.map((depth) => ({
    depth: depth.depth,
    points: Object.fromEntries(candidates.map((candidate) => [
      candidate,
      offloadMeasurement(depth.results.find((point) => point.n_cpu_moe === candidate)),
    ])),
  }));
  return { candidates, rows };
}

export function denseChartData(curve: DenseOffloadCurve): { candidates: number[]; rows: OffloadChartDatum[] } {
  const candidates = [...new Set(curve.by_depth.flatMap((depth) => depth.results.map((point) => point.n_gpu_layers)))]
    .sort((left, right) => left - right);
  const rows = curve.by_depth.map((depth) => ({
    depth: depth.depth,
    points: Object.fromEntries(candidates.map((candidate) => [
      candidate,
      offloadMeasurement(depth.results.find((point) => point.n_gpu_layers === candidate)),
    ])),
  }));
  return { candidates, rows };
}
