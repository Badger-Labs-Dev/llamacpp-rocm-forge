import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { RunEntry } from "../../../domain/results";

interface Props {
  runs: RunEntry[];
}

/** Primary chart: throughput at depth 0, one point per run_id (a
 * ROCm+llama.cpp version pairing), ordered by when the run actually
 * completed. Answers "how has performance changed over versions" -
 * the stated goal - rather than the depth-curve view. */
export function VersionChart({ runs }: Props) {
  const data = runs.map((run) => ({
    run_id: run.run_id,
    prefill: run.prefill_depth0_ts,
    generation: run.generation_depth0_ts,
    completed_at: run.run_completed_at,
  }));

  if (data.length < 2) {
    return (
      <p className="empty-note">
        Only one run recorded so far for this model. Once it's been
        benchmarked against another ROCm/llama.cpp version, this chart will
        show the trend.
      </p>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={320}>
      <LineChart data={data} margin={{ top: 8, right: 24, bottom: 8, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border, #333)" />
        <XAxis dataKey="run_id" tick={{ fontSize: 12 }} />
        <YAxis
          yAxisId="prefill"
          tick={{ fontSize: 12 }}
          label={{ value: "prefill tok/s", angle: -90, position: "insideLeft", fontSize: 12 }}
        />
        <YAxis
          yAxisId="generation"
          orientation="right"
          tick={{ fontSize: 12 }}
          label={{ value: "generation tok/s", angle: 90, position: "insideRight", fontSize: 12 }}
        />
        <Tooltip
          formatter={(value: unknown, name: unknown) => [
            typeof value === "number" ? value.toFixed(1) : String(value),
            String(name),
          ]}
        />
        <Legend />
        <Line
          yAxisId="prefill"
          type="monotone"
          dataKey="prefill"
          name="Prefill (depth 0)"
          stroke="#2563eb"
          strokeWidth={2}
          connectNulls
          dot={{ r: 4 }}
        />
        <Line
          yAxisId="generation"
          type="monotone"
          dataKey="generation"
          name="Generation (depth 0)"
          stroke="#e5484d"
          strokeWidth={2}
          connectNulls
          dot={{ r: 4 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
