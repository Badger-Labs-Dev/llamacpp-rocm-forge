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
import type { DenseOffloadCurve } from "../../../domain/results";

interface Props {
  curve: DenseOffloadCurve;
}

const LINE_COLORS = ["#2563eb", "#e5484d", "#12a594", "#f59e0b", "#8b5cf6", "#ec4899"];

/** Dense-model throughput as transformer/output layers move between CPU and GPU. */
export function DenseOffloadChart({ curve }: Props) {
  const allNgl = Array.from(
    new Set(curve.by_depth.flatMap((depth) => depth.results.map((point) => point.n_gpu_layers))),
  ).sort((a, b) => a - b);

  const data = allNgl.map((ngl) => {
    const row: Record<string, number | null> = { ngl };
    for (const depth of curve.by_depth) {
      const point = depth.results.find((result) => result.n_gpu_layers === ngl);
      row[`depth_${depth.depth}`] = point?.status === "ok" ? point.avg_ts : null;
    }
    return row;
  });

  return (
    <>
      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={data} margin={{ top: 8, right: 24, bottom: 24, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border, #333)" />
          <XAxis
            dataKey="ngl"
            type="number"
            domain={[0, curve.max_gpu_layers]}
            tick={{ fontSize: 12 }}
            label={{ value: "--ngl (layers on GPU)", position: "insideBottom", offset: -8, fontSize: 12 }}
          />
          <YAxis
            tick={{ fontSize: 12 }}
            label={{ value: "prefill tok/s", angle: -90, position: "insideLeft", fontSize: 12 }}
          />
          <Tooltip
            formatter={(value: unknown, name: unknown) => [
              typeof value === "number" ? value.toFixed(1) : "OOM / didn't fit",
              String(name),
            ]}
            labelFormatter={(ngl) => `--ngl ${ngl}`}
          />
          <Legend />
          {curve.by_depth.map((depth, index) => (
            <Line
              key={depth.depth}
              type="monotone"
              dataKey={`depth_${depth.depth}`}
              name={`depth ${depth.depth.toLocaleString()}`}
              stroke={LINE_COLORS[index % LINE_COLORS.length]}
              strokeWidth={2}
              connectNulls
              dot={{ r: 3 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
      <table className="moe-boundary-table">
        <thead>
          <tr>
            <th>Context depth</th>
            <th>Maximum --ngl that fits</th>
          </tr>
        </thead>
        <tbody>
          {curve.by_depth.map((depth) => (
            <tr key={depth.depth}>
              <td>{depth.depth.toLocaleString()}</td>
              <td>
                {depth.max_ngl_that_fits !== null
                  ? `${depth.max_ngl_that_fits} / ${curve.max_gpu_layers}`
                  : "doesn't fit even at --ngl 0"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="caveat-note">
        The exact fitting boundary is binary-searched at every depth. Throughput samples below
        that boundary show where moving more layers to CPU causes a nonlinear performance cliff.
        {curve.final_ngl !== null ? (
          <>The final depth curve holds <code>--ngl {curve.final_ngl}</code> fixed—the value safe at
          the deepest tested context—so context depth is the only changing variable there.</>
        ) : (
          <>No final depth curve was run because even <code>--ngl 0</code> could not fit the deepest requested context.</>
        )}
      </p>
    </>
  );
}
