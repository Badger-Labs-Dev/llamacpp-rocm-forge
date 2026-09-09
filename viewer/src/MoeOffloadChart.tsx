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
import type { MoeOffloadCurve } from "./types";

interface Props {
  curve: MoeOffloadCurve;
}

const LINE_COLORS = ["#2563eb", "#e5484d", "#12a594", "#f59e0b", "#8b5cf6", "#ec4899"];

/** MoE-only chart: throughput vs depth, one line per --n-cpu-moe value
 * tested, lines simply not drawn past the depth where that value first
 * OOMs (connectNulls=false) - visually, the "wall" where a line stops is
 * exactly the OOM boundary, and the vertical gap between the ncmoe=0
 * line and the highest-ncmoe line at any depth is the throughput cost
 * of offloading. Every expert has to be resident in VRAM regardless of
 * how few are active per token, which is why this curve exists at all -
 * see moe_params()/sweep_moe_offload_*() in run_bench.py. */
export function MoeOffloadChart({ curve }: Props) {
  // Every n_cpu_moe value that appears anywhere across all depths - the
  // union, not just one depth's candidates, since thorough mode tests
  // different values at different depths (binary search narrows in).
  const allNcmoe = Array.from(
    new Set(curve.by_depth.flatMap((d) => d.results.map((r) => r.n_cpu_moe))),
  ).sort((a, b) => a - b);

  const data = curve.by_depth.map((d) => {
    const row: Record<string, number | null> = { depth: d.depth };
    for (const ncmoe of allNcmoe) {
      const point = d.results.find((r) => r.n_cpu_moe === ncmoe);
      row[`ncmoe_${ncmoe}`] = point && point.status === "ok" ? point.avg_ts : null;
    }
    return row;
  });


  return (
    <>
      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={data} margin={{ top: 8, right: 24, bottom: 24, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border, #333)" />
          <XAxis
            dataKey="depth"
            tick={{ fontSize: 12 }}
            label={{ value: "depth (context size)", position: "insideBottom", offset: -8, fontSize: 12 }}
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
          />
          <Legend />
          {allNcmoe.map((ncmoe, i) => (
            <Line
              key={ncmoe}
              type="monotone"
              dataKey={`ncmoe_${ncmoe}`}
              name={`--n-cpu-moe ${ncmoe}`}
              stroke={LINE_COLORS[i % LINE_COLORS.length]}
              strokeWidth={2}
              connectNulls={false}
              dot={{ r: 3 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
      <table className="moe-boundary-table">
        <thead>
          <tr>
            <th>Context depth</th>
            <th>Min --n-cpu-moe that fits</th>
          </tr>
        </thead>
        <tbody>
          {curve.by_depth.map((d) => (
            <tr key={d.depth}>
              <td>{d.depth.toLocaleString()}</td>
              <td>
                {d.min_ncmoe_that_fits !== null
                  ? `${d.min_ncmoe_that_fits} / ${curve.block_count}`
                  : "doesn't fit even fully offloaded"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="caveat-note">
        {curve.mode === "quick"
          ? `Quick mode: tested fixed candidates (${curve.candidates_tested?.join(", ")}), so the true boundary may fall between two tested values rather than exactly at the reported one.`
          : "Thorough mode: binary-searched the exact boundary per depth."}
        {" "}Every expert has to be resident in VRAM regardless of how few are
        active per token (expert_used_count={curve.expert_used_count} of{" "}
        {curve.expert_count}) - --n-cpu-moe moves the rest to CPU RAM.
      </p>
    </>
  );
}
