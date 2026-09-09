import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { MoeOffloadCurve } from "../../../domain/results";
import { moeChartData } from "../../../domain/chartData";

interface Props { curve: MoeOffloadCurve }
const COLORS = ["#6ea8fe", "#ff7b7f", "#2dd4bf", "#fbbf24", "#c084fc", "#f472b6"];
const TICK = { fontSize: 12, fill: "#c4c8d4" };

export function MoeOffloadChart({ curve }: Props) {
  const { candidates, rows } = moeChartData(curve);
  const data = rows.map((row) => ({
    depth: row.depth,
    ...Object.fromEntries(candidates.map((candidate) => [`ncmoe_${candidate}`, row.points[candidate].value])),
  }));
  const failures = rows.flatMap((row) => candidates.filter((candidate) => row.points[candidate].state === "failed").map((candidate) => `${row.depth.toLocaleString()} / --n-cpu-moe ${candidate}`));
  return <>
    <div role="img" aria-label="Measured MoE offload throughput by context depth. Untested candidates are gaps; failures are listed separately.">
      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 28, left: 12 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis dataKey="depth" tick={TICK} label={{ value: "context depth", position: "insideBottom", offset: -10, fill: "#c4c8d4", fontSize: 12 }} />
          <YAxis tick={TICK} label={{ value: "prefill tok/s", angle: -90, position: "insideLeft", fill: "#c4c8d4", fontSize: 12 }} />
          <Tooltip formatter={(value: unknown) => [typeof value === "number" ? `${value.toFixed(1)} tok/s` : "not measured", ""]} />
          <Legend />
          {candidates.map((candidate, index) => <Line key={candidate} type="monotone" dataKey={`ncmoe_${candidate}`} name={`--n-cpu-moe ${candidate}`} stroke={COLORS[index % COLORS.length]} strokeWidth={2} connectNulls={false} dot={{ r: 3 }} />)}
        </LineChart>
      </ResponsiveContainer>
    </div>
    {failures.length > 0 && <p className="failure-note"><strong>Failed probes:</strong> {failures.join("; ")}. Untested combinations are not failures.</p>}
    <div className="table-scroll"><table className="data-table">
      <caption>{curve.mode === "thorough" ? "Exact" : "Approximate (fixed-candidate)"} MoE capacity boundaries and all candidate states</caption>
      <thead><tr><th scope="col">Context depth</th><th scope="col">Minimum that fits</th>{candidates.map((candidate) => <th scope="col" key={candidate}>ncmoe {candidate}</th>)}</tr></thead>
      <tbody>{curve.by_depth.map((depth, index) => <tr key={depth.depth}>
        <td>{depth.depth.toLocaleString()}</td>
        <td>{depth.min_ncmoe_that_fits === null ? "none fit" : `${depth.min_ncmoe_that_fits} / ${curve.block_count}`}</td>
        {candidates.map((candidate) => { const point = rows[index].points[candidate]; return <td key={candidate}>{point.state === "measured" ? `${point.value!.toFixed(1)} tok/s` : point.state}</td>; })}
      </tr>)}</tbody>
    </table></div>
    <p className="caveat-note">{curve.mode === "quick" ? `Quick mode tested fixed candidates (${curve.candidates_tested?.join(", ") ?? "not listed"}); untested gaps do not establish an OOM boundary.` : "Thorough mode binary-searched the exact per-depth boundary; intermediate candidates may be untested."} Expert usage: {curve.expert_used_count ?? "unknown"} of {curve.expert_count}.</p>
  </>;
}
