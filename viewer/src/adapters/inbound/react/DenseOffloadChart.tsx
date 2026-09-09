import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { DenseOffloadCurve } from "../../../domain/results";
import { denseChartData } from "../../../domain/chartData";

interface Props { curve: DenseOffloadCurve }
const COLORS = ["#6ea8fe", "#ff7b7f", "#2dd4bf", "#fbbf24", "#c084fc", "#f472b6"];
const TICK = { fontSize: 12, fill: "#c4c8d4" };

export function DenseOffloadChart({ curve }: Props) {
  const { candidates, rows } = denseChartData(curve);
  const byNgl = candidates.map((ngl) => ({
    ngl,
    ...Object.fromEntries(rows.map((row) => [`depth_${row.depth}`, row.points[ngl].value])),
  }));
  const failures = rows.flatMap((row) => candidates.filter((ngl) => row.points[ngl].state === "failed").map((ngl) => `${row.depth.toLocaleString()} / --ngl ${ngl}`));
  return <>
    <div role="img" aria-label="Measured dense offload throughput by GPU layer count. Untested candidates are gaps; failures are listed separately.">
      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={byNgl} margin={{ top: 8, right: 16, bottom: 28, left: 12 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis dataKey="ngl" type="number" domain={[0, curve.max_gpu_layers]} tick={TICK} label={{ value: "--ngl (layers on GPU)", position: "insideBottom", offset: -10, fill: "#c4c8d4", fontSize: 12 }} />
          <YAxis tick={TICK} label={{ value: "prefill tok/s", angle: -90, position: "insideLeft", fill: "#c4c8d4", fontSize: 12 }} />
          <Tooltip formatter={(value: unknown) => [typeof value === "number" ? `${value.toFixed(1)} tok/s` : "not measured", ""]} />
          <Legend />
          {rows.map((row, index) => <Line key={row.depth} type="monotone" dataKey={`depth_${row.depth}`} name={`depth ${row.depth.toLocaleString()}`} stroke={COLORS[index % COLORS.length]} strokeWidth={2} connectNulls={false} dot={{ r: 3 }} />)}
        </LineChart>
      </ResponsiveContainer>
    </div>
    {failures.length > 0 && <p className="failure-note"><strong>Failed probes:</strong> {failures.join("; ")}. Untested combinations are not failures.</p>}
    <div className="table-scroll"><table className="data-table">
      <caption>Exact dense capacity boundaries and all candidate states</caption>
      <thead><tr><th scope="col">Context depth</th><th scope="col">Maximum --ngl that fits</th>{candidates.map((ngl) => <th scope="col" key={ngl}>ngl {ngl}</th>)}</tr></thead>
      <tbody>{curve.by_depth.map((depth, index) => <tr key={depth.depth}>
        <td>{depth.depth.toLocaleString()}</td><td>{depth.max_ngl_that_fits === null ? "none fit" : `${depth.max_ngl_that_fits} / ${curve.max_gpu_layers}`}</td>
        {candidates.map((ngl) => { const point = rows[index].points[ngl]; return <td key={ngl}>{point.state === "measured" ? `${point.value!.toFixed(1)} tok/s` : point.state}</td>; })}
      </tr>)}</tbody>
    </table></div>
    <p className="caveat-note">The exact fitting boundary is binary-searched at every depth. {curve.final_ngl === null ? "No final curve ran because even --ngl 0 did not fit at the deepest context." : `The final depth curve fixes --ngl ${curve.final_ngl}, the value safe at the deepest tested context.`}</p>
  </>;
}
