import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { CurveRow } from "../../../domain/results";
import { depthCurveData } from "../../../domain/chartData";

interface Props { curve: CurveRow[]; depths: number[] }

const TICK = { fontSize: 12, fill: "#c4c8d4" };
const AXIS_LABEL = { fill: "#c4c8d4", fontSize: 12 };

function SeriesChart({ data, series, label, color }: {
  data: ReturnType<typeof depthCurveData>;
  series: "prefill" | "generation";
  label: string;
  color: string;
}) {
  const chartData = data.map((row) => ({ depth: row.depth, value: row[series].value }));
  return (
    <div className="small-multiple" role="img" aria-label={`${label} throughput by context depth. Failed and missing points are gaps.`}>
      <h3>{label}</h3>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={chartData} margin={{ top: 8, right: 16, bottom: 26, left: 12 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis dataKey="depth" tick={TICK} label={{ ...AXIS_LABEL, value: "context depth", position: "insideBottom", offset: -10 }} />
          <YAxis tick={TICK} label={{ ...AXIS_LABEL, value: "tok/s", angle: -90, position: "insideLeft" }} />
          <Tooltip formatter={(value: unknown) => [typeof value === "number" ? `${value.toFixed(1)} tok/s` : "missing", label]} />
          <Line type="monotone" dataKey="value" name={label} stroke={color} strokeWidth={2} connectNulls={false} dot={{ r: 4 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function DepthCurveChart({ curve, depths }: Props) {
  const data = depthCurveData(curve, depths);
  if (data.length === 0) return <p className="empty-note">No final benchmark curve was recorded for this run.</p>;
  const measuredCount = data.reduce((count, row) => count + Number(row.prefill.state === "measured") + Number(row.generation.state === "measured"), 0);
  return (
    <>
      {data.length === 1 && <p className="single-point-note">{measuredCount === 0 ? "One context depth was attempted, but no throughput was measured." : "One context depth was measured; the values below are point measurements, not a trend."}</p>}
      <div className="small-multiples">
        <SeriesChart data={data} series="prefill" label="Prefill" color="#6ea8fe" />
        <SeriesChart data={data} series="generation" label="Generation" color="#ff7b7f" />
      </div>
      <div className="table-scroll">
        <table className="data-table">
          <caption>Final benchmark curve values and probe states</caption>
          <thead><tr><th scope="col">Context depth</th><th scope="col">Prefill</th><th scope="col">Generation</th></tr></thead>
          <tbody>{data.map((row) => (
            <tr key={row.depth}>
              <td>{row.depth.toLocaleString()}</td>
              <td>{row.prefill.state === "measured" ? `${row.prefill.value!.toFixed(1)} tok/s` : row.prefill.state}</td>
              <td>{row.generation.state === "measured" ? `${row.generation.value!.toFixed(1)} tok/s` : row.generation.state}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
    </>
  );
}
