import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { RunEntry } from "../../../domain/results";
import { versionChartData } from "../../../domain/chartData";

interface Props { runs: RunEntry[] }
const TICK = { fontSize: 12, fill: "#c4c8d4" };
const AXIS_LABEL = { fill: "#c4c8d4", fontSize: 12 };

function VersionSeries({ data, field, label, color }: {
  data: ReturnType<typeof versionChartData>;
  field: "prefill" | "generation";
  label: string;
  color: string;
}) {
  return (
    <div className="small-multiple" role="img" aria-label={`${label} depth-zero throughput across versions. Failed and missing runs create gaps.`}>
      <h3>{label}</h3>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 26, left: 12 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis dataKey="runId" tick={TICK} label={{ ...AXIS_LABEL, value: "run / version", position: "insideBottom", offset: -10 }} />
          <YAxis tick={TICK} label={{ ...AXIS_LABEL, value: "tok/s", angle: -90, position: "insideLeft" }} />
          <Tooltip formatter={(value: unknown) => [typeof value === "number" ? `${value.toFixed(1)} tok/s` : "missing", label]} />
          <Line type="monotone" dataKey={field} name={label} stroke={color} strokeWidth={2} connectNulls={false} dot={{ r: 4 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function VersionChart({ runs }: Props) {
  const data = versionChartData(runs);
  const measurementCount = data.filter((row) => row.state === "measured").length;
  if (measurementCount === 0) return <p className="empty-note">No usable depth-0 measurements have been recorded for this model.</p>;
  return (
    <>
      {measurementCount === 1 && <p className="single-point-note">One run has usable measurements; these are point measurements, not a version trend yet.</p>}
      <div className="small-multiples">
        <VersionSeries data={data} field="prefill" label="Prefill (depth 0)" color="#6ea8fe" />
        <VersionSeries data={data} field="generation" label="Generation (depth 0)" color="#ff7b7f" />
      </div>
      <div className="table-scroll">
        <table className="data-table">
          <caption>Depth-zero values across runs, including failed and missing runs</caption>
          <thead><tr><th scope="col">Run</th><th scope="col">Status</th><th scope="col">Prefill</th><th scope="col">Generation</th></tr></thead>
          <tbody>{data.map((row) => (
            <tr key={row.runId}>
              <td>{row.runId}</td><td>{row.status}</td>
              <td>{row.measuredPrefill === null ? "missing" : `${row.measuredPrefill.toFixed(1)} tok/s${row.state === "failed" ? " (failed run; not trended)" : ""}`}</td>
              <td>{row.measuredGeneration === null ? "missing" : `${row.measuredGeneration.toFixed(1)} tok/s${row.state === "failed" ? " (failed run; not trended)" : ""}`}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
    </>
  );
}
