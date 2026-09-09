import {
  Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type { TuningStage } from "../../../domain/results";
import type { SensitivityImpact } from "../../../domain/sensitivity";
import { sensitivityBars } from "../../../domain/sensitivity";

interface Props { tuningLog: TuningStage[] }
const COLORS: Record<Exclude<SensitivityImpact, "non_comparable">, string> = { high: "#ff7b7f", medium: "#fbbf24", low: "#2dd4bf" };
const TICK = { fontSize: 12, fill: "#c4c8d4" };

export function SensitivityChart({ tuningLog }: Props) {
  const bars = sensitivityBars(tuningLog);
  if (bars.length === 0) return <p className="empty-note">No comparable sensitivity data for this run. Quick runs do not record per-parameter tuning scores, and failed probes are excluded.</p>;
  // Non-comparable bars (zero-valued worst candidate with a positive best)
  // have no meaningful percentage to plot; they are omitted from the chart
  // entirely and reported only in the table below.
  const chartable = bars.filter((bar): bar is typeof bar & { swingPct: number } => bar.swingPct !== null);
  const data = chartable.map((bar) => ({
    ...bar,
    detail: `${bar.bestLabel} (${bar.bestValue.toFixed(1)} tok/s) vs ${bar.worstLabel} (${bar.worstValue.toFixed(1)} tok/s)`,
  }));
  return <>
    {data.length > 0 && <div className="sensitivity-chart" role="img" aria-label="Throughput sensitivity by tuned parameter. Longer bars indicate greater impact.">
      <ResponsiveContainer width="100%" height={Math.max(160, data.length * 70)}>
        <BarChart data={data} layout="vertical" margin={{ top: 8, right: 24, bottom: 28, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis type="number" tick={TICK} label={{ value: "throughput swing (%)", position: "insideBottom", offset: -10, fill: "#c4c8d4", fontSize: 12 }} />
          <YAxis type="category" dataKey="label" tick={{ ...TICK, fontSize: 13 }} width={130} />
          <Tooltip formatter={(_value: unknown, _name: unknown, item) => [(item?.payload as { detail?: string })?.detail ?? "", ""]} />
          <Bar dataKey="swingPct" radius={[0, 4, 4, 0]}>{data.map((entry) => <Cell key={entry.stage} fill={COLORS[entry.impact as Exclude<SensitivityImpact, "non_comparable">]} />)}</Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>}
    <div className="table-scroll"><table className="data-table sensitivity-table">
      <caption>Sensitivity values</caption>
      <thead><tr><th scope="col">Parameter</th><th scope="col">Impact</th><th scope="col">Best</th><th scope="col">Worst</th></tr></thead>
      <tbody>{bars.map((bar) => <tr key={bar.stage}><td>{bar.label}</td><td>{bar.swingPct === null ? "non-comparable (zero baseline)" : `${bar.swingPct.toFixed(1)}% (${bar.impact})`}</td><td>{bar.bestLabel}: {bar.bestValue.toFixed(1)}</td><td>{bar.worstLabel}: {bar.worstValue.toFixed(1)}</td></tr>)}</tbody>
    </table></div>
    <p className="caveat-note">Sensitivity follows the auto-tuner's coordinate-descent path, not an independent full grid. Negative failed-probe sentinels are excluded. A zero-valued worst candidate against a positive best is reported as non-comparable in the table rather than charted.</p>
  </>;
}
