import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { TuningStage } from "./types";
import { sensitivityBars } from "./sensitivity";

interface Props {
  tuningLog: TuningStage[];
}

const BAR_COLOR_HIGH = "#e5484d"; // >20% swing - matters a lot
const BAR_COLOR_MED = "#f59e0b"; // 5-20% swing - worth checking
const BAR_COLOR_LOW = "#12a594"; // <5% swing - pick whatever's convenient

function colorFor(swingPct: number): string {
  if (swingPct >= 20) return BAR_COLOR_HIGH;
  if (swingPct >= 5) return BAR_COLOR_MED;
  return BAR_COLOR_LOW;
}

/** Tornado chart: one horizontal bar per swept parameter, length = the
 * throughput swing between that parameter's best and worst tested
 * candidate (from --full-sweep's tuning_log), sorted largest-first. */
export function SensitivityChart({ tuningLog }: Props) {
  const bars = sensitivityBars(tuningLog);

  if (bars.length === 0) {
    return (
      <p className="empty-note">
        No sensitivity data for this run - only <code>--full-sweep</code>{" "}
        runs record per-parameter tuning scores. Fixed-config or{" "}
        <code>--calibrate</code> runs only produce the version-over-time
        data point above.
      </p>
    );
  }

  const data = bars.map((bar) => ({
    name: bar.label,
    swingPct: Number(bar.swingPct.toFixed(1)),
    detail: `${bar.bestLabel} (${bar.bestValue.toFixed(0)} tok/s) vs ${bar.worstLabel} (${bar.worstValue.toFixed(0)} tok/s), ${bar.candidateCount} candidates tested`,
  }));

  return (
    <>
      <ResponsiveContainer width="100%" height={Math.max(160, bars.length * 70)}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 8, right: 32, bottom: 8, left: 8 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border, #333)" />
          <XAxis
            type="number"
            tick={{ fontSize: 12 }}
            label={{ value: "throughput swing (best vs worst, %)", position: "insideBottom", offset: -4, fontSize: 12 }}
          />
          <YAxis type="category" dataKey="name" tick={{ fontSize: 13 }} width={130} />
          <Tooltip
            formatter={(_value: unknown, _name: unknown, item) => [
              (item?.payload as { detail?: string } | undefined)?.detail ?? "",
              "",
            ]}
          />
          <Bar dataKey="swingPct" radius={[0, 4, 4, 0]}>
            {data.map((entry) => (
              <Cell key={entry.name} fill={colorFor(entry.swingPct)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="caveat-note">
        Sensitivity is measured along the auto-tuner's coordinate-descent
        search path, not a full independent grid: each stage's candidates
        were tested holding the <em>previous</em> stage's winner fixed, so
        this shows "how much did this knob matter given the choices already
        made," not a guaranteed independent effect.
      </p>
    </>
  );
}
