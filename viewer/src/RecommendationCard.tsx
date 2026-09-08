import type { FinalConfig, TuningStage } from "./types";
import { sensitivityBars } from "./sensitivity";

interface Props {
  config: FinalConfig;
  tuningLog: TuningStage[];
}

function sensitivityNote(swingPct: number | undefined): string {
  if (swingPct === undefined) return "";
  if (swingPct >= 20) return `— ${swingPct.toFixed(0)}% faster than the worst option tested. Matters a lot.`;
  if (swingPct >= 5) return `— ${swingPct.toFixed(0)}% faster than the worst option tested. Worth checking.`;
  return `— only ${swingPct.toFixed(1)}% different from the worst option tested. Pick whatever's convenient.`;
}

/** Recommended config, each field annotated with how much it actually
 * mattered (from the tornado sensitivity data) - the point is telling
 * you which choices you can safely ignore, not just what the winner was.
 * Flash attention isn't swept (always "auto" - see FinalConfig.flash_attn)
 * so it's shown informationally, without a sensitivity note. */
export function RecommendationCard({ config, tuningLog }: Props) {
  const bars = sensitivityBars(tuningLog);
  const byStage = Object.fromEntries(bars.map((b) => [b.stage, b]));

  const rows: { label: string; value: string; note: string }[] = [
    {
      label: "Flash attention",
      value: config.flash_attn,
      note: "— not swept; always \"auto\" (llama.cpp decides per model/backend).",
    },
    {
      label: "KV cache dtype",
      value: `${config.ctk} / ${config.ctv}`,
      note: sensitivityNote(byStage["kv_cache_dtype"]?.swingPct),
    },
    {
      label: "ubatch \u00d7 batch",
      value: `${config.ubatch} \u00d7 ${config.batch}`,
      note: sensitivityNote(byStage["ubatch_batch_grid"]?.swingPct),
    },
  ];

  return (
    <div className="recommendation-card">
      {rows.map((row) => (
        <div className="recommendation-row" key={row.label}>
          <span className="recommendation-label">{row.label}</span>
          <span className="recommendation-value">{row.value}</span>
          {row.note && <span className="recommendation-note">{row.note}</span>}
        </div>
      ))}
    </div>
  );
}
