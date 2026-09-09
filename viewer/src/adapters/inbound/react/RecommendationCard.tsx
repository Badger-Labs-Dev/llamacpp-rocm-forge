import type { FinalConfig, TuningStage } from "../../../domain/results";
import type { SensitivityBar } from "../../../domain/sensitivity";
import { sensitivityBars } from "../../../domain/sensitivity";

interface Props { config: FinalConfig; tuningLog: TuningStage[] }

function sensitivityNote(bar: SensitivityBar | undefined): string {
  if (!bar) return "";
  if (bar.impact === "non_comparable") return "— best was positive while the worst valid result was zero; relative impact is not comparable.";
  if (bar.impact === "high") return `— ${bar.swingPct!.toFixed(0)}% faster than the worst option tested. Matters a lot.`;
  if (bar.impact === "medium") return `— ${bar.swingPct!.toFixed(0)}% faster than the worst option tested. Worth checking.`;
  return `— only ${bar.swingPct!.toFixed(1)}% different from the worst option tested. Pick whatever's convenient.`;
}

export function RecommendationCard({ config, tuningLog }: Props) {
  const byStage = Object.fromEntries(sensitivityBars(tuningLog).map((bar) => [bar.stage, bar]));
  const rows = [
    { label: "Flash attention", value: config.flash_attn, note: "— not swept; llama.cpp selection or a KV-cache requirement." },
    { label: "KV cache dtype", value: `${config.ctk} / ${config.ctv}`, note: sensitivityNote(byStage.kv_cache_dtype) },
    { label: "ubatch × batch", value: `${config.ubatch} × ${config.batch}`, note: sensitivityNote(byStage.ubatch_batch_grid) },
    {
      label: "GPU layers / --ngl",
      value: String(config.gpu_layers),
      note: config.block_count === null
        ? "— full-offload request; no dense capacity boundary was recorded."
        : "— capacity-constrained for the deepest tested context; not sensitivity-swept.",
    },
    {
      label: "CPU layers",
      value: String(config.n_cpu_layers),
      note: config.block_count === null
        ? "— not active without a recorded dense block count."
        : `— ${config.n_cpu_layers === 0 ? "no dense layers offloaded" : "derived from the --ngl capacity boundary"}; not sensitivity-swept.`,
    },
    {
      label: "CPU MoE layers",
      value: String(config.n_cpu_moe),
      note: `— ${config.n_cpu_moe === 0 ? "no MoE expert layers offloaded" : "selected to satisfy the MoE capacity boundary"}; not sensitivity-swept.`,
    },
  ];
  return <div className="recommendation-card">{rows.map((row) => (
    <div className="recommendation-row" key={row.label}>
      <span className="recommendation-label">{row.label}</span>
      <span className="recommendation-value">{row.value}</span>
      <span className="recommendation-note">{row.note}</span>
    </div>
  ))}</div>;
}
