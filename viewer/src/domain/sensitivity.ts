import type { TuningStage } from "./results";

export type SensitivityImpact = "low" | "medium" | "high" | "non_comparable";

export interface SensitivityBar {
  stage: string;
  label: string;
  bestLabel: string;
  worstLabel: string;
  bestValue: number;
  worstValue: number;
  swingPct: number | null;
  impact: SensitivityImpact;
  candidateCount: number;
}

const STAGE_LABELS: Record<string, string> = {
  flash_attn: "Flash attention",
  kv_cache_dtype: "KV cache dtype",
  ubatch_batch_grid: "ubatch × batch",
};

const FLASH_ATTN_LABELS: Record<string, string> = { "0": "off", "1": "on" };

function labelForCandidate(stage: string, key: string): string {
  return stage === "flash_attn" ? (FLASH_ATTN_LABELS[key] ?? key) : key;
}

/** null swingPct means the worst valid candidate scored zero while the best
 * scored positive: a relative percentage would be meaningless (division by
 * zero), so this is reported as non-comparable rather than as some very
 * large or "unbounded-but-still-a-number" swing. */
export function impactForSwing(swingPct: number | null): SensitivityImpact {
  if (swingPct === null) return "non_comparable";
  if (swingPct >= 20) return "high";
  if (swingPct >= 5) return "medium";
  return "low";
}

export function stageSensitivity(stage: TuningStage): SensitivityBar | null {
  const entries = Object.entries(stage.scores).filter(([, value]) => value >= 0);
  if (entries.length < 2) return null;

  let best = entries[0];
  let worst = entries[0];
  for (const entry of entries) {
    if (entry[1] > best[1]) best = entry;
    if (entry[1] < worst[1]) worst = entry;
  }
  const swingPct = worst[1] === 0
    ? (best[1] > 0 ? null : 0)
    : ((best[1] - worst[1]) / worst[1]) * 100;
  return {
    stage: stage.stage,
    label: STAGE_LABELS[stage.stage] ?? stage.stage,
    bestLabel: labelForCandidate(stage.stage, best[0]),
    worstLabel: labelForCandidate(stage.stage, worst[0]),
    bestValue: best[1],
    worstValue: worst[1],
    swingPct,
    impact: impactForSwing(swingPct),
    candidateCount: entries.length,
  };
}

export function sensitivityBars(tuningLog: TuningStage[]): SensitivityBar[] {
  return tuningLog
    .map(stageSensitivity)
    .filter((bar): bar is SensitivityBar => bar !== null)
    .sort((left, right) => {
      if (left.swingPct === null) return -1;
      if (right.swingPct === null) return 1;
      return right.swingPct - left.swingPct;
    });
}
