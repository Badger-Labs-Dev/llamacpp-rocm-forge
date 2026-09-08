import type { TuningStage } from "./types";

export interface SensitivityBar {
  stage: string;
  label: string;
  bestLabel: string;
  worstLabel: string;
  bestValue: number;
  worstValue: number;
  swingPct: number; // (best - worst) / worst * 100
  candidateCount: number;
}

const STAGE_LABELS: Record<string, string> = {
  flash_attn: "Flash attention",
  kv_cache_dtype: "KV cache dtype",
  ubatch_batch_grid: "ubatch \u00d7 batch",
};

const FLASH_ATTN_LABELS: Record<string, string> = {
  "0": "off",
  "1": "on",
};

function labelForCandidate(stage: string, key: string): string {
  if (stage === "flash_attn") return FLASH_ATTN_LABELS[key] ?? key;
  return key;
}

/** Convert one tuning_log stage into a tornado-chart bar: the throughput
 * swing between that stage's best and worst tested candidate. This is
 * "sensitivity along the coordinate-descent search path", not a true
 * independent sensitivity - each stage's candidates were tested holding
 * the *previous* stage's winner fixed, not holding every other axis
 * constant. Worth surfacing that caveat in the UI, not just the numbers. */
export function stageSensitivity(stage: TuningStage): SensitivityBar | null {
  const entries = Object.entries(stage.scores).filter(([, v]) => v >= 0);
  if (entries.length < 2) return null;

  let best = entries[0];
  let worst = entries[0];
  for (const entry of entries) {
    if (entry[1] > best[1]) best = entry;
    if (entry[1] < worst[1]) worst = entry;
  }

  const swingPct = worst[1] > 0 ? ((best[1] - worst[1]) / worst[1]) * 100 : 0;

  return {
    stage: stage.stage,
    label: STAGE_LABELS[stage.stage] ?? stage.stage,
    bestLabel: labelForCandidate(stage.stage, best[0]),
    worstLabel: labelForCandidate(stage.stage, worst[0]),
    bestValue: best[1],
    worstValue: worst[1],
    swingPct,
    candidateCount: entries.length,
  };
}

export function sensitivityBars(tuningLog: TuningStage[]): SensitivityBar[] {
  return tuningLog
    .map(stageSensitivity)
    .filter((bar): bar is SensitivityBar => bar !== null)
    .sort((a, b) => b.swingPct - a.swingPct);
}
