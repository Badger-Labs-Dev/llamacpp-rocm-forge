import type { ModelEntry, ResultsFile, RunEntry } from "../domain/results";

export function selectModel(data: ResultsFile | null, slug: string | null): ModelEntry | null {
  return data?.models.find((model) => model.model_slug === slug) ?? null;
}

export function defaultModelSlug(data: ResultsFile): string | null {
  return data.models[0]?.model_slug ?? null;
}

export function selectRun(model: ModelEntry | null, runId: string | null): RunEntry | null {
  return model?.runs.find((run) => run.run_id === runId) ?? null;
}

function newestByCompletion(runs: RunEntry[], status: RunEntry["status"]): RunEntry | null {
  const matching = runs.filter((run) => run.status === status);
  if (matching.length === 0) return null;
  return matching.reduce((newest, run) => {
    const newestTime = newest.run_completed_at ?? "";
    const runTime = run.run_completed_at ?? "";
    return runTime >= newestTime ? run : newest;
  });
}

export function defaultRunId(model: ModelEntry | null): string | null {
  if (!model || model.runs.length === 0) return null;
  return newestByCompletion(model.runs, "finished")?.run_id
    ?? newestByCompletion(model.runs, "partial")?.run_id
    ?? newestByCompletion(model.runs, "failed")?.run_id
    ?? null;
}
