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

export function defaultRunId(model: ModelEntry | null): string | null {
  return model?.runs.at(-1)?.run_id ?? null;
}
