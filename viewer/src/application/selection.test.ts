import { describe, expect, it } from "vitest";
import type { ModelEntry, RunStatus } from "../domain/results";
import { completeRun } from "../testFixtures";
import { defaultRunId } from "./selection";

function model(runs: Array<{ status: RunStatus; completedAt: string; id: string }>): ModelEntry {
  return {
    model_slug: "example",
    model_filename: null,
    model_architecture: null,
    model_name: null,
    model_context_length: null,
    runs: runs.map(({ status, completedAt, id }) =>
      completeRun({ run_id: id, status, run_completed_at: completedAt })),
  };
}

describe("defaultRunId", () => {
  it("selects the newest finished run by completion time, not array order", () => {
    const withUnsortedOrder = model([
      { id: "newest", status: "finished", completedAt: "2026-09-09T00:00:00Z" },
      { id: "oldest", status: "finished", completedAt: "2026-09-01T00:00:00Z" },
      { id: "failed", status: "failed", completedAt: "2026-09-10T00:00:00Z" },
    ]);
    expect(defaultRunId(withUnsortedOrder)).toBe("newest");
  });

  it("falls back to the newest partial run when no finished run exists", () => {
    const runs = model([
      { id: "old-partial", status: "partial", completedAt: "2026-09-01T00:00:00Z" },
      { id: "new-partial", status: "partial", completedAt: "2026-09-05T00:00:00Z" },
      { id: "failed", status: "failed", completedAt: "2026-09-09T00:00:00Z" },
    ]);
    expect(defaultRunId(runs)).toBe("new-partial");
  });

  it("falls back to the newest failed run when no run is usable", () => {
    const runs = model([
      { id: "old-failed", status: "failed", completedAt: "2026-09-01T00:00:00Z" },
      { id: "new-failed", status: "failed", completedAt: "2026-09-05T00:00:00Z" },
    ]);
    expect(defaultRunId(runs)).toBe("new-failed");
  });
});
