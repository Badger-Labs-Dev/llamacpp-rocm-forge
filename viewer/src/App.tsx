import { useEffect, useMemo, useState } from "react";
import "./App.css";
import type { ResultsFile } from "./types";
import { VersionChart } from "./VersionChart";
import { SensitivityChart } from "./SensitivityChart";
import { RecommendationCard } from "./RecommendationCard";

function App() {
  const [data, setData] = useState<ResultsFile | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedSlug, setSelectedSlug] = useState<string | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}results.json`, { cache: "no-store" })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((json: ResultsFile) => {
        setData(json);
        if (json.models.length > 0) setSelectedSlug(json.models[0].model_slug);
      })
      .catch((err) => setError(String(err)));
  }, []);

  const selectedModel = useMemo(
    () => data?.models.find((m) => m.model_slug === selectedSlug) ?? null,
    [data, selectedSlug],
  );

  useEffect(() => {
    if (selectedModel && selectedModel.runs.length > 0) {
      setSelectedRunId(selectedModel.runs[selectedModel.runs.length - 1].run_id);
    }
  }, [selectedModel]);

  const selectedRun = useMemo(
    () => selectedModel?.runs.find((r) => r.run_id === selectedRunId) ?? null,
    [selectedModel, selectedRunId],
  );

  if (error) {
    return (
      <div className="page">
        <p className="error-note">
          Failed to load results.json: {error}. Run{" "}
          <code>benchmark/generate_results_json.py</code> first.
        </p>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="page">
        <p>Loading results\u2026</p>
      </div>
    );
  }

  if (data.models.length === 0) {
    return (
      <div className="page">
        <p>
          No benchmark results found. Run a{" "}
          <code>--full-sweep</code> benchmark, then{" "}
          <code>benchmark/generate_results_json.py</code>.
        </p>
      </div>
    );
  }

  return (
    <div className="page">
      <header className="page-header">
        <h1>r9700-llm-bench results</h1>
        <p className="subtitle">
          llama.cpp throughput on an AMD Radeon AI PRO R9700, tracked across
          ROCm/llama.cpp versions and swept parameters.
        </p>
      </header>

      <section className="model-picker">
        {data.models.map((model) => (
          <button
            key={model.model_slug}
            className={
              "model-button" + (model.model_slug === selectedSlug ? " active" : "")
            }
            onClick={() => setSelectedSlug(model.model_slug)}
          >
            {model.model_name ?? model.model_filename ?? model.model_slug}
          </button>
        ))}
      </section>

      {selectedModel && (
        <>
          <section className="card">
            <h2>Performance across versions</h2>
            <p className="section-note">
              Depth-0 throughput for every ROCm/llama.cpp version this model
              has been benchmarked against.{" "}
              {selectedModel.model_context_length && (
                <>Trained context: {selectedModel.model_context_length.toLocaleString()} tokens.</>
              )}
            </p>
            <VersionChart runs={selectedModel.runs} />
          </section>

          <section className="card">
            <div className="section-heading">
              <h2>Parameter sensitivity</h2>
              <select
                value={selectedRunId ?? ""}
                onChange={(e) => setSelectedRunId(e.target.value)}
              >
                {selectedModel.runs.map((run) => (
                  <option key={run.run_id} value={run.run_id}>
                    {run.run_id}
                  </option>
                ))}
              </select>
            </div>
            {selectedRun && (
              <>
                <p className="section-note">
                  How much each swept parameter moved throughput for{" "}
                  <strong>{selectedRun.run_id}</strong>.
                </p>
                <SensitivityChart tuningLog={selectedRun.tuning_log} />
              </>
            )}
          </section>

          {selectedRun && (
            <section className="card">
              <h2>Recommended settings</h2>
              <p className="section-note">
                For <strong>{selectedRun.run_id}</strong> — each setting
                annotated with how much it actually mattered.
              </p>
              <RecommendationCard
                config={selectedRun.final_config}
                tuningLog={selectedRun.tuning_log}
              />
            </section>
          )}
        </>
      )}

      <footer className="page-footer">
        Generated {new Date(data.generated_at).toLocaleString()}
      </footer>
    </div>
  );
}

export default App;
