import { useEffect, useMemo, useState } from "react";
import "./App.css";
import type { ResultsFile } from "./domain/results";
import { HttpResultsCatalog } from "./adapters/outbound/httpResultsCatalog";
import { defaultModelSlug, defaultRunId, selectModel, selectRun } from "./application/selection";
import { VersionChart } from "./adapters/inbound/react/VersionChart";
import { SensitivityChart } from "./adapters/inbound/react/SensitivityChart";
import { RecommendationCard } from "./adapters/inbound/react/RecommendationCard";
import { MoeOffloadChart } from "./adapters/inbound/react/MoeOffloadChart";
import { DenseOffloadChart } from "./adapters/inbound/react/DenseOffloadChart";

function App() {
  const [data, setData] = useState<ResultsFile | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedSlug, setSelectedSlug] = useState<string | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);

  useEffect(() => {
    const catalog = new HttpResultsCatalog(`${import.meta.env.BASE_URL}results.json`);
    catalog.load()
      .then((dataset) => {
        setData(dataset);
        setSelectedSlug(defaultModelSlug(dataset));
      })
      .catch((err) => setError(String(err)));
  }, []);

  const selectedModel = useMemo(
    () => selectModel(data, selectedSlug),
    [data, selectedSlug],
  );

  useEffect(() => {
    setSelectedRunId(defaultRunId(selectedModel));
  }, [selectedModel]);

  const selectedRun = useMemo(
    () => selectRun(selectedModel, selectedRunId),
    [selectedModel, selectedRunId],
  );

  if (error) {
    return (
      <div className="page">
        <p className="error-note">
          Failed to load results.json: {error}. Run{" "}
          <code>benchmark/generate_viewer_data.py</code> first.
        </p>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="page">
        <p>Loading results…</p>
      </div>
    );
  }

  if (data.models.length === 0) {
    return (
      <div className="page">
        <p>
          No benchmark results found. Run{" "}
          <code>uv run benchmark/run_bench.py --model ...</code>, then{" "}
          <code>benchmark/generate_viewer_data.py</code>.
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

          {selectedRun?.moe_offload_curve && (
            <section className="card">
              <h2>MoE expert offload (--n-cpu-moe)</h2>
              <p className="section-note">
                This model is MoE (expert_count={selectedRun.moe_offload_curve.expert_count}) -
                every expert has to be resident in VRAM regardless of how few
                are active per token, so a large MoE model can need CPU
                offload to fit even though its active-parameter count looks
                small. For <strong>{selectedRun.run_id}</strong>: minimum
                --n-cpu-moe that fits at each context depth, and how much
                throughput drops as more experts get offloaded.
              </p>
              <MoeOffloadChart curve={selectedRun.moe_offload_curve} />
            </section>
          )}

          {selectedRun?.dense_offload_curve && (
            <section className="card">
              <h2>Dense layer offload (--ngl)</h2>
              <p className="section-note">
                For <strong>{selectedRun.run_id}</strong>: the maximum GPU-resident
                layer count that fits at each context depth, plus throughput
                samples below that boundary to expose the CPU-offload performance cliff.
              </p>
              <DenseOffloadChart curve={selectedRun.dense_offload_curve} />
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
