import { useEffect, useState } from "react";
import "./App.css";
import type { ResultsFile } from "./domain/results";
import { HttpResultsCatalog } from "./adapters/outbound/httpResultsCatalog";
import { defaultModelSlug, defaultRunId, selectModel, selectRun } from "./application/selection";
import { VersionChart } from "./adapters/inbound/react/VersionChart";
import { DepthCurveChart } from "./adapters/inbound/react/DepthCurveChart";
import { SensitivityChart } from "./adapters/inbound/react/SensitivityChart";
import { RecommendationCard } from "./adapters/inbound/react/RecommendationCard";
import { MoeOffloadChart } from "./adapters/inbound/react/MoeOffloadChart";
import { DenseOffloadChart } from "./adapters/inbound/react/DenseOffloadChart";

function App() {
  const [data, setData] = useState<ResultsFile | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedSlug, setSelectedSlug] = useState<string | null>(null);
  const [runSelections, setRunSelections] = useState<Record<string, string>>({});

  useEffect(() => {
    const catalog = new HttpResultsCatalog(`${import.meta.env.BASE_URL}results.json`);
    catalog.load().then((dataset) => {
      setData(dataset);
      setSelectedSlug(defaultModelSlug(dataset));
    }).catch((err) => setError(String(err)));
  }, []);

  const selectedModel = data && (selectModel(data, selectedSlug) ?? selectModel(data, defaultModelSlug(data)));
  const selectedRunId = selectedModel
    ? (selectRun(selectedModel, runSelections[selectedModel.model_slug])?.run_id ?? defaultRunId(selectedModel))
    : null;
  const selectedRun = selectRun(selectedModel ?? null, selectedRunId);

  if (error) return <main className="page"><p className="error-note" role="alert">Failed to load results.json: {error}. Run <code>benchmark/generate_viewer_data.py</code> first.</p></main>;
  if (!data) return <main className="page"><p role="status" aria-live="polite">Loading results…</p></main>;
  if (data.models.length === 0) return <main className="page"><p role="status">No benchmark results found. Run <code>uv run benchmark/run_bench.py --model ...</code>, then <code>benchmark/generate_viewer_data.py</code>.</p></main>;

  return <main className="page">
    <header className="page-header">
      <h1>llamacpp-rocm-forge results</h1>
      <p className="subtitle">llama.cpp throughput on an AMD Radeon AI PRO R9700, tracked across ROCm/llama.cpp versions and swept parameters.</p>
    </header>

    <section className="model-section" role="group" aria-labelledby="model-picker-label">
      <h2 id="model-picker-label" className="picker-label">Model</h2>
      <div className="model-picker">{data.models.map((model) => {
        const active = model.model_slug === selectedModel?.model_slug;
        return <button key={model.model_slug} className={`model-button${active ? " active" : ""}`} aria-pressed={active} onClick={() => setSelectedSlug(model.model_slug)}>
          {model.model_name ?? model.model_filename ?? model.model_slug}
        </button>;
      })}</div>
    </section>

    {selectedModel && <>
      <section className="card">
        <h2>Performance across versions</h2>
        <p className="section-note">Depth-0 throughput for every ROCm/llama.cpp version. Failed and missing runs remain visible as gaps. {selectedModel.model_context_length !== null && selectedModel.model_context_length !== undefined && <>Trained context: {selectedModel.model_context_length.toLocaleString()} tokens.</>}</p>
        <VersionChart runs={selectedModel.runs} />
      </section>

      <section className="card">
        <div className="section-heading">
          <h2>Selected run</h2>
          <div className="run-picker">
            <label htmlFor="run-selector">Benchmark run</label>
            <select id="run-selector" value={selectedRunId ?? ""} onChange={(event) => setRunSelections((current) => ({ ...current, [selectedModel.model_slug]: event.target.value }))}>
              {selectedModel.runs.map((run) => <option key={run.run_id} value={run.run_id}>{run.run_id} — {run.status}</option>)}
            </select>
          </div>
        </div>
        {selectedRun ? <>
          <p className="selected-run-summary"><strong>{selectedRun.run_id}</strong> <span className={`status-badge status-${selectedRun.status}`}>{selectedRun.status}</span></p>
          {selectedRun.status !== "finished" && <p className="run-warning" role="status">{selectedRun.status === "partial" ? "This run is partial. Measured points remain viewable, but settings are provisional." : "This run failed. Any successful probe points remain viewable, but no settings are recommended."}</p>}
          <h3 className="subsection-title">Final benchmark curve</h3>
          <p className="section-note">Prefill and generation use separate scales because their magnitudes differ substantially.</p>
          <DepthCurveChart curve={selectedRun.curve} depths={selectedRun.depths_tested} />
        </> : <p className="empty-note">No runs recorded for this model.</p>}
      </section>

      {selectedRun && <section className="card">
        <h2>Parameter sensitivity</h2>
        <p className="section-note">How much each performance-tuned parameter moved throughput for <strong>{selectedRun.run_id}</strong>.</p>
        <SensitivityChart tuningLog={selectedRun.tuning_log} />
      </section>}

      {selectedRun && selectedRun.status !== "failed" && <section className="card">
        <h2>{selectedRun.status === "partial" ? "Provisional settings" : "Recommended settings"}</h2>
        <p className="section-note">For <strong>{selectedRun.run_id}</strong>. Performance-tuned choices include measured sensitivity; offload settings are capacity constraints and were not sensitivity-swept.</p>
        <RecommendationCard config={selectedRun.final_config} tuningLog={selectedRun.tuning_log} />
      </section>}

      {selectedRun?.moe_offload_curve && <section className="card">
        <h2>MoE expert offload (--n-cpu-moe)</h2>
        <p className="section-note">For <strong>{selectedRun.run_id}</strong>: measured throughput and the {selectedRun.moe_offload_curve.mode === "thorough" ? "exact" : "approximate (fixed-candidate)"} minimum offload that fits at each context depth.</p>
        <MoeOffloadChart curve={selectedRun.moe_offload_curve} />
      </section>}

      {selectedRun?.dense_offload_curve && <section className="card">
        <h2>Dense layer offload (--ngl)</h2>
        <p className="section-note">For <strong>{selectedRun.run_id}</strong>: measured throughput and the exact maximum GPU-resident layer count that fits at each context depth.</p>
        <DenseOffloadChart curve={selectedRun.dense_offload_curve} />
      </section>}
    </>}

    <footer className="page-footer">Generated {new Date(data.generated_at).toLocaleString()}</footer>
  </main>;
}

export default App;
