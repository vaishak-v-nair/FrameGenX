import React, { useCallback, useEffect, useRef, useState } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';
import { parseResearchList } from '../researchState';

const percent = value => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : 'Unavailable';

export default function ResearchExperiments({ apiBase }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const requestVersion = useRef(0);
  const activeRequest = useRef(null);
  const fetchEvidence = useCallback(async () => {
    activeRequest.current?.abort();
    const controller = new AbortController();
    activeRequest.current = controller;
    const signal = controller.signal;
    const version = ++requestVersion.current;
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${apiBase}/api/research/experiments?limit=5`, { signal });
      if (!response.ok) throw new Error(`Research evidence returned ${response.status}`);
      const body = parseResearchList(await response.json());
      if (!signal?.aborted && version === requestVersion.current) setData(body);
    } catch (failure) {
      if (failure.name !== 'AbortError' && version === requestVersion.current) setError(failure.message);
    } finally {
      if (!signal?.aborted && version === requestVersion.current) setLoading(false);
      if (activeRequest.current === controller) activeRequest.current = null;
    }
  }, [apiBase]);
  useEffect(() => {
    fetchEvidence();
    return () => activeRequest.current?.abort();
  }, [fetchEvidence]);

  return (
    <section className="glass-panel card research-evidence" aria-labelledby="research-evidence-title">
      <div className="card-header" style={{ justifyContent: 'space-between' }}>
        <h2 id="research-evidence-title" style={{ fontSize: '1rem' }}>ML / DL research evidence</h2>
        <button type="button" className="link-btn" disabled={loading} onClick={() => fetchEvidence()}>
          <RefreshCw size={14} aria-hidden="true" /> {loading ? 'Loading…' : 'Refresh research'}
        </button>
      </div>
      <div className="card-content">
        <p className="dim">VSM Cards experiments measure structured card validity. General image constraint satisfaction remains unverified. Research checkpoints are inactive in SDXL generation.</p>
        {error && <div role="alert" className="error-panel"><AlertTriangle size={18} aria-hidden="true" /> {error}</div>}
        <div role="status" aria-live="polite">
          {loading && <p>Reading saved experiment evidence…</p>}
          {!loading && !error && data?.experiments.length === 0 && <p>No research runs recorded. Run the I-HallA / VSM training notebook to create an experiment.</p>}
        </div>
        {data?.experiments.map(run => (
          <article key={run.id} className="glass-panel sub-card" style={{ marginTop: '1rem' }}>
            <h3 style={{ fontSize: '1rem' }}>{run.dataset || 'Research evidence'} · Recorded: {run.execution_status.replaceAll('_', ' ')}</h3>
            <p className="dim small-note">Run {run.id} · Qualification: unverified</p>
            {run.execution_status === 'evidence_error' && <p role="alert">Evidence is missing or damaged. Inspect the local run before retrying training.</p>}
            {run.execution_status === 'completed' && <>
              <dl className="research-metrics">
                <div><dt>ML synthetic-corruption diagnostic</dt><dd>{percent(run.classical_ml?.balanced_accuracy)} balanced accuracy · N={run.classical_ml?.test_n}</dd></div>
                <div><dt>DL baseline card validity</dt><dd>{percent(run.deep_learning?.baseline?.valid_rate)} · N={run.deep_learning?.baseline?.n}</dd></div>
                <div><dt>DL VSM card validity</dt><dd>{percent(run.deep_learning?.candidate?.valid_rate)} · N={run.deep_learning?.candidate?.n}</dd></div>
              </dl>
              <p className="dim small-note">Single training seed and bounded data subset. No I-HallA score was produced for this run. No general hallucination-reduction claim.</p>
            </>}
            {['failed', 'interrupted'].includes(run.execution_status) && <p>Experiment did not complete. Retained evidence is unverified; a deliberate new attempt restarts both training arms.</p>}
            {run.execution_status === 'running' && <p>This is saved progress. Process liveness is unverified; an interrupted process may leave this status behind.</p>}
          </article>
        ))}
        {data?.scan_truncated && <p className="dim">Showing a bounded research inventory; additional runs may exist locally.</p>}
      </div>
    </section>
  );
}
