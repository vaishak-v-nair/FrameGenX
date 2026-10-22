import React, { useCallback, useEffect, useRef, useState } from 'react';
import { History, CheckCircle, XCircle, ChevronLeft, ChevronRight, Eye } from 'lucide-react';

export default function RunHistory({ apiBase, onSelectRun }) {
  const [runs, setRuns] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [configFilter, setConfigFilter] = useState('');
  const [acceptedOnly, setAcceptedOnly] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const activeRequest = useRef(null);

  const fetchRuns = useCallback(async () => {
    activeRequest.current?.abort();
    const controller = new AbortController();
    activeRequest.current = controller;
    const signal = controller.signal;
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({
        page: page.toString(),
        per_page: '15',
      });
      if (configFilter) params.append('config', configFilter);
      if (acceptedOnly) params.append('accepted_only', 'true');

      const res = await fetch(`${apiBase}/api/runs?${params.toString()}`, { signal });
      if (!res.ok) throw new Error(`Run history returned ${res.status}`);
      const data = await res.json();
      if (signal?.aborted) return;
      setRuns(data.runs || []);
      setTotal(data.total || 0);
      setTotalPages(Math.max(1, data.total_pages || 1));
    } catch (err) {
      if (err.name !== 'AbortError' && !signal.aborted) setError(err.message);
    } finally {
      if (!signal?.aborted) setLoading(false);
      if (activeRequest.current === controller) activeRequest.current = null;
    }
  }, [apiBase, page, configFilter, acceptedOnly]);

  useEffect(() => {
    fetchRuns();
    return () => activeRequest.current?.abort();
  }, [fetchRuns]);

  return (
    <div className="glass-panel card run-history">
      <div className="card-header" style={{ justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <History size={20} color="var(--accent)" />
          <span>Historical Runs Audit ({total} runs in SQLite)</span>
        </div>
        <div className="filter-controls">
          <label className="sr-only" htmlFor="run-config-filter">Filter runs by configuration</label>
          <select
            id="run-config-filter"
            className="filter-select"
            value={configFilter}
            onChange={e => { setConfigFilter(e.target.value); setPage(1); }}
          >
            <option value="">All Configs</option>
            <option value="interactive">Interactive</option>
            <option value="baseline">Baseline</option>
            <option value="rag">+ RAG</option>
            <option value="rag_lora">+ RAG + LoRA</option>
            <option value="full">Full</option>
          </select>
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={acceptedOnly}
              onChange={e => { setAcceptedOnly(e.target.checked); setPage(1); }}
            />
            Hard passes only
          </label>
        </div>
      </div>

      <div className="card-content">
        {loading ? (
          <p className="dim" style={{ textAlign: 'center', padding: '2rem' }}>Loading run records...</p>
        ) : error ? (
          <div className="error-panel" role="alert">
            Failed to load run history: {error}
            <button type="button" className="link-btn" onClick={() => fetchRuns()}>Retry</button>
          </div>
        ) : runs.length === 0 ? (
          <p className="dim" style={{ textAlign: 'center', padding: '2rem' }}>No runs found matching filters.</p>
        ) : (
          <div className="table-responsive">
            <table className="runs-table">
              <thead>
                <tr>
                  <th>Status</th>
                  <th>Prompt</th>
                  <th>Config</th>
                  <th>Loops</th>
                  <th>Reward</th>
                  <th>Latency</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {runs.map(r => (
                  <tr key={r.run_id} className={r.status === 'accepted' ? 'row-accepted' : 'row-rejected'}>
                    <td>
                      {r.status === 'error' ? (
                        <span className="status-pill status-pill-bad"><XCircle size={14} /> Run error</span>
                      ) : r.accepted ? (
                        <span className="status-pill status-pill-ok"><CheckCircle size={14} /> Hard pass</span>
                      ) : (
                        <span className="status-pill status-pill-bad"><XCircle size={14} /> Rejected</span>
                      )}
                    </td>
                    <td className="prompt-cell" title={r.prompt}>
                      "{r.prompt}"
                    </td>
                    <td>
                      <span className="chip chip-neutral" style={{ fontSize: '0.75rem' }}>{r.config_name}</span>
                    </td>
                    <td>{r.repair_loops ?? 0}</td>
                    <td>{r.reward != null ? r.reward.toFixed(3) : '—'}</td>
                    <td>{r.duration != null ? `${r.duration}s` : '—'}</td>
                    <td>
                      {r.has_artifacts ? (
                        <button
                          className="link-btn"
                          style={{ fontSize: '0.8rem', padding: '0.2rem 0.5rem' }}
                          onClick={() => onSelectRun && onSelectRun(r.run_id)}
                          title="Inspect run artifacts and verification report"
                        >
                          <Eye size={13} /> View
                        </button>
                      ) : (
                        <span className="dim small-note">DB only</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {totalPages > 1 && (
          <div className="pagination">
            <button
              className="pagination-btn"
              disabled={page <= 1}
              onClick={() => setPage(p => Math.max(1, p - 1))}
            >
              <ChevronLeft size={16} /> Prev
            </button>
            <span className="dim small-note">Page {page} of {totalPages}</span>
            <button
              className="pagination-btn"
              disabled={page >= totalPages}
              onClick={() => setPage(p => Math.min(totalPages, p + 1))}
            >
              Next <ChevronRight size={16} />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
