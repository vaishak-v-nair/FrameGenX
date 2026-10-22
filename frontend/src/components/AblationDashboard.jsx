import React, { useCallback, useEffect, useState } from 'react';
import { BarChart3, AlertTriangle, RefreshCw } from 'lucide-react';

export default function AblationDashboard({ apiBase }) {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchStats = useCallback(async (signal) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${apiBase}/stats`, { signal });
      if (!res.ok) throw new Error(`Stats endpoint returned ${res.status}`);
      const data = await res.json();
      if (!signal?.aborted) setStats(data);
    } catch (err) {
      if (err.name !== 'AbortError') setError(err.message);
    } finally {
      if (!signal?.aborted) setLoading(false);
    }
  }, [apiBase]);

  useEffect(() => {
    const controller = new AbortController();
    fetchStats(controller.signal);
    return () => controller.abort();
  }, [fetchStats]);

  if (loading) {
    return (
      <div className="glass-panel card" style={{ textAlign: 'center', padding: '3rem' }}>
        <RefreshCw className="animate-spin" size={28} style={{ margin: '0 auto 1rem', color: 'var(--accent)' }} />
        <p className="dim">Loading ablation benchmark statistics from SQLite...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="glass-panel error-panel" role="alert">
        <AlertTriangle size={20} />
        <div>
          <strong>Failed to load statistics:</strong> {error}
          <button type="button" className="link-btn" onClick={() => fetchStats()} style={{ marginLeft: '1rem' }}>Retry</button>
        </div>
      </div>
    );
  }

  const configs = stats?.configs || {};
  const configOrder = ['baseline', 'rag', 'rag_lora', 'full'];
  const bestConfig = configOrder
    .filter(cfg => configs[cfg])
    .reduce((best, cfg) => best === null || configs[cfg].hard_pass_rate > configs[best].hard_pass_rate ? cfg : best, null);
  const configLabels = {
    baseline: 'Baseline',
    rag: '+ RAG',
    rag_lora: '+ RAG + LoRA',
    full: 'Full (Reflect hard-only)',
  };

  return (
    <div className="ablation-dashboard">
      <div className="glass-panel card">
        <div className="card-header" style={{ justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <BarChart3 size={20} color="var(--accent)" />
            <span>Ablation Benchmark Leaderboard</span>
          </div>
          <button type="button" className="link-btn" onClick={() => fetchStats()} title="Refresh live statistics">
            <RefreshCw size={14} /> Refresh ({stats?.total_runs || 0} runs total)
          </button>
        </div>

        <div className="card-content">
          <p className="dim" style={{ marginBottom: '1.25rem' }}>
            Latest fingerprinted benchmark suite comparing object-count and spatial pass rate, CLIP alignment, VQA attribute score, composite reward, and average repair latency. This pass rate does not establish full contract satisfaction. VQA and reward are unavailable when vision QA is incomplete.
          </p>

          {stats?.experiment_key && <p className="dim small-note">Suite ID: {stats.experiment_key.slice(13, 25)} · model weight revisions are not yet pinned.</p>}

          {bestConfig === null && <p className="dim">No fingerprinted benchmark rows are recorded yet.</p>}
          <div className="dashboard-grid">
            {configOrder.map(cfg => {
              const c = configs[cfg];
              if (!c) return null;
              const passPct = Math.round(c.hard_pass_rate * 100);
              const isBest = cfg === bestConfig;

              return (
                <div key={cfg} className={`glass-panel sub-card ${isBest ? 'highlight-card' : ''}`}>
                  <div className="sub-card-header">
                    <strong>{configLabels[cfg] || cfg}</strong>
                    <span className="dim small-note">N = {c.n}</span>
                  </div>

                  <div className="metric-large">
                    <span className="metric-number" style={{ color: passPct >= 70 ? 'var(--success)' : passPct >= 50 ? 'var(--warn)' : 'var(--error)' }}>
                      {passPct}%
                    </span>
                    <span className="metric-label">Count/spatial pass</span>
                  </div>

                  <div className="stat-bars">
                    <div className="stat-row">
                      <span className="dim">Reward:</span>
                      <strong>{c.reward != null ? c.reward.toFixed(3) : '—'}</strong>
                    </div>
                    <div className="stat-row">
                      <span className="dim">CLIP norm:</span>
                      <span>{c.clip_norm != null ? c.clip_norm.toFixed(3) : '—'}</span>
                    </div>
                    <div className="stat-row">
                      <span className="dim">VQA score:</span>
                      <span>{c.vqa_score != null ? c.vqa_score.toFixed(3) : '—'}</span>
                    </div>
                    <div className="stat-row">
                      <span className="dim">Avg attempts:</span>
                      <span>{c.avg_attempts != null ? c.avg_attempts.toFixed(2) : '—'}</span>
                    </div>
                    <div className="stat-row">
                      <span className="dim">Avg latency:</span>
                      <span>{c.avg_latency != null ? `${c.avg_latency}s` : '—'}</span>
                    </div>
                  </div>

                  {c.by_category && Object.keys(c.by_category).length > 0 && (
                    <div className="category-breakdown">
                      <p className="dim mini-title" style={{ fontSize: '0.75rem', marginTop: '0.75rem' }}>Categories:</p>
                      {Object.entries(c.by_category).map(([cat, val]) => {
                        const catPct = Math.round(val.hard_pass_rate * 100);
                        return (
                          <div key={cat} className="stat-row" style={{ fontSize: '0.75rem', flexDirection: 'column', alignItems: 'flex-start', gap: '0.2rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
                              <span className="dim">{cat}</span>
                              <span style={{ color: catPct >= 70 ? 'var(--success)' : 'inherit', fontWeight: 'bold' }}>
                                {catPct}% <span className="dim" style={{fontWeight:'normal'}}>({val.n})</span>
                              </span>
                            </div>
                            <div className="score-track" style={{ width: '100%', height: '4px', background: 'rgba(255,255,255,0.05)' }}>
                              <div className="score-fill" style={{ width: `${catPct}%`, background: catPct >= 70 ? 'var(--success)' : 'var(--warn)' }} />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
