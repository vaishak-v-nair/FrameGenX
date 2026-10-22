import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Sparkles, BrainCircuit, ScanLine, CheckCircle2,
  AlertCircle, Play, Loader2, Search, LayoutGrid, Palette, Wand2, RefreshCcw,
  FileJson, Gauge, BarChart3, History
} from 'lucide-react';
import AblationDashboard from './components/AblationDashboard';
import RunHistory from './components/RunHistory';
import ResearchExperiments from './components/ResearchExperiments';
import { apiErrorMessage, bestVisibleAttempt, isVqaUnavailable, runVerdict } from './runState';

// Served from FastAPI itself (:8000/app) -> same-origin, empty base.
// Vite dev server (:5173) -> talk to the API directly; 127.0.0.1 not
// localhost, because uvicorn binds IPv4 only and Windows may resolve
// localhost to ::1.
const API = import.meta.env.VITE_API_URL ||
  (window.location.port === '5173' ? 'http://127.0.0.1:8000' : '');

// Order matches config.yaml pipeline.stages
const STAGES = [
  { key: 'intent', label: 'Intent', desc: 'prompt → checklist', Icon: FileJson },
  { key: 'retrieval', label: 'Retrieval', desc: 'find real captions', Icon: Search },
  { key: 'plan', label: 'Plan', desc: 'layout + prompt', Icon: LayoutGrid },
  { key: 'adapt', label: 'Adapt', desc: 'LoRA style', Icon: Palette },
  { key: 'generate', label: 'Generate', desc: 'SDXL-Turbo draws', Icon: Wand2 },
  { key: 'verify', label: 'Verify', desc: 'count + score', Icon: ScanLine },
  { key: 'reflect', label: 'Reflect', desc: 'accept or retry', Icon: BrainCircuit },
];

const SHOWCASE_RUN = '4512bf99-e87e-4614-984c-c6d6c95f8b5d';
const ACTIVE_JOB_KEY = 'framegenx.activeJobId';

function stageState(job, stageKey) {
  if (!job) return 'pending';
  if (job.status === 'done') return 'done';
  const att = job.attempt ?? 0;
  // intent/retrieval run once before the loop; adapt runs only on attempt 0
  const runsOnce = ['intent', 'retrieval', 'adapt'].includes(stageKey);
  const finished = (job.events || []).some(
    e => e.event === 'stage_finished' && e.stage === stageKey &&
      (runsOnce || e.attempt === att)
  );
  if (finished) return 'done';
  if (job.stage === stageKey) return 'active';
  return 'pending';
}

function ScoreBar({ label, value, threshold, color }) {
  const pct = Math.max(0, Math.min(1, value ?? 0)) * 100;
  return (
    <div className="score-row">
      <span className="score-label">{label}</span>
      <div className="score-track">
        <div className="score-fill" style={{ width: `${pct}%`, background: color || 'var(--accent)' }} />
        {threshold != null && (
          <div className="score-threshold" style={{ left: `${threshold * 100}%` }} title={`τ = ${threshold}`} />
        )}
      </div>
      <span className="score-value">{value != null ? value.toFixed(3) : '—'}</span>
    </div>
  );
}

function CountChips({ verification }) {
  if (!verification?.signals) return null;
  const expected = verification.signals.expected_counts || {};
  const detected = verification.signals.detected_counts || {};
  const keys = [...new Set([...Object.keys(expected), ...Object.keys(detected)])];
  if (!keys.length) return null;
  return (
    <div className="chip-row">
      {keys.map(k => {
        const ok = (detected[k] ?? 0) === (expected[k] ?? 0);
        return (
          <span key={k} className={`chip ${ok ? 'chip-ok' : 'chip-bad'}`}>
            {k}: found {detected[k] ?? 0} / need {expected[k] ?? '?'}
          </span>
        );
      })}
    </div>
  );
}

function AttemptCard({ att, isLast, jobDone, job, setLightboxImage }) {
  const v = att.verification;
  const d = att.decision;
  const accepted = d?.accept === true;
  const decided = d != null;
  const vqaUnavailable = isVqaUnavailable(v);
  const evidence = v?.contract_evidence;

  const getVerificationMessage = () => {
    if (job?.stage === 'verify' && isLast && !jobDone) {
      return "Running Layout & CLIP checks, awaiting VQA (Gemini)...";
    }
    if (!jobDone && isLast) {
      return "waiting for generation to finish...";
    }
    return jobDone ? "Verification evidence unavailable." : "inspecting…";
  };

  return (
    <div className="glass-panel card attempt-card">
      <div className="card-header" style={{ justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          {att.attempt === 0 ? <Wand2 size={18} color="var(--accent)" /> : <RefreshCcw size={18} color="var(--warn)" />}
          <span style={{ fontSize: '1.1rem', fontWeight: '700' }}>
            {att.attempt === 0 ? 'Attempt 0 — First Draft' : `Attempt ${att.attempt} — Self-Repair`}
          </span>
        </div>
        {decided ? (
          <span className={`status-badge ${accepted ? 'status-success' : 'status-error'}`}>
            {accepted ? 'HARD CHECKS PASSED' : 'REJECTED'}
          </span>
        ) : (
          <span className="status-badge status-running">{jobDone ? 'NO DECISION' : 'IN PROGRESS'}</span>
        )}
      </div>

      <div className="showcase-pipeline" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', marginTop: '1rem' }}>

        {/* Stage 3: Plan */}
        <div style={{ padding: '1rem', background: 'rgba(0,0,0,0.2)', borderRadius: '12px' }}>
          <p className="mini-title" style={{ fontSize: '1rem', color: 'var(--accent)' }}><LayoutGrid size={16} style={{display:'inline', verticalAlign:'middle'}}/> Stage 3: Plan (Prompt Enhancement & Layout)</p>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '1rem' }}>
            <div className="glass-panel" style={{ padding: '1rem', background: 'rgba(0,0,0,0.2)' }}>
              <p className="mini-title">Final Diffusion Prompt</p>
              {att.final_prompt ? (
                <p className="mono" style={{ fontSize: '0.8rem', marginTop: '0.5rem' }}>{att.final_prompt}</p>
              ) : (
                <p className="dim">assembling…</p>
              )}
            </div>

            <div className="glass-panel" style={{ padding: '1rem', background: 'rgba(0,0,0,0.2)' }}>
              <p className="mini-title">Vector Layout Mask</p>
              <div className="image-container small" style={{ minHeight: '120px' }}>
                {att.mask_url ? <img src={`${API}${att.mask_url}`} alt="layout mask" onClick={() => setLightboxImage(`${API}${att.mask_url}`)} style={{ cursor: 'pointer' }} /> : <p className="dim">{jobDone ? 'No layout mask was used for this prompt.' : 'planning…'}</p>}
              </div>
            </div>
          </div>
        </div>

        {/* Stage 4: Generate */}
        <div style={{ padding: '1rem', background: 'rgba(0,0,0,0.2)', borderRadius: '12px' }}>
          <p className="mini-title" style={{ fontSize: '1rem', color: 'var(--accent)' }}><Wand2 size={16} style={{display:'inline', verticalAlign:'middle'}}/> Stage 4: Generate (Diffusion Steps)</p>
          <div className="diffusion-strip" style={{ marginTop: '1rem' }}>
            {att.intermediate_steps?.length > 0 ? (
              att.intermediate_steps.map((stepUrl, idx) => (
                <div key={idx} className="diffusion-step">
                  <img src={`${API}${stepUrl}`} alt="Saved diffusion preview" onClick={() => setLightboxImage(`${API}${stepUrl}`)} style={{ cursor: 'pointer' }} />
                  <span className="step-label">Saved preview</span>
                </div>
              ))
            ) : (
              <p className="dim">{jobDone ? 'No preview was captured.' : 'The decoded preview appears when generation completes.'}</p>
            )}
            <div className="diffusion-step">
              {att.image_url ? (
                <>
                  <img src={`${API}${att.image_url}`} alt="Generated draft" style={{ cursor: 'pointer' }} onClick={() => setLightboxImage(`${API}${att.image_url}`)} />
                  <span className="step-label">Generated draft</span>
                </>
              ) : (
                <p className="dim">drawing…</p>
              )}
            </div>
          </div>
        </div>

        {/* Stage 5: Verification & Reflection */}
        <div className="attempt-scores" style={{ padding: '1rem', background: 'rgba(0,0,0,0.2)', borderRadius: '12px' }}>
          <p className="mini-title" style={{ fontSize: '1rem', color: 'var(--accent)' }}><ScanLine size={16} style={{display:'inline', verticalAlign:'middle'}}/> Stage 5: Verification & Reflection</p>
          {v ? (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginTop: '1rem' }}>
              <div>
                <ScoreBar label="CLIP (prompt alignment)" value={v.clip_norm} color="#818cf8" />
                <ScoreBar label="VQA (Precision)" value={vqaUnavailable ? null : v.vqa_score} color="#34d399" />
                <ScoreBar label="Reward R" value={vqaUnavailable ? null : v.reward} threshold={d?.tau} color="#f59e0b" />
                {vqaUnavailable && (
                  <p className="dim small-note" role="status">Vision QA was unavailable or incomplete. Color and attribute checks remain unverified; reward is unavailable.</p>
                )}
                {accepted && d?.mode === 'hard_only' && (
                  <p className="dim small-note">Acceptance covers object counts and spatial checks. VQA is a separate diagnostic score.</p>
                )}
                {evidence ? (
                  <div className="small-note" role="status" style={{ marginTop: '0.75rem' }}>
                    <strong>Extracted contract:</strong> {evidence.counts.passed} checked pass · {evidence.counts.failed} checked fail · {evidence.counts.unverified} unverified.
                    {evidence.status !== 'passed' && (
                      <details className="prompt-details">
                        <summary>See requirements needing review</summary>
                        <ul>
                          {evidence.requirements.filter(item => item.status !== 'passed').map((item, index) => (
                            <li key={`${item.kind}-${item.target}-${index}`}>
                              {item.kind.replaceAll('_', ' ')}: {item.target} ({typeof item.expected === 'object' ? JSON.stringify(item.expected) : String(item.expected)}) — {item.status}
                              {item.reason ? `; ${item.reason}` : ''}
                            </li>
                          ))}
                        </ul>
                      </details>
                    )}
                    <p className="dim">These checks cover extracted requirements; prompt parsing and visual judges still need benchmark validation.</p>
                  </div>
                ) : (
                  <p className="dim small-note">Requirement coverage was not recorded for this run.</p>
                )}
                {d?.tau != null && (
                  <p className="dim small-note">accept needs: no count failures AND R ≥ τ ({d.tau})</p>
                )}
              </div>
              <div>
                <p className="mini-title" style={{ marginBottom: '0.25rem' }}>Entity Database Constraints</p>
                <CountChips verification={v} />
              </div>
            </div>
          ) : (
            <p className="dim" style={{ color: "var(--accent)", marginTop: '1rem' }}>{getVerificationMessage()}</p>
          )}
        </div>
      </div>

      {att.final_prompt && (
        <details className="prompt-details">
          <summary>Exact prompt sent to the model</summary>
          <p className="mono">{att.final_prompt}</p>
        </details>
      )}

      {decided && !accepted && d.prompt_edits?.length > 0 && (
        <div className="correction-box">
          <p className="mini-title" style={{ color: 'var(--warn)' }}>
            It wrote itself a correction{isLast && !jobDone ? '' : ' → next attempt'}:
          </p>
          <p className="mono">"{d.prompt_edits[0]}"</p>
        </div>
      )}
    </div>
  );
}

function App() {
  const [activeTab, setActiveTab] = useState('studio');
  const [prompt, setPrompt] = useState('');
  const [force, setForce] = useState(false);
  const [job, setJob] = useState(null);
  const [artifacts, setArtifacts] = useState(null);
  const [error, setError] = useState(null);
  const [elapsed, setElapsed] = useState(0);
  const [lightboxImage, setLightboxImage] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const pollRef = useRef(null);
  const pollInFlightRef = useRef(false);
  const pollFailuresRef = useRef(0);
  const submittingRef = useRef(false);
  const closeLightboxRef = useRef(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = null;
  }, []);

  const poll = useCallback(async (jobId) => {
    if (pollInFlightRef.current) return;
    pollInFlightRef.current = true;
    try {
      const response = await fetch(`${API}/jobs/${jobId}`);
      if (response.status === 404) {
        stopPolling();
        window.localStorage.removeItem(ACTIVE_JOB_KEY);
        setJob(null);
        setError('The saved job is no longer on the server. Start a new run.');
        return;
      }
      if (!response.ok) throw new Error(`Job status returned ${response.status}`);
      const j = await response.json();
      setJob(j);
      setElapsed(Math.round(j.elapsed ?? 0));
      if (j.run_id) {
        const artifactsResponse = await fetch(`${API}/jobs/${jobId}/artifacts`);
        if (!artifactsResponse.ok) throw new Error(`Artifacts returned ${artifactsResponse.status}`);
        setArtifacts(await artifactsResponse.json());
      }
      pollFailuresRef.current = 0;
      setError(null);
      if (j.status === 'done' || j.status === 'error') {
        stopPolling();
        window.localStorage.removeItem(ACTIVE_JOB_KEY);
        if (j.status === 'error') setError(j.error || 'Generation failed.');
      }
    } catch {
      pollFailuresRef.current += 1;
      if (pollFailuresRef.current >= 3) {
        setError('Connection to the API was lost. The run may still be active; this page will keep retrying.');
      }
    } finally {
      pollInFlightRef.current = false;
    }
  }, [stopPolling]);

  const startPolling = useCallback((jobId) => {
    stopPolling();
    pollFailuresRef.current = 0;
    poll(jobId);
    pollRef.current = setInterval(() => poll(jobId), 1500);
  }, [poll, stopPolling]);

  useEffect(() => {
    const activeJobId = window.localStorage.getItem(ACTIVE_JOB_KEY);
    if (activeJobId) {
      setJob({ status: 'running', job_id: activeJobId, events: [], attempt: 0 });
      startPolling(activeJobId);
    }
    return stopPolling;
  }, [startPolling, stopPolling]);

  useEffect(() => {
    if (!lightboxImage) return;
    const previousFocus = document.activeElement;
    closeLightboxRef.current?.focus();
    const onKeyDown = (event) => {
      if (event.key === 'Escape') setLightboxImage(null);
    };
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('keydown', onKeyDown);
      previousFocus?.focus?.();
    };
  }, [lightboxImage]);

  const handleGenerate = async (e) => {
    e.preventDefault();
    if (!prompt.trim() || submittingRef.current || job?.status === 'running') return;
    submittingRef.current = true;
    setSubmitting(true);
    setError(null); setJob(null); setArtifacts(null); setElapsed(0);
    stopPolling();
    try {
      const res = await fetch(`${API}/jobs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, force }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(apiErrorMessage(body, `API error ${res.status}`));
      }
      const { job_id } = await res.json();
      window.localStorage.setItem(ACTIVE_JOB_KEY, job_id);
      setJob({ status: 'running', job_id, prompt: prompt.trim(), events: [], attempt: 0 });
      startPolling(job_id);
    } catch (err) {
      setError(err.message);
    } finally {
      submittingRef.current = false;
      setSubmitting(false);
    }
  };

  const loadShowcase = async () => {
    setError(null); setJob(null); setArtifacts(null);
    stopPolling();
    setActiveTab('studio');
    try {
      const response = await fetch(`${API}/artifacts/${SHOWCASE_RUN}`);
      if (!response.ok) throw new Error(`Artifact request returned ${response.status}`);
      const a = await response.json();
      setArtifacts(a);
      setJob({ status: 'done', source: 'saved', accepted: runVerdict(a), prompt: a.contract?.prompt, events: [], elapsed: 0 });
      setPrompt(a.contract?.prompt || '');
    } catch {
      setError('Showcase run not found — is the API running?');
    }
  };

  const loadRun = async (runId) => {
    setError(null); setJob(null); setArtifacts(null);
    stopPolling();
    setActiveTab('studio');
    try {
      const response = await fetch(`${API}/artifacts/${runId}`);
      if (!response.ok) throw new Error(`Artifact request returned ${response.status}`);
      const a = await response.json();
      setArtifacts(a);
      setJob({ status: 'done', source: 'saved', accepted: runVerdict(a), prompt: a.contract?.prompt, events: [], elapsed: 0 });
      setPrompt(a.contract?.prompt || '');
    } catch {
      setError(`Run ${runId} artifacts not found.`);
    }
  };

  const running = job && job.status === 'running';
  const done = job && job.status === 'done';
  const attempts = artifacts?.attempts || [];
  const firstImg = attempts.find(a => a.image_url)?.image_url;
  const finalAttempt = bestVisibleAttempt(artifacts);
  const verdict = runVerdict(artifacts);

  const contract = artifacts?.contract;
  const entities = contract?.entities || [];
  const entityCounts = entities.reduce((m, e) => {
    const label = e.attributes?.length ? `${e.attributes.join(' ')} ${e.class_name}` : e.class_name;
    m[label] = (m[label] || 0) + 1;
    return m;
  }, {});

  return (
    <div className="container">
      {lightboxImage && (
        <div className="lightbox" role="dialog" aria-modal="true" aria-label="Image preview" onClick={() => setLightboxImage(null)}>
          <img src={lightboxImage} alt="Fullscreen" onClick={e => e.stopPropagation()} />
          <button ref={closeLightboxRef} type="button" aria-label="Close image preview" className="lightbox-close" onClick={() => setLightboxImage(null)}>×</button>
        </div>
      )}

      <header className="header">
        <h1>FrameGenX</h1>
        <p>Generate → Verify → Repair → Accept. Watch every step happen.</p>
      </header>

      {/* Navigation Tabs */}
      <nav className="nav-tabs">
        <button
          type="button"
          className={`nav-tab-btn ${activeTab === 'studio' ? 'active' : ''}`}
          aria-pressed={activeTab === 'studio'}
          onClick={() => setActiveTab('studio')}
        >
          <Wand2 size={16} /> Generation Studio
        </button>
        <button
          type="button"
          className={`nav-tab-btn ${activeTab === 'ablation' ? 'active' : ''}`}
          aria-pressed={activeTab === 'ablation'}
          onClick={() => setActiveTab('ablation')}
        >
          <BarChart3 size={16} /> Ablation Benchmark
        </button>
        <button
          type="button"
          className={`nav-tab-btn ${activeTab === 'history' ? 'active' : ''}`}
          aria-pressed={activeTab === 'history'}
          onClick={() => setActiveTab('history')}
        >
          <History size={16} /> Run History
        </button>
      </nav>

      {activeTab === 'ablation' && <><AblationDashboard apiBase={API} /><ResearchExperiments apiBase={API} /></>}
      {activeTab === 'history' && <RunHistory apiBase={API} onSelectRun={loadRun} />}

      {activeTab === 'studio' && (
        <>
          <form onSubmit={handleGenerate} className="glass-panel prompt-container">
            <label className="sr-only" htmlFor="framegenx-prompt">Describe the image to generate</label>
            <input
              id="framegenx-prompt"
              aria-describedby="prompt-network-note"
              type="text"
              maxLength={2000}
              className="prompt-input"
              placeholder='Try: "two cats sitting on a wall"'
              value={prompt}
              onChange={e => setPrompt(e.target.value)}
              disabled={running || submitting}
            />
            <button type="submit" className="generate-btn" disabled={running || submitting || !prompt.trim()}>
              {running || submitting ? <><Loader2 className="animate-spin" size={20} /> {running ? `${elapsed}s` : 'Starting…'}</> : <><Play size={20} /> Run pipeline</>}
            </button>
          </form>
          <p id="prompt-network-note" className="dim small-note">
            Prompts are sent to configured AI providers for contract parsing and vision QA. Retrieval may also search the web when local captions are insufficient.
          </p>

          <div className="toolbar">
            <label className="force-toggle">
              <input type="checkbox" checked={force} onChange={e => setForce(e.target.checked)} disabled={running || submitting} />
              fresh run (skip cache)
            </label>
            <button type="button" className="link-btn" onClick={loadShowcase} disabled={running || submitting}>
              <Sparkles size={14} /> load demo run (2 cats, self-repair) — no GPU needed
            </button>
          </div>

          {error && (
            <div className="glass-panel error-panel" role="alert">
              <AlertCircle size={20} /> {error}
            </div>
          )}

      {(running || done) && !job?.cached && !job?.source && (
        <div className="glass-panel card">
          <div className="stage-timeline">
            {STAGES.map(({ key, label, desc, Icon }) => {
              const st = stageState(job, key);
              return (
                <div key={key} className={`stage-chip stage-${st}`}>
                  {st === 'active' ? <Loader2 size={16} className="animate-spin" /> : st === 'done' ? <CheckCircle2 size={16} /> : <Icon size={16} />}
                  <div>
                    <span className="stage-name">{label}</span>
                    <span className="stage-desc">{desc}</span>
                  </div>
                </div>
              );
            })}
          </div>
          {running && (
            <p className="dim" style={{ textAlign: 'center' }}>
              attempt {job.attempt ?? 0} · {elapsed}s elapsed · first run of a session loads models (~1–2 min extra)
            </p>
          )}
        </div>
      )}

      {job?.cached && (
        <div className="glass-panel card cached-note">
          <CheckCircle2 size={18} color="var(--success)" />
          Served from a previous run — inspect its recorded verification below. Check "fresh run" to regenerate.
        </div>
      )}

      {job?.source === 'saved' && (
        <div className="glass-panel card cached-note">
          Viewing saved run evidence. The verdict below comes from its recorded decisions.
        </div>
      )}

      {contract && (
        <div className="glass-panel card">
          <div className="card-header"><FileJson size={20} color="var(--accent)" /> Stage 1: Intent & Contract parsing</div>
          <div className="card-content">
            <p className="dim">Original prompt: <strong className="bright">"{contract.prompt}"</strong></p>
            <div className="chip-row" style={{ marginTop: '0.75rem' }}>
              {Object.entries(entityCounts).map(([label, n]) => (
                <span key={label} className="chip chip-neutral">{n} × {label}</span>
              ))}
              {(contract.constraints || []).filter(c => c.type === 'object_count').map((c, i) => (
                <span key={i} className="chip chip-critical">must count: {c.value} ({c.target})</span>
              ))}
            </div>

          </div>
        </div>
      )}

      {artifacts?.rag_hits?.length > 0 && (
        <div className="glass-panel card">
          <div className="card-header"><Search size={20} color="var(--accent)" /> Stage 2: Visual Knowledge Retrieval (RAG)</div>
          <div className="card-content">
            <p className="mini-title" style={{ marginBottom: '0.75rem' }}>Retrieved captions for inspection</p>
            {artifacts.rag_hits.map((h, i) => (
              <div key={i} className="rag-row" style={{ marginBottom: '0.5rem' }}>
                <span className={`rag-score ${h.retrieval_mode === 'web_search' ? 'web-hit' : 'local-hit'}`} style={{ marginRight: '1rem', minWidth: '40px', display: 'inline-block' }}>
                  {h.retrieval_mode === 'web_search' ? 'WEB' : typeof h.score === 'number' ? `rank ${h.score.toFixed(3)}` : 'LOCAL'}
                </span>
                <span style={{ fontStyle: 'italic', color: '#e2e8f0' }}>"{h.caption}"</span>
              </div>
            ))}
            <p className="dim small-note" style={{ marginTop: '1rem' }}>
              Retrieved captions are shown as evidence. They are not inserted into the generation prompt.
            </p>
          </div>
        </div>
      )}

      {attempts.map((att, i) => (
        <AttemptCard key={att.attempt} att={att} isLast={i === attempts.length - 1} jobDone={done} job={job} setLightboxImage={setLightboxImage} />
      ))}

      {done && attempts.length > 0 && (
        <div className="glass-panel card">
          <div className="card-header" style={{ justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <Gauge size={20} color="var(--accent)" /> Verdict
            </div>
            <span className={`status-badge ${verdict === true ? 'status-success' : 'status-warn'}`}>
              {verdict === true ? 'HARD CHECKS PASSED' : verdict === false ? 'HARD CHECKS FAILED' : 'NO VERDICT RECORDED'}
            </span>
          </div>
          <div className="card-content">
            {attempts.length > 1 && firstImg && finalAttempt?.image_url ? (
              <div className="compare-grid">
                <div>
                  <p className="mini-title">First draft</p>
                  <div className="image-container"><img src={`${API}${firstImg}`} alt="first draft" onClick={() => setLightboxImage(`${API}${firstImg}`)} style={{ cursor: 'pointer' }} /></div>
                </div>
                <div>
                  <p className="mini-title">{verdict === true ? 'Accepted attempt' : 'Best recorded attempt'}</p>
                  <div className="image-container">
                    <img src={`${API}${finalAttempt.image_url}`} alt="final" onClick={() => setLightboxImage(`${API}${finalAttempt.image_url}`)} style={{ cursor: 'pointer' }} />
                  </div>
                </div>
              </div>
            ) : (
              firstImg && (
                <div className="image-container">
                  <img src={`${API}${finalAttempt?.image_url || firstImg}`} alt="result" onClick={() => setLightboxImage(`${API}${finalAttempt?.image_url || firstImg}`)} style={{ cursor: 'pointer' }} />
                </div>
              )
            )}
            {artifacts.reward_history?.length > 1 && (
              <div style={{ marginTop: '1rem' }}>
                <p className="mini-title">Reward by attempt</p>
                <div className="reward-bars">
                  {artifacts.reward_history.map(r => (
                    <div key={r.attempt} className="reward-bar-col" title={r.reward == null ? 'Reward unavailable' : `R=${r.reward.toFixed(3)}`}>
                      <div className="reward-bar" style={{ height: `${(r.reward ?? 0) * 100}%` }} />
                      <span className="dim">#{r.attempt}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            <p className="dim small-note" style={{ marginTop: '1rem' }}>
              run id {artifacts.run_id} · every panel above is a file in runs/{artifacts.run_id}/
              {job.elapsed ? ` · ${Math.round(job.elapsed)}s total` : ''}
            </p>
          </div>
        </div>
      )}
        </>
      )}
    </div>
  );
}

export default App;
