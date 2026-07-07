import React, { useState } from 'react';
import { Sparkles, BrainCircuit, ScanLine, CheckCircle2, Image as ImageIcon, AlertCircle, Play, Loader2 } from 'lucide-react';

function App() {
  const [prompt, setPrompt] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loadingStage, setLoadingStage] = useState(0);

  const STAGES = [
    "Extracting Intent Contract...",
    "Planning Scene Geometry...",
    "Drafting Base Image...",
    "Enforcing Structural Constraints...",
    "Running GroundingDINO QA Verification..."
  ];

  const handleGenerate = async (e) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    setIsLoading(true);
    setError(null);
    setResult(null);

    // Simulate notebook progression visually
    const interval = setInterval(() => {
      setLoadingStage(prev => (prev < 4 ? prev + 1 : prev));
    }, 5000);

    try {
      const response = await fetch("http://localhost:8000/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ prompt })
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.statusText}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      clearInterval(interval);
      setIsLoading(false);
      setLoadingStage(0);
    }
  };

  return (
    <div className="container">
      <header className="header">
        <h1>FrameGenX Studio</h1>
        <p>Enterprise RAG-Augmented Visual Generation & Verification Pipeline</p>
      </header>

      <form onSubmit={handleGenerate} className="glass-panel prompt-container">
        <input 
          type="text" 
          className="prompt-input" 
          placeholder="Enter prompt (e.g., three red apples on a wooden table)"
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          disabled={isLoading}
        />
        <button type="submit" className="generate-btn" disabled={isLoading || !prompt.trim()}>
          {isLoading ? (
            <><Loader2 className="animate-spin" size={20} /> Generating</>
          ) : (
            <><Play size={20} /> Execute Pipeline</>
          )}
        </button>
      </form>

      {error && (
        <div className="glass-panel" style={{ padding: '1rem', color: 'var(--error)', display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <AlertCircle size={20} /> {error}
        </div>
      )}

      {isLoading && (
        <div className="glass-panel card" style={{ alignItems: 'center', padding: '3rem' }}>
          <div className="loading-text">
            <BrainCircuit size={32} />
            <h2>{STAGES[loadingStage]}</h2>
          </div>
          <p style={{ color: 'var(--text-secondary)', marginTop: '1rem' }}>
            Processing via FrameGenX Autonomous Compiler...
          </p>
        </div>
      )}

      {result && (
        <div className="dashboard">
          
          <div className="glass-panel card">
            <div className="card-header">
              <ImageIcon size={20} color="var(--accent)" /> Generated Output
            </div>
            <div className="card-content image-container">
              {result.image_base64 ? (
                <img src={`data:image/png;base64,${result.image_base64}`} alt="Generated result" />
              ) : (
                <p>No image returned.</p>
              )}
            </div>
          </div>

          <div className="glass-panel card">
            <div className="card-header" style={{ justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <ScanLine size={20} color="var(--success)" /> QA Verification Report
              </div>
              <span className={`status-badge ${result.success ? 'status-success' : 'status-error'}`}>
                {result.success ? 'PASSED' : 'FAILED'}
              </span>
            </div>
            <div className="card-content">
              <pre className="json-viewer">
                {JSON.stringify(result.verification_report, null, 2)}
              </pre>
            </div>
          </div>

          <div className="glass-panel card" style={{ gridColumn: '1 / -1' }}>
            <div className="card-header">
              <BrainCircuit size={20} color="var(--accent)" /> Pipeline Metadata
            </div>
            <div className="card-content">
              <div style={{ display: 'flex', gap: '2rem', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                <p><strong>Run ID:</strong> {result.run_id}</p>
                <p><strong>Status:</strong> {result.success ? 'Artifacts Verified' : 'Failed QA constraints'}</p>
              </div>
            </div>
          </div>

        </div>
      )}
    </div>
  );
}

export default App;
