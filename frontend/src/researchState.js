const STATES = new Set(['running', 'completed', 'failed', 'interrupted', 'evidence_error']);
const UUID = /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/;
const rate = value => Number.isFinite(value) && value >= 0 && value <= 1;
const count = value => Number.isInteger(value) && value >= 0;

export function parseResearchList(body) {
  if (body?.schema !== 'framegenx-research-list-v1' || !Array.isArray(body.experiments)
      || body.experiments.length > 25 || typeof body.scan_truncated !== 'boolean') {
    throw new Error('Research evidence has an unsupported format.');
  }
  const ids = new Set();
  for (const run of body.experiments) {
    if (!UUID.test(run?.id) || ids.has(run.id) || !STATES.has(run.execution_status)
        || run.quality_status !== 'unverified') {
      throw new Error('Research evidence has an invalid identity or qualification.');
    }
    ids.add(run.id);
    if (run.execution_status !== 'completed') continue;
    if (run.study_scope !== 'cards_cpu_feasibility_only' || run.full_contract_CSR !== null
        || run.reduction_claim !== false || run.sdxl_activation !== 'blocked'
        || !rate(run.classical_ml?.balanced_accuracy) || !count(run.classical_ml?.test_n)) {
      throw new Error('Research evidence exceeds its declared experimental scope.');
    }
    for (const name of ['baseline', 'candidate']) {
      const arm = run.deep_learning?.[name];
      if (!arm || !count(arm.n) || arm.n < 1 || !rate(arm.valid_rate)
          || !['valid', 'invalid', 'unknown', 'error'].every(status => count(arm.counts?.[status]))
          || Object.values(arm.counts).reduce((sum, value) => sum + value, 0) !== arm.n
          || Math.abs(arm.valid_rate - arm.counts.valid / arm.n) > 1e-9) {
        throw new Error('Research evidence has inconsistent sample counts.');
      }
    }
  }
  return body;
}
