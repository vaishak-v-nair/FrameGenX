import test from 'node:test';
import assert from 'node:assert/strict';
import { parseResearchList } from './researchState.js';

const id = '12345678-1234-1234-1234-123456789abc';
const arm = { n: 32, valid_rate: 0, counts: { valid: 0, invalid: 31, unknown: 1, error: 0 } };
const fixture = () => ({ schema: 'framegenx-research-list-v1', scan_truncated: false, experiments: [{
  id, execution_status: 'completed', quality_status: 'unverified', study_scope: 'cards_cpu_feasibility_only',
  full_contract_CSR: null, reduction_claim: false, sdxl_activation: 'blocked',
  classical_ml: { balanced_accuracy: .9, test_n: 100 }, deep_learning: { baseline: structuredClone(arm), candidate: structuredClone(arm) },
}] });

test('finished training with no valid images stays experimental with its complete denominator', () => {
  const result = parseResearchList(fixture());
  assert.equal(result.experiments[0].quality_status, 'unverified');
  assert.equal(result.experiments[0].full_contract_CSR, null);
  assert.equal(result.experiments[0].deep_learning.candidate.n, 32);
});

test('fake qualification and generic accuracy claims cannot enter the research view', () => {
  for (const change of [{ quality_status: 'qualified' }, { full_contract_CSR: 1 }, { reduction_claim: true }, { sdxl_activation: 'active' }]) {
    const body = fixture();
    Object.assign(body.experiments[0], change);
    assert.throws(() => parseResearchList(body));
  }
});

test('missing and inconsistent results cannot become a success card', () => {
  const body = fixture();
  delete body.experiments[0].deep_learning.candidate;
  assert.throws(() => parseResearchList(body));
  const other = fixture();
  other.experiments[0].deep_learning.candidate.n = 30;
  assert.throws(() => parseResearchList(other));
});

test('empty, failed, interrupted and corrupt states are readable without invented metrics', () => {
  assert.deepEqual(parseResearchList({ schema: 'framegenx-research-list-v1', scan_truncated: false, experiments: [] }).experiments, []);
  for (const status of ['failed', 'interrupted', 'evidence_error']) {
    const body = { schema: 'framegenx-research-list-v1', scan_truncated: false, experiments: [{ id, execution_status: status, quality_status: 'unverified' }] };
    assert.equal(parseResearchList(body).experiments[0].execution_status, status);
  }
});

test('duplicate identities or malformed payloads are rejected before rendering', () => {
  const body = fixture();
  body.experiments.push(structuredClone(body.experiments[0]));
  assert.throws(() => parseResearchList(body));
  assert.throws(() => parseResearchList({ experiments: [{}] }));
});
