import test from 'node:test';
import assert from 'node:assert/strict';
import { apiErrorMessage, bestVisibleAttempt, isVqaUnavailable, runVerdict } from './runState.js';

test('API contention and validation errors remain readable', () => {
  assert.equal(apiErrorMessage({ detail: { code: 'device_busy', message: 'Another operation owns the device.' } }), 'Another operation owns the device.');
  assert.equal(apiErrorMessage({ detail: [{ msg: 'Prompt is too long', input: 'private prompt' }] }), 'Prompt is too long');
  assert.equal(apiErrorMessage({ detail: {} }, 'API error 503'), 'API error 503');
  assert.equal(apiErrorMessage({ detail: 'Prompt is empty.' }), 'Prompt is empty.');
});

test('a saved run with only rejected decisions cannot be shown as accepted', () => {
  const artifacts = { attempts: [
    { image_url: '/runs/a/generated_0.png', decision: { accept: false, reward: 0.9 }, verification: { hard_failures: [{}] } },
    { image_url: '/runs/a/generated_1.png', decision: { accept: false, reward: 0.2 }, verification: { hard_failures: [] } },
  ] };
  assert.equal(runVerdict(artifacts), false);
  assert.equal(bestVisibleAttempt(artifacts).image_url, '/runs/a/generated_1.png');
});

test('the accepted attempt is displayed even if another attempt has higher reward', () => {
  const artifacts = { attempts: [
    { image_url: '/first.png', decision: { accept: false, reward: 0.99 } },
    { image_url: '/accepted.png', decision: { accept: true, reward: 0.5 } },
  ] };
  assert.equal(runVerdict(artifacts), true);
  assert.equal(bestVisibleAttempt(artifacts).image_url, '/accepted.png');
});

test('missing decisions and images do not imply success', () => {
  assert.equal(runVerdict({ attempts: [{ image_url: '/draft.png' }] }), null);
  assert.equal(bestVisibleAttempt({ attempts: [{ decision: { accept: true } }] }), null);
});

test('vision provider errors are unavailable in new and legacy artifacts', () => {
  assert.equal(isVqaUnavailable({ vqa_status: 'unavailable', vqa_score: null }), true);
  assert.equal(isVqaUnavailable({ vqa_status: 'partial', vqa_score: null }), true);
  assert.equal(isVqaUnavailable({ vqa_score: 0, vqa_details: [
    { answer: 'error' }, { answer: 'error' },
  ] }), true);
  assert.equal(isVqaUnavailable({ vqa_status: 'observed', vqa_score: 0,
    vqa_details: [{ answer: 'no' }] }), false);
});
