import assert from 'node:assert/strict';
import test from 'node:test';
import { setImmediate } from 'node:timers/promises';
import { watchEvaluation } from '../src/evaluationRefresh.js';

test('completed evidence is downloaded once, even after a minute', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  let requests = 0;
  const stop = watchEvaluation(async () => { requests++; }, { poll: false, onError: assert.fail });
  await setImmediate();
  t.mock.timers.tick(60000);
  await setImmediate();
  assert.equal(requests, 1);
  stop();
});

test('slow live downloads never overlap and unmount aborts pending work', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  let finish, signal, requests = 0;
  const stop = watchEvaluation((currentSignal) => {
    requests++;
    signal = currentSignal;
    return new Promise((resolve) => { finish = resolve; });
  }, { poll: true, onError: assert.fail });
  t.mock.timers.tick(60000);
  await setImmediate();
  assert.equal(requests, 1, 'No extra request while the first response is pending');
  finish();
  await setImmediate();
  t.mock.timers.tick(14999);
  assert.equal(requests, 1);
  t.mock.timers.tick(1);
  assert.equal(requests, 2);
  stop();
  assert.equal(signal.aborted, true);
  finish();
  await setImmediate();
  t.mock.timers.tick(60000);
  assert.equal(requests, 2, 'No download after unmount');
});

test('live request failures retry, while cancelled requests do not report errors', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  let requests = 0, errors = 0;
  const stop = watchEvaluation(async (signal) => {
    requests++;
    if (requests > 1) await new Promise((resolve, reject) => signal.addEventListener('abort', () => reject(new Error('aborted'))));
    throw new Error('Connection lost');
  }, { poll: true, onError: () => { errors++; } });
  await setImmediate();
  assert.equal(errors, 1);
  t.mock.timers.tick(15000);
  assert.equal(requests, 2);
  stop();
  await setImmediate();
  assert.equal(errors, 1);
});
