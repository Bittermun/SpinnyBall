import test from 'node:test';
import assert from 'node:assert/strict';
import { nearestSampleAtTime } from '../plot-data.mjs';

test('nearestSampleAtTime returns null for empty or invalid samples', () => {
  assert.equal(nearestSampleAtTime([], 10), null);
  assert.equal(nearestSampleAtTime(null, 10), null);
});

test('nearestSampleAtTime rejects non-finite requested time', () => {
  const samples = [{ t: 0 }, { t: 1 }];
  assert.equal(nearestSampleAtTime(samples, NaN), null);
  assert.equal(nearestSampleAtTime(samples, Infinity), null);
  assert.equal(nearestSampleAtTime(samples, -Infinity), null);
  assert.equal(nearestSampleAtTime(samples, 'not-a-number'), null);
});

test('nearestSampleAtTime clamps to endpoints when time is beyond bounds', () => {
  const s0 = { t: 10, val: 'first' };
  const s1 = { t: 20, val: 'middle' };
  const s2 = { t: 30, val: 'last' };
  const samples = [s0, s1, s2];

  const before = nearestSampleAtTime(samples, 5);
  assert.equal(before.index, 0);
  assert.equal(before.sample, s0);

  const after = nearestSampleAtTime(samples, 45);
  assert.equal(after.index, 2);
  assert.equal(after.sample, s2);
});

test('nearestSampleAtTime selects exact match or nearest irregular time', () => {
  const samples = [
    { t: 0, val: 0 },
    { t: 1, val: 1 },
    { t: 1.6, val: 2 },
    { t: 4, val: 3 },
  ];

  // Exact match
  const exact = nearestSampleAtTime(samples, 1.6);
  assert.equal(exact.index, 2);
  assert.equal(exact.sample, samples[2]);

  // Target 1.4: |1.4 - 1.0| = 0.4, |1.4 - 1.6| = 0.2 -> chooses 1.6
  const near = nearestSampleAtTime(samples, 1.4);
  assert.equal(near.index, 2);
  assert.equal(near.sample, samples[2]);

  // Target 0.4: |0.4 - 0| = 0.4, |0.4 - 1| = 0.6 -> chooses 0
  const nearZero = nearestSampleAtTime(samples, 0.4);
  assert.equal(nearZero.index, 0);
  assert.equal(nearZero.sample, samples[0]);
});

test('nearestSampleAtTime breaks ties by choosing the earlier sample', () => {
  const s0 = { t: 1 };
  const s1 = { t: 3 };
  const samples = [s0, s1];

  // Target 2 is exactly midway between 1 and 3 (distance 1 to both)
  const tie = nearestSampleAtTime(samples, 2);
  assert.equal(tie.index, 0);
  assert.equal(tie.sample, s0);
});
