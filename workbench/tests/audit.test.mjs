import test from 'node:test';
import assert from 'node:assert/strict';
import { parseTrajectoryData, sampleOrbitInvariants, auditOrbitTrajectory } from '../audit.mjs';

const close = (actual, expected, tolerance, message = '') =>
  assert.ok(Math.abs(actual - expected) <= tolerance, `${message}: ${actual} versus ${expected} (±${tolerance})`);

test('parses SpinnyBall formatted CSV with standard headers', () => {
  const csv = `time_s,x,y,vx,vy,energy,signal,external_work_J,scaled_energy_balance_error,scaled_momentum_balance_error
0,2000000,0,0,1800,-688500,2000000,0,0,0
10,1999900,18000,-16.2,1799.8,-688500,2000000,0,0,0`;
  const samples = parseTrajectoryData(csv);
  assert.equal(samples.length, 2);
  assert.equal(samples[0].t, 0);
  assert.deepEqual(samples[0].r, [2000000, 0, 0]);
  assert.deepEqual(samples[0].v, [0, 1800, 0]);
});

test('parses generic 3D Cartesian CSV headers (t, x, y, z, vx, vy, vz)', () => {
  const csv = `t,x,y,z,vx,vy,vz
0,7000000,0,0,0,7500,100
1,7000000,7500,100,-8,7499,99.9`;
  const samples = parseTrajectoryData(csv);
  assert.equal(samples.length, 2);
  assert.equal(samples[1].t, 1);
  assert.deepEqual(samples[1].r, [7000000, 7500, 100]);
});

test('parses JSON trajectory format', () => {
  const json = JSON.stringify([
    { t: 0, r: [1000, 0, 0], v: [0, 50, 0] },
    { t: 1, r: [1000, 50, 0], v: [-1, 50, 0] }
  ]);
  const samples = parseTrajectoryData(json);
  assert.equal(samples.length, 2);
  assert.equal(samples[0].t, 0);
});

test('sampleOrbitInvariants calculates energy, angular momentum, and eccentricity vector', () => {
  const mu = 4.905e12;
  const r0 = 2000000;
  const v0 = Math.sqrt(mu / r0); // circular velocity
  const sample = { t: 0, r: [r0, 0, 0], v: [0, v0, 0] };
  const inv = sampleOrbitInvariants(sample, mu);
  
  // Specific energy for circular orbit: -mu / (2*r0)
  close(inv.energy, -mu / (2 * r0), 1e-4, 'circular specific energy');
  // Specific angular momentum magnitude: r0 * v0
  close(inv.hMag, r0 * v0, 1e-4, 'specific angular momentum');
  // Eccentricity for circular orbit: 0
  close(inv.eccentricity, 0, 1e-5, 'circular eccentricity');
});

test('auditOrbitTrajectory passes an exact Keplerian orbit within threshold', () => {
  const mu = 4.905e12;
  const r0 = 2000000;
  const v0 = Math.sqrt(mu / r0);
  const period = 2 * Math.PI * Math.sqrt(r0 ** 3 / mu);
  const steps = 100;
  const dt = period / steps;
  const samples = [];
  
  for (let i = 0; i <= steps; i++) {
    const t = i * dt;
    const theta = (2 * Math.PI * t) / period;
    samples.push({
      t,
      r: [r0 * Math.cos(theta), r0 * Math.sin(theta), 0],
      v: [-v0 * Math.sin(theta), v0 * Math.cos(theta), 0],
    });
  }

  const result = auditOrbitTrajectory(samples, { mu, r_body: 1737400 });
  assert.equal(result.passed, true);
  assert.ok(result.metrics.maxEnergyRelError < 1e-7);
  assert.ok(result.metrics.maxAngMomRelError < 1e-7);
  assert.ok(result.metrics.maxEccentricityDrift < 1e-7);
  assert.equal(result.warnings.length, 0);
});

test('auditOrbitTrajectory fails and warns when energy drifts over 0.1%', () => {
  const mu = 4.905e12;
  const r0 = 2000000;
  const v0 = Math.sqrt(mu / r0);
  const samples = [
    { t: 0, r: [r0, 0, 0], v: [0, v0, 0] },
    // Inject artificial 1% speed boost (which causes >1% energy drift)
    { t: 10, r: [r0, v0 * 10, 0], v: [0, v0 * 1.02, 0] },
  ];

  const result = auditOrbitTrajectory(samples, { mu });
  assert.equal(result.passed, false);
  assert.ok(result.metrics.maxEnergyRelError > 0.001);
  assert.ok(result.warnings.some(w => w.includes('Energy conservation failed')));
});

test('auditOrbitTrajectory handles near-parabolic orbits without division by zero', () => {
  const mu = 4.905e12;
  const r0 = 2000000;
  const vEscape = Math.sqrt((2 * mu) / r0);
  const samples = [
    { t: 0, r: [r0, 0, 0], v: [0, vEscape, 0] },
    { t: 10, r: [r0 + 10, vEscape * 10, 0], v: [0, vEscape, 0] },
  ];

  const result = auditOrbitTrajectory(samples, { mu });
  assert.ok(Number.isFinite(result.metrics.maxEnergyRelError));
});

test('auditOrbitTrajectory detects surface penetration', () => {
  const mu = 4.905e12;
  const r_body = 1737400; // lunar radius
  const samples = [
    { t: 0, r: [1800000, 0, 0], v: [0, 1000, 0] },
    { t: 10, r: [1700000, 0, 0], v: [0, 1000, 0] }, // below surface
  ];

  const result = auditOrbitTrajectory(samples, { mu, r_body });
  assert.equal(result.passed, false);
  assert.ok(result.warnings.some(w => w.includes('surface boundary')));
});

test('CLI verify-external executes and outputs audit results', async () => {
  const { execFile } = await import('node:child_process');
  const { promisify } = await import('node:util');
  const { writeFile, unlink } = await import('node:fs/promises');
  const exec = promisify(execFile);

  const testFile = 'workbench/tests/temp_test_orbit.csv';
  const csv = `time_s,x,y,vx,vy\n0,2000000,0,0,1566.04\n10,2000000,15660.4,-12.2,1566.04`;
  await writeFile(testFile, csv);

  try {
    const { stdout } = await exec('node', ['workbench/verify-external.mjs', testFile, '--mu', '4.905e12', '--json']);
    const parsed = JSON.parse(stdout);
    assert.ok(typeof parsed.passed === 'boolean');
    assert.equal(parsed.sampleCount, 2);
  } finally {
    await unlink(testFile).catch(() => {});
  }
});

