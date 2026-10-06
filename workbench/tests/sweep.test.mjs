import test from 'node:test';
import assert from 'node:assert/strict';
import { configFor, simulate } from '../physics.mjs';
import { validateOrbitSweep, runOrbitSweep, toSweepCSV, readOrbitSweep, MAX_SWEEP_RUNS, MAX_SWEEP_STEPS } from '../sweep.mjs';

const close = (actual, expected, tolerance, message = '') =>
  assert.ok(Math.abs(actual - expected) <= tolerance, `${message}: ${actual} versus ${expected} (±${tolerance})`);

test('validateOrbitSweep generates evenly spaced speeds and inclusive endpoints', () => {
  const baseConfig = configFor('orbit', 0); // circular preset
  const range = { minSpeed: 1.3, maxSpeed: 1.5, count: 5 };
  const { speeds, range: validatedRange } = validateOrbitSweep(baseConfig, range);

  assert.equal(speeds.length, 5);
  const expected = [1.3, 1.35, 1.4, 1.45, 1.5];
  for (let i = 0; i < expected.length; i++) {
    close(speeds[i], expected[i], 1e-12, `speed[${i}]`);
  }
  assert.equal(validatedRange.count, 5);
  assert.equal(validatedRange.minSpeed, 1.3);
  assert.equal(validatedRange.maxSpeed, 1.5);
});

test('runOrbitSweep produces correct initial energy, classifications, progress and residual maxima', () => {
  const baseConfig = configFor('orbit', 0);
  const range = { minSpeed: 1.3, maxSpeed: 1.5, count: 5 };
  const progressCalls = [];

  const sweep = runOrbitSweep(baseConfig, range, (completed, total) => {
    progressCalls.push({ completed, total });
  });

  assert.equal(sweep.kind, 'orbit-speed-sweep');
  assert.equal(sweep.sweepSchemaVersion, 1);
  assert.equal(sweep.rows.length, 5);

  // Progress callbacks 1/5 through 5/5
  assert.deepEqual(progressCalls, [
    { completed: 1, total: 5 },
    { completed: 2, total: 5 },
    { completed: 3, total: 5 },
    { completed: 4, total: 5 },
    { completed: 5, total: 5 },
  ]);

  const mu = baseConfig.mu;
  const radius = baseConfig.radius;

  for (let i = 0; i < sweep.rows.length; i++) {
    const row = sweep.rows[i];
    const speed = row.speed;
    const expectedEnergy = (mu / radius) * ((speed ** 2) / 2 - 1);
    close(row.initialEnergyJPerKg, expectedEnergy, 1e-4, `row ${i} initial energy`);

    // Verify residual maxima against direct simulate with sampleLimit 2
    const directSim = simulate({ ...baseConfig, speed }, 2);
    assert.equal(row.status, directSim.diagnostics.status);
    close(row.maxEnergyError, directSim.diagnostics.maxEnergyError, 1e-12, `row ${i} maxEnergyError`);
    close(row.maxMomentumError, directSim.diagnostics.maxMomentumError, 1e-12, `row ${i} maxMomentumError`);
    close(row.finalTimeS, directSim.diagnostics.finalTime, 1e-12, `row ${i} finalTimeS`);
    close(row.finalRadiusKm, directSim.samples.at(-1).signal, 1e-6, `row ${i} finalRadiusKm`);
  }

  // 1.4 is bound (< sqrt(2) ~ 1.4142)
  assert.equal(sweep.rows[2].speed, 1.4);
  assert.equal(sweep.rows[2].classification, 'bound');
  assert.ok(sweep.rows[2].initialEnergyJPerKg < 0);

  // 1.45 is unbound (> sqrt(2))
  assert.equal(sweep.rows[3].speed, 1.45);
  assert.equal(sweep.rows[3].classification, 'unbound');
  assert.ok(sweep.rows[3].initialEnergyJPerKg > 0);
});

test('analytic sqrt(2) speed yields non-negative initial energy (unbound)', () => {
  const baseConfig = configFor('orbit', 0);
  const sqrt2 = Math.SQRT2;
  const range = { minSpeed: sqrt2, maxSpeed: sqrt2 + 0.1, count: 3 };
  const sweep = runOrbitSweep(baseConfig, range);

  const row0 = sweep.rows[0];
  close(row0.speed, sqrt2, 1e-12);
  close(row0.initialEnergyJPerKg, 0, 1e-8);
  assert.equal(row0.classification, 'unbound');
});

test('surface-stop point stops before surface and records surface status', () => {
  // Low speed will plunge into the surface
  const baseConfig = { ...configFor('orbit', 0), duration: 5000, dt: 1 };
  const range = { minSpeed: 0.1, maxSpeed: 0.3, count: 3 };
  const sweep = runOrbitSweep(baseConfig, range);

  const plungingRow = sweep.rows[0];
  assert.equal(plungingRow.status, 'surface');
  // Stopped before surface crossing: finalRadiusKm must be strictly greater than surface/1000
  const surfaceKm = baseConfig.surface / 1000;
  assert.ok(plungingRow.finalRadiusKm > surfaceKm, `${plungingRow.finalRadiusKm} must be > ${surfaceKm}`);
});

test('validateOrbitSweep rejects invalid models, counts, ranges, and work budgets', () => {
  const orbitConfig = configFor('orbit', 0);
  const spinConfig = configFor('spin', 0);

  // Reject non-orbit model
  assert.throws(() => validateOrbitSweep(spinConfig, { minSpeed: 1, maxSpeed: 1.5, count: 5 }), /orbit/i);

  // Reject count out of 3-21 or non-integer
  assert.throws(() => validateOrbitSweep(orbitConfig, { minSpeed: 1, maxSpeed: 1.5, count: 2 }), /count/i);
  assert.throws(() => validateOrbitSweep(orbitConfig, { minSpeed: 1, maxSpeed: 1.5, count: 22 }), /count/i);
  assert.throws(() => validateOrbitSweep(orbitConfig, { minSpeed: 1, maxSpeed: 1.5, count: 5.5 }), /count/i);

  // Reject speed out of 0.1-2.0 bounds or unordered
  assert.throws(() => validateOrbitSweep(orbitConfig, { minSpeed: 0.05, maxSpeed: 1.5, count: 5 }), /speed/i);
  assert.throws(() => validateOrbitSweep(orbitConfig, { minSpeed: 1.2, maxSpeed: 2.1, count: 5 }), /speed/i);
  assert.throws(() => validateOrbitSweep(orbitConfig, { minSpeed: 1.5, maxSpeed: 1.2, count: 5 }), /speed/i);
  assert.throws(() => validateOrbitSweep(orbitConfig, { minSpeed: 1.2, maxSpeed: 1.2, count: 5 }), /speed/i);

  // Reject work budget exceeding MAX_SWEEP_STEPS (200000)
  // duration 100000, dt 0.1 -> 1,000,000 steps per run; with count 3 -> 3,000,000 steps > 200,000
  const heavyConfig = { ...orbitConfig, duration: 100000, dt: 0.1 };
  assert.throws(() => validateOrbitSweep(heavyConfig, { minSpeed: 1.0, maxSpeed: 1.5, count: 3 }), /steps|budget|work/i);
});

test('toSweepCSV produces expected headers and consistent column count', () => {
  const baseConfig = configFor('orbit', 0);
  const range = { minSpeed: 1.3, maxSpeed: 1.5, count: 3 };
  const sweep = runOrbitSweep(baseConfig, range);

  const csv = toSweepCSV(sweep);
  const lines = csv.trim().split('\n');
  assert.equal(lines.length, 4); // 1 header + 3 data rows

  const header = lines[0].split(',');
  assert.deepEqual(header, [
    'speed_ratio',
    'initial_specific_energy_J_per_kg',
    'classification',
    'status',
    'final_radius_km',
    'final_time_s',
    'max_scaled_energy_error',
    'max_scaled_momentum_error'
  ]);

  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(',');
    assert.equal(cols.length, header.length, `row ${i} column count matches header`);
  }
});

test('readOrbitSweep accepts valid sweep and ignores fabricated rows or malicious strings', () => {
  const baseConfig = configFor('orbit', 0);
  const range = { minSpeed: 1.3, maxSpeed: 1.5, count: 5 };
  const validSweep = runOrbitSweep(baseConfig, range);

  // Add forged rows and script-like properties
  const forgedSweep = {
    ...validSweep,
    rows: [
      { speed: 1.4, initialEnergyJPerKg: -999999, classification: '<script>alert(1)</script>', status: 'forged', finalRadiusKm: 0, finalTimeS: 0, maxEnergyError: 0, maxMomentumError: 0 }
    ],
    arbitraryUntrustedData: 'evil',
  };

  const parsed = readOrbitSweep(forgedSweep);
  assert.deepEqual(parsed.range, range);
  assert.equal(parsed.baseConfig.model, 'orbit');
  assert.equal(parsed.baseConfig.radius, baseConfig.radius);
  assert.equal(parsed.rows, undefined, 'forged rows must not be preserved or returned');
});

test('readOrbitSweep rejects foreign schema versions and malformed input', () => {
  const baseConfig = configFor('orbit', 0);
  const range = { minSpeed: 1.3, maxSpeed: 1.5, count: 5 };
  const validSweep = runOrbitSweep(baseConfig, range);

  // Rejects invalid schemaVersion or engineVersion
  assert.throws(() => readOrbitSweep({ ...validSweep, sweepSchemaVersion: 99 }), /schema/i);
  assert.throws(() => readOrbitSweep({ ...validSweep, engineVersion: '0.9.0' }), /engine/i);
  assert.throws(() => readOrbitSweep({ ...validSweep, kind: 'unknown-kind' }), /orbit speed sweep/i);

  // Rejects missing or malformed baseConfig/range
  assert.throws(() => readOrbitSweep({ ...validSweep, baseConfig: null }), /known experiment|config/i);
  assert.throws(() => readOrbitSweep({ ...validSweep, range: { minSpeed: 2, maxSpeed: 1, count: 5 } }), /less than/i);
});
