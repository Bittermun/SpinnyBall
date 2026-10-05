import test from 'node:test';
import assert from 'node:assert/strict';
import { configFor, simulate, validateConfig, readExperiment, toCSV, norm, orbitElements, rotate, keplerOrbitPoints } from '../physics.mjs';

const close = (actual, expected, tolerance, message = '') => assert.ok(Math.abs(actual - expected) <= tolerance, `${message}: ${actual} versus ${expected} (±${tolerance})`);
const last = r => r.samples.at(-1);

test('every supplied preset is finite and satisfies the displayed balance threshold', () => {
  for (const model of ['orbit', 'spin', 'exchange']) for (let i = 0; i < 3; i++) {
    const result = simulate(configFor(model, i));
    assert.equal(result.diagnostics.status, 'complete');
    assert.ok(result.diagnostics.maxEnergyError < 0.001);
    assert.ok(result.diagnostics.maxMomentumError < 0.001);
    assert.ok(result.samples.every(s => s.state.every(Number.isFinite)));
  }
});

test('circular Kepler orbit closes at its analytically predicted period', () => {
  const c = configFor('orbit', 0);
  c.duration = 2 * Math.PI * Math.sqrt(c.radius ** 3 / c.mu);
  const r = simulate(c), s = last(r).state;
  assert.ok(Math.hypot(s[0] - c.radius, s[1]) / c.radius < 3e-5);
  assert.ok(r.diagnostics.maxMomentumError < 1e-12);
});

// Independent analytic elliptic solution: Kepler's equation, no numerical ODE stepper.
function keplerPosition(c, t) {
  const { eccentricity: e, semiMajorAxis: a } = orbitElements(c);
  const mean = Math.sqrt(c.mu / a ** 3) * t;
  let E = mean;
  for (let i = 0; i < 15; i++) E -= (E - e * Math.sin(E) - mean) / (1 - e * Math.cos(E));
  return [a * (Math.cos(E) - e), a * Math.sqrt(1 - e * e) * Math.sin(E)];
}

test('elliptical positions agree with Kepler equation across the trajectory', () => {
  const c = configFor('orbit');
  const r = simulate(c);
  for (const sample of r.samples) {
    const exact = keplerPosition(c, sample.t);
    assert.ok(Math.hypot(sample.state[0] - exact[0], sample.state[1] - exact[1]) / c.radius < 0.0001);
  }
});

test('orbital position error converges at second order', () => {
  const c = { ...configFor('orbit'), duration: 5000 };
  const exact = keplerPosition(c, c.duration);
  const error = dt => { const s = last(simulate({ ...c, dt })).state; return Math.hypot(s[0] - exact[0], s[1] - exact[1]); };
  const ratio = error(8) / error(4);
  assert.ok(ratio > 3.8 && ratio < 4.2, `error ratio ${ratio}`);
});

test('escape has positive specific energy and growing distance', () => {
  const r = simulate(configFor('orbit', 2));
  assert.ok(last(r).energy > 0);
  assert.ok(last(r).signal > 4 * r.samples[0].signal);
});

test('surface interception terminates outside the body with an explicit warning', () => {
  const c = { ...configFor('orbit'), speed: 0.5 };
  const r = simulate(c);
  assert.equal(r.diagnostics.status, 'surface');
  assert.ok(r.diagnostics.finalTime < c.duration);
  assert.ok(r.samples.every(s => Math.hypot(...s.state.slice(0, 2)) > c.surface));
  assert.match(r.diagnostics.warnings[0], /Impact/);
});

test('spherical rotor follows exact quaternion rotation', () => {
  const c = { ...configFor('spin'), i1: 2, i2: 2, i3: 2, w1: 0.3, w2: -0.4, w3: 0.5, duration: 20, dt: 0.02 };
  const r = simulate(c), s = last(r).state, w = norm([c.w1, c.w2, c.w3]);
  const exact = [Math.cos(w * c.duration / 2), ...[c.w1, c.w2, c.w3].map(v => v / w * Math.sin(w * c.duration / 2))];
  s.slice(3).forEach((v, i) => close(v, exact[i], 2e-9, 'quaternion'));
  close(s[0], c.w1, 1e-14);
  assert.ok(r.diagnostics.maxMomentumError < 1e-12);
});

test('symmetric rotor agrees with analytic body-frame precession', () => {
  const c = { ...configFor('spin', 2), duration: 40 };
  const s = last(simulate(c)).state;
  const frequency = (c.i3 - c.i1) / c.i1 * c.w3;
  close(s[0], c.w1 * Math.cos(frequency * c.duration), 1e-10);
  close(s[1], c.w1 * Math.sin(frequency * c.duration), 1e-10);
  close(s[2], c.w3, 1e-12);
});

test('RK4 orientation converges at fourth order without artificial energy repair', () => {
  const c = { ...configFor('spin'), i1: 2, i2: 2, i3: 2, w1: 0, w2: 0, w3: 1, duration: 10 };
  const exact = [Math.cos(5), 0, 0, Math.sin(5)];
  const error = dt => norm(last(simulate({ ...c, dt })).state.slice(3).map((v, i) => v - exact[i]));
  const ratio = error(0.04) / error(0.02);
  assert.ok(ratio > 14 && ratio < 18, `error ratio ${ratio}`);
});

test('middle-axis spin reverses while inertial angular momentum stays fixed', () => {
  const r = simulate(configFor('spin', 1));
  assert.ok(Math.min(...r.samples.map(s => s.state[1])) < -0.9);
  assert.ok(r.diagnostics.maxEnergyError < 1e-8);
  assert.ok(r.diagnostics.maxMomentumError < 1e-8);
  for (const s of r.samples) close(norm(s.state.slice(3)), 1, 1e-14);
});

test('zero spin remains at rest and balance errors stay finite', () => {
  const r = simulate({ ...configFor('spin'), w1: 0, w2: 0, w3: 0 });
  assert.deepEqual(last(r).state, [0, 0, 0, 1, 0, 0, 0]);
  assert.equal(r.diagnostics.maxEnergyError, 0);
});

function exactExchange(c, t) {
  const M = c.m1 + c.m2, omega = Math.sqrt(c.k * (1 / c.m1 + 1 / c.m2));
  const equilibriumExtension = c.force / (c.m2 * omega ** 2);
  const separation = c.rest + equilibriumExtension + (c.extension - equilibriumExtension) * Math.cos(omega * t);
  const center = c.drift * t + 0.5 * c.force / M * t * t;
  return [center - c.m2 / M * separation, center + c.m1 / M * separation];
}

test('spring motion matches independent reduced-mass harmonic solution', () => {
  const c = configFor('exchange', 0), r = simulate(c);
  for (const s of r.samples) {
    const exact = exactExchange(c, s.t);
    close(s.state[0], exact[0], 0.0002);
    close(s.state[1], exact[1], 0.0002);
  }
  assert.ok(r.diagnostics.maxComError < 1e-12);
});

test('external force obeys impulse, work-energy and center-of-mass acceleration', () => {
  const c = { ...configFor('exchange', 2), drift: -0.3 }, r = simulate(c);
  const s = last(r), exact = exactExchange(c, s.t);
  close(s.momentum, (c.m1 + c.m2) * c.drift + c.force * s.t, 1e-10);
  close(s.state[0], exact[0], 0.0003);
  close(s.state[1], exact[1], 0.0003);
  assert.ok(r.diagnostics.maxComError < 1e-10);
  assert.ok(r.diagnostics.maxEnergyError < 0.0001);
});

test('equal masses have symmetric recoil and the equilibrium case is stationary', () => {
  const r = simulate({ ...configFor('exchange', 0), m1: 2, m2: 2 });
  r.samples.forEach(s => close(s.state[0], -s.state[1], 1e-12));
  const rest = simulate({ ...configFor('exchange', 0), extension: 0 });
  assert.deepEqual(last(rest).state, rest.initialState);
});

test('final partial steps land exactly on requested time, and sample caps retain endpoints', () => {
  const c = { ...configFor('exchange'), duration: 1.003, dt: 0.007 };
  const r = simulate(c, 20);
  assert.equal(last(r).t, c.duration);
  assert.equal(r.samples[0].t, 0);
  assert.ok(r.samples.length <= 20);
});

test('invalid inputs, unphysical inertias and excessive work are rejected', () => {
  for (const value of [NaN, Infinity, '4', null]) assert.throws(() => validateConfig({ ...configFor('spin'), dt: value }));
  assert.throws(() => validateConfig({ model: '__proto__' }));
  assert.throws(() => validateConfig({ ...configFor('spin'), i3: 4 }), /inertia/);
  assert.throws(() => validateConfig({ ...configFor('orbit'), duration: 200000, dt: 0.1 }), /limit/);
  assert.throws(() => validateConfig({ ...configFor('orbit'), mu: 0 }), /fixed/);
  assert.throws(() => validateConfig({ ...configFor('exchange'), rest: 0 }), /fixed/);
  assert.throws(() => validateConfig({ ...configFor('exchange'), m1: 0.1, m2: 0.1, k: 20, dt: 0.05 }), /step/);
});

test('JSON round trip recomputes identical results and ignores fabricated samples', () => {
  const r = simulate(configFor('orbit'));
  const imported = JSON.parse(JSON.stringify(r));
  imported.samples = [{ energy: 9001 }];
  imported.diagnostics.maxEnergyError = 0;
  assert.deepEqual(simulate(readExperiment(imported)), r);
  assert.throws(() => readExperiment({ ...r, engineVersion: 'unknown' }));
});

test('CSV contains all retained states and deterministic results do not depend on sampling', () => {
  const c = configFor('spin'), a = simulate(c, 30), b = simulate(c, 1000);
  assert.deepEqual(a.diagnostics, b.diagnostics);
  assert.deepEqual(last(a), last(b));
  const csv = toCSV(a).trim().split('\n');
  assert.equal(csv.length, a.samples.length + 1);
  assert.equal(csv[0].split(',').length, csv[1].split(',').length);
  assert.ok(!csv.join('').includes('NaN'));
});

test('quaternion rotation preserves vector length and uses body-to-inertial convention', () => {
  const q = [Math.SQRT1_2, 0, 0, Math.SQRT1_2], v = rotate(q, [1, 0, 0]);
  close(v[0], 0, 1e-15); close(v[1], 1, 1e-15); close(norm(v), 1, 1e-15);
});

test('keplerOrbitPoints generates a closed analytic ellipse matching orbit elements', () => {
  const c = configFor('orbit', 1); // Elliptical preset
  const points = keplerOrbitPoints(c, 60);
  assert.equal(points.length, 61); // Closed loop (0 to 2pi inclusive)
  
  // Starting point at periapsis / initial radius
  close(points[0][0], c.radius, 1e-4, 'start x');
  close(points[0][1], 0, 1e-4, 'start y');
  // Ending point closes the loop
  close(points.at(-1)[0], c.radius, 1e-4, 'end x');
  close(points.at(-1)[1], 0, 1e-4, 'end y');

  // Semi-major axis check
  const { semiMajorAxis, eccentricity } = orbitElements(c);
  // Apoapsis is at index 30 (pi): x = a * (cos(pi) - e) = -a * (1 + e)
  const apoapsisX = points[30][0];
  close(apoapsisX, -semiMajorAxis * (1 + eccentricity), 1e-3, 'apoapsis position');
});


