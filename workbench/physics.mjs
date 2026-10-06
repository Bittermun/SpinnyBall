/** SpinnyBall's deterministic SI-unit model core. No DOM, randomness or dependencies. */
export const ENGINE_VERSION = '1.0.0';
export const SCHEMA_VERSION = 1;
export const MAX_STEPS = 100000;
const TAU = 2 * Math.PI;
export const norm = v => Math.hypot(...v);

export const MODELS = {
  orbit: {
    title: 'Orbit laboratory', question: 'How much speed does it take to leave an orbit?',
    description: 'Give a packet a sideways velocity. Gravity writes the rest of the story.',
    method: 'Velocity Verlet',
    defaults: { radius: 2000000, speed: 1.18, mu: 4.905e12, surface: 1737400, duration: 16000, dt: 4 },
    fields: [
      ['radius', 'Starting radius', 'km', 1800, 20000, 50, 1000],
      ['speed', 'Speed / circular speed', '×', 0.1, 2, 0.01, 1],
      ['duration', 'Experiment duration', 's', 100, 200000, 100, 1],
      ['dt', 'Integration step', 's', 0.1, 10, 0.1, 1],
    ],
    presets: [
      { name: 'Circular', detail: 'A reference orbit', values: { speed: 1, duration: 16000 } },
      { name: 'Elliptical', detail: 'More speed, a larger orbit', values: { speed: 1.18, duration: 16000 } },
      { name: 'Escape', detail: 'Above √2 × circular speed', values: { speed: 1.48, duration: 16000 } },
    ],
    assumptions: 'Planar test particle in a fixed, spherical lunar-scale gravity field (μ = 4.905 × 10¹² m³/s²). No Earth, Sun, terrain, drag or thrust. The surface is a spherical stop boundary. This is a two-body approximation, not a lunar mission predictor.',
    challenge: 'Pin the circular run, then raise the speed to 1.18×. Does the packet come back? Try 1.42× and compare the sign of its energy.',
    equation: 'r̈ = −μ r / |r|³',
    reference: 'https://ocw.mit.edu/courses/16-346-astrodynamics-fall-2008/resources/lec_01/',
  },
  spin: {
    title: 'Spin laboratory', question: 'Why does a freely spinning body sometimes flip?',
    description: 'An asymmetric rotor. No torque. A surprisingly unstable middle axis.',
    method: 'RK4 + quaternion orientation',
    defaults: { i1: 1, i2: 2, i3: 2.7, w1: 0.03, w2: 1, w3: 0.01, duration: 40, dt: 0.01 },
    fields: [
      ['i1', 'Principal inertia I₁', 'kg m²', 0.1, 10, 0.1, 1],
      ['i2', 'Principal inertia I₂', 'kg m²', 0.1, 10, 0.1, 1],
      ['i3', 'Principal inertia I₃', 'kg m²', 0.1, 10, 0.1, 1],
      ['w1', 'Initial spin ω₁', 'rad/s', -5, 5, 0.01, 1],
      ['w2', 'Initial spin ω₂', 'rad/s', -5, 5, 0.01, 1],
      ['w3', 'Initial spin ω₃', 'rad/s', -5, 5, 0.01, 1],
      ['duration', 'Experiment duration', 's', 1, 200, 1, 1],
      ['dt', 'Integration step', 's', 0.001, 0.05, 0.001, 1],
    ],
    presets: [
      { name: 'Stable axis', detail: 'Spin about the largest inertia', values: { w1: 0.03, w2: 0.01, w3: 1 } },
      { name: 'Middle axis', detail: 'Watch the rotor tumble', values: { w1: 0.03, w2: 1, w3: 0.01 } },
      { name: 'Symmetric', detail: 'Two equal principal inertias', values: { i1: 2, i2: 2, i3: 3, w1: 0.3, w2: 0, w3: 1 } },
    ],
    assumptions: 'Rigid body in free space, with constant principal inertias and zero external torque. The displayed ellipsoid matches the inertia ratios for a uniform solid; its size is illustrative. No bearings, magnetic fields, structural failure or damping.',
    challenge: 'Compare the stable and middle-axis presets. The body can tumble while energy and angular momentum remain almost constant. Which quantity is actually conserved?',
    equation: 'I ω̇ + ω × (I ω) = 0',
    reference: 'https://mitp-content-server.mit.edu/books/content/sectbyfn/books_pres_0/9579/sicm_edition_2.zip/chapter002.html',
  },
  exchange: {
    title: 'Momentum laboratory', question: 'Can motion inside a system move its center of mass?',
    description: 'Two packets exchange momentum. Account for the whole system.',
    method: 'Velocity Verlet',
    defaults: { m1: 2, m2: 5, k: 3, rest: 4, extension: 1, drift: 0, force: 0, duration: 20, dt: 0.01 },
    fields: [
      ['m1', 'Packet A mass', 'kg', 0.1, 20, 0.1, 1],
      ['m2', 'Packet B mass', 'kg', 0.1, 20, 0.1, 1],
      ['k', 'Spring stiffness', 'N/m', 0.1, 20, 0.1, 1],
      ['extension', 'Initial extension', 'm', 0, 3, 0.1, 1],
      ['drift', 'Initial common velocity', 'm/s', -1, 1, 0.1, 1],
      ['force', 'External force on B', 'N', -2, 2, 0.1, 1],
      ['duration', 'Experiment duration', 's', 1, 100, 1, 1],
      ['dt', 'Integration step', 's', 0.001, 0.05, 0.001, 1],
    ],
    presets: [
      { name: 'Closed system', detail: 'Only internal forces', values: { force: 0, drift: 0 } },
      { name: 'Drifting', detail: 'An unchanged common velocity', values: { force: 0, drift: 0.25 } },
      { name: 'External push', detail: 'Add an external momentum source', values: { force: 0.6, drift: 0 } },
    ],
    assumptions: 'Two point masses on a line, joined by an ideal massless linear spring with 4 m rest length. No damping, contacts or gravity. A constant external force may act on B. Spring separation is signed; particles have no collision geometry. This isolates momentum accounting, not a magnetic bearing model.',
    challenge: 'Change the mass ratio in the closed system. Watch the center of mass. Then add an external push: total momentum must change by force × time.',
    equation: 'dP/dt = Fexternal',
    reference: 'https://ocw.mit.edu/courses/16-07-dynamics-fall-2009/resources/mit16_07f09_lec11/',
  },
};

export function configFor(model, preset = 1) {
  const m = MODELS[model];
  if (!m) throw new Error('Unknown experiment.');
  return { model, ...m.defaults, ...m.presets[preset].values };
}

export function validateConfig(input) {
  if (!input || typeof input !== 'object' || Array.isArray(input) || !Object.hasOwn(MODELS, input.model)) throw new Error('Choose a known experiment.');
  const m = MODELS[input.model];
  const c = { model: input.model };
  for (const [key, fallback] of Object.entries(m.defaults)) {
    const value = input[key] === undefined ? fallback : input[key];
    if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error(`${key} must be a finite number.`);
    c[key] = value;
  }
  for (const [key, label, , min, max, , scale] of m.fields) {
    if (c[key] < min * scale || c[key] > max * scale) throw new Error(`${label} must be between ${min} and ${max} in the displayed units.`);
  }
  const steps = Math.ceil(c.duration / c.dt);
  if (steps > MAX_STEPS) throw new Error(`This needs ${steps.toLocaleString()} steps. Increase the step or shorten the duration (limit ${MAX_STEPS.toLocaleString()}).`);
  if (c.model === 'orbit') {
    if (c.mu !== m.defaults.mu || c.surface !== m.defaults.surface) throw new Error('This model uses the documented fixed lunar gravity and surface constants.');
    if (c.radius <= c.surface) throw new Error('Start outside the central body.');
  } else if (c.model === 'spin') {
    const inertia = [c.i1, c.i2, c.i3];
    if (2 * Math.max(...inertia) >= inertia.reduce((a, b) => a + b)) throw new Error('A solid body needs each inertia smaller than the sum of the other two.');
    const frequency = norm([c.w1, c.w2, c.w3]) * Math.max(...inertia) / Math.min(...inertia);
    if (c.dt * frequency > 0.1) throw new Error('The step is too large for this spin and inertia ratio. Reduce the integration step.');
  } else {
    if (c.rest !== m.defaults.rest) throw new Error('The spring rest length is fixed at 4 m.');
    if (c.dt * Math.sqrt(c.k * (1 / c.m1 + 1 / c.m2)) > 0.1) throw new Error('The step is too large for this spring. Reduce the integration step.');
  }
  return c;
}

export function rotate(q, v) {
  const [w, x, y, z] = q;
  const tx = 2 * (y * v[2] - z * v[1]);
  const ty = 2 * (z * v[0] - x * v[2]);
  const tz = 2 * (x * v[1] - y * v[0]);
  return [v[0] + w * tx + y * tz - z * ty, v[1] + w * ty + z * tx - x * tz, v[2] + w * tz + x * ty - y * tx];
}

function rk4(y, h, f) {
  const add = (a, b, k) => a.map((v, i) => v + b[i] * k);
  const a = f(y), b = f(add(y, a, h / 2)), c = f(add(y, b, h / 2)), d = f(add(y, c, h));
  return y.map((v, i) => v + h / 6 * (a[i] + 2 * b[i] + 2 * c[i] + d[i]));
}

function verlet(y, h, acceleration) {
  const n = y.length / 2;
  const q = y.slice(0, n), v = y.slice(n), a = acceleration(q);
  const next = q.map((x, i) => x + h * v[i] + 0.5 * h * h * a[i]);
  const b = acceleration(next);
  return [...next, ...v.map((x, i) => x + h / 2 * (a[i] + b[i]))];
}

function modelSystem(c) {
  if (c.model === 'orbit') {
    const v = c.speed * Math.sqrt(c.mu / c.radius);
    return {
      initial: [c.radius, 0, 0, v],
      step: (y, h) => verlet(y, h, ([x, z]) => { const r3 = Math.hypot(x, z) ** 3; return [-c.mu * x / r3, -c.mu * z / r3]; }),
      observe: y => ({ energy: (y[2] ** 2 + y[3] ** 2) / 2 - c.mu / Math.hypot(y[0], y[1]), momentum: y[0] * y[3] - y[1] * y[2], signal: Math.hypot(y[0], y[1]) / 1000 }),
      energyScale: c.mu / c.radius,
      momentumScale: c.radius * v,
      signalLabel: 'Distance from center (km)',
      stateNames: ['x_m', 'y_m', 'vx_m_s', 'vy_m_s'],
    };
  }
  if (c.model === 'spin') {
    const I = [c.i1, c.i2, c.i3];
    const f = y => {
      const [a, b, d, w, x, z, u] = y;
      return [(I[1] - I[2]) / I[0] * b * d, (I[2] - I[0]) / I[1] * d * a, (I[0] - I[1]) / I[2] * a * b,
        -0.5 * (x * a + z * b + u * d), 0.5 * (w * a + z * d - u * b), 0.5 * (w * b + u * a - x * d), 0.5 * (w * d + x * b - z * a)];
    };
    const initial = [c.w1, c.w2, c.w3, 1, 0, 0, 0];
    return {
      initial,
      step: (y, h) => { const z = rk4(y, h, f); const qn = norm(z.slice(3)); return [...z.slice(0, 3), ...z.slice(3).map(v => v / qn)]; },
      observe: y => ({ energy: I.reduce((e, a, i) => e + 0.5 * a * y[i] ** 2, 0), momentum: rotate(y.slice(3), I.map((a, i) => a * y[i])), signal: y[1] }),
      energyScale: Math.max(0.5 * I.reduce((e, a, i) => e + a * initial[i] ** 2, 0), 1e-12),
      momentumScale: Math.max(norm(I.map((a, i) => a * initial[i])), 1e-12),
      signalLabel: 'Body-axis spin ω₂ (rad/s)',
      stateNames: ['omega1_rad_s', 'omega2_rad_s', 'omega3_rad_s', 'qw', 'qx', 'qy', 'qz'],
    };
  }
  const M = c.m1 + c.m2, separation = c.rest + c.extension;
  const initial = [-c.m2 / M * separation, c.m1 / M * separation, c.drift, c.drift];
  return {
    initial,
    step: (y, h) => verlet(y, h, ([a, b]) => { const force = c.k * (b - a - c.rest); return [force / c.m1, (-force + c.force) / c.m2]; }),
    observe: y => ({ energy: 0.5 * (c.m1 * y[2] ** 2 + c.m2 * y[3] ** 2 + c.k * (y[1] - y[0] - c.rest) ** 2), momentum: c.m1 * y[2] + c.m2 * y[3], signal: (c.m1 * y[0] + c.m2 * y[1]) / M }),
    energyScale: Math.max(0.5 * c.k * c.extension ** 2 + 0.5 * M * c.drift ** 2, Math.abs(c.force * c.rest), 1),
    momentumScale: Math.max(M * Math.abs(c.drift), Math.sqrt(c.k * M) * c.extension, Math.abs(c.force * c.duration), 1),
    signalLabel: 'Center of mass (m)',
    stateNames: ['xA_m', 'xB_m', 'vA_m_s', 'vB_m_s'],
  };
}

export function simulate(input, sampleLimit = 1200) {
  const config = validateConfig(input), c = config, sys = modelSystem(c);
  if (!Number.isInteger(sampleLimit) || sampleLimit < 2 || sampleLimit > 100001) throw new Error('Sample limit must be an integer from 2 to 100001.');
  let y = [...sys.initial], t = 0;
  const initial = sys.observe(y), samples = [];
  let maxEnergyError = 0, maxMomentumError = 0, maxComError = 0, status = 'complete', steps = 0;
  const totalSteps = Math.ceil(c.duration / c.dt), stride = Math.max(1, Math.ceil(totalSteps / (sampleLimit - 1)));
  const record = (save) => {
    const o = sys.observe(y);
    const work = c.model === 'exchange' ? c.force * (y[1] - sys.initial[1]) : 0;
    const energyError = (o.energy - initial.energy - work) / sys.energyScale;
    const impulse = c.model === 'exchange' ? c.force * t : 0;
    const momentumError = (Array.isArray(o.momentum) ? norm(o.momentum.map((v, i) => v - initial.momentum[i])) : Math.abs(o.momentum - initial.momentum - impulse)) / sys.momentumScale;
    if (!Number.isFinite(energyError) || !Number.isFinite(momentumError)) throw new Error('The integration became non-finite. Reduce the step or revise the parameters.');
    maxEnergyError = Math.max(maxEnergyError, Math.abs(energyError));
    maxMomentumError = Math.max(maxMomentumError, momentumError);
    if (c.model === 'exchange') maxComError = Math.max(maxComError, Math.abs(o.signal - c.drift * t - 0.5 * c.force / (c.m1 + c.m2) * t * t));
    if (save) samples.push({ t, state: [...y], ...o, work, energyError, momentumError });
  };
  record(true);
  for (let i = 1; i <= totalSteps; i++) {
    const nextTime = Math.min(i * c.dt, c.duration), h = nextTime - t;
    const next = sys.step(y, h);
    if (!next.every(Number.isFinite)) throw new Error('Non-finite state. Try a smaller integration step.');
    if (c.model === 'orbit' && Math.hypot(next[0], next[1]) <= c.surface) {
      // Return the last exterior state; no made-up underground trajectory or precise impact time.
      status = 'surface';
      if (samples.at(-1).t !== t) record(true);
      break;
    }
    y = next; t = nextTime; steps = i;
    record(i % stride === 0 || i === totalSteps);
  }
  const warnings = [];
  if (status === 'surface') warnings.push(`Stopped before the surface crossing. Impact lies within the next ${c.dt} s step; no collision model is included.`);
  if (maxEnergyError > 0.001 || maxMomentumError > 0.001) warnings.push('Balance error exceeds 0.1% of its reference scale. Reduce the integration step and check convergence.');
  return { schemaVersion: SCHEMA_VERSION, engineVersion: ENGINE_VERSION, config, method: MODELS[c.model].method, assumptions: MODELS[c.model].assumptions, reference: MODELS[c.model].reference, units: 'SI; signal units specified by signalLabel', stateNames: sys.stateNames, signalLabel: sys.signalLabel, initialState: sys.initial, samples,
    diagnostics: { status, steps, finalTime: t, maxEnergyError, maxMomentumError, maxComError, energyScale: sys.energyScale, momentumScale: sys.momentumScale, warnings } };
}

export function readExperiment(value) {
  if (!value || value.schemaVersion !== SCHEMA_VERSION || value.engineVersion !== ENGINE_VERSION) throw new Error('Use a SpinnyBall experiment exported by engine 1.0.0 / schema 1.');
  // Recompute from parameters: imported samples and diagnostics are never trusted.
  return validateConfig(value.config);
}

export function toCSV(result) {
  const lines = [['time_s', ...result.stateNames, 'energy', 'signal', 'external_work_J', 'scaled_energy_balance_error', 'scaled_momentum_balance_error'].join(',')];
  for (const s of result.samples) lines.push([s.t, ...s.state, s.energy, s.signal, s.work, s.energyError, s.momentumError].join(','));
  return lines.join('\n') + '\n';
}

export function orbitElements(c) {
  const energy = c.mu / c.radius * (c.speed ** 2 / 2 - 1);
  const eccentricity = Math.abs(c.speed ** 2 - 1);
  const a = energy < 0 ? -c.mu / (2 * energy) : null;
  return { energy, eccentricity, semiMajorAxis: a, period: a ? TAU * Math.sqrt(a ** 3 / c.mu) : null };
}

/**
 * Computes exact Keplerian orbit coordinates (x, y) around the central body.
 * For bound orbits (e < 1), returns an array of [x, y] points forming a closed ellipse.
 * @param {object} c Orbit configuration object
 * @param {number} numPoints Number of segments along the ellipse (default 120)
 * @returns {Array<[number, number]>}
 */
export function keplerOrbitPoints(c, numPoints = 120) {
  const { energy, eccentricity: e, semiMajorAxis: a } = orbitElements(c);
  if (energy >= 0 || !a || e >= 1) return []; // Open / unbound trajectory
  
  const b = a * Math.sqrt(1 - e * e);
  const points = [];
  const isSubCircular = c.speed < 1;
  
  for (let i = 0; i <= numPoints; i++) {
    const E = (TAU * i) / numPoints; // Eccentric anomaly from 0 to 2pi
    const x = isSubCircular ? a * (e + Math.cos(E)) : a * (Math.cos(E) - e);
    const y = b * Math.sin(E);
    points.push([x, y]);
  }
  return points;
}


