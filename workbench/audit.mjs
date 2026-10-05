/**
 * SpinnyBall Trajectory Invariant Audit Core.
 * Independent verification oracle for external trajectories.
 * Pure ES module, SI units, zero external dependencies.
 */

export const AUDIT_VERSION = '1.0.0';
export const INVARIANT_ERROR_THRESHOLD = 0.001; // 0.1% strict threshold

const norm = v => Math.hypot(...v);
const sub = (a, b) => a.map((val, i) => val - b[i]);
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0]
];
const scale = (v, s) => v.map(x => x * s);

/**
 * Computes specific orbital invariants for a single state vector in a central gravity field (μ).
 * @param {{ t: number, r: number[], v: number[] }} sample
 * @param {number} mu Standard gravitational parameter (m^3/s^2)
 */
export function sampleOrbitInvariants(sample, mu) {
  const r = sample.r.length === 2 ? [...sample.r, 0] : [...sample.r];
  const v = sample.v.length === 2 ? [...sample.v, 0] : [...sample.v];
  const rMag = norm(r);
  const vMag = norm(v);

  if (rMag === 0) throw new Error('Singularity: position vector magnitude is zero.');

  // Specific mechanical energy: E = v^2 / 2 - mu / r
  const energy = 0.5 * (vMag ** 2) - mu / rMag;

  // Specific angular momentum: h = r x v
  const h = cross(r, v);
  const hMag = norm(h);

  // Laplace-Runge-Lenz (eccentricity) vector: e = (v x h) / mu - r / |r|
  const vCrossH = cross(v, h);
  const eVec = sub(scale(vCrossH, 1 / mu), scale(r, 1 / rMag));
  const eccentricity = norm(eVec);

  return { energy, h, hMag, eVec, eccentricity, rMag, vMag };
}

/**
 * Audits a trajectory time-series against two-body Keplerian conservation laws.
 * @param {Array<{ t: number, r: number[], v: number[] }>} samples
 * @param {{ mu: number, r_body?: number }} options
 */
export function auditOrbitTrajectory(samples, options) {
  if (!Array.isArray(samples) || samples.length < 2) {
    throw new Error('Audit requires at least 2 state samples.');
  }
  const { mu, r_body = 0 } = options;
  if (typeof mu !== 'number' || !Number.isFinite(mu) || mu <= 0) {
    throw new Error('Valid gravitational parameter mu > 0 is required.');
  }

  // Baseline invariants at t0
  const initial = sampleOrbitInvariants(samples[0], mu);
  // Energy scale: avoid division by zero near parabolic trajectories (E -> 0)
  const energyScale = Math.max(Math.abs(initial.energy), mu / initial.rMag);
  const momScale = initial.hMag > 0 ? initial.hMag : 1;

  let maxEnergyRelError = 0;
  let maxAngMomRelError = 0;
  let maxEccentricityDrift = 0;
  let maxParasiticAccel = 0;
  let impactDetected = false;
  let minDistance = initial.rMag;

  for (let i = 0; i < samples.length; i++) {
    const s = samples[i];
    const cur = sampleOrbitInvariants(s, mu);
    minDistance = Math.min(minDistance, cur.rMag);

    if (r_body > 0 && cur.rMag <= r_body) {
      impactDetected = true;
    }

    // 1. Specific energy drift scaled by reference potential/kinetic scale
    const dE = Math.abs(cur.energy - initial.energy) / energyScale;
    maxEnergyRelError = Math.max(maxEnergyRelError, dE);

    // 2. Specific angular momentum vector drift (norm of 3D difference)
    const dh = norm(sub(cur.h, initial.h)) / momScale;
    maxAngMomRelError = Math.max(maxAngMomRelError, dh);

    // 3. Laplace-Runge-Lenz eccentricity vector drift
    const de = norm(sub(cur.eVec, initial.eVec));
    maxEccentricityDrift = Math.max(maxEccentricityDrift, de);

    // 4. Parasitic acceleration (non-uniform finite difference between consecutive steps)
    if (i > 0 && i < samples.length - 1) {
      const prev = samples[i - 1];
      const next = samples[i + 1];
      const dtTotal = next.t - prev.t;
      if (dtTotal > 0) {
        const vDiff = sub(
          next.v.length === 2 ? [...next.v, 0] : next.v,
          prev.v.length === 2 ? [...prev.v, 0] : prev.v
        );
        const aNum = scale(vDiff, 1 / dtTotal);
        const r3D = s.r.length === 2 ? [...s.r, 0] : s.r;
        const aGrav = scale(r3D, -mu / (cur.rMag ** 3));
        const aRes = norm(sub(aNum, aGrav));
        maxParasiticAccel = Math.max(maxParasiticAccel, aRes);
      }
    }
  }

  const warnings = [];
  if (maxEnergyRelError > INVARIANT_ERROR_THRESHOLD) {
    warnings.push(`Energy conservation failed (<0.1% threshold): drift is ${(maxEnergyRelError * 100).toFixed(4)}%`);
  }
  if (maxAngMomRelError > INVARIANT_ERROR_THRESHOLD) {
    warnings.push(`Angular momentum conservation failed (<0.1% threshold): drift is ${(maxAngMomRelError * 100).toFixed(4)}%`);
  }
  if (impactDetected) {
    warnings.push(`Trajectory penetrated central body surface boundary (r_body = ${r_body} m, closest approach = ${minDistance.toFixed(1)} m).`);
  }

  return {
    passed: warnings.length === 0,
    metrics: {
      initialEnergy: initial.energy,
      initialEccentricity: initial.eccentricity,
      closestApproach: minDistance,
      maxEnergyRelError,
      maxAngMomRelError,
      maxEccentricityDrift,
      maxParasiticAccel,
    },
    sampleCount: samples.length,
    duration: samples.at(-1).t - samples[0].t,
    warnings,
  };
}

/**
 * Parses CSV or JSON trajectory text into normalized state samples.
 * @param {string} text
 * @returns {Array<{ t: number, r: [number, number, number], v: [number, number, number] }>}
 */
export function parseTrajectoryData(text) {
  const trimmed = text.trim();
  if (trimmed.startsWith('[') || trimmed.startsWith('{')) {
    const parsed = JSON.parse(trimmed);
    const list = Array.isArray(parsed) ? parsed : (parsed.samples || []);
    return list.map(item => {
      const t = item.t ?? item.time ?? item.time_s ?? 0;
      let r, v;
      if (item.r && item.v) {
        r = item.r.length === 2 ? [item.r[0], item.r[1], 0] : [item.r[0], item.r[1], item.r[2]];
        v = item.v.length === 2 ? [item.v[0], item.v[1], 0] : [item.v[0], item.v[1], item.v[2]];
      } else if (item.state && Array.isArray(item.state)) {
        r = [item.state[0], item.state[1], 0];
        v = [item.state[2], item.state[3], 0];
      } else {
        r = [item.x ?? 0, item.y ?? 0, item.z ?? 0];
        v = [item.vx ?? 0, item.vy ?? 0, item.vz ?? 0];
      }
      return { t: Number(t), r, v };
    });
  }

  // Parse CSV
  const lines = trimmed.split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0 && !l.startsWith('#'));
  if (lines.length < 2) throw new Error('CSV must have a header row and at least one data row.');

  const header = lines[0].toLowerCase().split(',').map(h => h.trim());
  const col = name => header.indexOf(name);

  const tIdx = [col('time_s'), col('t'), col('time')].find(i => i !== -1);
  const xIdx = col('x');
  const yIdx = col('y');
  const zIdx = col('z');
  const vxIdx = col('vx');
  const vyIdx = col('vy');
  const vzIdx = col('vz');

  if (xIdx === -1 || yIdx === -1 || vxIdx === -1 || vyIdx === -1) {
    throw new Error('CSV must include at least position (x, y) and velocity (vx, vy) columns.');
  }

  const samples = [];
  for (let i = 1; i < lines.length; i++) {
    const parts = lines[i].split(',').map(p => Number(p.trim()));
    const t = tIdx !== undefined && tIdx !== -1 ? parts[tIdx] : (i - 1);
    const r = [parts[xIdx], parts[yIdx], zIdx !== -1 ? parts[zIdx] : 0];
    const v = [parts[vxIdx], parts[vyIdx], vzIdx !== -1 ? parts[vzIdx] : 0];
    if (r.some(x => !Number.isFinite(x)) || v.some(x => !Number.isFinite(x))) {
      throw new Error(`Non-finite numeric value on CSV line ${i + 1}`);
    }
    samples.push({ t, r, v });
  }

  return samples;
}
