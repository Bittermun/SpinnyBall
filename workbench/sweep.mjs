/**
 * Pure orchestration and validation for orbital launch speed parameter sweeps.
 * No DOM or Worker globals.
 */
import { ENGINE_VERSION, validateConfig, simulate } from './physics.mjs';

export const SWEEP_SCHEMA_VERSION = 1;
export const MIN_SWEEP_RUNS = 3;
export const MAX_SWEEP_RUNS = 21;
export const MAX_SWEEP_STEPS = 200000;

/**
 * Validates base configuration and sweep range parameters.
 * @param {object} baseConfig Orbit experiment configuration
 * @param {object} range Sweep range { minSpeed, maxSpeed, count }
 * @returns {{ baseConfig: object, range: { minSpeed: number, maxSpeed: number, count: number }, speeds: number[] }}
 */
export function validateOrbitSweep(baseConfig, range) {
  const config = validateConfig(baseConfig);
  if (config.model !== 'orbit') {
    throw new Error('Orbit speed sweep is only available for the orbit laboratory.');
  }

  if (!range || typeof range !== 'object' || Array.isArray(range)) {
    throw new Error('Provide a valid sweep range object.');
  }

  const { minSpeed, maxSpeed, count } = range;
  if (typeof minSpeed !== 'number' || !Number.isFinite(minSpeed) || minSpeed < 0.1 || minSpeed > 2) {
    throw new Error('Minimum speed must be between 0.1 and 2.');
  }
  if (typeof maxSpeed !== 'number' || !Number.isFinite(maxSpeed) || maxSpeed < 0.1 || maxSpeed > 2) {
    throw new Error('Maximum speed must be between 0.1 and 2.');
  }
  if (minSpeed >= maxSpeed) {
    throw new Error('Minimum speed must be strictly less than maximum speed.');
  }

  if (!Number.isInteger(count) || count < MIN_SWEEP_RUNS || count > MAX_SWEEP_RUNS) {
    throw new Error(`Count must be an integer between ${MIN_SWEEP_RUNS} and ${MAX_SWEEP_RUNS}.`);
  }

  const stepsPerRun = Math.ceil(config.duration / config.dt);
  const totalSteps = count * stepsPerRun;
  if (totalSteps > MAX_SWEEP_STEPS) {
    throw new Error(`This sweep requires ${totalSteps.toLocaleString()} steps. Reduce count, increase dt, or shorten duration (limit ${MAX_SWEEP_STEPS.toLocaleString()}).`);
  }

  const speeds = [];
  for (let i = 0; i < count; i++) {
    const s = minSpeed + (maxSpeed - minSpeed) * (i / (count - 1));
    speeds.push(s);
  }

  return {
    baseConfig: config,
    range: { minSpeed, maxSpeed, count },
    speeds,
  };
}

/**
 * Runs a deterministic parameter sweep across launch speeds.
 * @param {object} baseConfig Base orbit experiment configuration
 * @param {object} range Range object { minSpeed, maxSpeed, count }
 * @param {Function} [onProgress] Callback receiving (completed, total)
 * @returns {object} SweepResult
 */
export function runOrbitSweep(baseConfig, range, onProgress = () => {}) {
  const { baseConfig: config, range: validatedRange, speeds } = validateOrbitSweep(baseConfig, range);
  const rows = [];

  for (let i = 0; i < speeds.length; i++) {
    const speed = speeds[i];
    const pointConfig = { ...config, speed };
    const sim = simulate(pointConfig, 2);

    const initialEnergyJPerKg = (config.mu / config.radius) * ((speed ** 2) / 2 - 1);
    const classification = initialEnergyJPerKg < 0 ? 'bound' : 'unbound';
    const finalSample = sim.samples.at(-1);

    rows.push({
      speed,
      initialEnergyJPerKg,
      classification,
      status: sim.diagnostics.status,
      finalRadiusKm: finalSample.signal,
      finalTimeS: sim.diagnostics.finalTime,
      maxEnergyError: sim.diagnostics.maxEnergyError,
      maxMomentumError: sim.diagnostics.maxMomentumError,
    });

    onProgress(i + 1, speeds.length);
  }

  return {
    kind: 'orbit-speed-sweep',
    sweepSchemaVersion: SWEEP_SCHEMA_VERSION,
    engineVersion: ENGINE_VERSION,
    baseConfig: config,
    range: validatedRange,
    rows,
    method: 'Velocity Verlet parameter sweep',
    assumptions: 'Planar test particle in a fixed, spherical lunar-scale gravity field. Each point integrates from the same initial radius with sample limit 2; balance maxima span all integration steps.',
    units: 'speed: dimensionless multiplier of circular speed; initialEnergyJPerKg: J/kg; finalRadiusKm: km; finalTimeS: s; maxEnergyError: fraction of scale; maxMomentumError: fraction of scale',
  };
}

/**
 * Serializes a sweep result into unit-bearing CSV format.
 * @param {object} sweep SweepResult
 * @returns {string} CSV text
 */
export function toSweepCSV(sweep) {
  if (!sweep || !Array.isArray(sweep.rows)) throw new Error('Valid sweep result with rows required.');
  const lines = [[
    'speed_ratio',
    'initial_specific_energy_J_per_kg',
    'classification',
    'status',
    'final_radius_km',
    'final_time_s',
    'max_scaled_energy_error',
    'max_scaled_momentum_error'
  ].join(',')];

  for (const r of sweep.rows) {
    lines.push([
      r.speed,
      r.initialEnergyJPerKg,
      r.classification,
      r.status,
      r.finalRadiusKm,
      r.finalTimeS,
      r.maxEnergyError,
      r.maxMomentumError
    ].join(','));
  }
  return lines.join('\n') + '\n';
}

/**
 * Validates an imported orbit speed sweep payload and extracts trusted configuration and range.
 * Does NOT trust or copy imported rows, status, or diagnostics.
 * @param {object} value Imported JSON payload
 * @returns {{ baseConfig: object, range: { minSpeed: number, maxSpeed: number, count: number } }}
 */
export function readOrbitSweep(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('Choose a valid JSON orbit speed sweep file.');
  }
  if (value.kind !== 'orbit-speed-sweep') {
    throw new Error('Use a SpinnyBall orbit speed sweep exported by engine 1.0.0 / sweep schema 1.');
  }
  if (value.sweepSchemaVersion !== SWEEP_SCHEMA_VERSION || value.engineVersion !== ENGINE_VERSION) {
    throw new Error('Use a SpinnyBall orbit speed sweep exported by engine 1.0.0 / sweep schema 1.');
  }

  const { baseConfig, range } = validateOrbitSweep(value.baseConfig, value.range);
  return { baseConfig, range };
}
