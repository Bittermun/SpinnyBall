import { MODELS, configFor, validateConfig, readExperiment, toCSV, norm, rotate, orbitElements, keplerOrbitPoints } from './physics.mjs';
import { parseTrajectoryData, auditOrbitTrajectory } from './audit.mjs';
import { nearestSampleAtTime } from './plot-data.mjs';
import { validateOrbitSweep } from './sweep.mjs';

const $ = id => document.getElementById(id);
const colors = { green: '#cbe9a2', orange: '#e4a66f', blue: '#94c8ce', muted: '#66827b', cyan: '#5dd8ce' };
let model = 'orbit', draft = configFor(model), result = null, pinned = null, auditedTrajectory = null;

let playing = false, fraction = 0, replaySpeed = 1, previousFrame = 0, requestId = 0;
let busy = false, dirty = false, worker;
let inspectedIndex = null, inspectingPlot = 'signal';
let sweepWorker = null, sweepRequestId = 0, sweepBusy = false, displayedSweep = null;
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
const fmt = (value, digits = 2) => Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits }) : '—';
const small = value => value === 0 ? '0' : Math.abs(value) < 0.001 ? value.toExponential(1) : fmt(value, 3);
const status = (message, error = false) => { $('status').textContent = message; $('status').classList.toggle('error', error); };
const setSweepStatus = (message, error = false) => { if ($('sweepStatus')) { $('sweepStatus').textContent = message; $('sweepStatus').classList.toggle('error', error); } };

function field(item) {
  const [key, label, unit, min, max, step, scale] = item;
  const wrap = document.createElement('div'); wrap.className = 'field';
  const header = document.createElement('div'); header.className = 'field-header';
  const text = document.createElement('label'); text.htmlFor = `param-${key}`; text.textContent = label;
  const units = document.createElement('span'); units.className = 'unit'; units.textContent = unit;
  const input = document.createElement('input');
  // Imported configurations accept every finite value within the model bounds.
  // A suggested step must not make a valid exported experiment unrunnable.
  Object.assign(input, { id: `param-${key}`, name: key, type: 'number', min, max, step: 'any', value: draft[key] / scale, required: true });
  input.addEventListener('input', () => { dirty = true; $('runButton').firstChild.textContent = 'Apply & run '; status('Parameters changed. Run to update the results.'); });
  header.append(text, units); wrap.append(header, input); return wrap;
}

function configure(nextModel, config, presetIndex = 1) {
  model = nextModel; draft = config;
  result = null; fraction = 0; playing = false; dirty = false;
  inspectedIndex = null;
  const m = MODELS[model];
  document.querySelectorAll('[data-model]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.model === model)));
  $('modelNumber').textContent = { orbit: '01', spin: '02', exchange: '03' }[model];
  for (const [id, value] of Object.entries({ modelTitle: m.title, description: m.description, question: m.question, challenge: m.challenge, equation: m.equation, assumptions: m.assumptions })) $(id).textContent = value;
  $('sourceLink').href = m.reference;
  $('preset').replaceChildren(...m.presets.map((p, i) => { const option = document.createElement('option'); option.value = i; option.textContent = `${p.name} · ${p.detail}`; return option; }));
  if (presetIndex === -1) { const custom = document.createElement('option'); custom.value = -1; custom.textContent = 'Imported experiment'; $('preset').append(custom); }
  $('preset').value = presetIndex;
  $('fields').replaceChildren(); $('numericalFields').replaceChildren();
  for (const item of m.fields) (['dt', 'duration'].includes(item[0]) ? $('numericalFields') : $('fields')).append(field(item));
  $('sceneCaption').textContent = { orbit: 'LUNAR-SCALE GRAVITY · INERTIAL FRAME', spin: 'TORQUE-FREE ROTOR · INERTIAL VIEW', exchange: '1D POINT MASSES · INERTIAL FRAME' }[model];
  if ($('sweepPanel')) {
    $('sweepPanel').hidden = model !== 'orbit';
    if (model !== 'orbit') cancelSweep(true);
  }
  updatePin(); run(config, false);
}

function run(config, autoplay = true) {
  let validated;
  try { validated = validateConfig(config); } catch (error) { status(error.message, true); return; }
  busy = true; playing = false; dirty = false;
  status('Calculating the trajectory and checking balances…');
  $('runButton').disabled = true;
  for (const id of ['play', 'pin', 'exportJSON', 'exportCSV']) $(id).disabled = true;
  const id = ++requestId;
  worker.postMessage({ id, config: validated });
  run.autoplay = autoplay && !reducedMotion;
  draft = validated;
}

function receive({ data }) {
  if (data.id !== requestId) return;
  busy = false; $('runButton').disabled = false;
  if (data.error) { status(data.error, true); return; }
  result = data.result; fraction = 0; playing = run.autoplay;
  if (inspectedIndex !== null) {
    if (result.samples?.length) inspectedIndex = Math.min(inspectedIndex, result.samples.length - 1);
    else inspectedIndex = null;
  }
  for (const id of ['play', 'pin', 'exportJSON', 'exportCSV']) $(id).disabled = false;
  $('runButton').firstChild.textContent = dirty ? 'Apply & run ' : 'Run experiment ';
  status(dirty ? 'Parameters changed. Run to update the results.' : `${result.diagnostics.steps.toLocaleString()} steps · ${result.method}`);
  $('signalTitle').textContent = result.signalLabel;
  const d = result.diagnostics, good = d.maxEnergyError <= 0.001 && d.maxMomentumError <= 0.001;
  $('balanceBadge').textContent = good ? 'BALANCE CHECK < 0.1%' : 'CHECK RESOLUTION';
  $('balanceBadge').classList.toggle('warn', !good);
  $('balanceText').textContent = `Max energy ${small(d.maxEnergyError * 100)}% · momentum ${small(d.maxMomentumError * 100)}%`;
  $('warnings').hidden = d.warnings.length === 0; $('warnings').textContent = d.warnings.join(' ');
  updatePin(); updateReadout();
}

function updatePin() {
  const visible = pinned?.config.model === model;
  const legend = model === 'orbit' ? [['Packet trajectory', colors.green], ['Central body', '#a0b7ac']] : model === 'spin' ? [['Body axes', colors.green], ['Angular momentum L', colors.orange]] : [['Packet A', colors.green], ['Packet B', colors.blue], ['Center of mass', colors.orange]];
  if (visible) legend.push([model === 'spin' ? 'Pinned experiment (plots)' : 'Pinned experiment', colors.orange]);
  $('sceneLegend').replaceChildren(...legend.map(([text, color]) => { const line = document.createElement('span'); const dot = document.createElement('i'); dot.className = 'legend-dot'; dot.style.background = color; line.append(dot, document.createTextNode(text)); return line; }));
  $('clearPin').hidden = !pinned;
  $('pinLabel').textContent = pinned ? `${visible ? '●' : 'Stored:'} ${MODELS[pinned.config.model].title.replace(' laboratory', '')} · ${pinned.config.dt} s step` : '';
  $('pinLabel').style.color = visible ? '#a15d2e' : '';
  $('pin').textContent = visible ? 'Replace comparison' : '＋ Pin comparison';
}

function sampleAt(run, t) {
  const samples = run.samples;
  let low = 0, high = samples.length - 1;
  while (low < high) { const mid = Math.ceil((low + high) / 2); if (samples[mid].t <= t) low = mid; else high = mid - 1; }
  return samples[low];
}

function selected() { return result ? sampleAt(result, fraction * result.diagnostics.finalTime) : null; }

function updateReadout() {
  const s = selected(); if (!s) return;
  const c = result.config;
  let values;
  if (model === 'orbit') {
    const elements = orbitElements(c);
    values = [['Distance from center', `${fmt(s.signal, 0)} <small>km</small>`], ['Speed', `${fmt(Math.hypot(s.state[2], s.state[3]) / 1000, 3)} <small>km/s</small>`], ['Specific orbital energy', `${fmt(s.energy / 1e6, 3)} <small>MJ/kg</small>`]];
    $('scene').setAttribute('aria-label', `Orbital trajectory, eccentricity ${fmt(elements.eccentricity, 3)}. Current radius ${fmt(s.signal, 0)} kilometers.`);
  } else if (model === 'spin') {
    values = [['Body-axis spin ω₂', `${fmt(s.state[1], 3)} <small>rad/s</small>`], ['Rotational energy', `${fmt(s.energy, 3)} <small>J</small>`], ['Angular momentum |L|', `${fmt(norm(s.momentum), 3)} <small>kg m²/s</small>`]];
    $('scene').setAttribute('aria-label', `Rotating rigid ellipsoid. Body-axis spin omega 2 is ${fmt(s.state[1], 3)} radians per second.`);
  } else {
    values = [['Center of mass', `${fmt(s.signal, 3)} <small>m</small>`], ['Total momentum', `${fmt(s.momentum, 3)} <small>kg m/s</small>`], ['External impulse', `${fmt(c.force * s.t, 3)} <small>N s</small>`]];
    $('scene').setAttribute('aria-label', `Two masses coupled by a spring. Center of mass at ${fmt(s.signal, 3)} meters; external force ${c.force} newtons.`);
  }
  values.forEach(([label, value], i) => { $(`metricLabel${i + 1}`).textContent = label; $(`metric${i + 1}`).innerHTML = value; });
  $('clock').textContent = `t = ${fmt(s.t, model === 'orbit' ? 0 : 2)} s`;
  $('timeline').value = fraction;
  $('play').textContent = playing ? 'Ⅱ' : fraction >= 1 ? '↻' : '▶';
  $('play').setAttribute('aria-label', playing ? 'Pause simulation' : fraction >= 1 ? 'Replay simulation' : 'Play simulation');
}

function context(id) {
  const canvas = $(id), { width, height } = canvas.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) { canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr); }
  const ctx = canvas.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height);
  return [ctx, width, height];
}
function line(ctx, points, color, width = 1, dashed = false) {
  if (!points.length) return;
  ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dashed ? [4, 5] : []);
  points.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.stroke(); ctx.setLineDash([]);
}
function dot(ctx, x, y, r, color, glow = false) {
  if (glow) { ctx.shadowBlur = 18; ctx.shadowColor = color; }
  ctx.beginPath(); ctx.fillStyle = color; ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill(); ctx.shadowBlur = 0;
}
function label(ctx, text, x, y, color = '#9bb9aa', align = 'left') { ctx.fillStyle = color; ctx.font = '9px Consolas, monospace'; ctx.textAlign = align; ctx.fillText(text, x, y); ctx.textAlign = 'left'; }
function arrow(ctx, from, to, color) {
  line(ctx, [from, to], color, 1.4); const angle = Math.atan2(to[1] - from[1], to[0] - from[0]);
  line(ctx, [[to[0] - 7 * Math.cos(angle - 0.4), to[1] - 7 * Math.sin(angle - 0.4)], to, [to[0] - 7 * Math.cos(angle + 0.4), to[1] - 7 * Math.sin(angle + 0.4)]], color, 1.4);
}

function drawScene() {
  const [ctx, w, h] = context('scene'), s = selected();
  ctx.fillStyle = '#132a2c'; ctx.fillRect(0, 0, w, h);
  // Fixed coordinate grid. It is decoration, never stochastic physical data.
  ctx.strokeStyle = '#234043'; ctx.lineWidth = 0.5;
  for (let x = 0; x < w; x += 38) { ctx.beginPath(); ctx.moveTo(x, 80); ctx.lineTo(x, h - 30); ctx.stroke(); }
  for (let y = 85; y < h - 30; y += 38) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
  if (!s) return;
  if (model === 'orbit') drawOrbit(ctx, w, h, s);
  else if (model === 'spin') drawSpin(ctx, w, h, s);
  else drawExchange(ctx, w, h, s);
}

function drawOrbit(ctx, w, h, s) {
  const comparison = pinned?.config.model === 'orbit' ? pinned : null;
  const all = [...result.samples, ...(comparison?.samples || []), ...(auditedTrajectory?.samples || [])].map(s => s.state || [s.r[0], s.r[1]]);
  const radius = result.config.surface;
  let minX = -radius, maxX = radius, minY = -radius, maxY = radius;
  for (const state of all) { minX = Math.min(minX, state[0]); maxX = Math.max(maxX, state[0]); minY = Math.min(minY, state[1]); maxY = Math.max(maxY, state[1]); }
  const scale = Math.min((w - 95) / (maxX - minX), (h - 132) / (maxY - minY)) * 0.9;
  const cx = (maxX + minX) / 2, cy = (maxY + minY) / 2;
  const project = (x, y) => [w / 2 + (x - cx) * scale, (h + 54) / 2 - (y - cy) * scale];

  // 1. Exact Kepler analytic reference ellipse (dashed faint sage)
  const analyticPoints = keplerOrbitPoints(result.config);
  if (analyticPoints.length) {
    line(ctx, analyticPoints.map(([x, y]) => project(x, y)), '#78978055', 1, true);
  }

  // 2. Numerical trajectory and comparison
  line(ctx, result.samples.map(s => project(s.state[0], s.state[1])), '#54786c', 1);
  if (comparison) line(ctx, comparison.samples.map(s => project(s.state[0], s.state[1])), colors.orange, 1, true);

  // 3. External audited trajectory overlay (vibrant cyan dashed)
  if (auditedTrajectory?.samples) {
    line(ctx, auditedTrajectory.samples.map(s => project(s.r[0], s.r[1])), colors.cyan, 1.4, true);
  }

  line(ctx, result.samples.filter(p => p.t <= s.t).map(p => project(p.state[0], p.state[1])), colors.green, 2);
  const [mx, my] = project(0, 0), r = radius * scale;
  const gradient = ctx.createRadialGradient(mx - r * 0.35, my - r * 0.3, 0, mx, my, r);
  gradient.addColorStop(0, '#9bafa5'); gradient.addColorStop(0.75, '#647e75'); gradient.addColorStop(1, '#3c5752');
  dot(ctx, mx, my, r, gradient);
  ctx.save(); ctx.beginPath(); ctx.arc(mx, my, r, 0, Math.PI * 2); ctx.clip();
  for (const [a, b, k] of [[-.35, -.25, .17], [.4, .25, .23], [-.1, .55, .1], [.3, -.5, .08], [-.6, .38, .12]]) { dot(ctx, mx + a * r, my + b * r, k * r, '#455e5555'); }
  ctx.restore();
  const pos = project(s.state[0], s.state[1]);
  line(ctx, [[mx, my], pos], '#78978066', 1, true);
  dot(ctx, ...pos, 4.5, colors.green, true);
  label(ctx, 'PACKET', pos[0] + 11, pos[1] - 7, colors.green);
  const bar = Math.min(75, w / 5), km = bar / scale / 1000;
  line(ctx, [[w - bar - 25, h - 47], [w - 25, h - 47]], '#90aaa1');
  label(ctx, `${fmt(km, 0)} km`, w - 25, h - 53, '#90aaa1', 'right');
}


function drawSpin(ctx, w, h, s) {
  const c = result.config, q = s.state.slice(3), origin = [w * 0.54, (h + 48) / 2];
  const radii = [Math.sqrt(c.i2 + c.i3 - c.i1), Math.sqrt(c.i1 + c.i3 - c.i2), Math.sqrt(c.i1 + c.i2 - c.i3)];
  const factor = Math.min(w * 0.24, h * 0.29) / Math.max(...radii);
  const project = v => [origin[0] + factor * (v[0] + 0.45 * v[1]), origin[1] - factor * (v[2] + 0.3 * v[1]), v[1] - 0.4 * v[0] - 0.25 * v[2]];
  const faces = [], lat = 12, lon = 24;
  const point = (i, j) => rotate(q, [radii[0] * Math.sin(Math.PI * i / lat) * Math.cos(2 * Math.PI * j / lon), radii[1] * Math.sin(Math.PI * i / lat) * Math.sin(2 * Math.PI * j / lon), radii[2] * Math.cos(Math.PI * i / lat)]);
  for (let i = 0; i < lat; i++) for (let j = 0; j < lon; j++) { const points = [point(i, j), point(i + 1, j), point(i + 1, j + 1), point(i, j + 1)].map(project); faces.push({ points, depth: points.reduce((n, p) => n + p[2], 0) / 4, stripe: j % 6 === 0 }); }
  faces.sort((a, b) => b.depth - a.depth);
  for (const { points, depth, stripe } of faces) {
    const shade = Math.max(0, Math.min(1, 0.5 - depth / Math.max(...radii) * 0.3));
    ctx.beginPath(); points.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.closePath();
    ctx.fillStyle = stripe ? '#c0dc97' : `rgb(${45 + shade * 65},${95 + shade * 75},${79 + shade * 46})`; ctx.fill(); ctx.strokeStyle = '#b6dc9728'; ctx.lineWidth = 0.4; ctx.stroke();
  }
  for (let i = 0; i < 3; i++) { const vector = [0, 0, 0]; vector[i] = radii[i] * 1.45; const end = project(rotate(q, vector)); arrow(ctx, origin, end, [colors.green, colors.blue, '#c9ccc0'][i]); label(ctx, `e${i + 1}`, end[0] + 5, end[1] - 4); }
  if (norm(s.momentum) > 1e-12) {
    const end = project(s.momentum.map(x => x / norm(s.momentum) * Math.max(...radii) * 1.65));
    arrow(ctx, origin, end, colors.orange); label(ctx, 'L', end[0] + 7, end[1], colors.orange);
  }
  label(ctx, 'Orientation from integrated quaternion', w / 2, h - 48, '#94afa3', 'center');
}

function drawExchange(ctx, w, h, s) {
  const c = result.config, y = h * 0.6;
  let lo = 0, hi = 0;
  for (const p of result.samples) { lo = Math.min(lo, p.state[0], p.state[1]); hi = Math.max(hi, p.state[0], p.state[1]); }
  if (pinned?.config.model === 'exchange') for (const p of pinned.samples) { lo = Math.min(lo, p.state[0], p.state[1]); hi = Math.max(hi, p.state[0], p.state[1]); }
  const scale = (w - 100) / Math.max(hi - lo + 2, 6), mid = (lo + hi) / 2;
  const px = x => w / 2 + (x - mid) * scale;
  const a = px(s.state[0]), b = px(s.state[1]), com = px(s.signal);
  line(ctx, [[35, y + 40], [w - 25, y + 40]], '#678578', 1);
  const tick = 10 ** Math.floor(Math.log10(Math.max(hi - lo, 1) / 5));
  for (let x = Math.ceil((lo - 1) / tick) * tick; x <= hi + 1; x += tick) { if ((hi - lo) / tick > 30) break; line(ctx, [[px(x), y + 37], [px(x), y + 43]], '#678578'); label(ctx, fmt(x, tick < 1 ? 1 : 0), px(x), y + 57, '#91aa9c', 'center'); }
  const coil = [[a, y]];
  for (let i = 1; i < 36; i++) coil.push([a + (b - a) * i / 36, y + (i % 2 ? 7 : -7)]);
  coil.push([b, y]); line(ctx, coil, '#83a69a', 1.2);
  dot(ctx, a, y, 10 + Math.sqrt(c.m1) * 2, colors.green); dot(ctx, b, y, 10 + Math.sqrt(c.m2) * 2, colors.blue);
  label(ctx, `A · ${c.m1} kg`, a, y - 28, colors.green, 'center'); label(ctx, `B · ${c.m2} kg`, b, y - 28, colors.blue, 'center');
  line(ctx, [[com, y - 70], [com, y + 32]], colors.orange, 1, true);
  dot(ctx, com, y + 40, 3.5, colors.orange); label(ctx, 'COM', com, y - 80, colors.orange, 'center');
  if (c.force !== 0) { const end = b + Math.sign(c.force) * 45; arrow(ctx, [b, y - 50], [end, y - 50], colors.orange); label(ctx, `${c.force} N external`, b, y - 63, colors.orange, 'center'); }
  if (pinned?.config.model === 'exchange' && s.t <= pinned.diagnostics.finalTime) { const p = sampleAt(pinned, s.t); for (const x of p.state.slice(0, 2)) { ctx.beginPath(); ctx.strokeStyle = colors.orange; ctx.setLineDash([3, 3]); ctx.arc(px(x), y, 18, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]); } }
}

function drawPlot(id, key) {
  const [ctx, w, h] = context(id); if (!result) return;
  const isEnergy = key === 'energyError', series = [result];
  if (pinned?.config.model === model) series.push(pinned);
  const value = s => s[key] * (isEnergy ? 100 : 1);
  let lo = Infinity, hi = -Infinity;
  for (const run of series) for (const s of run.samples) { const v = value(s); lo = Math.min(lo, v); hi = Math.max(hi, v); }
  if (isEnergy) { lo = Math.min(0, lo); hi = Math.max(0, hi); }
  if (hi - lo < 1e-10) { lo -= isEnergy ? 1e-6 : 0.01; hi += isEnergy ? 1e-6 : 0.01; }
  const pad = (hi - lo) * 0.12; lo -= pad; hi += pad;
  const left = 54, right = w - 7, top = 9, bottom = h - 19;
  const timeMax = Math.max(...series.map(r => r.diagnostics.finalTime), 1e-9);
  const point = s => [left + s.t / timeMax * (right - left), bottom - (value(s) - lo) / (hi - lo) * (bottom - top)];
  for (let i = 0; i <= 2; i++) {
    const y = top + (bottom - top) * i / 2;
    line(ctx, [[left, y], [right, y]], '#d9dfd0', 0.7);
    const number = hi - i / 2 * (hi - lo); label(ctx, Math.abs(number) > 100 ? fmt(number, 0) : small(number), left - 7, y + 3, '#75816e', 'right');
  }
  series.forEach((run, i) => line(ctx, run.samples.map(point), i ? '#bd8150' : '#416e4e', 1.4, i === 1));
  const s = selected(), p = point(s);
  line(ctx, [[p[0], top], [p[0], bottom]], '#b8c4b1', 1);
  dot(ctx, ...p, 2.5, '#3e6a49');

  if (inspectedIndex !== null && result.samples[inspectedIndex]) {
    const isamp = result.samples[inspectedIndex];
    const ip = point(isamp);
    line(ctx, [[ip[0], top], [ip[0], bottom]], '#d77845', 1, true);
    dot(ctx, ...ip, 4, '#d77845');
    dot(ctx, ...ip, 2, '#f5f3ec');
    if (pinned?.config.model === model && pinned.samples?.length) {
      const pinHit = nearestSampleAtTime(pinned.samples, isamp.t);
      if (pinHit) {
        const pp = point(pinHit.sample);
        dot(ctx, ...pp, 3.5, '#bd8150');
      }
    }
  }

  label(ctx, '0', left, h - 4, '#75816e'); label(ctx, fmt(timeMax, timeMax > 100 ? 0 : 1), right, h - 4, '#75816e', 'right');
  const baseAria = `${isEnergy ? 'Energy balance error in percent of scale' : result.signalLabel}. Minimum ${small(Math.min(...result.samples.map(value)))}, maximum ${small(Math.max(...result.samples.map(value)))}. Time 0 to ${result.diagnostics.finalTime} seconds. ${series.length > 1 ? 'Dashed amber line is the pinned run.' : ''}`;
  const inspectAria = inspectedIndex !== null && result.samples[inspectedIndex]
    ? ` Inspected sample ${inspectedIndex + 1} of ${result.samples.length}: time ${fmt(result.samples[inspectedIndex].t, 2)} s, value ${small(value(result.samples[inspectedIndex]))}.`
    : '';
  $(id).setAttribute('aria-label', baseAria + inspectAria);
}

function updatePlotInspection() {
  const el = $('plotInspection');
  if (!el) return;
  if (!result || !result.samples?.length || inspectedIndex === null || inspectedIndex < 0 || inspectedIndex >= result.samples.length) {
    el.textContent = 'Inspect sample: hover or focus a plot (arrow keys step, Home/End, Esc to clear).';
    return;
  }
  const s = result.samples[inspectedIndex];
  const isEnergy = inspectingPlot === 'energyError';
  const label = isEnergy ? 'Energy balance error' : result.signalLabel;
  const val = isEnergy ? `${small(s.energyError * 100)}%` : small(s.signal);
  let text = `Sample ${inspectedIndex + 1}/${result.samples.length} · t = ${fmt(s.t, model === 'orbit' ? 1 : 2)} s · ${label}: ${val}`;
  if (pinned && pinned.config.model === model && pinned.samples?.length) {
    const pinHit = nearestSampleAtTime(pinned.samples, s.t);
    if (pinHit) {
      const pinS = pinHit.sample;
      const pinVal = isEnergy ? `${small(pinS.energyError * 100)}%` : small(pinS.signal);
      text += ` | Pinned: t = ${fmt(pinS.t, model === 'orbit' ? 1 : 2)} s · ${label}: ${pinVal} (nearest sample)`;
    }
  }
  el.textContent = text;
}

let needsDraw = true;
function frame(now) {
  const elapsed = Math.min((now - previousFrame) / 1000, 0.1); previousFrame = now;
  if (playing && result && !busy) {
    fraction = Math.min(1, fraction + elapsed / 24 * replaySpeed);
    if (fraction === 1) playing = false;
    needsDraw = true;
  }
  if (needsDraw) {
    drawScene();
    drawPlot('signalPlot', 'signal');
    drawPlot('energyPlot', 'energyError');
    updateReadout();
    updatePlotInspection();
    needsDraw = false;
  }
  requestAnimationFrame(frame);
}

function download(filename, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const a = document.createElement('a'); a.href = url; a.download = filename; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function initSweepWorker() {
  sweepWorker = new Worker(new URL('./worker.mjs', import.meta.url), { type: 'module' });
  sweepWorker.addEventListener('message', ({ data }) => handleSweepWorkerMessage(data));
  sweepWorker.addEventListener('error', () => {
    sweepBusy = false;
    $('runSweepButton').disabled = false;
    $('cancelSweepButton').hidden = true;
    setSweepStatus('The sweep worker encountered an error.', true);
  });
}

function handleSweepWorkerMessage(data) {
  if (data.id !== sweepRequestId) return;
  if (data.kind === 'sweep-progress') {
    setSweepStatus(`Running sweep: completed ${data.completed} of ${data.total} points…`);
    return;
  }
  if (data.kind === 'sweep-result') {
    sweepBusy = false;
    $('runSweepButton').disabled = false;
    $('cancelSweepButton').hidden = true;
    if (data.error) {
      setSweepStatus(`Sweep error: ${data.error}`, true);
      return;
    }
    displayedSweep = data.result;
    renderSweep(data.result);
    setSweepStatus(`Sweep complete: ${data.result.rows.length} points calculated.`);
    $('exportSweepJSON').disabled = false;
    $('exportSweepCSV').disabled = false;
  }
}

function cancelSweep(quiet = false) {
  if (sweepBusy) {
    sweepWorker?.terminate();
    sweepBusy = false;
    sweepRequestId++;
    initSweepWorker();
    $('runSweepButton').disabled = false;
    $('cancelSweepButton').hidden = true;
    if (!quiet) setSweepStatus('Sweep cancelled.');
  }
}

function startSweep() {
  if (sweepBusy) return;
  if (dirty) {
    setSweepStatus('Apply & run parameter changes first before starting a sweep.', true);
    return;
  }
  const baseConfig = result ? result.config : draft;
  const minSpeed = $('sweepMinSpeed').valueAsNumber;
  const maxSpeed = $('sweepMaxSpeed').valueAsNumber;
  const count = $('sweepCount').valueAsNumber;
  const range = { minSpeed, maxSpeed, count };
  try {
    validateOrbitSweep(baseConfig, range);
  } catch (err) {
    setSweepStatus(err.message, true);
    return;
  }
  sweepBusy = true;
  $('runSweepButton').disabled = true;
  $('cancelSweepButton').hidden = false;
  $('exportSweepJSON').disabled = true;
  $('exportSweepCSV').disabled = true;
  setSweepStatus(`Starting sweep: 0 of ${count} points…`);
  const id = ++sweepRequestId;
  sweepWorker.postMessage({ id, kind: 'orbit-speed-sweep', config: baseConfig, range });
}

function drawSweepPlot(sweep) {
  if (!sweep || !sweep.rows?.length) return;
  const [ctx, w, h] = context('sweepPlot');
  const rows = sweep.rows;
  const left = 65, right = w - 15, top = 20, bottom = h - 25;
  const speeds = rows.map(r => r.speed);
  const radii = rows.map(r => r.finalRadiusKm);
  const minSpeed = Math.min(...speeds);
  const maxSpeed = Math.max(...speeds);
  let minRadius = Math.min(...radii);
  let maxRadius = Math.max(...radii);
  if (maxRadius - minRadius < 1e-4) {
    minRadius -= 10;
    maxRadius += 10;
  }
  const padR = (maxRadius - minRadius) * 0.12;
  minRadius -= padR;
  maxRadius += padR;

  const speedX = s => left + ((s - minSpeed) / (maxSpeed - minSpeed || 1)) * (right - left);
  const radiusY = r => bottom - ((r - minRadius) / (maxRadius - minRadius || 1)) * (bottom - top);

  for (let i = 0; i <= 2; i++) {
    const y = top + (bottom - top) * i / 2;
    line(ctx, [[left, y], [right, y]], '#d9dfd0', 0.7);
    const val = maxRadius - (i / 2) * (maxRadius - minRadius);
    label(ctx, fmt(val, val > 100 ? 0 : 1) + ' km', left - 7, y + 3, '#75816e', 'right');
  }

  line(ctx, [[left, bottom], [right, bottom]], '#d9dfd0', 0.7);
  label(ctx, `${fmt(minSpeed, 2)}×`, left, h - 6, '#75816e');
  label(ctx, `${fmt(maxSpeed, 2)}×`, right, h - 6, '#75816e', 'right');
  label(ctx, 'Speed (× circular)', (left + right) / 2, h - 6, '#75816e', 'center');

  const sqrt2 = Math.SQRT2;
  if (sqrt2 >= minSpeed && sqrt2 <= maxSpeed) {
    const xRef = speedX(sqrt2);
    line(ctx, [[xRef, top], [xRef, bottom]], '#bd8150', 1, true);
    label(ctx, '√2', xRef, top - 6, '#bd8150', 'center');
  }

  const curvePoints = rows.map(r => [speedX(r.speed), radiusY(r.finalRadiusKm)]);
  line(ctx, curvePoints, '#416e4e', 1.5);

  rows.forEach(r => {
    const x = speedX(r.speed);
    const y = radiusY(r.finalRadiusKm);
    if (r.status === 'surface') {
      line(ctx, [[x - 4, y - 4], [x + 4, y + 4]], '#9e4928', 1.5);
      line(ctx, [[x - 4, y + 4], [x + 4, y - 4]], '#9e4928', 1.5);
    } else if (r.classification === 'unbound') {
      dot(ctx, x, y, 3.5, '#bd8150');
    } else {
      dot(ctx, x, y, 3.5, '#416e4e');
    }
  });

  $('sweepPlot').setAttribute('aria-label', `Orbital speed sweep chart. ${rows.length} points from speed ${fmt(minSpeed, 2)} to ${fmt(maxSpeed, 2)} times circular speed. Final distance from center ${fmt(minRadius + padR, 0)} to ${fmt(maxRadius - padR, 0)} km. Vertical dashed line at square root of 2 marks bound and unbound threshold.`);
}

function renderSweep(sweep) {
  if (!sweep) return;
  $('sweepOutput').hidden = false;
  drawSweepPlot(sweep);
  const tbody = $('sweepTableBody');
  tbody.replaceChildren();
  sweep.rows.forEach(row => {
    const tr = document.createElement('tr');

    const tdAction = document.createElement('td');
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = 'Select & run';
    btn.addEventListener('click', () => {
      $('param-speed').value = row.speed;
      draft.speed = row.speed;
      dirty = false;
      $('runButton').firstChild.textContent = 'Run experiment ';
      run({ ...draft, speed: row.speed });
    });
    tdAction.append(btn);

    const tdSpeed = document.createElement('td');
    tdSpeed.textContent = fmt(row.speed, 3);

    const tdEnergy = document.createElement('td');
    tdEnergy.textContent = fmt(row.initialEnergyJPerKg, 1);

    const tdClass = document.createElement('td');
    const badge = document.createElement('span');
    badge.className = `badge ${row.classification}`;
    badge.textContent = row.classification;
    tdClass.append(badge);

    const tdStatus = document.createElement('td');
    tdStatus.textContent = row.status === 'surface' ? 'stopped before surface crossing' : 'complete';

    const tdRadius = document.createElement('td');
    tdRadius.textContent = fmt(row.finalRadiusKm, 1);

    const tdTime = document.createElement('td');
    tdTime.textContent = fmt(row.finalTimeS, 1);

    const tdEnergyErr = document.createElement('td');
    tdEnergyErr.textContent = `${small(row.maxEnergyError * 100)}%`;

    const tdMomErr = document.createElement('td');
    tdMomErr.textContent = `${small(row.maxMomentumError * 100)}%`;

    tr.append(tdAction, tdSpeed, tdEnergy, tdClass, tdStatus, tdRadius, tdTime, tdEnergyErr, tdMomErr);
    tbody.append(tr);
  });
}

try {
  worker = new Worker(new URL('./worker.mjs', import.meta.url), { type: 'module' });
  worker.addEventListener('message', event => { receive(event); needsDraw = true; });
  worker.addEventListener('error', () => { busy = false; $('runButton').disabled = false; status('The simulation worker could not start. Launch with python -m workbench and use the localhost URL.', true); });
  initSweepWorker();

  if ($('sweepForm')) {
    $('sweepForm').addEventListener('submit', event => {
      event.preventDefault();
      startSweep();
    });
  }
  if ($('cancelSweepButton')) {
    $('cancelSweepButton').addEventListener('click', () => cancelSweep(false));
  }
  $('parameters').addEventListener('submit', event => {
    event.preventDefault();
    const config = { ...draft };
    for (const [key, , , , , , scale] of MODELS[model].fields) config[key] = $(`param-${key}`).valueAsNumber * scale;
    run(config);
  });
  // Invalid hidden numerical fields must become visible before native focus.
  $('parameters').addEventListener('invalid', event => {
    const details = event.target.closest('details');
    if (details) details.open = true;
    status('Check the highlighted parameter and its allowed range.', true);
  }, true);
  document.querySelectorAll('[data-model]').forEach(button => button.addEventListener('click', () => { const m = button.dataset.model; configure(m, configFor(m, m === 'exchange' ? 0 : 1), m === 'exchange' ? 0 : 1); needsDraw = true; }));
  $('preset').addEventListener('change', event => { const i = Number(event.target.value); configure(model, configFor(model, i), i); needsDraw = true; });
  $('play').addEventListener('click', () => { if (!result || busy) return; if (fraction >= 1) fraction = 0; playing = !playing; needsDraw = true; });
  $('reset').addEventListener('click', () => { fraction = 0; playing = false; needsDraw = true; });
  $('timeline').addEventListener('input', event => { fraction = Number(event.target.value); playing = false; needsDraw = true; });
  $('speed').addEventListener('change', event => { replaySpeed = Number(event.target.value); });
  $('pin').addEventListener('click', () => { if (!result) return; pinned = result; updatePin(); needsDraw = true; status('Current result pinned. Change a parameter or scenario, then compare.'); });
  $('clearPin').addEventListener('click', () => { pinned = null; updatePin(); needsDraw = true; });
  $('exportJSON').addEventListener('click', () => { if (!result) return; download(`spinnyball-${model}.json`, JSON.stringify({ ...result, exportedAt: new Date().toISOString() }, null, 2), 'application/json'); if (dirty) status('Saved the displayed run. Unapplied parameter edits are not included.'); });
  $('exportCSV').addEventListener('click', () => { if (result) download(`spinnyball-${model}.csv`, toCSV(result), 'text/csv'); });
  $('importButton').addEventListener('click', () => $('importFile').click());
  $('importFile').addEventListener('change', async event => {
    const file = event.target.files[0]; if (!file) return;
    try { if (file.size > 8 * 1024 * 1024) throw new Error('Choose an experiment file smaller than 8 MB.'); const config = readExperiment(JSON.parse(await file.text())); configure(config.model, config, -1); }
    catch (error) { status(`Import failed: ${error.message}`, true); }
    finally { event.target.value = ''; }
  });

  ['signalPlot', 'energyPlot'].forEach(id => {
    const key = id === 'energyPlot' ? 'energyError' : 'signal';
    const canvas = $(id);
    const handlePointer = e => {
      if (!result || !result.samples?.length) return;
      const rect = canvas.getBoundingClientRect();
      const left = 54, right = rect.width - 7;
      const series = [result];
      if (pinned?.config.model === model) series.push(pinned);
      const timeMax = Math.max(...series.map(r => r.diagnostics.finalTime), 1e-9);
      const frac = Math.max(0, Math.min(1, (e.clientX - rect.left - left) / (right - left)));
      const hit = nearestSampleAtTime(result.samples, frac * timeMax);
      if (hit) {
        inspectedIndex = hit.index;
        inspectingPlot = key;
        needsDraw = true;
      }
    };
    canvas.addEventListener('pointermove', e => {
      if (e.buttons === 0 || e.buttons === 1) handlePointer(e);
    });
    canvas.addEventListener('pointerdown', handlePointer);
    canvas.addEventListener('focus', () => {
      inspectingPlot = key;
      if (inspectedIndex === null && result?.samples?.length) {
        const s = selected();
        const hit = nearestSampleAtTime(result.samples, s ? s.t : 0);
        inspectedIndex = hit ? hit.index : 0;
      }
      needsDraw = true;
    });
    canvas.addEventListener('keydown', e => {
      if (!result || !result.samples?.length) return;
      if (inspectedIndex === null) inspectedIndex = 0;
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        inspectedIndex = Math.max(0, inspectedIndex - 1);
        inspectingPlot = key;
        needsDraw = true;
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        inspectedIndex = Math.min(result.samples.length - 1, inspectedIndex + 1);
        inspectingPlot = key;
        needsDraw = true;
      } else if (e.key === 'Home') {
        e.preventDefault();
        inspectedIndex = 0;
        inspectingPlot = key;
        needsDraw = true;
      } else if (e.key === 'End') {
        e.preventDefault();
        inspectedIndex = result.samples.length - 1;
        inspectingPlot = key;
        needsDraw = true;
      } else if (e.key === 'Escape') {
        e.preventDefault();
        inspectedIndex = null;
        needsDraw = true;
      }
    });
  });

  // External Trajectory Audit Handler
  async function handleAuditFile(file) {
    if (!file) return;
    try {
      if (file.size > 8 * 1024 * 1024) throw new Error('Choose an audit file smaller than 8 MB.');
      const text = await file.text();
      const samples = parseTrajectoryData(text);
      const mu = result?.config.mu || 4.905e12;
      const r_body = result?.config.surface || 0;
      const audit = auditOrbitTrajectory(samples, { mu, r_body });
      auditedTrajectory = { file: file.name, samples, audit };
      
      const badgeClass = audit.passed ? 'audit-pass' : 'audit-fail';
      const badgeText = audit.passed ? '✓ PASSED (< 0.1% drift)' : '✗ FAILED (invariants exceeded threshold)';
      const badge = document.createElement('span');
      badge.className = `audit-badge ${badgeClass}`;
      badge.textContent = badgeText;

      const fileP = document.createElement('p');
      fileP.className = 'muted';
      fileP.append('File: ');
      const bold = document.createElement('b');
      bold.textContent = file.name;
      fileP.append(bold, ` (${audit.sampleCount} samples, ${audit.duration.toFixed(1)} s duration, μ = ${mu.toExponential(3)} m³/s²)`);

      const table = document.createElement('table');
      table.className = 'audit-table';
      table.innerHTML = `
        <tr><th>Invariant / Diagnostic</th><th>Value</th><th>Status</th></tr>
        <tr><td>Max Energy Error</td><td>${(audit.metrics.maxEnergyRelError * 100).toFixed(4)}%</td><td>${audit.metrics.maxEnergyRelError <= 0.001 ? '✓ &lt; 0.1%' : '⚠ Exceeded'}</td></tr>
        <tr><td>Max Angular Momentum Error</td><td>${(audit.metrics.maxAngMomRelError * 100).toFixed(4)}%</td><td>${audit.metrics.maxAngMomRelError <= 0.001 ? '✓ &lt; 0.1%' : '⚠ Exceeded'}</td></tr>
        <tr><td>Eccentricity Drift (|Δe|)</td><td>${audit.metrics.maxEccentricityDrift.toExponential(3)}</td><td>—</td></tr>
        <tr><td>Max Residual Acceleration</td><td>${audit.metrics.maxParasiticAccel.toExponential(3)} m/s²</td><td>—</td></tr>
        <tr><td>Closest Approach</td><td>${(audit.metrics.closestApproach / 1000).toFixed(1)} km</td><td>${audit.metrics.closestApproach > r_body ? 'Clear' : '⚠ Penetrated'}</td></tr>
      `;

      $('auditContent').replaceChildren(badge, fileP, table);

      if (audit.warnings.length) {
        const warnBox = document.createElement('div');
        warnBox.className = 'audit-warning-box';
        const strong = document.createElement('strong');
        strong.textContent = 'Warnings:';
        warnBox.append(strong);
        for (const w of audit.warnings) {
          warnBox.append(document.createElement('br'));
          warnBox.append(document.createTextNode(`• ${w}`));
        }
        $('auditContent').append(warnBox);
      }
      $('auditModal').showModal();
      needsDraw = true;
      status(`Audited ${file.name}: ${audit.passed ? 'passed conservation checks' : 'invariants drifted'}.`);
    } catch (err) {
      status(`Audit error: ${err.message}`, true);
    }
  }

  $('auditButton').addEventListener('click', () => $('auditFile').click());
  $('auditFile').addEventListener('change', async event => {
    await handleAuditFile(event.target.files[0]);
    event.target.value = '';
  });
  $('closeAudit').addEventListener('click', () => $('auditModal').close());

  // Drag and drop onto stage
  const stage = $('stage');
  stage.addEventListener('dragover', event => { event.preventDefault(); stage.classList.add('dragover'); });
  stage.addEventListener('dragleave', () => stage.classList.remove('dragover'));
  stage.addEventListener('drop', async event => {
    event.preventDefault(); stage.classList.remove('dragover');
    const file = event.dataTransfer.files[0];
    if (file) await handleAuditFile(file);
  });

  const showNotes = () => $('notes').showModal();
  $('notesButton').addEventListener('click', showNotes); $('balanceInfo').addEventListener('click', showNotes);
  $('closeNotes').addEventListener('click', () => $('notes').close());
  $('notes').addEventListener('click', event => { if (event.target === $('notes')) { const r = $('notes').getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) $('notes').close(); } });
  new ResizeObserver(() => { needsDraw = true; if (displayedSweep) drawSweepPlot(displayedSweep); }).observe($('scene'));
  document.addEventListener('visibilitychange', () => { if (document.hidden) { playing = false; needsDraw = true; } });
  configure(model, draft); requestAnimationFrame(frame);
} catch (error) { status(`Could not start: ${error.message}. Serve this folder over localhost with python -m workbench.`, true); }

