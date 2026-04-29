"""
Static interactive dashboard for reduced-order anchor exploration.
"""

from __future__ import annotations

import json
from pathlib import Path


def build_dashboard_payload(manifest: dict, summaries: dict[str, dict]) -> dict:
    experiments = []
    for item in manifest.get("experiments", []):
        slug = item["slug"]
        summary = summaries[slug]
        experiments.append(
            {
                "name": item["name"],
                "slug": slug,
                "params": summary.get("params", {}),
                "profile": summary.get("profile", {}),
                "trade_study": summary.get("trade_study", {}),
                "claim_context": summary.get("claim_context", {}),
                "calibration": summary.get("calibration", {}),
            }
        )
    return {
        "run_label": manifest.get("run_label"),
        "generated_at_utc": manifest.get("generated_at_utc"),
        "experiments": experiments,
    }


def _dashboard_html(payload_json: str) -> str:
    template = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SGMS Anchor Dashboard</title>
<style>
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: "Trebuchet MS", Arial, sans-serif;
  color: #112;
  background:
    radial-gradient(circle at top left, rgba(255, 206, 126, 0.28), transparent 36%),
    linear-gradient(180deg, #eef1ea 0%, #dfe5db 100%);
}}
main {{
  max-width: 1180px;
  margin: 0 auto;
  padding: 28px 18px 36px;
}}
.hero, .panel, .plot-card {{
  background: rgba(255,255,255,0.78);
  border: 1px solid rgba(17,18,34,0.12);
  border-radius: 18px;
  box-shadow: 0 16px 42px rgba(31, 41, 32, 0.08);
}}
.hero {{
  padding: 22px;
  margin-bottom: 18px;
}}
.layout {{
  display: grid;
  grid-template-columns: minmax(300px, 360px) 1fr;
  gap: 18px;
}}
.panel {{
  padding: 18px;
}}
.controls {{
  display: grid;
  gap: 14px;
}}
.control label {{
  display: flex;
  justify-content: space-between;
  font-size: 13px;
  margin-bottom: 5px;
}}
.metrics {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px;
  margin-bottom: 18px;
}}
.metric {{
  padding: 12px 14px;
  background: rgba(244, 246, 240, 0.9);
  border-radius: 14px;
}}
.metric .label {{
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #4d5c50;
}}
.metric .value {{
  margin-top: 6px;
  font-size: 22px;
}}
.plot-card {{
  padding: 18px;
}}
canvas {{
  width: 100%;
  height: 320px;
  display: block;
  background: linear-gradient(180deg, rgba(245,247,243,0.95), rgba(227,232,224,0.9));
  border-radius: 14px;
}}
.small {{
  font-size: 12px;
  color: #51615a;
}}
@media (max-width: 880px) {{
  .layout {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<main>
  <section class="hero">
    <h1>SGMS Anchor Dashboard</h1>
    <p class="small" id="runMeta"></p>
  </section>
  <div class="layout">
    <section class="panel">
      <div class="controls">
        <div class="control">
          <label for="profileSelect"><span>Profile</span><span id="profileMeta"></span></label>
          <select id="profileSelect"></select>
        </div>
        <div class="control">
          <label for="uSlider"><span>u (m/s)</span><span id="uVal"></span></label>
          <input id="uSlider" type="range" min="5" max="520" step="1">
        </div>
        <div class="control">
          <label for="gSlider"><span>g_gain (rad/m)</span><span id="gVal"></span></label>
          <input id="gSlider" type="range" min="0.02" max="0.2" step="0.001">
        </div>
        <div class="control">
          <label for="epsSlider"><span>eps</span><span id="epsVal"></span></label>
          <input id="epsSlider" type="range" min="0" max="0.001" step="0.00001">
        </div>
        <div class="control">
          <label for="dampSlider"><span>c_damp</span><span id="dampVal"></span></label>
          <input id="dampSlider" type="range" min="0" max="20" step="0.1">
        </div>
      </div>
      <h3>Claim Context</h3>
      <p class="small" id="claimText"></p>
      <h3>Calibration</h3>
      <div class="small" id="calibrationText"></div>
    </section>
    <section>
      <div class="metrics" id="metrics"></div>
      <div class="plot-card">
        <h2>Open vs Proportional Control Response</h2>
        <canvas id="plot" width="820" height="320"></canvas>
        <p class="small">Static browser model uses the same reduced-order anchor equations as the pipeline and a proportional trim controller derived from the current stiffness.</p>
      </div>
    </section>
  </div>
</main>
<script id="dashboard-data" type="application/json">__PAYLOAD_JSON__</script>
<script>
const payload = JSON.parse(document.getElementById('dashboard-data').textContent);
const profileSelect = document.getElementById('profileSelect');
const uSlider = document.getElementById('uSlider');
const gSlider = document.getElementById('gSlider');
const epsSlider = document.getElementById('epsSlider');
const dampSlider = document.getElementById('dampSlider');
const metricsEl = document.getElementById('metrics');
const claimText = document.getElementById('claimText');
const calibrationText = document.getElementById('calibrationText');
const profileMeta = document.getElementById('profileMeta');
document.getElementById('runMeta').textContent =
  `Run ${payload.run_label} generated ${payload.generated_at_utc || 'n/a'} with ${payload.experiments.length} experiment(s).`;

function fmt(value, digits = 4) {{
  if (value === null || value === undefined || Number.isNaN(value)) return 'n/a';
  if (!Number.isFinite(value)) return String(value);
  return Number(value).toFixed(digits);
}}

payload.experiments.forEach((exp, index) => {{
  const option = document.createElement('option');
  option.value = String(index);
  option.textContent = `${exp.name} (${exp.profile?.category || 'unspecified'})`;
  profileSelect.appendChild(option);
}});

function currentExperiment() {{
  return payload.experiments[Number(profileSelect.value) || 0];
}}

function omega(k, m) {{
  return k > 0 ? Math.sqrt(k / m) : 0;
}}

function computeMetrics(params) {{
  const k = params.lam * params.u * params.u * params.g_gain;
  const om = omega(k, params.ms);
  const period = om > 0 ? 2 * Math.PI / om : Infinity;
  const staticOffset = k > 0 ? 2 * params.eps * params.lam * params.u * params.u * params.theta_bias / k : Infinity;
  return {{ k, om, period, staticOffset }};
}}

function simulate(params, closedLoop = false) {{
  const dt = 0.25;
  const tMax = 180;
  const steps = Math.floor(tMax / dt);
  const xs = [];
  let x = 0.1;
  let v = 0;
  const kp = closedLoop ? 0.5 * params.lam * params.u * params.u * params.g_gain : 0;
  for (let i = 0; i <= steps; i++) {{
    const t = i * dt;
    xs.push({{ t, x }});
    const k = params.lam * params.u * params.u * params.g_gain;
    const bias = 2 * params.eps * params.lam * params.u * params.u * params.theta_bias;
    const a = (bias - (k + kp) * x - params.c_damp * v) / params.ms;
    v += a * dt;
    x += v * dt;
  }}
  return xs;
}}

function drawPlot(openData, closedData) {{
  const canvas = document.getElementById('plot');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#f4f6f0';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const allY = openData.concat(closedData).map(p => p.x);
  const minY = Math.min(...allY) - 0.01;
  const maxY = Math.max(...allY) + 0.01;
  const minX = 0;
  const maxX = openData[openData.length - 1].t;
  const pad = {{ l: 50, r: 20, t: 20, b: 30 }};
  const plotW = canvas.width - pad.l - pad.r;
  const plotH = canvas.height - pad.t - pad.b;
  const xToCanvas = t => pad.l + (t - minX) / (maxX - minX) * plotW;
  const yToCanvas = y => pad.t + plotH - (y - minY) / (maxY - minY || 1) * plotH;

  ctx.strokeStyle = '#8aa08f';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, yToCanvas(0));
  ctx.lineTo(canvas.width - pad.r, yToCanvas(0));
  ctx.stroke();

  function trace(data, color) {{
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    data.forEach((point, idx) => {{
      const x = xToCanvas(point.t);
      const y = yToCanvas(point.x);
      if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }});
    ctx.stroke();
  }}

  trace(openData, '#ff7b72');
  trace(closedData, '#4c8c5b');

  ctx.fillStyle = '#334';
  ctx.font = '12px Arial';
  ctx.fillText('Open-loop', pad.l + 10, pad.t + 14);
  ctx.fillText('Proportional trim', pad.l + 90, pad.t + 14);
}}

function update() {{
  const exp = currentExperiment();
  const params = {{
    ...exp.params,
    u: Number(uSlider.value),
    g_gain: Number(gSlider.value),
    eps: Number(epsSlider.value),
    c_damp: Number(dampSlider.value),
  }};
  const metrics = computeMetrics(params);
  document.getElementById('uVal').textContent = fmt(params.u, 1);
  document.getElementById('gVal').textContent = fmt(params.g_gain, 3);
  document.getElementById('epsVal').textContent = fmt(params.eps, 5);
  document.getElementById('dampVal').textContent = fmt(params.c_damp, 2);
  profileMeta.textContent = `${exp.profile?.name || 'direct-defaults'}`;
  claimText.textContent = `${exp.claim_context?.phase_decision?.decision || 'n/a'} | ${exp.claim_context?.profile_claim?.claim_level || 'n/a'}`;

  const calibrationLines = Object.entries(exp.calibration?.provenance || {{}})
    .slice(0, 6)
    .map(([key, value]) => `<p><strong>${{key}}</strong>: ${{value.status}} via ${{value.source}}</p>`)
    .join('');
  calibrationText.innerHTML = calibrationLines || '<p>No calibration entries.</p>';

  metricsEl.innerHTML = `
    <div class="metric"><div class="label">k_eff</div><div class="value">${fmt(metrics.k, 3)} N/m</div></div>
    <div class="metric"><div class="label">Period</div><div class="value">${fmt(metrics.period, 2)} s</div></div>
    <div class="metric"><div class="label">Static Offset</div><div class="value">${fmt(metrics.staticOffset * 1000, 3)} mm</div></div>
    <div class="metric"><div class="label">Damping Ratio Proxy</div><div class="value">${fmt(params.c_damp / (2 * Math.sqrt(Math.max(metrics.k * params.ms, 1e-12))), 3)}</div></div>
  `;

  const openData = simulate(params, false);
  const closedData = simulate(params, true);
  drawPlot(openData, closedData);
}}

function loadExperiment(index) {{
  const exp = payload.experiments[index];
  uSlider.value = exp.params.u;
  gSlider.value = exp.params.g_gain;
  epsSlider.value = exp.params.eps;
  dampSlider.value = exp.params.c_damp;
  update();
}}

[uSlider, gSlider, epsSlider, dampSlider].forEach(el => el.addEventListener('input', update));
profileSelect.addEventListener('change', () => loadExperiment(Number(profileSelect.value) || 0));
loadExperiment(0);
</script>
</body>
</html>"""
    return template.replace("__PAYLOAD_JSON__", payload_json)


def write_dashboard_html(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload_json = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(_dashboard_html(payload_json), encoding="utf-8")
