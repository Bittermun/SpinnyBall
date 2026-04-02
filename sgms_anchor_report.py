"""
Static, self-contained HTML report for config-driven anchor runs.
"""

from __future__ import annotations

import json
from pathlib import Path


def build_report_payload(manifest: dict, summaries: dict[str, dict]) -> dict:
    experiments = []
    for item in manifest.get("experiments", []):
        slug = item["slug"]
        summary = summaries[slug]
        experiments.append(
            {
                "name": item["name"],
                "slug": slug,
                "summary_path": item["summary_path"],
                "files": item.get("files", []),
                "profile": summary.get("profile", {}),
                "anchor": summary.get("anchor", {}),
                "trade_study": summary.get("trade_study", {}),
                "robustness": summary.get("robustness", {}),
                "sensitivity": summary.get("sensitivity", {}),
                "claim_context": summary.get("claim_context", {}),
                "calibration": summary.get("calibration", {}),
            }
        )
    return {
        "run_label": manifest.get("run_label"),
        "generated_at_utc": manifest.get("generated_at_utc"),
        "experiments": experiments,
    }


def _report_html(payload_json: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SGMS Anchor Report</title>
<style>
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: Georgia, "Times New Roman", serif;
  background: linear-gradient(180deg, #f4f1e8 0%, #ece7dc 100%);
  color: #1b1c1d;
}}
main {{
  max-width: 1120px;
  margin: 0 auto;
  padding: 32px 20px 48px;
}}
h1, h2, h3 {{ margin: 0 0 12px; }}
h1 {{ font-size: 40px; letter-spacing: 0.02em; }}
h2 {{ font-size: 24px; margin-top: 28px; }}
h3 {{ font-size: 18px; }}
p {{ margin: 0 0 12px; line-height: 1.5; }}
.hero {{
  background: rgba(255,255,255,0.7);
  border: 1px solid rgba(27,28,29,0.12);
  border-radius: 18px;
  padding: 24px;
  box-shadow: 0 18px 50px rgba(60, 53, 42, 0.08);
}}
.grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px;
  margin-top: 18px;
}}
.card {{
  background: rgba(255,255,255,0.78);
  border: 1px solid rgba(27,28,29,0.1);
  border-radius: 14px;
  padding: 14px 16px;
}}
.label {{
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #5c645d;
}}
.value {{
  font-size: 24px;
  margin-top: 6px;
}}
.experiment {{
  margin-top: 26px;
  background: rgba(255,255,255,0.76);
  border: 1px solid rgba(27,28,29,0.1);
  border-radius: 18px;
  padding: 22px;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  margin-top: 10px;
}}
th, td {{
  text-align: left;
  padding: 8px 10px;
  border-bottom: 1px solid rgba(27,28,29,0.12);
  font-size: 14px;
}}
th {{
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #5c645d;
}}
.files {{
  margin-top: 12px;
  font-family: "Courier New", monospace;
  font-size: 12px;
  white-space: pre-wrap;
  word-break: break-all;
  color: #3f433f;
}}
@media (max-width: 720px) {{
  h1 {{ font-size: 30px; }}
  main {{ padding: 18px 14px 30px; }}
}}
</style>
</head>
<body>
<main>
  <section class="hero">
    <h1>SGMS Anchor Report</h1>
    <p id="runMeta"></p>
    <div id="heroGrid" class="grid"></div>
  </section>
  <section id="experiments"></section>
</main>
<script id="report-data" type="application/json">{payload_json}</script>
<script>
const payload = JSON.parse(document.getElementById('report-data').textContent);

function fmt(value, digits = 4) {{
  if (value === null || value === undefined || Number.isNaN(value)) return 'n/a';
  if (!Number.isFinite(value)) return String(value);
  return Number(value).toFixed(digits);
}}

document.getElementById('runMeta').textContent =
  `Run ${{payload.run_label}} generated ${{payload.generated_at_utc || 'n/a'}} with ${{payload.experiments.length}} experiment(s).`;

const heroGrid = document.getElementById('heroGrid');
payload.experiments.forEach((exp) => {{
  const metrics = exp.anchor?.continuum_metrics || {{}};
  const card = document.createElement('div');
  card.className = 'card';
  card.innerHTML = `
    <div class="label">${{exp.name}}</div>
    <div class="value">${{fmt(metrics.k_eff, 3)}} N/m</div>
    <p>k_eff with period ${{fmt(exp.anchor?.continuum_period_s, 2)}} s</p>
  `;
  heroGrid.appendChild(card);
}});

const experimentsEl = document.getElementById('experiments');
payload.experiments.forEach((exp) => {{
  const section = document.createElement('section');
  section.className = 'experiment';
  const metrics = exp.anchor?.continuum_metrics || {{}};
  const profile = exp.profile || {{}};
  const claim = exp.claim_context || {{}};
  const tradeRows = exp.trade_study?.rows || [];
  const robustnessRows = exp.robustness?.rows || [];
  const sensitivity = exp.sensitivity || {{}};
  const files = (exp.files || []).join('\\n');
  const provenanceLines = Object.entries(profile.provenance || {{}}).map(([key, value]) =>
    `<p><strong>${{key}}</strong>: ${{value}}</p>`
  ).join('');
  const calibrationLines = Object.entries(exp.calibration?.provenance || {{}}).map(([key, value]) =>
    `<p><strong>${{key}}</strong>: ${{value.status}} via ${{value.source}}</p>`
  ).join('');

  let tradeHtml = '<p>No trade study rows.</p>';
  if (tradeRows.length) {{
    tradeHtml = `
      <table>
        <thead><tr><th>Controller</th><th>Peak |x| (m)</th><th>Area |x|</th><th>Peak |u| (N)</th></tr></thead>
        <tbody>
          ${{tradeRows.map((row) => `
            <tr>
              <td>${{row.controller}}</td>
              <td>${{fmt(row.peak_abs_x_m, 4)}}</td>
              <td>${{fmt(row.area_abs_x, 4)}}</td>
              <td>${{fmt(row.peak_abs_u_n, 4)}}</td>
            </tr>`).join('')}}
        </tbody>
      </table>`;
  }}

  let robustHtml = '<p>No robustness rows.</p>';
  if (robustnessRows.length) {{
    robustHtml = `
      <table>
        <thead><tr><th>Scenario</th><th>Peak |x| (m)</th><th>Area |x|</th></tr></thead>
        <tbody>
          ${{robustnessRows.map((row) => `
            <tr>
              <td>${{row.scenario}}</td>
              <td>${{fmt(row.peak_abs_x_m, 4)}}</td>
              <td>${{fmt(row.area_abs_x, 4)}}</td>
            </tr>`).join('')}}
        </tbody>
      </table>`;
  }}

  const sensitivityLines = Object.entries(sensitivity).map(([output, data]) => {{
    const st = data.ST || {{}};
    const top = Object.entries(st).sort((a, b) => b[1] - a[1])[0];
    return `<p><strong>${{output}}</strong>: dominant ST = ${{top ? `${{top[0]}} (${{fmt(top[1], 4)}})` : 'n/a'}}</p>`;
  }}).join('');

  section.innerHTML = `
    <h2>${{exp.name}}</h2>
    <p><strong>Profile:</strong> ${{profile.name || 'direct-defaults'}} (${{profile.category || 'unspecified'}})</p>
    <div class="grid">
      <div class="card"><div class="label">k_eff</div><div class="value">${{fmt(metrics.k_eff, 3)}}</div></div>
      <div class="card"><div class="label">Period (s)</div><div class="value">${{fmt(exp.anchor?.continuum_period_s, 2)}}</div></div>
      <div class="card"><div class="label">Static Offset (mm)</div><div class="value">${{fmt((metrics.static_offset_m || 0) * 1000, 3)}}</div></div>
      <div class="card"><div class="label">Packet Rate (Hz)</div><div class="value">${{fmt(metrics.packet_rate_hz, 2)}}</div></div>
    </div>
    <h3>Profile Notes</h3>
    <p>${{(profile.notes || []).join(' ') || 'No profile notes.'}}</p>
    <h3>Parameter Provenance</h3>
    ${{provenanceLines || '<p>No provenance entries.</p>'}}
    <h3>Calibration Context</h3>
    ${{calibrationLines || '<p>No calibration entries.</p>'}}
    <h3>Claim Context</h3>
    <p><strong>Phase decision:</strong> ${{claim.phase_decision?.decision || 'n/a'}}</p>
    <p><strong>Claim level:</strong> ${{claim.profile_claim?.claim_level || 'n/a'}}</p>
    <h3>Controller Trade Study</h3>
    ${{tradeHtml}}
    <h3>Robustness</h3>
    ${{robustHtml}}
    <h3>Sensitivity Summary</h3>
    ${{sensitivityLines || '<p>No sensitivity summary.</p>'}}
    <h3>Files</h3>
    <div class="files">${{files}}</div>
  `;
  experimentsEl.appendChild(section);
}});
</script>
</body>
</html>"""


def write_report_html(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload_json = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(_report_html(payload_json), encoding="utf-8")
