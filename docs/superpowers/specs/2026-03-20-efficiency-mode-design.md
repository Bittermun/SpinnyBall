# SGMS Demo — Efficiency Mode Design
**Date:** 2026-03-20
**Feature:** Section 24 — Efficiency proof visualization

---

## Problem

The demo proves lateral steering *works* (ball deflects sideways, Δvx ≠ 0, Δvz ≈ 0) but does not prove *why it matters*. The core claim of the paper is that lateral steering is orders of magnitude more energy-efficient than conventional drag braking. Without a comparison, the viewer has no frame of reference.

---

## Goal

Add a 4th mode — **EFFICIENCY** — that makes the energy cost ratio visceral without any equations. Two vertical bars, same impulse, wildly different heights.

---

## Architecture

### Mode system change
- Current modes: 0=FORCES, 1=TRAJECTORY, 2=FIELD
- New modes:     0=FORCES, 1=TRAJECTORY, 2=FIELD, 3=EFFICIENCY
- MODE button cycles 0→1→2→3→0
- MODE button label array updated: `['FORCES','TRAJECTORY','FIELD','EFFICIENCY']`
- `updateModeAssets()` gains a branch for `currentMode === 3`

### Layout in mode 3
- WebGL canvas: `display:none` (hidden, not destroyed — resumes when switching away)
- Efficiency panel (`#efficiencyPanel`): takes full canvas area, `display:flex`
- Panel layout:
  - Header row: plain text "EFFICIENCY COMPARISON" (no emoji, no icon)
  - Bar section: two columns side by side, LATERAL (left, green) | DRAG (right, red), bars grow upward
  - Stream speed row: single horizontal bar below the vertical bars
  - Ratio text: large, centered, computed from simulation output

### HTML elements to add

```html
<!-- inserted after #canvasWrap -->
<div id="efficiencyPanel" style="display:none">
  <div id="effHeader">EFFICIENCY COMPARISON</div>
  <div id="effBars">
    <div class="eff-col">
      <div class="eff-bar-wrap"><div class="eff-bar" id="latBar"></div></div>
      <div class="eff-label">LATERAL</div>
      <div class="eff-value" id="latVal">—</div>
    </div>
    <div class="eff-col">
      <div class="eff-bar-wrap"><div class="eff-bar" id="dragBar"></div></div>
      <div class="eff-label">DRAG</div>
      <div class="eff-value" id="dragVal">—</div>
    </div>
  </div>
  <div id="effStreamRow">
    <span class="eff-stream-label">Stream speed retained</span>
    <div id="effStreamBar"><div id="effStreamFill"></div></div>
    <span id="effStreamVal">—</span>
  </div>
  <div id="effRatio">—</div>
</div>
```

---

## Physics (analytical — no second simulation)

### Derivation of E_drag

To deliver the same lateral impulse `J = m|Δvx|` using drag braking instead of lateral deflection, the coils must oppose the ball's axial motion and transfer impulse `J_drag = m|Δvz_equiv|` where `|Δvz_equiv| = |Δvx|` (same impulse magnitude, different axis). The energy extracted from the axial degree of freedom is:

```
E_drag = ½m(vz0² − vz_drag_final²)
       = ½m(vz0 + vz_drag_final)(vz0 − vz_drag_final)
       ≈ ½m · 2vz0 · |Δvz_equiv|   (since vz0 >> |Δvz_equiv|)
       = m · vz0 · |Δvx|
```

This is exact to first order in `|Δvx|/vz0` (~2.7×10⁻⁷ at reference params).

### Formulas

| Quantity | Formula | Reference params result |
|---|---|---|
| `E_lateral` | `½ × m × Δvx²` | ~1.6 nJ |
| `E_drag` | `m × vz0 × |Δvx|` | ~12 mJ |
| `ratio` | `E_drag / E_lateral = 2 × vz0 / |Δvx|` | ~7,500,000× |
| `stream_retained` | `VZ_snap[exitIdx] / VZ_snap[0]` | ~99.9997% |

**Important:** `vz0` must be read from the simulation snapshot (`VZ_snap[0]`, i.e. `transitData.VZ[0]`) not from the live `currentP.vz0`, so that the stream_retained fraction is self-consistent even if params are changed after the last transit.

**`exitIdx` semantics:** The existing `augmentWithHistory()` sets `exitIdx` to the step where `Z[i]` is closest to `P.zend` — i.e., the last step inside the coil array (inclusive). `VZ[exitIdx]` is therefore the final in-coil axial velocity, which is what we want for `vz_final`.

---

## Bar visualization

### Scale: logarithmic

Linear scale makes the lateral bar invisible (ratio ~10⁷). Log scale keeps both bars visible and makes the difference dramatic.

```js
function logBarFrac(E, logMin, logMax) {
  // Returns fraction 0.0–1.0 for bar height.
  // Hard floor at 0.04 (4%) so the lateral bar always shows as a visible sliver.
  if (E <= 0 || logMax <= logMin) return 0.04;  // degenerate: equal energies or zero
  const frac = (Math.log10(E) - logMin) / (logMax - logMin);
  return Math.max(0.04, Math.min(1.0, frac));
}
```

Scale anchors (computed fresh each call to `updateEfficiencyPanel`):
```js
// If E_lat ≈ E_drag (symmetric or zero steering) logMax ≈ logMin + 1.5, still valid.
// The degenerate guard in logBarFrac handles logMax <= logMin defensively.
const logMin = Math.log10(Math.max(E_lat,  1e-40)) - 1.0;
const logMax = Math.log10(Math.max(E_drag, 1e-40)) + 0.5;
```

Bar heights applied as:
```js
latBar.style.height  = (logBarFrac(E_lat,  logMin, logMax) * 100) + '%';
dragBar.style.height = (logBarFrac(E_drag, logMin, logMax) * 100) + '%';
```

CSS `transition: height 0.8s ease-out` on `.eff-bar` provides the animate-in effect.

### Stream speed bar
Horizontal bar, width = `(vz_final / vz0_snap) * 100`%. Always ~99.9997% — visually nearly full. Color: `#79c0ff`.

---

## updateEfficiencyPanel() — full implementation

```js
function updateEfficiencyPanel() {
  // Guard: do nothing if simulation hasn't run or produced valid data
  if (!transitData
      || transitData.exitIdx === undefined
      || !transitData.VZ
      || transitData.VZ.length <= transitData.exitIdx
      || !transitData.metrics) {
    // Show dash placeholders (set by HTML defaults, nothing to do)
    return;
  }

  const m        = currentP.m;
  const vz0_snap = transitData.VZ[0];
  const dvx      = transitData.metrics.delta_vx;
  const vzf      = transitData.VZ[transitData.exitIdx];

  // Zero-steering guard: if dvx ≈ 0, show zero-cost placeholder
  if (Math.abs(dvx) < 1e-15) {
    latVal.textContent  = '0 J';
    dragVal.textContent = '0 J';
    effRatio.textContent = 'No steering applied';
    latBar.style.height  = '4%';
    dragBar.style.height = '4%';
    effStreamFill.style.width = '100%';
    effStreamVal.textContent  = vz0_snap.toFixed(1) + ' m/s';
    return;
  }

  const E_lat  = 0.5 * m * dvx * dvx;
  const E_drag = m * vz0_snap * Math.abs(dvx);
  const ratio  = E_drag / E_lat;   // = 2*vz0/|dvx|

  const logMin = Math.log10(Math.max(E_lat,  1e-40)) - 1.0;
  const logMax = Math.log10(Math.max(E_drag, 1e-40)) + 0.5;

  latBar.style.height  = (logBarFrac(E_lat,  logMin, logMax) * 100) + '%';
  dragBar.style.height = (logBarFrac(E_drag, logMin, logMax) * 100) + '%';

  latVal.textContent  = formatJoules(E_lat);
  dragVal.textContent = formatJoules(E_drag);

  // stream_retained = fraction of initial stream speed preserved by lateral steering
  const stream_retained = vzf / vz0_snap;
  effStreamFill.style.width = (stream_retained * 100).toFixed(4) + '%';
  effStreamVal.textContent  = vzf.toFixed(1) + ' m/s';

  const ratioStr = ratio >= 1e6
    ? (ratio / 1e6).toFixed(2) + ' million'
    : Math.round(ratio).toLocaleString();
  effRatio.textContent =
    `Lateral steering is ${ratioStr}x more energy-efficient than drag braking`;
}
```

### formatJoules() helper

```js
function formatJoules(E) {
  if (E <= 0)    return '0 J';
  if (E < 1e-9)  return (E * 1e12).toFixed(1) + ' pJ';
  if (E < 1e-6)  return (E * 1e9).toFixed(2) + ' nJ';
  if (E < 1e-3)  return (E * 1e6).toFixed(2) + ' μJ';
  if (E < 1)     return (E * 1e3).toFixed(2) + ' mJ';
  return E.toFixed(2) + ' J';
}
```

---

## Call sites

| Call site | When | Why |
|---|---|---|
| `showFinalStats()` | After transit animation ends | Bars animate in with final values |
| `updateModeAssets()` case 3 | When user switches to EFFICIENCY mode | Re-draw with existing transitData (guard prevents crash if no sim yet) |

**Slider update chain (criterion 9):** `onParamChange()` → `simulate()` → `augmentWithHistory()` → `autoFire()` → animation → `showFinalStats()` → `updateEfficiencyPanel()`. This means bars update automatically whenever a slider changes and the transit re-runs. No extra wiring needed beyond the `showFinalStats()` call site — but this dependency must be preserved if `showFinalStats()` is ever refactored.

---

## updateModeAssets() change

```js
// Existing: show/hide 3D elements for modes 0–2
// Add:
case 3:
  renderer.domElement.style.display = 'none';
  efficiencyPanel.style.display = 'flex';
  // Reset bars to 0% before redraw so re-entry always animates
  latBar.style.height = '0%';
  dragBar.style.height = '0%';
  effStreamFill.style.width = '0%';
  // Brief delay so CSS transition fires from 0
  setTimeout(updateEfficiencyPanel, 50);
  break;
default: // modes 0–2
  renderer.domElement.style.display = '';
  efficiencyPanel.style.display = 'none';
  break;
```

---

## CSS additions

```css
#efficiencyPanel {
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 28px;
  background: #0a0a0f;
  color: #c0c8d8;
  font-family: system-ui, sans-serif;
  padding: 40px;
  width: 100%;
  height: 100%;
}
#effHeader {
  font-size: 12px;
  letter-spacing: 0.12em;
  color: #8b949e;
  text-transform: uppercase;
}
#effBars {
  display: flex;
  gap: 64px;
  align-items: flex-end;
  height: 280px;
}
.eff-col {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  width: 90px;
}
.eff-bar-wrap {
  flex: 1;
  width: 100%;
  display: flex;
  align-items: flex-end;
  background: #1c2030;
  border-radius: 4px 4px 0 0;
  overflow: hidden;
}
.eff-bar {
  width: 100%;
  border-radius: 4px 4px 0 0;
  transition: height 0.8s ease-out;
  height: 0%;
}
#latBar  { background: #3fb950; }
#dragBar { background: #f85149; }
.eff-label {
  font-size: 11px;
  letter-spacing: 0.08em;
  color: #8b949e;
  text-transform: uppercase;
}
.eff-value { font-size: 13px; font-family: monospace; color: #c0c8d8; }
#effStreamRow {
  display: flex;
  align-items: center;
  gap: 14px;
  font-size: 12px;
  width: 100%;
  max-width: 520px;
}
.eff-stream-label { color: #8b949e; white-space: nowrap; }
#effStreamBar {
  flex: 1;
  height: 10px;
  background: #1c2030;
  border-radius: 3px;
  overflow: hidden;
}
#effStreamFill {
  height: 100%;
  background: #79c0ff;
  transition: width 0.8s ease-out;
  width: 0%;
}
#effStreamVal { font-family: monospace; color: #79c0ff; white-space: nowrap; }
#effRatio {
  font-size: 17px;
  color: #f2cc60;
  text-align: center;
  max-width: 520px;
  line-height: 1.6;
}
```

---

## Files changed

- `index.html` — single file, all changes inline

**Sections touched (10):**
1. CSS block — add efficiency panel styles (above)
2. HTML body — add `#efficiencyPanel` div after `#canvasWrap`
3. MODE label array — add `'EFFICIENCY'` as 4th entry
4. `cycleMode()` — change `% 3` → `% 4` so mode 3 is reachable
5. `applyMode()` — add `case 3` branch (show panel, hide canvas, setTimeout(updateEfficiencyPanel, 50)) and update `default` clause to hide panel for modes 0–2. **Note:** the live code splits mode logic between `updateModeAssets()` (rebuilds geometry) and `applyMode()` (switches UI). The case 3 branch targets `applyMode()`.
6. `showFinalStats()` — call `updateEfficiencyPanel()` after transit
7. `logBarFrac()` — new helper function
8. `formatJoules()` — new helper function
9. `updateEfficiencyPanel()` — new function

---

## Acceptance criteria

1. Clicking MODE cycles: FORCES → TRAJECTORY → FIELD → EFFICIENCY → FORCES. Button label updates accordingly.
2. In EFFICIENCY mode, the WebGL canvas is hidden (`display:none`). The `#efficiencyPanel` is visible (`display:flex`) and fills the canvas area.
3. After autoFire completes in EFFICIENCY mode, both vertical bars animate upward (CSS ease-out, ~0.8 s). LATERAL bar is green and short; DRAG bar is red and tall.
4. The LATERAL bar height is at least 4% (never invisible), despite being 7,500,000× smaller in energy.
5. DRAG bar label shows a value in mJ range (e.g. "12.00 mJ") at reference params; LATERAL shows a value in nJ range (e.g. "1.74 nJ").
6. Stream speed bar is visually ~99.99% full (nearly complete blue bar) with actual m/s value shown.
7. Ratio text reads "Lateral steering is X.XXx million x more energy-efficient than drag braking" with X derived from live simulation output (changes when params change).
8. Switching away from EFFICIENCY mode to any other mode restores the 3D canvas and resumes the normal mode view without errors.
9. Changing a slider while in EFFICIENCY mode triggers `onParamChange()`, re-runs the simulation, and updates all bars and the ratio text.
10. No emoji characters appear anywhere in the efficiency panel.
11. Switching to EFFICIENCY mode before any simulation has run (on first page load with no autoFire) shows "—" placeholders, not NaN or a crash.
12. When all coil amplitudes are symmetric and Δvx ≈ 0, efficiency panel shows "No steering applied" text rather than NaN or divide-by-zero.
13. Stream speed bar width equals `(VZ[exitIdx] / VZ[0]) × 100%` — verified by reading these two values from transitData in browser console and checking the rendered width matches.
