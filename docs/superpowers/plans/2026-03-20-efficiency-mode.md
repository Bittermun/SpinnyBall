# Efficiency Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 4th demo mode ("EFFICIENCY") that replaces the 3D canvas with two vertical log-scale bars showing lateral vs drag energy cost side by side, making the efficiency argument visceral without equations.

**Architecture:** All changes are in the single file `index.html`. The mode system cycles 0→1→2→3→0. In mode 3, the WebGL canvas is hidden and `#efficiencyPanel` fills its place. Physics is analytical — no second simulation; energy values are derived from the existing `transitData` after each `simulate()` run.

**Tech Stack:** Vanilla JS, Three.js r152, inline CSS/HTML in `index.html`. No build step. Test by opening `index.html` directly in a browser and running checks in the browser console.

---

## File Map

| File | Change |
|---|---|
| `index.html` lines 7–380 (CSS block) | Add efficiency panel CSS |
| `index.html` lines 290–297 (HTML body, inside `#canvasWrap`) | Add `#efficiencyPanel` div |
| `index.html` line 900 (`MODE_NAMES`) | Add `'EFFICIENCY'` as 4th entry |
| `index.html` line 903 (`cycleMode`) | Change `% 3` → `% 4` |
| `index.html` lines 908–936 (`applyMode`) | Add `else if (currentMode === 3)` branch |
| `index.html` lines 1070–1083 (`showFinalStats`) | Call `updateEfficiencyPanel()` at end |
| `index.html` after `showFinalStats` (~line 1084) | Add `logBarFrac`, `formatJoules`, `updateEfficiencyPanel` functions |

---

## Task 1: Add CSS for the efficiency panel

**Files:**
- Modify: `index.html` — CSS block (ends around line 380, before `</style>`)

- [ ] **Step 1: Locate the closing `</style>` tag**

  Open `index.html`, find the line containing `</style>` that closes the main CSS block (around line 380). All new CSS goes directly before that line.

- [ ] **Step 2: Insert the efficiency panel CSS**

  Insert before `</style>`:

  ```css
  /* ── Efficiency Mode Panel ─────────────────────────────── */
  #efficiencyPanel {
    display: none;
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
    position: absolute;
    top: 0; left: 0;
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

- [ ] **Step 3: Verify — no syntax error**

  Open `index.html` in a browser. Open DevTools console. If there are no red CSS errors, this step is done.

- [ ] **Step 4: Commit**

  ```bash
  git add index.html
  git commit -m "feat(efficiency): add CSS for efficiency panel"
  ```

---

## Task 2: Add the efficiency panel HTML

**Files:**
- Modify: `index.html` lines 290–297 (inside `#canvasWrap`)

- [ ] **Step 1: Locate `#canvasWrap`**

  Find the line `<div id="canvasWrap">` (~line 290). The block currently contains:
  ```html
  <canvas id="threeCanvas"></canvas>
  <canvas id="insetCanvas" width="220" height="320"></canvas>
  <div id="overlayLabels"></div>
  <div id="centerResultText"></div>
  <div id="challengeOverlay"></div>
  <div id="scoreDisplay"></div>
  ```

- [ ] **Step 2: Insert `#efficiencyPanel` div as last child of `#canvasWrap`**

  Add immediately before the closing `</div>` of `#canvasWrap`:

  ```html
  <div id="efficiencyPanel">
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

- [ ] **Step 3: Verify — panel exists in DOM, hidden by default**

  Reload page. In console:
  ```js
  document.getElementById('efficiencyPanel').style.display  // should be "" (empty = hidden by CSS default)
  document.getElementById('latBar')  // should not be null
  document.getElementById('dragBar') // should not be null
  ```

- [ ] **Step 4: Commit**

  ```bash
  git add index.html
  git commit -m "feat(efficiency): add efficiency panel HTML"
  ```

---

## Task 3: Extend the mode system to include EFFICIENCY

**Files:**
- Modify: `index.html` lines 900–906 (`MODE_NAMES`, `cycleMode`)

- [ ] **Step 1: Update `MODE_NAMES`**

  Find line ~900:
  ```js
  const MODE_NAMES = ['FORCES', 'TRAJECTORY', 'FIELD'];
  ```
  Change to:
  ```js
  const MODE_NAMES = ['FORCES', 'TRAJECTORY', 'FIELD', 'EFFICIENCY'];
  ```

- [ ] **Step 2: Update `cycleMode` modulo**

  Find line ~903:
  ```js
  currentMode = (currentMode + 1) % 3;
  ```
  Change to:
  ```js
  currentMode = (currentMode + 1) % 4;
  ```

- [ ] **Step 3: Verify — mode cycles correctly**

  Reload page. In console:
  ```js
  cycleMode(); document.getElementById('modeBtn').textContent  // "MODE: TRAJECTORY"
  cycleMode(); document.getElementById('modeBtn').textContent  // "MODE: FIELD"
  cycleMode(); document.getElementById('modeBtn').textContent  // "MODE: EFFICIENCY"
  cycleMode(); document.getElementById('modeBtn').textContent  // "MODE: FORCES"
  ```

- [ ] **Step 4: Commit**

  ```bash
  git add index.html
  git commit -m "feat(efficiency): add EFFICIENCY to mode cycle (4 modes)"
  ```

---

## Task 4: Add mode 3 branch to `applyMode()`

**Files:**
- Modify: `index.html` lines 908–936 (`applyMode`)

The current `applyMode` ends after the `else if (currentMode === 2)` block and falls through silently for any other mode value. We need to add mode 3 handling and ensure the efficiency panel is hidden in all other modes.

- [ ] **Step 1: Add efficiency panel variable reference**

  At the top of the `<script>` section, or near the other DOM element references (search for `getElementById` assignments), ensure `efficiencyPanel` is accessible. Since we call `document.getElementById` inline in `applyMode`, no separate variable is strictly required — we'll call it inline.

- [ ] **Step 2: Add `else if (currentMode === 3)` branch to `applyMode`**

  Find the end of the `applyMode` function. The last block is:
  ```js
    } else if (currentMode === 2) {
      inset.style.display = 'none';
      if (arrowRed)  arrowRed.visible  = false;
      if (arrowBlue) arrowBlue.visible = false;
      ol.innerHTML = '';
      buildDipoleFieldLines(currentP);
      if (dipoleGroup) dipoleGroup.visible = true;
      if (streamlineGroup) streamlineGroup.visible = false;
      orbitControls.setView(0.5, Math.PI / 3, 16);
    }
  }
  ```

  Change the closing `}` of the mode 2 block + function to:
  ```js
    } else if (currentMode === 2) {
      inset.style.display = 'none';
      if (arrowRed)  arrowRed.visible  = false;
      if (arrowBlue) arrowBlue.visible = false;
      ol.innerHTML = '';
      buildDipoleFieldLines(currentP);
      if (dipoleGroup) dipoleGroup.visible = true;
      if (streamlineGroup) streamlineGroup.visible = false;
      orbitControls.setView(0.5, Math.PI / 3, 16);
    } else if (currentMode === 3) {
      // EFFICIENCY mode: hide 3D canvas, show comparison panel
      inset.style.display = 'none';
      if (arrowRed)  arrowRed.visible  = false;
      if (arrowBlue) arrowBlue.visible = false;
      if (dipoleGroup)     dipoleGroup.visible     = false;
      if (streamlineGroup) streamlineGroup.visible = false;
      ol.innerHTML = '';
      renderer.domElement.style.display = 'none';
      document.getElementById('efficiencyPanel').style.display = 'flex';
      // Reset bars to 0% so the CSS transition always animates from bottom
      document.getElementById('latBar').style.height   = '0%';
      document.getElementById('dragBar').style.height  = '0%';
      document.getElementById('effStreamFill').style.width = '0%';
      // 50 ms delay gives the CSS reset a frame to apply before animating up
      setTimeout(updateEfficiencyPanel, 50);
    }

    // Always hide efficiency panel when not in mode 3.
    // This block lives INSIDE applyMode(), after all the mode branches.
    // It ensures mode 0/1/2 always restores the canvas even if mode 3 was previously active.
    if (currentMode !== 3) {
      renderer.domElement.style.display = '';
      document.getElementById('efficiencyPanel').style.display = 'none';
    }
  }
  ```

- [ ] **Step 3: Verify — switching to mode 3 hides canvas and shows panel**

  Reload page. Let autoFire complete. Then in console:
  ```js
  currentMode = 3;
  document.getElementById('modeBtn').textContent = 'MODE: EFFICIENCY';
  applyMode();
  // Check:
  document.getElementById('threeCanvas').style.display         // "none"
  document.getElementById('efficiencyPanel').style.display     // "flex"
  ```

- [ ] **Step 4: Verify — switching back to mode 0 restores canvas**

  ```js
  currentMode = 0;
  document.getElementById('modeBtn').textContent = 'MODE: FORCES';
  applyMode();
  document.getElementById('threeCanvas').style.display         // ""
  document.getElementById('efficiencyPanel').style.display     // "none"
  ```

- [ ] **Step 5: Commit**

  ```bash
  git add index.html
  git commit -m "feat(efficiency): applyMode handles mode 3, hides canvas shows panel"
  ```

---

## Task 5: Add helper functions `logBarFrac` and `formatJoules`

**Files:**
- Modify: `index.html` — insert new functions after `showFinalStats` (~line 1084)

- [ ] **Step 1: Insert both helpers immediately after the closing `}` of `showFinalStats`**

  Find the closing `}` of `showFinalStats` (~line 1083). Insert:

  **Log scale anchors** (supplied by `updateEfficiencyPanel` in Task 6, not hardcoded here):
  - `logMin = log10(max(E_lat, 1e-40)) - 1.0` — 1 decade below lateral energy
  - `logMax = log10(max(E_drag, 1e-40)) + 0.5` — half decade above drag energy
  - At reference params (~1.74 nJ lateral, ~12 mJ drag): lateral bar ≈ 7%, drag bar ≈ 100%.

  ```js
  // ── Efficiency panel helpers ────────────────────────────
  function logBarFrac(E, logMin, logMax) {
    // Returns 0.04–1.0 for bar height fraction.
    // Floor at 0.04 so the lateral bar is always a visible sliver.
    if (E <= 0 || logMax <= logMin) return 0.04;
    const frac = (Math.log10(E) - logMin) / (logMax - logMin);
    return Math.max(0.04, Math.min(1.0, frac));
  }

  function formatJoules(E) {
    if (E <= 0)   return '0 J';
    if (E < 1e-9) return (E * 1e12).toFixed(1) + ' pJ';
    if (E < 1e-6) return (E * 1e9).toFixed(2) + ' nJ';
    if (E < 1e-3) return (E * 1e6).toFixed(2) + ' uJ';
    if (E < 1)    return (E * 1e3).toFixed(2) + ' mJ';
    return E.toFixed(2) + ' J';
  }
  ```

  Note: `μJ` uses plain `uJ` to avoid Unicode issues.

- [ ] **Step 2: Verify helpers in console**

  Reload page. In console:
  ```js
  formatJoules(1.74e-9)   // "1.74 nJ"
  formatJoules(0.012)     // "12.00 mJ"
  formatJoules(0)         // "0 J"
  logBarFrac(1e-9, -10, 1.5)   // value between 0.04 and 1.0
  logBarFrac(0, 0, 1)          // 0.04  (floor)
  logBarFrac(100, 2, 1)        // 0.04  (degenerate: logMax < logMin)
  ```

- [ ] **Step 3: Commit**

  ```bash
  git add index.html
  git commit -m "feat(efficiency): add logBarFrac and formatJoules helpers"
  ```

---

## Task 6: Add `updateEfficiencyPanel()`

**Files:**
- Modify: `index.html` — insert after `formatJoules` (added in Task 5)

- [ ] **Step 1: Insert `updateEfficiencyPanel` immediately after `formatJoules`**

  ```js
  function updateEfficiencyPanel() {
    // Guard: bail out gracefully if simulation hasn't produced valid data
    if (!transitData
        || transitData.exitIdx === undefined
        || !transitData.VZ
        || transitData.VZ.length <= transitData.exitIdx
        || !transitData.metrics) {
      return; // HTML defaults show "—" placeholders
    }

    const m        = currentP.m;
    const vz0_snap = transitData.VZ[0];               // vz0 from sim snapshot
    const dvx      = transitData.metrics.delta_vx;
    const vzf      = transitData.VZ[transitData.exitIdx];

    // Zero-steering guard
    if (Math.abs(dvx) < 1e-15) {
      document.getElementById('latVal').textContent  = '0 J';
      document.getElementById('dragVal').textContent = '0 J';
      document.getElementById('effRatio').textContent = 'No steering applied';
      document.getElementById('latBar').style.height  = '4%';
      document.getElementById('dragBar').style.height = '4%';
      document.getElementById('effStreamFill').style.width = '100%';
      document.getElementById('effStreamVal').textContent  = vz0_snap.toFixed(1) + ' m/s';
      return;
    }

    const E_lat  = 0.5 * m * dvx * dvx;
    const E_drag = m * vz0_snap * Math.abs(dvx);
    const ratio  = E_drag / E_lat;  // = 2 * vz0 / |dvx|

    const logMin = Math.log10(Math.max(E_lat,  1e-40)) - 1.0;
    const logMax = Math.log10(Math.max(E_drag, 1e-40)) + 0.5;

    document.getElementById('latBar').style.height  =
      (logBarFrac(E_lat,  logMin, logMax) * 100) + '%';
    document.getElementById('dragBar').style.height =
      (logBarFrac(E_drag, logMin, logMax) * 100) + '%';

    document.getElementById('latVal').textContent  = formatJoules(E_lat);
    document.getElementById('dragVal').textContent = formatJoules(E_drag);

    const stream_retained = vzf / vz0_snap;
    document.getElementById('effStreamFill').style.width =
      (stream_retained * 100).toFixed(4) + '%';
    document.getElementById('effStreamVal').textContent = vzf.toFixed(1) + ' m/s';

    const ratioStr = ratio >= 1e6
      ? (ratio / 1e6).toFixed(2) + ' million'
      : Math.round(ratio).toLocaleString();
    document.getElementById('effRatio').textContent =
      'Lateral steering is ' + ratioStr + 'x more energy-efficient than drag braking';
  }
  ```

- [ ] **Step 2: Verify function exists and guard works before sim runs**

  Reload page (before autoFire). In console:
  ```js
  typeof updateEfficiencyPanel   // "function"
  updateEfficiencyPanel()        // should not throw — guard returns early
  document.getElementById('effRatio').textContent  // still "—"
  ```

- [ ] **Step 3: Commit**

  ```bash
  git add index.html
  git commit -m "feat(efficiency): add updateEfficiencyPanel function"
  ```

---

## Task 7: Wire `updateEfficiencyPanel` into `showFinalStats`

**Files:**
- Modify: `index.html` lines 1070–1083 (`showFinalStats`)

- [ ] **Step 1: Add call at end of `showFinalStats`**

  Find `showFinalStats`. Current body:
  ```js
  function showFinalStats() {
    setStatus('Transit complete. Δvx = ' + ...);
    checkBoreField(transitData);
    if (currentMode === 1) { ... }
    if (currentMode === 0) { showCenterResult(...); }
    if (challengeState.active) { evaluateChallengeResult(...); }
  }
  ```

  Add one line before the closing `}`:
  ```js
    if (currentMode === 3) updateEfficiencyPanel();
  ```

  Full result:
  ```js
  function showFinalStats() {
    setStatus('Transit complete. Δvx = ' + (transitData.metrics.delta_vx * 1000).toFixed(4) + ' mm/s');
    checkBoreField(transitData);
    if (currentMode === 1) {
      document.getElementById('insetCanvas').style.display = 'block';
      drawInsetProgressive(transitData, transitData.T.length - 1);
    }
    if (currentMode === 0) {
      showCenterResult(transitData.metrics.delta_vx);
    }
    if (challengeState.active) {
      evaluateChallengeResult(transitData.metrics.delta_vx);
    }
    if (currentMode === 3) updateEfficiencyPanel();
  }
  ```

- [ ] **Step 2: Verify end-to-end — bars populate after transit in mode 3**

  Reload page. Click MODE button 3 times to reach EFFICIENCY mode. Wait for autoFire to complete (or click FIRE). In console:
  ```js
  document.getElementById('latVal').textContent   // e.g. "1.74 nJ" (not "—")
  document.getElementById('dragVal').textContent  // e.g. "12.00 mJ" (not "—")
  document.getElementById('effRatio').textContent // "Lateral steering is X.XXx million x more..."
  ```

- [ ] **Step 3: Verify bars have non-zero height**

  ```js
  document.getElementById('latBar').style.height   // e.g. "7.14%"  (> 4%, > 0)
  document.getElementById('dragBar').style.height  // e.g. "100%"
  ```

- [ ] **Step 4: Verify stream bar is nearly full**

  ```js
  parseFloat(document.getElementById('effStreamFill').style.width)  // >= 99.99
  ```

- [ ] **Step 5: Verify switching modes doesn't crash**

  ```js
  cycleMode()  // → FORCES
  cycleMode()  // → TRAJECTORY
  cycleMode()  // → FIELD
  cycleMode()  // → EFFICIENCY (bars animate back in from 0)
  // No errors in console
  ```

- [ ] **Step 6: Verify slider update refreshes bars**

  The existing slider path is: `onSlider()` → `onParamChange()` → `simulate()` → `augmentWithHistory()` → `autoFire()` → animation → `showFinalStats()` → `updateEfficiencyPanel()`. No extra wiring is needed — wiring `updateEfficiencyPanel` into `showFinalStats` (this task) is sufficient.

  Move the RPM slider while in EFFICIENCY mode. Wait for autoFire to complete. Verify `effRatio` textContent updates (ratio changes as dvx changes). Check console for errors.

- [ ] **Step 7: Commit**

  ```bash
  git add index.html
  git commit -m "feat(efficiency): wire updateEfficiencyPanel into showFinalStats"
  ```

---

## Task 8: Edge cases and final acceptance check

**Files:**
- `index.html` (read-only verification)

- [ ] **Step 1: Test zero-steering case**

  In console, set all coil amplitudes equal (symmetric), then re-fire:
  ```js
  // Set amp to symmetric via URL or sliders — or directly:
  currentP.amp = [1,1,1,1,1,1];
  currentP.delta = [0,0,0,0,0,0];
  onParamChange();
  // After transit completes, switch to EFFICIENCY mode
  currentMode = 3;
  applyMode();
  // In ~50ms, effRatio should say "No steering applied"
  ```

- [ ] **Step 2: Verify first-load pre-simulation state**

  Open a fresh tab (no autoFire yet — disable by commenting out `autoFire()` temporarily, or just check immediately on load before the animation completes):
  - Switch to EFFICIENCY mode immediately
  - Panel should show "—" placeholders, not NaN, not crash

- [ ] **Step 3: Verify no emoji in panel**

  ```js
  document.getElementById('efficiencyPanel').textContent.includes('🔋')  // false
  document.getElementById('efficiencyPanel').textContent.includes('⚡')  // false
  // Search for any non-ASCII in the panel
  /[^\x00-\x7F]/.test(document.getElementById('efficiencyPanel').innerHTML)  // false
  ```

  Note: the `formatJoules` function uses `uJ` (not `μJ`) to keep this clean.

- [ ] **Step 4: Final visual check**

  Open demo in browser. Click MODE until EFFICIENCY. After transit:
  - Green LATERAL bar is short (visible sliver, not invisible)
  - Red DRAG bar is tall (fills most of column height)
  - Stream bar is almost completely full (blue, ~99.99%)
  - Ratio text is yellow, readable, shows "million x"
  - No emoji anywhere in the panel

- [ ] **Step 5: Commit and push**

  ```bash
  git add index.html
  git commit -m "feat(efficiency): efficiency mode complete, all acceptance criteria verified"
  git push
  ```

---

## Acceptance Criteria (from spec)

Before marking the feature done, confirm all 13 criteria pass:

| # | Criterion | How to verify |
|---|---|---|
| 1 | MODE cycles FORCES→TRAJECTORY→FIELD→EFFICIENCY→FORCES, label updates | Console: cycle 4x, check button text each time |
| 2 | Canvas hidden, panel visible in mode 3 | Console: `threeCanvas.style.display === 'none'`, `efficiencyPanel.style.display === 'flex'` |
| 3 | Bars animate after autoFire | Visual: watch bars grow upward ~0.8s after transit |
| 4 | LATERAL bar >= 4% height | Console: `parseFloat(latBar.style.height) >= 4` |
| 5 | Energy values in correct SI prefix range | Console: latVal shows nJ/pJ, dragVal shows mJ |
| 6 | Stream bar visually ~99.99% full | Visual |
| 7 | Ratio derived from live sim, changes with params | Move a slider, re-fire, check ratio text changes |
| 8 | Switching away restores 3D view | Console: switch to mode 0, check `threeCanvas.style.display === ''` |
| 9 | Slider changes in EFFICIENCY mode update bars | Move slider, wait for autoFire, check bars refresh |
| 10 | No emoji | Console: `document.getElementById('efficiencyPanel').textContent` — visually inspect |
| 11 | First-load pre-sim shows "—" not NaN | Reload, immediately switch to mode 3 before transit completes |
| 12 | Zero-steering shows "No steering applied" | Set symmetric amps, re-fire, check effRatio text |
| 13 | Stream bar width = `VZ[exitIdx]/VZ[0] × 100%` | Console: compute manually, compare to rendered width |
