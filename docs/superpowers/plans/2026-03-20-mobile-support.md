# Mobile Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `index.html` usable on mobile phones by replacing the fixed sidebar with a bottom sheet, adding touch orbit controls to the 3D canvas, and fixing tap target sizes.

**Architecture:** All changes are CSS media queries + JS additions inside the single `index.html` file. Desktop layout is untouched. A `@media (max-width: 768px)` block hides the existing sidebar/bottom-bar and shows new mobile-only HTML elements. Touch listeners are added alongside existing mouse listeners in `makeOrbitControls`.

**Tech Stack:** Vanilla HTML/CSS/JS, Three.js (already loaded from CDN)

---

## File Structure

| File | Change |
|------|--------|
| `index.html` | Only file modified. Four additions: (1) mobile HTML elements, (2) mobile CSS block, (3) touch listeners in `makeOrbitControls`, (4) sheet drag JS + `setStatus` update. |

---

### Task 1: Add mobile HTML elements

**Files:**
- Modify: `index.html` — HTML body only

These two elements are hidden on desktop via CSS (added in Task 2). They must exist in the DOM before any JS runs.

- [ ] **Step 1: Add `#mobileActionBar` after `</div>` closing `#mainArea`**

Insert this block between the closing `</div>` of `#mainArea` (line ~353 in current file) and the opening `<div id="bottomBar">`:

```html
<!-- Mobile action bar: FIRE + RESET + status (hidden on desktop) -->
<div id="mobileActionBar">
  <button class="btn fire" id="mobileFireBtn" onclick="onFire()">&#9654; FIRE</button>
  <button class="btn" onclick="onReset()">RESET</button>
  <span id="mobileStatus">Ready &#8212; press FIRE to launch.</span>
</div>
```

- [ ] **Step 2: Add `#mobileSheet` just before `</body>`**

Insert this block immediately before `</body>`:

```html
<!-- Mobile bottom sheet (hidden on desktop) -->
<div id="mobileSheet">
  <div id="sheetHandle"></div>
  <div id="sheetSecondaryRow">
    <button class="btn" id="mobileSlowBtn" onclick="toggleSlow()">SLOW 0.1&#215;</button>
    <button class="btn" onclick="applyPresetByName('symmetric')">SYMMETRIC</button>
    <button class="btn" onclick="onValidate()">VALIDATE</button>
    <button class="btn" onclick="onShare()">SHARE</button>
    <button class="btn" id="mobileChallengeBtn" onclick="toggleChallenge()">CHALLENGE</button>
  </div>
  <div id="sheetContent">
    <div class="panel-section">
      <div class="section-title">Parameters</div>
      <div class="param-row">
        <div class="param-label">&#956; (magnetic moment) <span id="muVal2">60</span> A&#183;m&#178;</div>
        <input type="range" id="muSlider2" min="0" max="200" step="1" value="60" oninput="onSlider('mu',this.value,'muVal');onSlider('mu',this.value,'muVal2')">
      </div>
      <div class="param-row">
        <div class="param-label">RPM <span id="rpmVal2">50000</span></div>
        <input type="range" id="rpmSlider2" min="1000" max="150000" step="1000" value="50000" oninput="onSlider('rpm',this.value,'rpmVal');onSlider('rpm',this.value,'rpmVal2')">
      </div>
      <div class="param-row">
        <div class="param-label">k_drag <span id="kdVal2">0</span></div>
        <input type="range" id="kdSlider2" min="0" max="0.2" step="0.005" value="0" oninput="onSlider('kd',this.value,'kdVal',3);onSlider('kd',this.value,'kdVal2',3)">
      </div>
      <div class="param-row">
        <div class="param-label">Gradient (T/m) <span id="gradVal2">50</span></div>
        <input type="range" id="gradSlider2" min="1" max="200" step="1" value="50" oninput="onSlider('gradient',this.value,'gradVal');onSlider('gradient',this.value,'gradVal2')">
      </div>
      <div class="param-row">
        <div class="param-label">v_z0 (m/s) <span id="vzVal2">15000</span></div>
        <input type="range" id="vzSlider2" min="1000" max="50000" step="500" value="15000" oninput="onSlider('vz0',this.value,'vzVal');onSlider('vz0',this.value,'vzVal2')">
      </div>
      <div class="param-row">
        <div class="param-label">&#963; pulse (&#956;s) <span id="sigVal2">50</span></div>
        <input type="range" id="sigSlider2" min="5" max="200" step="5" value="50" oninput="onSlider('sigma_us',this.value,'sigVal');onSlider('sigma_us',this.value,'sigVal2')">
      </div>
      <div class="param-row">
        <div class="param-label">Amplitudes (comma sep)</div>
        <input type="text" id="ampText2" value="1.0,1.1,1.2,1.0,0.9,0.8" oninput="document.getElementById('ampText').value=this.value;onTextParamDebounced('amp',this.value)">
      </div>
      <div class="param-row">
        <div class="param-label">Timing offsets (&#956;s)</div>
        <input type="text" id="deltaText2" value="0,2,-1,1,-2,0" oninput="document.getElementById('deltaText').value=this.value;onTextParamDebounced('delta',this.value)">
      </div>
    </div>
    <div class="panel-section">
      <div class="section-title">Stats</div>
      <div class="stat-row"><span class="stat-label">&#916;vx (physics)</span><span class="stat-val" id="s_dvx2">&#8212;</span></div>
      <div class="stat-row"><span class="stat-label">&#916;vz (drag loss)</span><span class="stat-val" id="s_dvz2">&#8212;</span></div>
      <div class="stat-row"><span class="stat-label">Displacement x</span><span class="stat-val" id="s_dx2">&#8212;</span></div>
      <div class="stat-row"><span class="stat-label">Impulse |p|</span><span class="stat-val" id="s_imp2">&#8212;</span></div>
      <div class="stat-row"><span class="stat-label">Steering &#951;</span><span class="stat-val" id="s_eta2">&#8212;</span></div>
      <div class="stat-row"><span class="stat-label">Peak force</span><span class="stat-val" id="s_pkf2">&#8212;</span></div>
      <div class="stat-row"><span class="stat-label">E_loss</span><span class="stat-val" id="s_eloss2">&#8212;</span></div>
      <hr style="border-color:#223; margin: 4px 0">
      <div class="stat-row"><span class="stat-label">Rim speed</span><span class="stat-val" id="s_rim2">&#8212;</span></div>
      <div id="warningBanner2"></div>
    </div>
  </div>
</div>
```

**Note on duplicate controls:** The mobile sheet has its own slider/input elements with `id`s suffixed `2`. Their `oninput` handlers write to both the desktop element value and call the same `onSlider`/`onTextParamDebounced` functions. This keeps the desktop hidden-panel sliders in sync (needed because `readParams()` reads from the desktop element IDs). Stats panels with `2`-suffixed IDs are updated in Task 4.

- [ ] **Step 3: Verify HTML is valid — open `index.html` in browser, check no JS errors in console**

---

### Task 2: Add mobile CSS

**Files:**
- Modify: `index.html` — inside the `<style>` block, appended before `</style>`

- [ ] **Step 1: Append the mobile CSS block**

Add this entire block immediately before `</style>`:

```css
/* ── Mobile layout (≤768px) ──────────────────────────────── */
@media (max-width: 768px) {
  /* Hide desktop sidebar and bottom bar */
  #rightPanel { display: none; }
  #bottomBar  { display: none; }

  /* Stack canvas above mobile controls */
  #mainArea { flex-direction: column; position: relative; }

  /* Mobile action bar: FIRE + RESET + status */
  #mobileActionBar {
    display: flex;
    align-items: center;
    background: #111118;
    border-top: 1px solid #223;
    padding: 6px 12px;
    gap: 8px;
    flex-shrink: 0;
    z-index: 10;
  }
  #mobileStatus {
    flex: 1;
    font-size: 11px;
    color: #556;
    font-family: 'Courier New', monospace;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  /* Bottom sheet */
  #mobileSheet {
    display: flex;
    flex-direction: column;
    position: fixed;
    bottom: 0; left: 0; right: 0;
    background: #111118;
    border-top: 1px solid #334;
    border-radius: 12px 12px 0 0;
    height: 85vh;
    transform: translateY(calc(100% - 180px));
    transition: transform 0.3s ease;
    z-index: 50;
  }
  #mobileSheet.expanded  { transform: translateY(0); }
  #mobileSheet.collapsed { transform: translateY(calc(100% - 48px)); }

  #sheetHandle {
    width: 40px; height: 4px;
    background: #445;
    border-radius: 2px;
    margin: 10px auto 8px;
    flex-shrink: 0;
    touch-action: none;
    cursor: grab;
  }

  #sheetSecondaryRow {
    display: flex;
    gap: 6px;
    padding: 0 10px 8px;
    flex-wrap: wrap;
    flex-shrink: 0;
  }

  #sheetContent {
    overflow-y: auto;
    flex: 1;
    padding: 0 10px 20px;
    touch-action: pan-y;
  }

  /* Inset canvas: move to top so sheet peek doesn't cover it */
  #insetCanvas {
    bottom: auto;
    top: 10px;
    left: 10px;
  }

  /* Score display: push below topbar height (~40px) */
  #scoreDisplay {
    top: 50px;
    right: 8px;
    font-size: 11px;
    padding: 6px 8px;
  }

  /* Efficiency panel: narrower layout for reduced canvas height */
  #efficiencyPanel {
    padding: 16px 12px;
    gap: 14px;
  }
  #effBars {
    height: 120px;
    gap: 24px;
  }
  .eff-col { width: 70px; }
  #effStreamRow { font-size: 11px; }
  #effRatio { font-size: 13px; }

  /* Larger tap targets */
  .btn, .top-btn {
    min-height: 44px;
    padding: 10px 16px;
    font-size: 13px;
  }
  input[type=range] { height: 28px; }
  .param-label   { font-size: 13px; }
  .stat-row      { font-size: 12px; }
  .section-title { font-size: 11px; }
  #topbar h1 { font-size: 12px; }
  #topbar    { padding: 4px 10px; gap: 6px; }
}

/* Desktop: hide mobile-only elements */
@media (min-width: 769px) {
  #mobileActionBar { display: none; }
  #mobileSheet     { display: none; }
}
```

- [ ] **Step 2: Verify in browser at desktop width — no visual change to existing layout**

Open DevTools, confirm `#mobileActionBar` and `#mobileSheet` are `display: none` at >768px.

- [ ] **Step 3: Verify in browser at 375px width (DevTools mobile emulation)**

Switch DevTools to iPhone SE size. Confirm:
- Right panel gone, canvas fills width
- `#mobileActionBar` visible at bottom of canvas with FIRE + RESET
- Bottom sheet visible peeking ~180px from bottom
- Sheet has drag handle at top
- Secondary buttons (SLOW, SYMMETRIC, etc.) visible in sheet
- Sliders visible and scrollable in sheet content

---

### Task 3: Add touch orbit controls

**Files:**
- Modify: `index.html` — inside `makeOrbitControls` function

- [ ] **Step 1: Locate `makeOrbitControls` in the script**

Find the line: `domElement.addEventListener('wheel', ...` (currently the last event listener in `makeOrbitControls`, around line 445). Insert the touch listeners immediately after the `wheel` listener and before the `return` statement.

- [ ] **Step 2: Add touch listeners**

```js
// Touch orbit controls
let lastPinchDist = 0;
let touchStartX = 0, touchStartY = 0, touchStartTime = 0;

domElement.addEventListener('touchstart', e => {
  if (e.touches.length === 1) {
    isDragging = true;
    lastX = e.touches[0].clientX;
    lastY = e.touches[0].clientY;
    touchStartX = lastX;
    touchStartY = lastY;
    touchStartTime = Date.now();
  }
  if (e.touches.length === 2) {
    isDragging = false;
    const dx = e.touches[0].clientX - e.touches[1].clientX;
    const dy = e.touches[0].clientY - e.touches[1].clientY;
    lastPinchDist = Math.sqrt(dx*dx + dy*dy);
  }
}, { passive: true });

domElement.addEventListener('touchmove', e => {
  if (e.touches.length === 1 && isDragging) {
    theta -= (e.touches[0].clientX - lastX) * 0.01;
    phi = Math.max(0.1, Math.min(Math.PI - 0.1,
          phi - (e.touches[0].clientY - lastY) * 0.01));
    lastX = e.touches[0].clientX;
    lastY = e.touches[0].clientY;
    update();
  } else if (e.touches.length === 2) {
    isDragging = false;
    const dx = e.touches[0].clientX - e.touches[1].clientX;
    const dy = e.touches[0].clientY - e.touches[1].clientY;
    const dist = Math.sqrt(dx*dx + dy*dy);
    if (lastPinchDist > 0) {
      radius = Math.max(5, Math.min(40, radius - (dist - lastPinchDist) * 0.05));
      update();
    }
    lastPinchDist = dist;
  }
}, { passive: true });

domElement.addEventListener('touchend', e => {
  if (e.touches.length === 0) {
    const moved = Math.abs(e.changedTouches[0].clientX - touchStartX) +
                  Math.abs(e.changedTouches[0].clientY - touchStartY);
    if (Date.now() - touchStartTime < 200 && moved < 10) {
      // Tap: collapse sheet
      const sheet = document.getElementById('mobileSheet');
      if (sheet && !sheet.classList.contains('collapsed')) {
        sheet.classList.remove('expanded');
        sheet.classList.add('collapsed');
      }
    }
    isDragging = false;
    lastPinchDist = 0;
  }
}, { passive: true });
```

- [ ] **Step 3: Verify in DevTools touch emulation**

Enable touch emulation. Single-finger drag on canvas should rotate the scene. Two-finger pinch should zoom. Short tap on canvas should collapse the sheet.

---

### Task 4: Add sheet drag JS, update setStatus, update stats mirroring

**Files:**
- Modify: `index.html` — JS section (end of `<script>`, and `setStatus` function)

- [ ] **Step 1: Update `setStatus` to mirror to `#mobileStatus`**

Find `function setStatus(msg)` and replace it:

```js
function setStatus(msg) {
  document.getElementById('statusBar').textContent = msg;
  const ms = document.getElementById('mobileStatus');
  if (ms) ms.textContent = msg;
}
```

- [ ] **Step 2: Update `updateStatsPanel` to mirror to `*2` elements**

Find `function updateStatsPanel()` and add mirroring at the end of the function, after all existing `document.getElementById` calls:

```js
  // Mirror to mobile sheet stats
  ['dvx','dvz','dx','imp','eta','pkf','eloss','rim'].forEach(key => {
    const src = document.getElementById('s_' + key);
    const dst = document.getElementById('s_' + key + '2');
    if (src && dst) {
      dst.textContent = src.textContent;
      dst.className = src.className;
    }
  });
  const wb  = document.getElementById('warningBanner');
  const wb2 = document.getElementById('warningBanner2');
  if (wb && wb2) {
    wb2.innerHTML = wb.innerHTML;
    wb2.style.cssText = wb.style.cssText;
  }
  // end mirroring block
}
```

- [ ] **Step 3: Add `initSheetDrag` function and call it from `init()`**

Append this function before the closing `</script>` tag:

```js
// ============================================================
// MOBILE — sheet drag
// ============================================================
function initSheetDrag() {
  const sheet  = document.getElementById('mobileSheet');
  const handle = document.getElementById('sheetHandle');
  if (!sheet || !handle) return;

  let startY = 0, startTranslate = 0;

  function getTranslate() {
    const matrix = new DOMMatrixReadOnly(window.getComputedStyle(sheet).transform);
    return matrix.m42;
  }

  handle.addEventListener('touchstart', e => {
    startY = e.touches[0].clientY;
    startTranslate = getTranslate();
    sheet.style.transition = 'none';
  }, { passive: true });

  handle.addEventListener('touchmove', e => {
    const dy = e.touches[0].clientY - startY;
    const newY = Math.max(0, startTranslate + dy);
    sheet.style.transform = `translateY(${newY}px)`;
  }, { passive: true });

  handle.addEventListener('touchend', e => {
    sheet.style.transition = '';
    const dy = e.changedTouches[0].clientY - startY;
    sheet.classList.remove('expanded', 'collapsed');
    sheet.style.transform = '';
    if (dy < -60)      sheet.classList.add('expanded');
    else if (dy > 60)  sheet.classList.add('collapsed');
    // else: snap back to peek (no class = default CSS transform)
  }, { passive: true });
}
```

Then find the call to `autoFire()` near the bottom of `init()` and add `initSheetDrag()` after it:

```js
  autoFire();
  initSheetDrag();   // ← add this line
  setStatus('Ready — launching...');
```

- [ ] **Step 4: Update `applyPresetByName` and `toggleChallenge` to sync mobile `*2` inputs**

`applyPresetByName` sets slider values via `document.getElementById('muSlider').value = ...` etc. After each desktop slider sync block, add mirroring for the `2` variants. Find the block in `applyPresetByName` that sets slider values and add at the end:

```js
  // Sync mobile sheet sliders
  ['mu','rpm','kd','grad','vz','sig','amp','delta'].forEach(key => {
    const srcId = key === 'amp' ? 'ampText' : key === 'delta' ? 'deltaText' : key + 'Slider';
    const dstId = key === 'amp' ? 'ampText2' : key === 'delta' ? 'deltaText2' : key + 'Slider2';
    const lblId = (key === 'amp' || key === 'delta') ? null : key + 'Val2';
    const src = document.getElementById(srcId);
    const dst = document.getElementById(dstId);
    const lbl = lblId ? document.getElementById(lblId) : null;
    if (src && dst) dst.value = src.value;
    if (lbl) lbl.textContent = document.getElementById(key + 'Val')?.textContent || '';
  });
```

Similarly, `toggleChallenge` disables sliders by ID — add the `2` variants to its disable/enable arrays:

Find:
```js
['muSlider','rpmSlider','kdSlider','gradSlider','vzSlider','sigSlider','ampText'].forEach(id => {
```

Replace with (both occurrences — enable and disable):
```js
['muSlider','rpmSlider','kdSlider','gradSlider','vzSlider','sigSlider','ampText',
 'muSlider2','rpmSlider2','kdSlider2','gradSlider2','vzSlider2','sigSlider2','ampText2'].forEach(id => {
```

- [ ] **Step 5: Final verification in DevTools mobile emulation (375px)**

Checklist:
- [ ] FIRE button launches animation, status bar updates in mobile action bar
- [ ] Sliders in sheet respond, simulation reruns (ball moves differently)
- [ ] Drag handle up → sheet expands; drag down → collapses; release mid-drag → snaps to peek
- [ ] Tap on canvas → sheet collapses
- [ ] Single-finger drag on canvas → 3D scene rotates
- [ ] Two-finger pinch on canvas → zoom in/out
- [ ] VALIDATE shows badge in stats section of sheet
- [ ] CHALLENGE mode disables sliders in mobile sheet too
- [ ] Switch to desktop width → layout unchanged, original sidebar visible

- [ ] **Step 6: Commit**

```bash
git add index.html
git commit -m "feat: add mobile support (bottom sheet, touch orbit, responsive layout)"
```

---

## Notes

- `readParams()` reads from the desktop element IDs (`muSlider`, not `muSlider2`). The mobile sliders mirror their values to the desktop hidden elements via `oninput`, so physics always reads consistent values.
- No changes to physics kernel, simulation math, or Three.js scene.
- Desktop layout is completely unchanged — all mobile additions are gated by `@media (max-width: 768px)` or hidden via `@media (min-width: 769px)`.
