# Mobile Support Design — SGMS
**Date:** 2026-03-20

## Problem

The app currently loads on mobile but is unusable:
- Fixed 280px right sidebar leaves ~95px for the 3D canvas on a 375px phone
- Orbit controls only handle mouse events; touch does nothing
- Buttons have 4–5px padding; font sizes are 10–12px — too small for touch
- 7 bottom-bar buttons overflow off-screen on narrow devices

## Decisions

- **Breakpoint:** `@media (max-width: 768px)` for all mobile overrides
- **Layout:** bottom sheet replaces the right sidebar
- **Bottom sheet default state:** partially open (~180px peek)
- **Action buttons:** FIRE + RESET + status bar pinned above sheet handle; SLOW, SYMMETRIC, VALIDATE, SHARE, CHALLENGE inside sheet
- **Touch orbit:** single-finger rotate, two-finger pinch zoom; sheet collapses on tap (not rotate) via tap heuristic

---

## Section 1 — Layout

On mobile the two-column flex layout collapses to a single column with a bottom sheet.

**HTML additions:**
- `#mobileActionBar` — new `<div>` between `#mainArea` and `#bottomBar`. Contains FIRE button, RESET button, and a `#mobileStatus` span (mirrors `setStatus()` output). Visible only on mobile.
- `#mobileSheet` — new `<div>` at the bottom of `<body>`. Contains: `#sheetHandle`, `#sheetSecondaryRow` (SLOW, SYMMETRIC, VALIDATE, SHARE, CHALLENGE buttons), and `#sheetContent` (scrollable area with cloned parameter + stats sections). Visible only on mobile.
- The original `#rightPanel` and `#bottomBar` are hidden on mobile via CSS.

**`setStatus()` updated** to also write to `#mobileStatus` when present:
```js
function setStatus(msg) {
  document.getElementById('statusBar').textContent = msg;
  const ms = document.getElementById('mobileStatus');
  if (ms) ms.textContent = msg;
}
```

**CSS — mobile layout:**
```css
@media (max-width: 768px) {
  #rightPanel { display: none; }
  #bottomBar  { display: none; }

  #mainArea {
    flex-direction: column;
    position: relative;
  }

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

  #mobileSheet {
    display: flex;
    flex-direction: column;
    position: fixed;
    bottom: 0; left: 0; right: 0;
    background: #111118;
    border-top: 1px solid #334;
    border-radius: 12px 12px 0 0;
    height: 85vh;
    transform: translateY(calc(100% - 180px));  /* peek: 180px visible */
    transition: transform 0.3s ease;
    z-index: 50;
  }
  #mobileSheet.expanded  { transform: translateY(0); }
  #mobileSheet.collapsed { transform: translateY(calc(100% - 48px)); }

  /* Handle drag: touch-action only on the handle, not the whole sheet */
  #sheetHandle {
    width: 40px; height: 4px;
    background: #445;
    border-radius: 2px;
    margin: 10px auto 8px;
    flex-shrink: 0;
    touch-action: none;   /* only the handle intercepts vertical drag */
    cursor: grab;
  }

  #sheetSecondaryRow {
    display: flex;
    gap: 6px;
    padding: 0 10px 8px;
    flex-wrap: wrap;
    flex-shrink: 0;
  }

  /* Inner scroll area: allow native pan-y */
  #sheetContent {
    overflow-y: auto;
    flex: 1;
    padding: 0 10px 20px;
    touch-action: pan-y;
  }

  /* Inset canvas: move to top-left so it clears the sheet peek */
  #insetCanvas {
    bottom: auto;
    top: 10px;
    left: 10px;
  }

  /* Score display: push below topbar */
  #scoreDisplay {
    top: 50px;
    right: 8px;
    font-size: 11px;
    padding: 6px 8px;
  }

  /* Efficiency panel: smaller padding for reduced canvas height */
  #efficiencyPanel {
    padding: 16px 12px;
    gap: 14px;
  }
  #effBars {
    flex-direction: row;
    height: 120px;
    gap: 24px;
  }
  .eff-col { width: 70px; }
  #effStreamRow { font-size: 11px; }
  #effRatio { font-size: 13px; }
}

/* Desktop: hide mobile-only elements */
@media (min-width: 769px) {
  #mobileActionBar { display: none; }
  #mobileSheet     { display: none; }
}
```

**Sheet drag JS (attached to `#sheetHandle` only):**
```js
function initSheetDrag() {
  const sheet = document.getElementById('mobileSheet');
  const handle = document.getElementById('sheetHandle');
  if (!sheet || !handle) return;

  let startY = 0, startTranslate = 0;

  function getTranslate() {
    const style = window.getComputedStyle(sheet);
    const matrix = new DOMMatrixReadOnly(style.transform);
    return matrix.m42; // translateY
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
    if (dy < -60) {
      sheet.classList.remove('collapsed');
      sheet.classList.add('expanded');
    } else if (dy > 60) {
      sheet.classList.remove('expanded');
      sheet.classList.add('collapsed');
    } else {
      // snap back to peek (remove both classes)
      sheet.classList.remove('expanded', 'collapsed');
      sheet.style.transform = '';
    }
  }, { passive: true });
}
```

Call `initSheetDrag()` at the end of `init()`.

---

## Section 2 — Touch Orbit Controls

`makeOrbitControls` gains touch event listeners. All touch handling is consolidated into a single `touchmove` listener to avoid race conditions when finger count changes.

```js
// Inside makeOrbitControls, after existing mouse listeners:

let lastPinchDist = 0;
let touchStartX = 0, touchStartY = 0;
let touchStartTime = 0;

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

// Single consolidated touchmove handler
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
    // Tap detection: short duration + minimal movement → collapse sheet
    const dt = Date.now() - touchStartTime;
    const moved = Math.abs(e.changedTouches[0].clientX - touchStartX) +
                  Math.abs(e.changedTouches[0].clientY - touchStartY);
    if (dt < 200 && moved < 10) {
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

---

## Section 3 — Tap Targets & Typography

All inside `@media (max-width: 768px)`:

```css
.btn, .top-btn {
  min-height: 44px;
  padding: 10px 16px;
  font-size: 13px;
}

input[type=range] {
  height: 28px;
}

.param-label   { font-size: 13px; }
.stat-row      { font-size: 12px; }
.section-title { font-size: 11px; }

#topbar h1 { font-size: 12px; }
#topbar    { padding: 4px 10px; gap: 6px; }
```

---

## Files Changed

| File | Change |
|------|--------|
| `index.html` | Add `#mobileActionBar` and `#mobileSheet` HTML; add mobile CSS block; add touch listeners to `makeOrbitControls`; add sheet drag JS; update `setStatus()` to write to `#mobileStatus` |

No new files. No changes to physics kernel, simulation logic, or desktop layout.

## Out of Scope

- Landscape-specific layout (bottom sheet works acceptably in landscape)
- PWA / installable app
- Offline support
