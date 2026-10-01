# SpinnyBall Workbench — UI Architecture & Design Tokens

Scoped rules for the standalone interactive physics laboratory in `workbench/`.

## Architecture & Zero-Build Contract
1. **Zero External Dependencies**: Pure browser ES modules (`.mjs`), HTML5 Canvas, and CSS. No bundler, no npm install, no framework.
2. **Worker Boundary**:
   - `worker.mjs` runs calculations off the UI thread via `validateConfig` and `simulate` from `physics.mjs`.
   - UI thread (`app.mjs`) handles form state, Canvas animation loop, scrubable timeline, and JSON/CSV import-export.
3. **No Invented Interpolation**: Replay displays only retained states or exact linear progress between discrete samples.

## Archival Scientific Design Tokens (`style.css`)
```css
:root {
  --paper:  #f5f3ec;  /* Warm archival paper background */
  --ink:    #243a34;  /* Deep forest slate text */
  --muted:  #65716b;  /* Technical commentary / secondary labels */
  --line:   #d9ddd1;  /* Structural hairline borders */
  --green:  #214d3e;  /* Primary active tab / execution buttons */
  --lime:   #d4eaa7;  /* High-visibility diagnostic accent */
  --orange: #d77845;  /* Escape velocity / focus ring / comparison overlay */
  --night:  #132a2c;  /* Deep space canvas viewport */
  --mono:   ui-monospace, SFMono-Regular, Consolas, monospace;
}
```

## Canvas & Interaction Rules
- **Stage Viewport**: Dark `#132a2c` background. Trajectory curves in high contrast (`#f4f4e2`, pinned run in amber dashed `#d77845`).
- **Accessible Controls**: Focus rings must use `outline: 3px solid var(--orange); outline-offset: 3px`.
- **Keyboard Operable**: Play/pause on Space, scrub on Arrow keys, tabs with `aria-pressed`.
