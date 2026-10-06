# Interactive diagnostics and orbital speed sweep — design

## Purpose and scope

Give a curious user a complete, reproducible way to ask two questions: “What happened at this point in the run?” and “How does changing launch speed alter the result?” The release adds exact-sample inspection to the existing diagnostic plots for all three laboratories, plus a deterministic speed sweep in the orbital laboratory. It does not claim statistical uncertainty, a mission prediction, or physical feasibility of the historical mass-stream proposal.

The existing Kepler overlay and external-trajectory audit remain. Correct the overlay for sub-circular initial speeds: those launches begin at apoapsis, whereas the current ellipse generator assumes periapsis. Also remove untrusted file names and audit warnings from `innerHTML` in the external-audit dialog before extending the interface.

## Dependency decision from the Context7 review

Keep the shipped `workbench/` free of external JavaScript, CDN calls, build steps, and Python research imports. [Chart.js](https://www.chartjs.org/docs/latest/charts/scatter) could plot independent `{x,y}` samples; [uPlot](https://github.com/leeoniya/uPlot/blob/master/docs/README.md) has excellent cursors but requires aligned x values; [Plotly.js](https://plotly.com/graphs/) offers more interaction with a larger payload; [Three.js](https://threejs.org/manual/pages/installation.html) would improve camera control for the spin view. None supplies the physical assumptions or validation of a sweep. The current two Canvas plots contain a small amount of code, and zero dependencies are an explicit repository contract. Implement the specified inspection in Canvas now. A library adoption or 3D view is a separate decision and implementation plan.

## Exact-sample inspection

Both existing time plots gain a visible, keyboard-reachable “Inspect sample” readout. Hovering or focusing a plot selects the nearest retained sample in the displayed run, with time, plotted value, and units. The displayed point and readout must always come from a retained sample; no interpolated physical state is implied. If a comparison is pinned, show its independently nearest sample with its **own** time, because the runs may have different steps and retained time grids. Arrow keys move one retained sample at a time; Home/End select the first/last. Escape clears inspection. The scene replay cursor remains independent and continues to show its current sampled state. Touch users can tap a plot to inspect. Text remains available when Canvas is not visible to assistive technology.

## Orbital speed sweep

Show a “Sweep launch speed” panel only in Orbital motion. A sweep uses the **currently displayed and applied** orbit configuration as its base, changing only `speed`. Unapplied form edits must be applied first. User inputs are minimum speed, maximum speed (both in × circular speed, within the existing 0.1–2 range), and an integer point count from 3 to 21. Require min < max and a conservative total-work cap of `count * ceil(duration / dt) <= 200000`; reject before starting if exceeded. Include both endpoints and generate evenly spaced speeds deterministically.

Each point runs the same `simulate` engine in a dedicated module Worker with a sample limit of 2. The engine still checks every integration step and includes maxima over every step; the sweep retains only its initial and final states. Store speed, the analytically known initial specific energy, energy classification (`bound` if negative, `unbound` otherwise), run status, last exterior distance in km, final time, and maximum scaled energy and angular-momentum balance residuals. A surface stop is explicitly labeled “stopped before surface crossing,” never “impact at finalTime.” The √2 speed ratio is displayed as an analytic bound/unbound threshold, not an empirical discovery. The finite-duration final distance must not be labeled “escaped” merely because it is large or because initial energy is positive.

Report progress after each point. Cancel terminates the dedicated sweep Worker; switching laboratories also cancels it. The ordinary simulation Worker and displayed run remain usable. A sweep chart shows speed on x and final distance from center (km) on y, with point status and a √2 reference line. An adjacent table gives exact numbers, units, energy classification, surface status, and both balance maxima. Selecting a sweep row sets that speed in the ordinary orbit controls and runs a full-resolution ordinary experiment, preserving the existing pin/compare behavior. The sweep is a distinct result, not a replacement for the displayed run.

## Provenance and interchange

Sweep JSON contains `kind: "orbit-speed-sweep"`, `sweepSchemaVersion: 1`, the engine version, the validated base configuration, range, generated rows, assumptions, method, units, and export time. A sweep CSV contains one row per speed with named unit-bearing columns. The JSON import path recognizes sweep files separately from ordinary experiments; it validates only version and configuration/range, then **recomputes** rows. Uploaded rows or diagnostics never become evidence. Incompatible versions and malformed files get actionable errors. Exports describe the displayed sweep, not unapplied edits.

## Acceptance and limits

- Independent tests establish the analytic √2 boundary, endpoint inclusion, deterministic replay, full-step residual maxima despite two retained samples, validation limits, surface-stop wording, and rejection of forged imported rows.
- Existing 19 physics tests, audit tests, launcher tests, browser import/export, and ordinary replay continue to pass. Test sub-circular Kepler overlay alignment against the actual initial position.
- Browser inspection covers mouse, touch, keyboard, mobile width, dissimilar comparison time grids, cancellation, repeated runs, and no console errors.
- Document the sweep's numerical meaning and its absence of probability distributions or uncertainty bounds. Do not call a numerical balance pass physical validation.
