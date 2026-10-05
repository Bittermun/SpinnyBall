# UI Features: External Audit Drag-and-Drop & Kepler Analytic Overlay

## Context & Motivation
Following the creation of the Trajectory Invariant Audit Core (`workbench/audit.mjs`), two high-leverage UI enhancements will directly elevate the SpinnyBall interactive workbench:
1. **Analytic Reference Ellipse Overlay**: Renders the exact closed Keplerian orbit as a faint dashed reference on the orbital canvas so users immediately visualize numerical integrator phase distortion or apsidal precession when varying $\Delta t$.
2. **External Trajectory Audit Drag-and-Drop / File Picker**: Allows developers to drag any simulation CSV/JSON directly onto the workbench interface to overlay the trajectory and inspect live conservation metrics in the browser.

---

## Component 1: Kepler Analytic Reference Overlay (`workbench/app.mjs`)

### Exact Visual Reference & Design
In `workbench/app.mjs` (`drawOrbit` function):
- The existing canvas uses:
  - Moon: radial gradient circle with craters at `(0, 0)`.
  - Pinned trajectory: amber dashed line (`colors.orange`, `[4, 5]` dash).
  - Computed trajectory: green line (`colors.green`).
  - Base trajectory path: dark sage (`#54786c`).
- **New Overlay**:
  - Computes the exact closed Keplerian ellipse for bound orbits ($E < 0$) using standard Kepler elements:
    $$r(\theta) = \frac{a(1 - e^2)}{1 + e \cos(\theta)}$$
    or parametric Kepler coordinates:
    $$x(E) = a(\cos E - e), \quad y(E) = a\sqrt{1 - e^2}\sin E$$
  - Render style: faint dashed cyan/light-slate curve (`#6bb3a055`, dash `[2, 4]`, width `1`).
  - Legend indicator in `.scene-legend`: adds "Exact Keplerian orbit" indicator when in Orbit mode.

---

## Component 2: External Trajectory Audit Drag-and-Drop (`workbench/index.html` & `workbench/app.mjs`)

### UI Structure & Integration
1. **HTML Additions** (`workbench/index.html`):
   - Add an "Audit file" button in `.result-actions`:
     ```html
     <button id="auditButton">Audit external run</button>
     <input id="auditFile" type="file" accept=".csv,.json,text/csv,application/json" hidden>
     ```
   - Add a modal dialog or toast panel (`#auditDialog`) displaying the scorecard:
     - Overall status (`PASSED < 0.1%` vs `FAILED`).
     - Specific Energy drift %, Angular Momentum drift %, Eccentricity drift, and Parasitic acceleration.
     - List of warnings.
2. **Drag-and-Drop Handler**:
   - Attach `dragover` and `drop` event listeners to `#scene` / `#stage`.
   - When a CSV or JSON file is dropped:
     - Parses the file using `parseTrajectoryData` from `./audit.mjs`.
     - Computes invariants with `auditOrbitTrajectory`.
     - Displays the audit results modal and overlays the external trajectory path on the canvas in vibrant cyan (`#4ecdc4`).

---

## Edge Cases & Precautions
1. **Never mutate core simulation state**: Uploaded external trajectories are audits/overlays only. They must never overwrite `result` in a way that tricks SpinnyBall into claiming it simulated the external file.
2. **Coordinate frame scaling**: If the imported external trajectory has coordinates on Earth-scale ($r \approx 7 \times 10^6$ m) while SpinnyBall is set to lunar scale ($r \approx 2 \times 10^6$ m), the canvas bounding-box scaler in `drawOrbit` must adapt gracefully without NaN or clipping.
3. **Memory & Parsing Limits**: Cap uploaded files at 8 MB (consistent with existing `importFile` limit).
