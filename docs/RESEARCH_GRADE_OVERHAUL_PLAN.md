# Research-Grade Overhaul Plan (Architecture + Physics)

## 0) Scope and Success Standard

This plan defines a full-system redesign path to move SpinnyBall from a mixed heuristic/prototype state to a **research-grade (9/10–10/10)** simulation and analysis platform. It explicitly preserves Halbach-based SGMS as a candidate baseline while testing alternatives that may be more physically defensible or easier to validate.

**Definition of done (research-grade):**
- Every major equation has provenance (derivation, assumption envelope, reference)
- Every simulation subsystem has benchmark tests and uncertainty quantification
- Heuristics are replaced with first-principles models, calibrated empirical fits, or explicitly bounded surrogate models
- Sparse and clustered packet architectures are both modeled and compared on common metrics
- CI gates enforce physics invariants, validation thresholds, and reproducibility

---

## 1) Architecture Decision Framing: Sparse vs Clustered Packet Regimes

## 1.1 Regime definitions
- **Sparse stream:** packet spacing `s >> packet diameter`; interactions dominated by nearest neighbors and station encounters.
- **Clustered stream:** `s ~ O(packet diameter)` to a few diameters; strong near-field coupling, collective modes, and increased collision/cascade risk.

## 1.2 Decision criteria
1. **Stability margin:** modal damping ratio, Floquet multiplier bounds, Lyapunov exponents
2. **Guidance authority:** required station force/impulse per pass
3. **Fault containment:** single-packet perturbation propagation rate
4. **Thermal and material load:** peak temperature/stress margins
5. **Validation tractability:** number of uncertain parameters and measurable observables

## 1.3 Recommended architecture stance
- Keep **sparse operation** as baseline for early research-grade validation (cleaner identifiability, reduced unmodeled near-field coupling).
- Keep **clustered operation** as an explicit advanced-mode branch for throughput studies once sparse model is validated.
- Introduce a **regime map** over `(u, s, m, station spacing, dipole moment, control bandwidth)` with boundaries for: stable sparse, quasi-clustered, unstable/cascade-prone.

---

## 2) System Decomposition and Modeling Stack

## 2.1 Subsystems
1. Packet rigid-body + spin dynamics
2. Magnetic interaction model (Halbach baseline + alternatives)
3. Stream continuum/tension dynamics (ring/string approximation)
4. Station interaction model (capture/trim/pass-through)
5. Thermal/material model (losses, demagnetization/quench margins)
6. Fault, insertion, and reliability model
7. Control/state-estimation model
8. Numerical integrator/UQ infrastructure

## 2.2 Layered architecture target
- **L0: Governing equations** (first principles)
- **L1: Reduced-order models** (validated simplifications)
- **L2: Surrogate/fit models** (only with error bars)
- **L3: Mission metrics and decision outputs**

Every L1/L2 model must trace to L0 + calibration dataset + validity envelope.

---

## 3) Literature-Backed Modeling + Validation Matrix

| Subsystem | Primary equations/models | Candidate references | Known weak points | Validation/benchmark criteria |
|---|---|---|---|---|
| Rigid-body spin dynamics | Euler rigid-body equations, torque coupling | Goldstein; Landau & Lifshitz | Numerical drift at high spin rates | Angular momentum drift <1e-6 per 100 revs on torque-free benchmark |
| Magnetics (baseline) | Dipole field/energy/force/torque, Halbach equivalent moment | Jackson; Halbach; Yonnet | Point-dipole misuse in near field | Compare against FEM/BEM for `r/a` sweep; require <5% force error in stated validity region |
| Magnetics (alternatives) | Magnetic bearing/lens formulations, induced-current models | Post et al.; Earnshaw constraints literature | Overreliance on one architecture | Cross-architecture A/B: Halbach vs active EM trim vs hybrid |
| Stream tension/continuum | `T = λu^2`, perturbation PDE/mode analysis | Hoyt & Forward; tether dynamics texts | Continuum model may fail in clustered regime | Discrete N-body vs continuum mode frequencies (<10% mismatch in sparse regime) |
| Station dynamics | Quadrupole lens linearization + nonlinear pass geometry | Accelerator beam optics analogs | Linear model outside small-offset zone | Define interaction envelope; Monte Carlo pass-through miss probability |
| Thermal/material | Eddy, hysteresis, radiative balance, demag curves | magnet materials handbooks; HTS literature | Simplified thermal constants and lumping | Thermal transient benchmarks; margin to material limits with uncertainty intervals |
| Reliability/insertion | Stochastic process + queue/inventory + latency | reliability engineering texts; point-process models | Ad hoc probability scaling with time horizon | Replace linear scaling with hazard/survival model validated against long-horizon runs |
| Controls/UQ | Linearized state-space + nonlinear MPC/LQR + robust margins | Åström & Murray; Zhou robust control | Mode switching assumptions implicit | Gain/phase and robust stability margins per mode; estimator consistency tests |

---

## 4) Heuristic Replacement Program (Keep / Replace / Research)

| Current heuristic pattern | Action | Replacement target |
|---|---|---|
| Stiffness calibration multipliers | **Replace** | Derive from geometry/material/FEM-informed correction factors with calibration dataset and posterior uncertainty |
| Velocity correction factors with hard caps | **Replace** | Loss-channel model (eddy/hysteresis/drag) or empirical fit with confidence intervals and validity limits |
| Short-horizon probability scaling to long horizons | **Replace** | Survival/hazard framework (non-homogeneous Poisson or renewal model) + goodness-of-fit tests |
| Binary mode assumptions without guard conditions | **Replace** | Explicit finite-state mode machine with transition guards/hysteresis |
| Pure point-dipole in close interactions | **Research then constrain** | Multipole/finite-size correction or minimum-separation validity gate |
| Halbach-centered architecture direction | **Keep (baseline)** | Maintain as baseline branch with explicit comparison to alternatives |

---

## 5) Packet Insertion and Sparse-System Reliability Guidance

## 5.1 Insertion model requirements
- Model insertion as a stochastic controlled event with state-dependent acceptance probability.
- Include synchronization error, station latency, packet spin state, and local stream density.
- Distinguish **hard failures** (collision/loss) from **soft failures** (timing miss, re-queue).

## 5.2 Reliability metrics
- Packet survival function `S(t)` and hazard `h(t)`
- Mean packets between interventions (MPBI)
- Insertion success probability conditioned on mode and local state
- Cascade amplification factor from single injection fault

## 5.3 Sparse-specific policy
- Keep minimum spacing guard `s_min` from validated interaction zone limits.
- Add insertion hold logic when local phase window uncertainty exceeds threshold.
- Require two independent acceptance checks: deterministic geometric gate + probabilistic reliability gate.

---

## 6) Modes, Control Assumptions, and Critical Interaction Zones

## 6.1 Required mode set
1. **Nominal cruise (sparse)**
2. **Station approach/interaction**
3. **Insertion/reinjection**
4. **Disturbance rejection**
5. **Degraded operation (sensor/actuator loss)**
6. **Safe mode / stream rundown**

## 6.2 Control/stability assumptions (must be explicit)
- Time-delay bounds, sensor noise model, estimator observability assumptions
- Linearization validity region for each mode
- Actuator saturation and slew limits
- Cross-mode transition conditions and forbidden transitions

## 6.3 Critical interaction zones
- Station lens capture volume
- Near-neighbor magnetic close-approach region
- Thermal stress hotspots (high-loss velocity bands)
- Mode-transition boundary layers (where controllers switch)

Each zone must have: monitored state variables, hard/soft limits, automatic mitigation action.

---

## 7) Benchmark and Validation Campaign (equation trust ladder)

## 7.1 Tiered validation
- **Tier A: Analytic checks** (closed-form invariants, dimensional checks)
- **Tier B: Canonical numerics** (manufactured solutions, convergence order)
- **Tier C: Cross-model** (reduced model vs high-fidelity/FEM)
- **Tier D: Scenario stress tests** (rare-event Monte Carlo, long horizon)

## 7.2 Mandatory benchmarks by subsystem
- Rigid-body torque-free conservation test
- Dipole force/torque sweep versus high-fidelity solver
- Continuum mode frequencies versus N-body simulation
- Station pass-through acceptance map versus nonlinear simulation
- Thermal transient against reference lumped/distributed models
- Reliability hazard model fit diagnostics (AIC/BIC, residual tests)

## 7.3 Acceptance thresholds
- Numerical convergence order achieved within ±0.2 of expected
- Conserved quantities drift below pre-set tolerances
- Prediction intervals calibrated (coverage within ±5%)
- Reproducibility: seeded runs reproduce statistics within CI bands

---

## 8) Prioritized Overhaul Roadmap

## Phase 1 (Weeks 1–3): Foundation hardening
- Build model registry with equation provenance and validity envelopes
- Add deterministic unit benchmarks for all major equations
- Introduce mode machine skeleton and explicit assumptions

## Phase 2 (Weeks 4–7): Magnetics + dynamics credibility
- Implement near-field correction path (or strict separation validity guard)
- Validate Halbach baseline against cross-model benchmark
- Add sparse-vs-clustered regime map and automated sweeps

## Phase 3 (Weeks 8–11): Reliability + insertion rigor
- Replace time-scaling heuristics with hazard/survival model
- Implement insertion state machine and reliability gates
- Run long-horizon rare-event campaign with uncertainty decomposition

## Phase 4 (Weeks 12–15): Alternatives and comparative science
- Add at least two architecture alternatives (e.g., hybrid EM trim, non-Halbach guidance variant)
- Run A/B comparisons under common benchmark protocol
- Select baseline by evidence, not architectural preference

## Phase 5 (Weeks 16–20): Research-grade packaging
- Freeze benchmark suite as CI physics gates
- Publish validation report with assumptions, limits, and error budgets
- Produce reproducibility bundle (configs, seeds, dataset manifests)

---

## 9) Keep / Replace / Research-Further Summary

## Keep now
- Halbach-based SGMS branch as baseline candidate
- Momentum-flux framing (`T = λu^2`) where regime assumptions hold
- Existing sweep infrastructure as execution scaffold (after statistical corrections)

## Replace now
- Calibration multipliers and capped correction heuristics without provenance
- Linear time-horizon probability scaling rules
- Implicit/undocumented mode transitions

## Research further before locking design
- Sparse-vs-clustered crossover boundaries
- Near-field finite-size magnetic interactions
- Architecture alternatives that may reduce validation burden

---

## 10) Immediate actionable backlog (first 10 tickets)

1. Add `model_provenance.yaml` for all core equations and assumptions.
2. Add benchmark tests for conservation + convergence in dynamics core.
3. Add validity guard for dipole approximation (`r/a` threshold + warning/error mode).
4. Implement mode enum + transition guard table.
5. Add insertion reliability state machine and telemetry outputs.
6. Replace time-horizon scaling with hazard-model module + fit tests.
7. Add sparse/clustered regime classifier utility and sweep report.
8. Add cross-model magnetics benchmark harness (analytic vs numerical high-fidelity).
9. Add uncertainty budget reporting for mission outputs.
10. Add CI “physics trust gates” with hard fail thresholds.

---

## 11) Reference starter list (for implementation work packages)

- J. D. Jackson, *Classical Electrodynamics* (dipole fields/interactions)
- K. Halbach (permanent-magnet array formulations)
- R. M. Yonnet (permanent magnet bearings and stiffness behavior)
- R. P. Hoyt & R. L. Forward (momentum-exchange tether dynamics)
- H. Goldstein et al., *Classical Mechanics* (rigid-body dynamics)
- K. J. Åström & R. M. Murray, *Feedback Systems* (control fundamentals)
- K. Zhou et al., *Robust and Optimal Control* (robust margins)

Use these as anchors; each implemented equation should cite the exact section/equation in code docs.
