# Workbench models and numerical contracts

All integration uses SI units. Display conversions are explicit. These are idealizations for concept testing, with no empirical validation of a spacecraft, bearing, material or launch system.

## Orbital motion

State: [x, y, vx, vy] in m and m/s. **a = −μ r / |r|³**. Initial position is [r₀, 0] and velocity [0, s √(μ/r₀)], where s is the speed/circular-speed ratio.

Rounded illustrative lunar-scale constants: μ = 4.905 × 10¹² m³/s² and spherical stop radius R = 1,737,400 m. They are fixed model parameters, not high-precision ephemerides. There is no Earth/Sun perturbation, terrain, gravity harmonic, drag or thrust.

Velocity Verlet advances position and velocity using endpoint accelerations. Specific energy is ε = |v|²/2 − μ/r (J/kg); specific angular momentum is h = x vy − y vx (m²/s). Energy residual scale is μ/r₀, avoiding division by nearly zero energy in a parabolic orbit. Momentum residual scale is |h₀|.

For a bound orbit a = −μ/(2ε), e = |s² − 1| and T = 2π √(a³/μ). Tests independently solve Kepler's equation for a periapsis-started ellipse and compare trajectory positions. At ε ≥ 0 the conic is unbound.

If the next position enters the surface, the run stops at the last exterior state and reports a crossing in the next step. This is not precise event timing or a collision response. The allowed steps are small relative to the surface dynamical time.

Source: [Richard Battin, MIT 16.346, The Two Body Problem](https://ocw.mit.edu/courses/16-346-astrodynamics-fall-2008/resources/lec_01/). [JPL astrodynamic parameters](https://ssd.jpl.nasa.gov/astro_par.html) provide context for physical constants versus ephemerides.

## Torque-free spin

State: body-frame angular velocity ω (rad/s) and a scalar-first unit quaternion q mapping body vectors to the inertial frame. Principal inertias I are in kg m².

**I ω̇ + ω × (I ω) = 0**, with **q̇ = ½ q ⊗ [0, ω]**.

Classical RK4 advances ω and q together. Only quaternion norm is restored after each step; energy and momentum are not artificially repaired. Energy E = ½ Σ Iᵢωᵢ²; inertial angular momentum L = rotate(q, Iω). The full vector is compared to L₀. Reference scales are E₀ and |L₀|, with 10⁻¹² floors for rest states.

Positive inertias satisfy strict triangle inequalities for a solid ellipsoid. Displayed relative axis lengths are √(I₂+I₃−I₁), √(I₁+I₃−I₂), √(I₁+I₂−I₃); absolute size is illustrative. Inputs require dt |ω₀| Imax/Imin ≤ 0.1. This is an input guard, not an adaptive error estimator.

Tests include exact spherical rotation, analytic symmetric-body precession, fourth-order orientation convergence and intermediate-axis reversal while inertial L stays nearly fixed. Minimum/maximum inertia axes are stable to small perturbations; the intermediate axis is unstable.

Source: [Sussman and Wisdom, Structure and Interpretation of Classical Mechanics, chapter 2](https://mitp-content-server.mit.edu/books/content/sectbyfn/books_pres_0/9579/sicm_edition_2.zip/chapter002.html).

## Momentum exchange

State: [xA, xB, vA, vB]. Two point masses move on a line, connected by an ideal massless linear spring (k in N/m, rest length ℓ = 4 m). **FA = k(xB − xA − ℓ)** and **FB = −FA + Fext**.

Velocity Verlet advances both masses. Initial separation is ℓ + extension, the initial center of mass is zero, and both masses have the configured common velocity. Separation is signed; there is no collision geometry.

Mechanical energy E = ½ mA vA² + ½ mB vB² + ½ k(extension)². External work W = Fext(xB − xB₀). Momentum P = mA vA + mB vB; external impulse J = Fext t. Diagnostics use E − E₀ − W and P − P₀ − J.

Energy scale: max(E₀, |Fext ℓ|, 1 J). Momentum scale: max(M|v₀|, √(kM) extension, |Fext duration|, 1 kg m/s), where M = mA+mB. Analytic center of mass: v₀t + Fext t²/(2M). Relative oscillation frequency: Ω = √(k(1/mA+1/mB)); equilibrium extension: Fext/(mBΩ²). Independent oscillator tests compare individual positions and center-of-mass motion. Inputs require dt Ω ≤ 0.1.

Internal forces redistribute momentum without changing the total. This example isolates system boundaries and external momentum sources; it does not establish or rule out every orbital control mechanism.

Source: [MIT 16.07, Conservation Laws for Systems of Particles](https://ocw.mit.edu/courses/16-07-dynamics-fall-2009/resources/mit16_07f09_lec11/).

## Accuracy and exports

Maximum residuals include every step. Normally at most 1,200 samples are retained, including endpoints. Plots and replay use those samples. The chart's percent is 100 × scaled energy residual; the 0.1% indicator is a diagnostic convention, not calibrated uncertainty.

JSON carries column names, units, method, assumptions and provenance. CSV state columns are SI, its signal uses the JSON signalLabel units, and its energy is J/kg for orbit or J for the other models. Save JSON alongside CSV. External work is relevant only for momentum exchange.

Implementations are original project code; no external code was copied. [Daniel Schroeder's interactive molecular dynamics](https://physics.weber.edu/schroeder/md/) informed the interaction pattern of immediate controls and visible physical diagnostics.
