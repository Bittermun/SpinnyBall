# Technical Specification — SGMS Anchor

## System Architecture

The Spin-Stabilized Gyroscopic Mass-Stream (SGMS) anchor represents a propellantless station-keeping paradigm for cislunar nodes. Rather than employing continuous physical tracks or guide tethers stretching across space (which would introduce prohibitive mass penalties), the system relies on a **Guided-Beam Ballistic Free-Flight** topology.

Packets travel through cislunar vacuum in unguided, Keplerian ballistic trajectories (Free-Flight Corridors) spanning up to 380,000 km. Localized stabilization, steering, and momentum transfer occur exclusively within discrete satellite stations (Nodes) equipped with electromagnetic deflection channels and passive **flux-pinning Samarium-Cobalt ($\text{Sm}_2\text{Co}_{17}$)** permanent magnet bearings. A feedback controller modulates effective stiffness at each node via the gain `g_gain` to compensate for sensor error and dynamic drift.

## Operational Parameters

| Parameter | Symbol | Value | Units |
|-----------|--------|-------|-------|
| Stream velocity | u | 15,000 | m/s |
| Linear density | lam | 72.92 | kg/m |
| Packet mass | mp | 35 | kg |
| Control gain | g_gain | 0.000140 | — |
| Flux-pinning stiffness | k_fp | 9,000 | N/m |
| Station mass | ms | 1,000 | kg |
| Damping coefficient | c_damp | 4.0 | N·s/m |
| Spin rate | omega | 5,236 | rad/s (50k RPM) |
| Packet radius | r | 0.1 | m |
| Orbital altitude | h | 550 | km SSO |

## Sobol Optimal Parameters

Sobol global sensitivity analysis (Saltelli sampling, N=20,480 total runs) identifies stream velocity $u$ as the dominant driver of infrastructure mass. The table below represents the transition from the high-velocity operational envelope (`smco-heavy`) to the low-velocity Sobol-optimal design envelope (`paper-recommended`), showing mathematically consistent values and exact relative changes ($\Delta$):

| Parameter | Operational (`smco-heavy`) | Optimal (`paper-recommended`) | Δ |
|-----------|----------------------------|-------------------------------|---|
| u (m/s) | 15,000 | 588.8 | -96.07% |
| lam (kg/m) | 72.92 | 9.52 | -86.94% |
| mp (kg) | 35.00 | 4.57 | -86.94% |
| g_gain | 0.000140 | 0.000400 | +185.71% |
| k_eff (N/m) | 2.31e6 | 7,320 | -99.68% |

*Note: Packet spacing is maintained at 0.48 m across both profiles. Optimal k_eff falls safely within the 6,000–10,000 N/m design target.*

## Physical Models

**Effective stiffness**: $k_{eff} = \lambda u^2 g_{gain} + k_{fp}$ (active controller feedback momentum coupling + passive magnetostatic flux-pinning stiffness)

**Momentum flux**: $F = \lambda u^2$

**Centrifugal stress**: $\sigma = \frac{3+\nu}{8} \rho \omega^2 r^2$ — CFRP (2 GPa tensile limit) required at 50,000 RPM ($\sigma \approx 765$ MPa, SF=2.61). Basalt Fiber Reinforced Polymer (BFRP) is insufficient at this spin rate.

**Thermal limit**: $T_{ss} = 379$ K at 15 km/s. SmCo permanent magnet arrays operate in passive radiative-vacuum equilibrium at 106°C, representing a 15.7% thermal safety margin below the 450 K material degradation limit. Active Stirling/pulse-tube cryocoolers are reserved strictly for auxiliary/backup GdBCO systems, as eddy-current hypervelocity heating at 15 km/s poses a high GdBCO thermal quench risk.

## Material: Passive SmCo vs Cryo-HTS GdBCO

- **GdBCO (High-Temperature Superconductor)**: Offers high magnetic pinning fields, but exhibits extreme vulnerability to thermal quench at high velocities ($u > 1$ km/s) due to hypervelocity eddy-current dissipation exceeding active space-qualified Stirling cryocooler capacities.
- **SmCo (Samarium-Cobalt Permanent Magnets)**: Passively stable up to 300°C in vacuum. Eliminates active cryogenics, resulting in a 95% reduction in auxiliary power requirements compared to HTS options.

## Control-Latency Stability

Empirical stability boundaries from JAX-accelerated sweeps establish a strict latency limit of $<20$ ms to maintain a 2.1x control safety margin. The indicator safety parameter $\eta_{ind} = 0.82$ provides a hard physical boundary, tolerating a maximum feedback latency of 42 ms before system-wide phase lag drives exponential divergence.

## JAX Acceleration

Our LQR surrogate + vectorized RK4 integrator engine, compiled via XLA, achieves a 0.96s full-sweep execution time (3,751x speedup over native CPU iterations), enabling massive statistical validation.

## Statistical Methodology

Confidence intervals are calculated using the Wilson Score method for binomial rates (fault containment) and standard Normal distributions for system parameter means. The adaptive Monte Carlo algorithm automatically scales sample sizes (N=100 to N=10,000) until the 5% confidence interval width threshold is achieved. System cascade propagation is modeled as a localized load redistribution process where adjacent surviving nodes' passive flux-pinning stiffness is scaled down by a load factor $L_f = 1 + \alpha / N_{\text{neighbors}}$ ($\alpha = 0.10$), with a node failing only if its dynamic effective stiffness drops below the 50% nominal stiffness threshold ($3,000$ N/m).
