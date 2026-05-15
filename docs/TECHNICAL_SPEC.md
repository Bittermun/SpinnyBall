# Technical Specification — SGMS Anchor

## System Architecture

Spin-stabilized gyroscopic mass-stream anchor for cislunar station-keeping. Stiffness from momentum-flux reaction + flux-pinning superconducting bearings. Feedback controller modulates effective stiffness via gain `g_gain`.

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

| Parameter | Operational | Optimal | Δ |
|-----------|-------------|---------|---|
| u (m/s) | 15,000 | 588.8 | -63% |
| lam (kg/m) | 72.92 | 15.47 | -7% |
| mp (kg) | 35 | 4.57 | -43% |
| g_gain | 0.000140 | 0.0004 | +186% |
| k_eff | 2.3e6 | 8,145 | — |

k_eff target: 6,000–10,000 N/m.

## Physical Models

**Effective stiffness**: `k_eff = λu²g_gain + k_fp`

**Momentum flux**: `F = λu²`

**Centrifugal stress**: `σ = mω²/(4πr)` — CFRP (2 GPa) required at 50k RPM (σ ≈ 765 MPa, SF=2.61). BFRP (800 MPa) insufficient.

**Thermal limit**: T_ss = 379 K at 15 km/s. SmCo operating at 106°C with 15.7% margin below 450 K limit. GdBCO quench risk at this velocity.

## Material: SmCo vs GdBCO

- **GdBCO**: Extreme pinning, quench risk at 15 km/s (eddy heating exceeds radiative cooling)
- **SmCo**: Passively stable to 300°C. 95% less auxiliary power than cryogenic HTS systems

## Control-Latency Stability

Empirical stability boundary (JAX sweep): latency < 20ms for 2.1x margin. η_ind = 0.82 safety limit tolerates 42ms delay.

## JAX Acceleration

LQR surrogate + vectorized RK4 achieves 0.96s per full sweep (3,751x speedup).

## Statistical Methodology

Wilson score CI (binomial), normal CI (means), 5% CI width convergence threshold, adaptive MC (N=100 to N=10,000), containment threshold = 2 nodes, cascade threshold = 1.05x stiffness reduction.
