# SGMS vs Conventional Station-Keeping Comparison

## Assumptions
- Station mass: 1000 kg
- Orbit: 550 km SSO
- Required station-keeping force: 4.2 N (from J2 + SRP + drag with 10x margin)
- Mission duration: 1 year (8,760 hours)

## Propellant mass formula
For chemical/electric propulsion:
`m_prop = F * t / (Isp * g0)` where `t = 3.156e7 s/yr`, `g0 = 9.81 m/s²`

## Comparison Table

| Method | Thrust (N) | Isp (s) | Propellant (kg/yr) | Power (kW) | Infra Mass (kg) | Lifetime | TRL |
|--------|-----------|---------|---------------------|------------|------------------|----------|-----|
| Cold gas (N₂) | 0.1–10 | 65 | 20,870 | 0 | ~50 (tank) | Propellant-limited | 9 |
| Hydrazine | 0.5–22 | 220 | 6,170 | 0 | ~30 (tank) | Propellant-limited | 9 |
| Hall effect (SPT-100) | 0.01–1 | 1,500 | 905 | 1.5 | ~20 (thruster+PPU) | Propellant-limited | 9 |
| Ion (NSTAR) | 0.01–0.1 | 3,100 | 437 | 2.3 | ~30 (thruster+PPU) | Propellant-limited | 8 |
| SGMS (589 m/s, GdBCO) | 4.2 | N/A | 0 | 4,360 | 37,200 | 69,600 | 1–2 |
| SGMS (15 km/s, SmCo) | 4.2 | N/A | 0 | 0.00565 | 280 | 102 | 1–2 |

## SGMS Values

Populated by running:
```python
from src.sgms_anchor_v1 import mission_level_metrics

# Sobol-optimal GdBCO
r1 = mission_level_metrics(
    u=588.8, mp=4.57, r=0.05, omega=5236,
    h_km=550, ms=1000, g_gain=0.0004, k_fp=6000,
    magnet_material="GdBCO", jacket_material="CFRP", spacing=0.48
)

# SmCo 15 km/s
r2 = mission_level_metrics(
    u=15000, mp=35, r=0.1, omega=5236,
    h_km=550, ms=1000, g_gain=0.00014, k_fp=9000,
    magnet_material="SmCo", jacket_material="CFRP", spacing=0.48
)
```

- Power: `r['P_total_kW']`
- Infra Mass: `r['M_total_kg']`
- Lifetime: `r['service_lifetime_hr']` hours

## Sources
- Cold gas / Hydrazine: Sutton & Biblarz, "Rocket Propulsion Elements", 9th ed., Table 1-2
- Hall effect: Fakel SPT-100 datasheet
- Ion: NASA NSTAR thruster specifications (DS1 mission)
- SGMS: SpinnyBall `mission_level_metrics()` output
