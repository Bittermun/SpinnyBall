# Control & Thermal

## MPC

Smith predictor delay compensation. Configurable delay_steps (0-20), effective range 0-200ms.

JAX LQR surrogate: 256k realizations in ~1s via XLA.

Stable up to 65ms latency at η=0.90. Safety limit: 42ms at η=0.82.

## Cryocooler

For GdBCO superconductors.

| Temp | Cooling | Power |
|------|---------|-------|
| Temperature | Cooling Power ($Q_{\text{cool}}$) | Input Power ($P_{\text{input}}$, 12% Carnot Bound @ 300 K) | Actual COP |
|:---|:---|:---|:---|
| 70 K | 5 W | 137 W | 0.0365 |
| 80 K | 8 W | 183 W | 0.0436 |
| 90 K | 12 W | 233 W | 0.0514 |

Cooldown: 1hr from 300K to 77K.

## SmCo vs GdBCO

| | SmCo | GdBCO |
|--|------|-------|
| Operating temp | 280-573K | 77-90K |
| Cooling | Passive radiative | Cryogenic required |
| Quench risk | Low | High |

At 15 km/s: SmCo achieves 379K steady-state (passive). GdBCO requires ~2 MW cryogenic power.