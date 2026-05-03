
from src.sgms_anchor_v1 import mission_level_metrics
import numpy as np

# Baseline operational point that should be feasible (with GdBCO)
# From MISSION_LEVEL_ANALYSIS.md
res = mission_level_metrics(
    u=11192.8,
    mp=7.65,
    r=0.1,
    omega=2877, # rad/s (~27.4k RPM)
    h_km=841,
    ms=1000,
    g_gain=0.014,
    k_fp=6000,
    spacing=0.48,
    magnet_material="GdBCO",
    jacket_material="CFRP"
)

print(f"Feasible: {res['feasible']}")
for k, v in res.items():
    if k != 'feasible':
        print(f"  {k}: {v}")
