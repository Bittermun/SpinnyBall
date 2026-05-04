import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

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

def print_metrics(name, r):
    print(f"--- {name} ---")
    print(f"Power: {r['P_total_kW']:.3g} kW")
    print(f"Infra Mass: {r['M_total_kg']:.3g} kg")
    print(f"Lifetime: {r['service_lifetime_hr']:.3g} hours")

print_metrics("GdBCO", r1)
print_metrics("SmCo", r2)
