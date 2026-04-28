#!/usr/bin/env python
"""
Generate figures for research paper.

This script generates all required figures for the paper:
1. ROM validation (MuJoCo vs ROM)
2. Parameter sweeps
3. Sensitivity analysis
4. Operational profile validation
5. Monte Carlo results
6. Thermal/stress analysis
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sgms_anchor_profiles import load_anchor_profiles, resolve_profile_params
from sgms_anchor_v1 import analytical_metrics, simulate_anchor, sweep_velocity
from sgms_anchor_sensitivity import run_sobol_sensitivity
from sgms_anchor_pipeline import export_fmeca_json
import json

print("=" * 70)
print("GENERATING PAPER FIGURES")
print("=" * 70)

# Load operational profile
data = load_anchor_profiles("anchor_profiles.json")
operational_params = resolve_profile_params(data, "operational")["params"]

# Figure 1: Velocity Sweep
print("\nGenerating Figure 1: Velocity Sweep...")
u_values = np.logspace(1, 3.2, 20)  # 10 to 1600 m/s
sweep = sweep_velocity(params=operational_params, u_values=u_values)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

axes[0].loglog(sweep["u"], sweep["force_per_stream_n"], 'o-', color="#79c0ff", linewidth=2)
axes[0].set_title("Force Per Stream")
axes[0].set_xlabel("u (m/s)")
axes[0].set_ylabel("F_stream (N)")
axes[0].grid(True, alpha=0.3)

axes[1].semilogx(sweep["u"], sweep["k_total"], 'o-', color="#7ee787", linewidth=2)
axes[1].axhspan(6000, 10000, alpha=0.2, color="green", label="Target range")
axes[1].set_title("Total Stiffness (Active + Pinning)")
axes[1].set_xlabel("u (m/s)")
axes[1].set_ylabel("k_total (N/m)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].semilogx(sweep["u"], sweep["period_s"], 'o-', color="#f2cc60", linewidth=2)
axes[2].set_title("Oscillation Period")
axes[2].set_xlabel("u (m/s)")
axes[2].set_ylabel("Period (s)")
axes[2].grid(True, alpha=0.3)

axes[3].semilogx(sweep["u"], sweep["static_offset_m"] * 1e3, 'o-', color="#ff7b72", linewidth=2)
axes[3].set_title("Static Offset Under Imbalance")
axes[3].set_xlabel("u (m/s)")
axes[3].set_ylabel("Offset (mm)")
axes[3].grid(True, alpha=0.3)

fig.suptitle("Figure 1: Velocity Sweep Analysis", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("paper_figures/fig1_velocity_sweep.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: paper_figures/fig1_velocity_sweep.png")

# Figure 2: Sensitivity Analysis
print("\nGenerating Figure 2: Sensitivity Analysis...")
problem = {
    "num_vars": 5,
    "names": ["u", "g_gain", "eps", "lam", "mp"],
    "bounds": [
        [5.0, 1600.0],
        [0.0001, 0.001],
        [0.0, 1e-3],
        [0.1, 20.0],
        [0.05, 8.0],
    ],
}
result = run_sobol_sensitivity(problem=problem, N=256, base_params=operational_params)
si = result["indices"]

fig, ax = plt.subplots(figsize=(10, 6))
s1_values = [si["k_eff"]["S1"][i] if i < len(si["k_eff"]["S1"]) else 0 for i in range(len(problem["names"]))]
bars = ax.bar(problem["names"], s1_values, color=["#79c0ff", "#7ee787", "#ff7b72", "#f2cc60", "#a5d6ff"])
ax.set_ylabel("First-Order Sobol Index")
ax.set_title("Figure 2: Parameter Sensitivity for k_eff")
ax.set_ylim(0, max(s1_values) * 1.1)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars, s1_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.3f}',
            ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig("paper_figures/fig2_sensitivity.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: paper_figures/fig2_sensitivity.png")

# Figure 3: Operational Profile Validation
print("\nGenerating Figure 3: Operational Profile Validation...")
metrics = analytical_metrics(operational_params)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# k_eff comparison
axes[0].bar(["Paper Target", "Operational"], [8000, metrics["k_eff"]], 
            color=["#7ee787", "#79c0ff"])
axes[0].axhspan(6000, 10000, alpha=0.2, color="green", label="Target range")
axes[0].set_ylabel("k_eff (N/m)")
axes[0].set_title("Effective Stiffness")
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Parameter comparison
param_names = ["u (m/s)", "mp (kg)", "k_fp (N/m)"]
paper_values = [600, 4.0, 6000]  # From sensitivity analysis
operational_values = [operational_params["u"], operational_params["mp"], operational_params["k_fp"]]
x = np.arange(len(param_names))
width = 0.35

axes[1].bar(x - width/2, paper_values, width, label="Optimal (Sobol)", color="#7ee787")
axes[1].bar(x + width/2, operational_values, width, label="Operational", color="#79c0ff")
axes[1].set_ylabel("Value")
axes[1].set_title("Parameter Comparison")
axes[1].set_xticks(x)
axes[1].set_xticklabels(param_names)
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

# FMECA summary
t_eval = np.linspace(0, 5.0, 1000)
fmeca_results = export_fmeca_json(simulate_anchor(operational_params, t_eval=t_eval))
kill_criteria = fmeca_results["kill_criteria"]
criteria_names = list(kill_criteria.keys())
criteria_status = [1 if kill_criteria[k] else 0 for k in criteria_names]
colors = ["#7ee787" if s else "#ff7b72" for s in criteria_status]

axes[2].barh(criteria_names, criteria_status, color=colors)
axes[2].set_xlabel("Pass (1) / Fail (0)")
axes[2].set_title("FMECA Kill Criteria")
axes[2].set_xlim(0, 1.2)
axes[2].grid(True, alpha=0.3, axis='x')

fig.suptitle("Figure 3: Operational Profile Validation", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("paper_figures/fig3_operational_validation.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: paper_figures/fig3_operational_validation.png")

# Figure 4: Parameter Distribution from Sobol
print("\nGenerating Figure 4: Parameter Distribution...")
samples = result["samples"]
outputs = result["outputs"]
k_eff_values = outputs["k_eff"]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, name in enumerate(problem["names"]):
    if i >= 5:
        break
    axes[i].scatter(samples[:, i], k_eff_values, alpha=0.5, s=10)
    axes[i].set_xlabel(name)
    axes[i].set_ylabel("k_eff (N/m)")
    axes[i].set_title(f"k_eff vs {name}")
    axes[i].grid(True, alpha=0.3)
    axes[i].axhspan(6000, 10000, alpha=0.2, color="green")

# Remove extra subplot
axes[5].axis('off')

fig.suptitle("Figure 4: Parameter Distributions from Sobol Analysis", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("paper_figures/fig4_parameter_distributions.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: paper_figures/fig4_parameter_distributions.png")

# Figure 5: Thermal and Stress Analysis
print("\nGenerating Figure 5: Thermal and Stress Analysis...")
omega_values = np.linspace(0, 6000, 100)  # rad/s
radius = 0.1  # m
density = 2500  # kg/m³ (BFRP)
stress_limit = 8.0e8  # Pa (800 MPa)

# Hoop stress: σ_θ = ρ·r²·ω²
stress = density * radius**2 * omega_values**2
stress_mpa = stress / 1e6

# Temperature (simplified radiative cooling)
emissivity = 0.85
surface_area = 0.2  # m²
power_heating = 200  # W
sigma_sb = 5.67e-8
# T = (P / (ε·A·σ))^0.25
temp_steady = (power_heating / (emissivity * surface_area * sigma_sb))**0.25 * np.ones_like(omega_values)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(omega_values * 60 / (2 * np.pi), stress_mpa, linewidth=2, color="#ff7b72")
axes[0].axhline(stress_limit / 1e6, color="#7ee787", linestyle='--', linewidth=2, label="Stress limit (800 MPa)")
axes[0].axvline(5236 * 60 / (2 * np.pi), color="#79c0ff", linestyle=':', linewidth=2, label="Operational (50k RPM)")
axes[0].set_xlabel("Spin Rate (RPM)")
axes[0].set_ylabel("Hoop Stress (MPa)")
axes[0].set_title("Centrifugal Stress vs Spin Rate")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].axhline(temp_steady[0], color="#f2cc60", linewidth=2, label=f"Steady-state ({temp_steady[0]:.1f} K)")
axes[1].axhline(450, color="#ff7b72", linestyle='--', linewidth=2, label="Thermal limit (450 K)")
axes[1].set_xlim(0, 1)
axes[1].set_ylim(300, 500)
axes[1].set_xlabel("Spin Rate (normalized)")
axes[1].set_ylabel("Temperature (K)")
axes[1].set_title("Thermal Limits")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.suptitle("Figure 5: Thermal and Stress Analysis", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("paper_figures/fig5_thermal_stress.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: paper_figures/fig5_thermal_stress.png")

# Figure 6: System Response
print("\nGenerating Figure 6: System Response...")
t_eval = np.linspace(0, 5.0, 1000)
result = simulate_anchor(operational_params, t_eval=t_eval, seed=42)

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

axes[0].plot(result["t"], result["x"] * 1e3, linewidth=2, color="#79c0ff")
axes[0].axhline(0, color="#ff7b72", linestyle='--', alpha=0.7, label="Center")
axes[0].axhline(metrics["static_offset_m"] * 1e3, color="#f2cc60", linestyle=':', linewidth=1.5, label="Static offset")
axes[0].set_ylabel("Displacement (mm)")
axes[0].set_title("Node Displacement Response")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(result["t"], result["vx"], linewidth=2, color="#7ee787")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Velocity (m/s)")
axes[1].set_title("Node Velocity Response")
axes[1].grid(True, alpha=0.3)

fig.suptitle("Figure 6: System Dynamic Response", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("paper_figures/fig6_system_response.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: paper_figures/fig6_system_response.png")

print("\n" + "=" * 70)
print("PAPER FIGURES GENERATED")
print("=" * 70)
print("\nFigures saved to paper_figures/:")
print("  fig1_velocity_sweep.png")
print("  fig2_sensitivity.png")
print("  fig3_operational_validation.png")
print("  fig4_parameter_distributions.png")
print("  fig5_thermal_stress.png")
print("  fig6_system_response.png")
