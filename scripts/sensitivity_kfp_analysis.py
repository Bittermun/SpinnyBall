"""
Sensitivity analysis for flux-pinning stiffness k_fp impact on mission feasibility.

Determines whether exact k_fp value matters for feasibility conclusions.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, 'c:\\Users\\msunw\\Desktop\\SpinnyBall')

from src.sgms_anchor_v1 import mission_level_metrics


def run_kfp_sensitivity(
    k_fp_range=(1e4, 1e9),
    n_points=50,
    base_params=None,
    magnet_material='GdBCO',
    jacket_material='CNT_yarn',
):
    """Run sensitivity analysis on k_fp parameter."""
    if base_params is None:
        base_params = {
            'u': 15000,      # m/s
            'mp': 0.5,       # kg
            'r': 0.05,       # m
            'omega': 5000,   # rad/s
            'h_km': 400,     # km
            'ms': 1000,      # kg
            'g_gain': 0.05,
            'spacing': 10,   # m
        }

    k_fp_values = np.logspace(np.log10(k_fp_range[0]), np.log10(k_fp_range[1]), n_points)

    results = []
    for k_fp in k_fp_values:
        try:
            metrics = mission_level_metrics(
                u=base_params['u'],
                mp=base_params['mp'],
                r=base_params['r'],
                omega=base_params['omega'],
                h_km=base_params['h_km'],
                ms=base_params['ms'],
                g_gain=base_params['g_gain'],
                k_fp=k_fp,
                magnet_material=magnet_material,
                jacket_material=jacket_material,
                spacing=base_params['spacing'],
            )
            results.append({
                'k_fp': k_fp,
                'feasible': metrics['feasible'],
                'k_eff': metrics['k_eff'],
                'N_packets': metrics['N_packets'],
                'M_total_kg': metrics['M_total_kg'],
                'stress_margin': metrics['stress_margin'],
                'thermal_margin': metrics['thermal_margin'],
                'P_total_kW': metrics['P_total_kW'],
            })
        except Exception as e:
            print(f"Error at k_fp={k_fp:.2e}: {e}")
            results.append({
                'k_fp': k_fp,
                'feasible': False,
                'k_eff': np.nan,
                'N_packets': np.nan,
                'M_total_kg': np.nan,
                'stress_margin': np.nan,
                'thermal_margin': np.nan,
                'P_total_kW': np.nan,
            })

    return results


def analyze_results(results):
    """Analyze sensitivity results."""
    print('=== k_fp SENSITIVITY ANALYSIS RESULTS ===')
    print()

    # Print table
    print('k_fp (N/m)    | Feasible | k_eff (N/m) | N_packets | Mass (kg) | Stress Marg | Therm Marg')
    print('-' * 100)

    for r in results[::5]:  # Print every 5th for readability
        feas_str = 'YES' if r['feasible'] else 'NO'
        print(f"{r['k_fp']:13.2e} | {feas_str:8s} | {r['k_eff']:11.2e} | "
              f"{r['N_packets']:9.1f} | {r['M_total_kg']:9.1f} | "
              f"{r['stress_margin']:11.2f} | {r['thermal_margin']:10.2f}")

    # Find feasibility range
    feasible_results = [r for r in results if r['feasible']]

    print()
    if feasible_results:
        min_kfp = min(r['k_fp'] for r in feasible_results)
        max_kfp = max(r['k_fp'] for r in feasible_results)
        print(f'Feasibility range: k_fp in [{min_kfp:.2e}, {max_kfp:.2e}] N/m')
        print(f'Number of feasible configurations: {len(feasible_results)}/{len(results)}')
    else:
        print('No feasible configurations found in tested range')

    # Key stiffness values for context
    print()
    print('Context: Key stiffness values')
    print(f'  Paper claim (GdBCO):     15,000 N/m')
    print(f'  Paper claim (SmCo):       9,000 N/m')
    print(f'  Original Bean-London:     ~2.75e8 N/m')
    print(f'  Workstream A at 15 km/s:  ~1.4e6 N/m')

    # Check if paper values are in feasible range
    paper_gdbco = 15000
    paper_smco = 9000

    if feasible_results:
        in_range = min_kfp <= paper_gdbco <= max_kfp
        print()
        print(f'Paper GdBCO value (15,000 N/m) in feasible range: {in_range}')

    return feasible_results


def plot_results(results, output_file='kfp_sensitivity.png'):
    """Plot sensitivity results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    k_fp = [r['k_fp'] for r in results]
    feasible = [r['feasible'] for r in results]
    k_eff = [r['k_eff'] for r in results]
    mass = [r['M_total_kg'] for r in results]

    # Plot 1: Feasibility vs k_fp
    ax = axes[0, 0]
    colors = ['green' if f else 'red' for f in feasible]
    ax.scatter(k_fp, [1 if f else 0 for f in feasible], c=colors, alpha=0.6)
    ax.set_xscale('log')
    ax.set_xlabel('k_fp (N/m)')
    ax.set_ylabel('Feasible')
    ax.set_title('Feasibility vs Flux-Pinning Stiffness')
    ax.axvline(15000, color='blue', linestyle='--', label='Paper claim (GdBCO)')
    ax.axvline(9000, color='orange', linestyle='--', label='Paper claim (SmCo)')
    ax.legend()

    # Plot 2: k_eff vs k_fp
    ax = axes[0, 1]
    ax.plot(k_fp, k_eff, 'b-', linewidth=2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('k_fp (N/m)')
    ax.set_ylabel('k_eff (N/m)')
    ax.set_title('Effective Stiffness vs Flux-Pinning Stiffness')
    ax.axvline(15000, color='blue', linestyle='--', alpha=0.5)

    # Plot 3: Mass vs k_fp
    ax = axes[1, 0]
    mass_feasible = [m if f else np.nan for m, f in zip(mass, feasible)]
    ax.plot(k_fp, mass_feasible, 'g-', linewidth=2, label='Feasible')
    ax.set_xscale('log')
    ax.set_xlabel('k_fp (N/m)')
    ax.set_ylabel('Total Mass (kg)')
    ax.set_title('Infrastructure Mass vs Stiffness')
    ax.axvline(15000, color='blue', linestyle='--', alpha=0.5, label='Paper claim')
    ax.legend()

    # Plot 4: Histogram of feasible k_fp
    ax = axes[1, 1]
    feasible_kfp = [r['k_fp'] for r in results if r['feasible']]
    if feasible_kfp:
        ax.hist(np.log10(feasible_kfp), bins=20, color='green', alpha=0.6)
        ax.set_xlabel('log10(k_fp)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Feasible k_fp Values')
        ax.axvline(np.log10(15000), color='blue', linestyle='--', linewidth=2, label='Paper claim')
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"\nPlot saved to: {output_file}")


if __name__ == '__main__':
    print('Running k_fp sensitivity analysis...')
    print()

    results = run_kfp_sensitivity(
        k_fp_range=(1e4, 1e9),
        n_points=100,
    )

    feasible_results = analyze_results(results)
    plot_results(results)

    print()
    print('=== KEY FINDING ===')
    if feasible_results:
        print('Feasibility IS sensitive to k_fp value.')
        print('Exact stiffness matters for mission success.')
    else:
        print('No feasible configurations found.')
        print('May indicate other constraints are binding.')
