#!/usr/bin/env python3
"""Check damping ratio discrepancy with operational parameters."""

from src.sgms_anchor_v1 import analytical_metrics
import math

def main():
    # Operational parameters from TECH_SPEC
    operational_params = {
        "u": 1600.0,  # Operational velocity
        "lam": 72.92,  # Linear density from TECH_SPEC
        "mp": 35.0,    # Packet mass from TECH_SPEC
        "ms": 1000.0,  # Station mass
        "c_damp": 4.0, # Damping coefficient
        "g_gain": 0.000140,  # Control gain from TECH_SPEC
        "k_fp": 9000.0,  # Flux-pinning stiffness from TECH_SPEC
        "theta_bias": 0.087,
        "eps": 0.0,
    }
    
    metrics = analytical_metrics(operational_params)

    print('=== OPERATIONAL PARAMETERS FROM TECH_SPEC ===')
    for key in ['u', 'lam', 'mp', 'ms', 'c_damp', 'g_gain', 'k_fp']:
        value = operational_params.get(key, 'N/A')
        print(f'{key}: {value}')

    print('\n=== CALCULATED METRICS ===')
    print(f'k_control: {metrics["k_control"]:.1f} N/m')
    print(f'k_total: {metrics["k_total"]:.1f} N/m')
    print(f'Natural frequency: {metrics["omega_n_rad_s"]:.3f} rad/s')
    print(f'Period: {metrics["period_s"]:.3f} s')
    print(f'Damping ratio (zeta): {metrics["zeta"]:.6f}')

    print('\n=== TECH SPEC CLAIMS ===')
    print('Natural frequency: 3.45 rad/s')
    print('Damping ratio: 0.047')

    print('\n=== DISCREPANCY ANALYSIS ===')
    print(f'zeta discrepancy: {metrics["zeta"] / 0.047:.1f}x difference')
    print(f'omega_n discrepancy: {metrics["omega_n_rad_s"] / 3.45:.1f}x difference')
    
    # User's manual calculation verification
    print('\n=== MANUAL CALCULATION VERIFICATION ===')
    k_eff = operational_params["lam"] * operational_params["u"]**2 * operational_params["g_gain"] + operational_params["k_fp"]
    zeta_manual = operational_params["c_damp"] / (2 * math.sqrt(k_eff * operational_params["ms"]))
    print(f'k_eff (manual): {k_eff:.1f} N/m')
    print(f'zeta (manual): {zeta_manual:.6f}')
    print(f'Match with code: {abs(zeta_manual - metrics["zeta"]) < 1e-9}')

if __name__ == '__main__':
    main()
