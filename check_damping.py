#!/usr/bin/env python3
"""Check damping ratio discrepancy."""

from src.sgms_anchor_v1 import analytical_metrics, DEFAULT_PARAMS
import math

def main():
    # Get the actual metrics calculation
    params = DEFAULT_PARAMS.copy()
    metrics = analytical_metrics(params)

    print('=== DEFAULT OPERATIONAL PARAMETERS ===')
    for key in ['u', 'lam', 'mp', 'ms', 'c_damp', 'g_gain', 'k_fp']:
        value = params.get(key, 'N/A')
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

if __name__ == '__main__':
    main()
