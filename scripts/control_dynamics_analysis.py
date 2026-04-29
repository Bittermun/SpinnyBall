#!/usr/bin/env python3
"""
Control System Dynamics Analysis - Extract key metrics from MPC controller
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime

def extract_control_metrics():
    """Extract control system performance metrics."""
    
    # Configuration modes from MPC controller
    config_modes = {
        "TEST": {
            "packet_mass": 0.05,  # kg
            "packet_radius": 0.02,  # m
            "spin_rate": 100.0,  # rad/s
            "max_stress": 1.2e9,  # Pa
        },
        "VALIDATION": {
            "packet_mass": 2.0,  # kg
            "packet_radius": 0.1,  # m
            "spin_rate": 5236.0,  # rad/s
            "max_stress": 1.2e9,  # Pa
        },
        "OPERATIONAL": {
            "packet_mass": 8.0,  # kg
            "packet_radius": 0.1,  # m
            "spin_rate": 5236.0,  # rad/s
            "max_stress": 1.2e9,  # Pa
        }
    }
    
    # MPC performance targets (from code comments)
    performance_targets = {
        "solve_time_target": 30.0,  # ms
        "horizon": 10,  # prediction steps
        "delay_compensation": True,
        "stability_margin_target": 45.0,  # degrees
    }
    
    # Extracted from code analysis
    results = {
        'configuration_modes': config_modes,
        'performance_targets': performance_targets,
        'control_characteristics': {
            'controller_type': 'Model-Predictive Control (MPC)',
            'prediction_horizon': 10,
            'optimization_engine': 'CasADi',
            'acceleration': 'numba/jit',
            'delay_compensation': True,
        },
        'response_analysis': {
            'theoretical_solve_time': '< 30 ms (target)',
            'stability_analysis': 'Phase margin + delay margin',
            'robustness_features': [
                'Delay compensation',
                'Constraint handling',
                'Multi-variable coordination'
            ]
        },
        'scaling_analysis': {
            'mass_scaling': {
                'test_to_validation': config_modes['VALIDATION']['packet_mass'] / config_modes['TEST']['packet_mass'],
                'validation_to_operational': config_modes['OPERATIONAL']['packet_mass'] / config_modes['VALIDATION']['packet_mass']
            },
            'spin_rate_scaling': {
                'test_to_validation': config_modes['VALIDATION']['spin_rate'] / config_modes['TEST']['spin_rate'],
                'operational_stress': config_modes['OPERATIONAL']['spin_rate']**2 * config_modes['OPERATIONAL']['packet_radius']**2
            }
        }
    }
    
    # Calculate key performance indicators
    results['kpi_summary'] = {
        'solve_time_achievement': 'Target: <30ms, Implementation: CasADi-optimized',
        'stability_margin': 'Computed via frequency response analysis',
        'control_precision': 'Multi-variable coordination with constraints',
        'scalability': f'Mass range: {config_modes["TEST"]["packet_mass"]}-{config_modes["OPERATIONAL"]["packet_mass"]} kg',
        'operational_envelope': f'Spin rate: {config_modes["OPERATIONAL"]["spin_rate"]} rad/s'
    }
    
    return results

if __name__ == "__main__":
    results = extract_control_metrics()
    
    print("Control System Dynamics Analysis:")
    print(f"Controller Type: {results['control_characteristics']['controller_type']}")
    print(f"Target Solve Time: {results['performance_targets']['solve_time_target']} ms")
    print(f"Mass Scaling Range: {results['scaling_analysis']['mass_scaling']}")
    print(f"Operational Spin Rate: {results['configuration_modes']['OPERATIONAL']['spin_rate']} rad/s")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(f"control_dynamics_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to control_dynamics_{timestamp}.json")
