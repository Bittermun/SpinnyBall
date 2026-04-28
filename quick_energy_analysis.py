#!/usr/bin/env python3
"""
Quick Energy Budget Analysis - Extract key metrics from cryocooler model
"""

import numpy as np
from dynamics.cryocooler_model import CryocoolerModel, DEFAULT_CRYOCOOLER_SPECS

def analyze_energy_budget():
    """Extract energy budget metrics from cryocooler model."""
    
    model = CryocoolerModel(DEFAULT_CRYOCOOLER_SPECS)
    
    # Temperature range for analysis
    temps = np.linspace(70, 90, 21)  # 70K to 90K
    
    results = {
        'temperature_range': [70.0, 90.0],
        'cooling_power': {},
        'input_power': {},
        'cop': {},
        'energy_efficiency': {}
    }
    
    for T in temps:
        results['cooling_power'][f'{T:.1f}K'] = model.cooling_power(T)
        results['input_power'][f'{T:.1f}K'] = model.input_power(T)
        results['cop'][f'{T:.1f}K'] = model.cop(T)
    
    # Key performance metrics
    results['key_metrics'] = {
        'max_cooling_power': model.cooling_power(70.0),
        'max_input_power': model.input_power(90.0),
        'best_cop': max(results['cop'].values()),
        'cop_at_operating_temp': model.cop(77.0),  # Liquid nitrogen temp
        'cooldown_time': DEFAULT_CRYOCOOLER_SPECS.cooldown_time,
        'warmup_time': DEFAULT_CRYOCOOLER_SPECS.warmup_time,
        'system_mass': DEFAULT_CRYOCOOLER_SPECS.mass,
        'power_density': DEFAULT_CRYOCOOLER_SPECS.cooling_power_at_70k / DEFAULT_CRYOCOOLER_SPECS.mass
    }
    
    return results

if __name__ == "__main__":
    results = analyze_energy_budget()
    print("Energy Budget Analysis Results:")
    print(f"Max Cooling Power: {results['key_metrics']['max_cooling_power']:.1f} W")
    print(f"Max Input Power: {results['key_metrics']['max_input_power']:.1f} W")
    print(f"Best COP: {results['key_metrics']['best_cop']:.3f}")
    print(f"COP at 77K: {results['key_metrics']['cop_at_operating_temp']:.3f}")
    print(f"Power Density: {results['key_metrics']['power_density']:.1f} W/kg")
    print(f"Cooldown Time: {results['key_metrics']['cooldown_time']/3600:.1f} hours")
    
    import json
    with open("energy_budget_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to energy_budget_results.json")
