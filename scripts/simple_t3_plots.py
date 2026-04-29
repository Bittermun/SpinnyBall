#!/usr/bin/env python3
"""
Simple T3 plots - avoid encoding issues.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

def load_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def plot_all_profiles():
    """Plot all 4 profiles together."""
    
    # Load all profile data
    profiles = ['paper-baseline', 'operational', 'engineering-screen', 'resilience']
    data_dir = Path('profile_sweep_quick_20260428-095855')
    
    plt.figure(figsize=(12, 8))
    
    for profile in profiles:
        file_path = data_dir / f't3_sweep_{profile}.json'
        if file_path.exists():
            data = load_data(file_path)
            
            fault_rates = np.array(data['fault_rates'])
            cascade_prob = np.array(data['cascade_probabilities'])
            ci_upper = np.array(data['cascade_ci_upper'])
            
            plt.semilogx(fault_rates, cascade_prob, 'o-', 
                         label=profile, linewidth=2, markersize=6)
            
            # Add CI shading
            plt.fill_between(fault_rates, np.zeros_like(cascade_prob), ci_upper,
                            alpha=0.2)
    
    plt.xlabel('Fault Rate (/hr)', fontsize=12)
    plt.ylabel('Cascade Probability', fontsize=12)
    plt.title('T3 Fault Rate Sweep - All Profiles', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.ylim(-0.01, 0.05)
    
    # Save
    output_dir = Path('t3_plots_simple')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'cascade_all_profiles.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {output_dir / 'cascade_all_profiles.png'}")

def create_simple_table():
    """Create summary table for all profiles."""
    
    data_dir = Path('profile_sweep_quick_20260428-095855')
    profiles = ['paper-baseline', 'operational', 'engineering-screen', 'resilience']
    
    table_lines = [
        "# T3 Fault Rate Sweep - All Profiles Summary",
        "",
        "| Profile | Max Cascade Prob | 95% CI Upper | Min Containment | 95% CI Lower | Status |",
        "|---------|-----------------|---------------|-----------------|---------------|--------|"
    ]
    
    for profile in profiles:
        file_path = data_dir / f't3_sweep_{profile}.json'
        if file_path.exists():
            data = load_data(file_path)
            
            max_cascade = max(data['cascade_probabilities'])
            ci_upper = max(data['cascade_ci_upper'])
            min_containment = min(data['containment_rates'])
            ci_lower = min(data['containment_ci_lower'])
            
            status = "Robust" if max_cascade == 0.0 else "Risk"
            
            table_lines.append(
                f"| {profile} | {max_cascade:.4f} | {ci_upper:.4f} | "
                f"{min_containment:.4f} | {ci_lower:.4f} | {status} |"
            )
    
    # Save table
    output_dir = Path('t3_plots_simple')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'summary_table.md', 'w') as f:
        f.write('\n'.join(table_lines))
    
    print(f"Table saved to: {output_dir / 'summary_table.md'}")

def main():
    print("Generating simple T3 plots...")
    
    plot_all_profiles()
    create_simple_table()
    
    print("Done!")

if __name__ == "__main__":
    main()
