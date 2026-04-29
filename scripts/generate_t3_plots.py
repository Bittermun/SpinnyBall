#!/usr/bin/env python3
"""
Generate publication-ready plots from T3 sweep data.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

def load_t3_data(json_file: Path):
    """Load T3 sweep data from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    return {
        'fault_rates': np.array(data['fault_rates']),
        'cascade_prob': np.array(data['cascade_probabilities']),
        'cascade_ci_lower': np.array(data['cascade_ci_lower']),
        'cascade_ci_upper': np.array(data['cascade_ci_upper']),
        'containment_rate': np.array(data['containment_rates']),
        'containment_ci_lower': np.array(data['containment_ci_lower']),
        'containment_ci_upper': np.array(data['containment_ci_upper']),
        'nodes_affected_mean': np.array(data['nodes_affected_mean']),
        'nodes_affected_std': np.array(data['nodes_affected_std']),
        'profile': data['profile']
    }

def plot_cascade_probability(data_list, output_dir: Path):
    """Plot cascade probability with confidence intervals."""
    
    plt.figure(figsize=(10, 6))
    
    for data in data_list:
        plt.semilogx(data['fault_rates'], data['cascade_prob'], 
                    'o-', label=f"{data['profile']}", linewidth=2, markersize=6)
        
        # Add confidence interval shading
        plt.fill_between(data['fault_rates'], 
                        data['cascade_ci_lower'], 
                        data['cascade_ci_upper'],
                        alpha=0.2)
    
    plt.xlabel('Fault Rate (/hr)', fontsize=12)
    plt.ylabel('Cascade Probability', fontsize=12)
    plt.title('Cascade Probability vs Fault Rate', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.ylim(-0.01, 0.1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cascade_probability.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_containment_rate(data_list, output_dir: Path):
    """Plot containment rate with confidence intervals."""
    
    plt.figure(figsize=(10, 6))
    
    for data in data_list:
        plt.semilogx(data['fault_rates'], data['containment_rate'], 
                    's-', label=f"{data['profile']}", linewidth=2, markersize=6)
        
        # Add confidence interval shading
        plt.fill_between(data['fault_rates'], 
                        data['containment_ci_lower'], 
                        data['containment_ci_upper'],
                        alpha=0.2)
    
    plt.xlabel('Fault Rate (/hr)', fontsize=12)
    plt.ylabel('Containment Rate', fontsize=12)
    plt.title('Containment Rate vs Fault Rate', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.ylim(0.95, 1.01)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'containment_rate.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_nodes_affected(data_list, output_dir: Path):
    """Plot nodes affected (if any)."""
    
    plt.figure(figsize=(10, 6))
    
    for data in data_list:
        # Calculate mean ± std for error bars
        nodes_mean = np.array(data['nodes_affected_mean'])
        nodes_std = np.array(data['nodes_affected_std'])
        
        plt.semilogx(data['fault_rates'], nodes_mean, 
                    '^-', label=f"{data['profile']}", linewidth=2, markersize=6)
        
        # Add error bars
        plt.errorbar(data['fault_rates'], nodes_mean, yerr=nodes_std, 
                   fmt='none', ecolor='gray', alpha=0.5, capsize=3)
    
    plt.xlabel('Fault Rate (/hr)', fontsize=12)
    plt.ylabel('Nodes Affected (mean ± std)', fontsize=12)
    plt.title('Nodes Affected vs Fault Rate', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.ylim(-0.1, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'nodes_affected.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_table(data_list, output_dir: Path):
    """Create summary table for publication."""
    
    # Create markdown table
    table_lines = [
        "# T3 Fault Rate Sweep - Summary Table\n",
        "\n",
        "| Profile | Fault Rate Range | Max Cascade Prob | 95% CI Upper | Min Containment | 95% CI Lower | Status |",
        "|---------|----------------|-----------------|---------------|-----------------|---------------|--------|"
    ]
    
    for data in data_list:
        max_cascade = np.max(data['cascade_prob'])
        ci_upper = np.max(data['cascade_ci_upper'])
        min_containment = np.min(data['containment_rate'])
        ci_lower = np.min(data['containment_ci_lower'])
        
        status = "✅ Robust" if max_cascade == 0.0 else "⚠️ Risk"
        
        table_lines.append(
            f"| {data['profile']} | 10⁻⁶ - 10⁻³ /hr | {max_cascade:.4f} | {ci_upper:.4f} | "
            f"{min_containment:.4f} | {ci_lower:.4f} | {status} |"
        )
    
    # Save table
    with open(output_dir / 'summary_table.md', 'w') as f:
        f.write('\n'.join(table_lines))

def main():
    """Generate plots from existing T3 data."""
    
    # Load research data
    research_dir = Path("research_data/20260428-093002")
    
    if not research_dir.exists():
        print(f"Research data not found at {research_dir}")
        return
    
    # Load operational profile data
    t3_file = research_dir / "t3_fault_rate_sweep.json"
    if not t3_file.exists():
        print(f"T3 data not found at {t3_file}")
        return
    
    data = load_t3_data(t3_file)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(f"t3_plots_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    
    # Generate plots
    print("Generating T3 plots...")
    
    plot_cascade_probability([data], output_dir)
    print("  ✓ Cascade probability plot")
    
    plot_containment_rate([data], output_dir)
    print("  ✓ Containment rate plot")
    
    plot_nodes_affected([data], output_dir)
    print("  ✓ Nodes affected plot")
    
    create_summary_table([data], output_dir)
    print("  ✓ Summary table")
    
    print(f"\nPlots saved to: {output_dir}")
    print("Files generated:")
    print("  - cascade_probability.png")
    print("  - containment_rate.png") 
    print("  - nodes_affected.png")
    print("  - summary_table.md")

if __name__ == "__main__":
    main()
