#!/usr/bin/env python3
"""
Visualize sweep results from aggregated data.

This script creates plots showing the relationship between fault rates and cascade probabilities
across different profiles and runs.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_and_plot_sweep_results():
    """Load aggregated sweep results and create visualizations."""
    
    # Try to load the aggregated results
    summary_file = Path("sweep_summary.json")
    if not summary_file.exists():
        print(f"Error: {summary_file} not found. Run summarize_sweep_results.py first.")
        return
    
    with open(summary_file, 'r') as f:
        summary_data = json.load(f)
    
    # Load detailed data for plotting
    detailed_file = Path("sweep_aggregated_summary.json")
    if not detailed_file.exists():
        print(f"Error: {detailed_file} not found. Run summarize_sweep_results.py first.")
        return
    
    with open(detailed_file, 'r') as f:
        detailed_data = json.load(f)
    
    # Extract fault rates and cascade probabilities from each run
    all_series = {}
    
    # Process each run directory
    for run in detailed_data['runs']:
        if run.get('error_count', 0) > 0:
            continue  # Skip runs with JSON errors
            
        run_dir = Path(run['directory'])
        
        # Get all JSON files in the directory
        for json_file in run_dir.glob("t3_sweep_*.json"):
            profile_name = json_file.stem.replace("t3_sweep_", "")
            
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                fault_rates = data.get("fault_rates", [])
                cascade_probs = data.get("cascade_probabilities", [])
                
                if profile_name not in all_series:
                    all_series[profile_name] = []
                    
                all_series[profile_name].append({
                    'fault_rates': fault_rates,
                    'cascade_probs': cascade_probs,
                    'directory': run['directory']
                })
            except Exception as e:
                print(f"Warning: Could not load {json_file} - {e}")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Sweep Results Analysis: Fault Rate vs Cascade Probability', fontsize=16)
    
    # Plot 1: All profiles in one plot
    ax1 = axes[0, 0]
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    color_map = {}
    for idx, profile in enumerate(all_series.keys()):
        color_map[profile] = colors[idx % len(colors)]
    
    for profile, series_list in all_series.items():
        for series in series_list[:1]:  # Just plot the first occurrence of each profile
            if series['fault_rates'] and series['cascade_probs']:
                ax1.semilogx(
                    series['fault_rates'], 
                    series['cascade_probs'], 
                    label=profile,
                    color=color_map[profile],
                    alpha=0.7,
                    marker='o',
                    markersize=4
                )
    
    ax1.set_xlabel('Fault Rate (log scale)')
    ax1.set_ylabel('Cascade Probability')
    ax1.set_title('Cascade Probability vs Fault Rate\n(by Profile)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Convergence rates by run
    ax2 = axes[0, 1]
    run_names = [r['directory'] for r in detailed_data['runs'] if r.get('error_count', 0) == 0]
    convergence_rates = []
    
    for r in detailed_data['runs']:
        if r.get('error_count', 0) == 0 and r['profiles_found']:
            # Calculate convergence rate for this run
            total_converged = sum(p['converged_count'] for p in r['profiles_found'].values())
            total_points = sum(p['total_points'] for p in r['profiles_found'].values())
            rate = total_converged / total_points if total_points > 0 else 0
            convergence_rates.append(rate)
    
    if run_names and convergence_rates:
        bars = ax2.bar(range(len(run_names)), convergence_rates, color='skyblue')
        ax2.set_xlabel('Run Directory')
        ax2.set_ylabel('Convergence Rate')
        ax2.set_title('Convergence Rate by Run')
        ax2.set_xticks(range(len(run_names)))
        ax2.set_xticklabels([name.split('-')[-1][:4] for name in run_names], rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, rate in zip(bars, convergence_rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                     f'{rate:.2f}',
                     ha='center', va='bottom')
    else:
        ax2.text(0.5, 0.5, 'No valid data available', horizontalalignment='center', 
                 verticalalignment='center', transform=ax2.transAxes)
        ax2.set_title('Convergence Rate by Run (No Data)')
    
    # Plot 3: Average cascade probability by profile
    ax3 = axes[1, 0]
    profiles = list(summary_data['profile_performance'].keys())
    avg_cascade_probs = [summary_data['profile_performance'][p]['avg_cascade_prob'] for p in profiles]
    
    if profiles and avg_cascade_probs:
        bars = ax3.bar(range(len(profiles)), avg_cascade_probs, color='lightcoral')
        ax3.set_xlabel('Profile')
        ax3.set_ylabel('Average Cascade Probability')
        ax3.set_title('Average Cascade Probability by Profile')
        ax3.set_xticks(range(len(profiles)))
        ax3.set_xticklabels(profiles, rotation=45, ha='right')
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, prob in zip(bars, avg_cascade_probs):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                     f'{prob:.4f}',
                     ha='center', va='bottom')
    else:
        ax3.text(0.5, 0.5, 'No profile data available', horizontalalignment='center', 
                 verticalalignment='center', transform=ax3.transAxes)
        ax3.set_title('Average Cascade Probability by Profile (No Data)')
    
    # Plot 4: Convergence rate by profile
    ax4 = axes[1, 1]
    profile_convergence_rates = [summary_data['profile_performance'][p]['avg_convergence_rate'] for p in profiles]
    
    if profiles and profile_convergence_rates:
        bars = ax4.bar(range(len(profiles)), profile_convergence_rates, color='lightgreen')
        ax4.set_xlabel('Profile')
        ax4.set_ylabel('Average Convergence Rate')
        ax4.set_title('Average Convergence Rate by Profile')
        ax4.set_xticks(range(len(profiles)))
        ax4.set_xticklabels(profiles, rotation=45, ha='right')
        ax4.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, rate in zip(bars, profile_convergence_rates):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                     f'{rate:.2f}',
                     ha='center', va='bottom')
    else:
        ax4.text(0.5, 0.5, 'No profile data available', horizontalalignment='center', 
                 verticalalignment='center', transform=ax4.transAxes)
        ax4.set_title('Average Convergence Rate by Profile (No Data)')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = "sweep_visualization.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    plt.show()
    
    # Print summary insights
    print("\nKey Insights from Visualization:")
    print(f"- Total profiles analyzed: {len(all_series)}")
    valid_runs = len([r for r in detailed_data['runs'] if r.get('error_count', 0) == 0])
    print(f"- Total valid runs processed: {valid_runs}")
    
    for profile, data in summary_data['profile_performance'].items():
        print(f"- {profile}: {data['avg_cascade_prob']:.4f} avg cascade prob, "
              f"{data['avg_convergence_rate']:.2%} convergence rate")


def main():
    """Main function to run visualization."""
    print("Creating sweep results visualization...")
    load_and_plot_sweep_results()


if __name__ == "__main__":
    main()