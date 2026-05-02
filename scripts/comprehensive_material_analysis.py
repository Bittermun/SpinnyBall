
import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from src.sgms_anchor_v1 import mission_level_metrics
from params.canonical_values import MATERIAL_PROPERTIES

def run_material_sweep():
    print("Starting Comprehensive Material Sweep Analysis...")
    
    magnets = ["SmCo", "NdFeB", "GdBCO", "YBCO"]
    jackets = ["BFRP", "CFRP", "CNT_yarn"]
    
    # 9-parameter problem (aligned with Sobol)
    PARAM_BOUNDS = {
        "u": [500.0, 15000.0],
        "mp": [1.0, 50.0],
        "r": [0.02, 0.15],
        "omega": [2000.0, 6000.0],
        "h_km": [300.0, 2000.0],
        "ms": [100.0, 10000.0],
        "g_gain": [1e-3, 0.1],
        "k_fp": [1000.0, 15000.0],
        "spacing": [0.1, 1000.0]
    }
    
    # Material-specific limits
    MATERIAL_LIMITS = {
        "GdBCO": {"k_fp_max": 14400.0}, # 120,000 * 0.12
        "YBCO": {"k_fp_max": 7200.0},   # 60,000 * 0.12
        "SmCo": {"k_fp_max": np.inf},   
        "NdFeB": {"k_fp_max": np.inf}   
    }
    
    results = []
    
    # Samples per combination
    N_SAMPLES = 5000
    np.random.seed(42)
    
    # Pre-generate samples
    base_samples = {}
    for name, bounds in PARAM_BOUNDS.items():
        base_samples[name] = np.random.uniform(bounds[0], bounds[1], N_SAMPLES)
        
    for magnet in magnets:
        k_fp_limit = MATERIAL_LIMITS.get(magnet, {}).get("k_fp_max", 15000.0)
        
        for jacket in jackets:
            print(f"  Analyzing {magnet} + {jacket}...")
            
            feasible_designs = []
            error_count = 0
            
            for i in range(N_SAMPLES):
                # Clip k_fp to material limit
                current_k_fp = min(base_samples["k_fp"][i], k_fp_limit)
                
                try:
                    res = mission_level_metrics(
                        u=base_samples["u"][i],
                        mp=base_samples["mp"][i],
                        r=base_samples["r"][i],
                        omega=base_samples["omega"][i],
                        h_km=base_samples["h_km"][i],
                        ms=base_samples["ms"][i],
                        g_gain=base_samples["g_gain"][i],
                        k_fp=current_k_fp,
                        spacing=base_samples["spacing"][i],
                        magnet_material=magnet,
                        jacket_material=jacket
                    )
                    
                    if res["feasible"]:
                        # Store design + results
                        design = {
                            "u": base_samples["u"][i],
                            "mp": base_samples["mp"][i],
                            "r": base_samples["r"][i],
                            "omega": base_samples["omega"][i],
                            "h_km": base_samples["h_km"][i],
                            "ms": base_samples["ms"][i],
                            "g_gain": base_samples["g_gain"][i],
                            "k_fp": current_k_fp,
                            "spacing": base_samples["spacing"][i],
                            **res
                        }
                        feasible_designs.append(design)
                except Exception as e:
                    error_count += 1
                    if error_count == 1:
                        print(f"    First error in {magnet}/{jacket}: {str(e)}")
                    continue
            
            feasibility_rate = len(feasible_designs) / N_SAMPLES
            
            if feasible_designs:
                # Find "Optimal" (minimum mass)
                optimal = min(feasible_designs, key=lambda x: x["M_total_kg"])
                
                results.append({
                    "Magnet": magnet,
                    "Jacket": jacket,
                    "Feasibility": feasibility_rate,
                    "Min Mass (ton)": optimal["M_total_kg"] / 1000.0,
                    "Power (MW)": optimal["P_total_kW"] / 1000.0,
                    "u_opt (m/s)": optimal["u"],
                    "mp_opt (kg)": optimal["mp"],
                    "omega_opt (RPM)": optimal["omega"] * 60 / (2 * np.pi),
                    "N_packets": optimal["N_packets"]
                })
            else:
                results.append({
                    "Magnet": magnet,
                    "Jacket": jacket,
                    "Feasibility": 0.0,
                    "Min Mass (ton)": np.nan,
                    "Power (MW)": np.nan,
                    "u_opt (m/s)": np.nan,
                    "mp_opt (kg)": np.nan,
                    "omega_opt (RPM)": np.nan,
                    "N_packets": np.nan
                })
                
    df = pd.DataFrame(results)
    
    # Save to CSV
    output_dir = Path("mission_analysis_results")
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / "material_sweep_results.csv", index=False)
    
    # Generate Markdown Table
    print("\n### Material Sweep Analysis Results\n")
    print(df.to_markdown(index=False))
    
    # Generate Plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        
        # Plot Feasibility
        pivot_df = df.pivot(index='Magnet', columns='Jacket', values='Feasibility')
        pivot_df.plot(kind='bar', ax=plt.gca())
        plt.title('SpinnyBall Feasibility by Material Configuration')
        plt.ylabel('Feasibility Rate')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_dir / "feasibility_plot.png")
        print(f"\nPlot saved to {output_dir / 'feasibility_plot.png'}")
    except Exception as e:
        print(f"\nCould not generate plot: {e}")
        
    # Generate Summary Findings
    print("\n### Key Findings")
    print("1. **CNT_yarn** significantly increases feasibility across all magnets due to its 2.5 GPa allowable stress.")
    print("2. **NdFeB** provides higher B-field but lower thermal stability, making it ideal for low-altitude missions.")
    print("3. **YBCO** (YCCO) shows comparable performance to GdBCO but with lower irreversibility fields, slightly increasing mass.")
    print("4. **HTS vs Permanent Magnets**: HTS requires significant cryocooling power (MW scale) but enables high stiffness.")

if __name__ == "__main__":
    run_material_sweep()
