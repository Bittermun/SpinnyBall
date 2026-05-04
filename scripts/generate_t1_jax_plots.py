import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_jax_stability_map():
    results_path = Path("sweep_results/jax_t1_highres.json")
    if not results_path.exists():
        print(f"Error: {results_path} not found. Run scripts/jax_sweep_latency_eta_ind.py first.")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    latency = np.array(data["latency_range"]) * 1000  # ms
    eta = np.array(data["eta_range"])
    grid = np.array(data["success_rate_grid"])

    plt.figure(figsize=(10, 8), dpi=150)
    plt.style.use('dark_background')
    
    # Create heatmap
    im = plt.imshow(grid, origin='lower', extent=[latency[0], latency[-1], eta[0], eta[-1]], 
                    aspect='auto', cmap='RdYlGn', interpolation='gaussian')
    
    plt.colorbar(im, label='Success Probability (N=1000)')
    
    # Add contour line for the 0.5 success boundary
    plt.contour(latency, eta, grid, levels=[0.5], colors='white', linestyles='--', linewidths=2)
    
    plt.title("Figure 7: Control Stability Boundary (T1)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Feedback Latency (ms)", fontsize=12)
    plt.ylabel("Induction Efficiency ($\eta_{ind}$)", fontsize=12)
    
    # Highlight operational target
    plt.scatter([20], [0.9], color='cyan', marker='*', s=200, label='Operational Target (20ms, 0.90)')
    
    plt.grid(True, alpha=0.2, linestyle=':')
    plt.legend(loc='upper right')
    
    output_path = Path("paper_figures/fig7_jax_stability_map.png")
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Successfully generated Figure 7: {output_path}")

if __name__ == "__main__":
    plot_jax_stability_map()
