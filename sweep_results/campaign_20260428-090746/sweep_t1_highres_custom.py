
import sys
sys.path.insert(0, 'c:\\Users\\msunw\\Desktop\\SpinnyBall')

from sweep_latency_eta_ind import run_t1_sweep, plot_t1_results, analyze_stability_boundary

# High-resolution sweep
results = run_t1_sweep(
    latency_range=(1.0, 100.0),  # Expanded: 1-100ms
    eta_ind_range=(0.75, 0.98),  # Expanded: 0.75-0.98
    n_latency_points=20,  # Higher resolution
    n_eta_points=15,  # Higher resolution
    n_realizations_per_point=100,  # More MC runs
)

plot_t1_results(results, output_file='sweep_t1_highres.png')
analysis = analyze_stability_boundary(results)

# Save results
import json
with open('sweep_t1_highres_results.json', 'w') as f:
    json.dump({k: v.tolist() if hasattr(v, 'tolist') else v 
               for k, v in results.items()}, f, indent=2)

print("\n=== T1 HIGH-RES SWEEP COMPLETE ===")
