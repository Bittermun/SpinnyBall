
import sys
sys.path.insert(0, 'c:\\Users\\msunw\\Desktop\\SpinnyBall')

from sweep_fault_cascade import run_t3_sweep, plot_t3_results, analyze_containment_threshold

# High-resolution sweep
results = run_t3_sweep(
    fault_rate_range=(1e-8, 1e-2),  # Expanded: 10^-8 to 10^-2
    n_fault_rate_points=15,  # Higher resolution
    cascade_threshold=1.05,
    containment_threshold=2,
    n_nodes=10,
    n_realizations_per_point=200,  # More MC runs
    time_horizon=10.0,
)

plot_t3_results(results, output_file='sweep_t3_highres.png')
analysis = analyze_containment_threshold(results)

# Save results
import json
with open('sweep_t3_highres_results.json', 'w') as f:
    json.dump({k: v.tolist() if hasattr(v, 'tolist') else v 
               for k, v in results.items()}, f, indent=2)

print("\n=== T3 HIGH-RES SWEEP COMPLETE ===")
