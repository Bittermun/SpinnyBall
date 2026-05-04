"""
High-resolution 2x3 material sweep (parallel, N=512).
GdBCO/SmCo × BFRP/CFRP/CNT_yarn → 6 configs, fully parallelised.
Run from SpinnyBall root: python scripts/run_highres_sweep.py
"""
import sys, time
import numpy as np
from pathlib import Path

sys.path.insert(0, '.')

from src.sgms_anchor_sensitivity import (
    run_2x2_material_sweep,
    print_mission_summary,
    export_sobol_indices_csv,
    plot_mission_results,
)

print("=" * 70)
print("HIGH-RES 2x3 MATERIAL SWEEP  (N=512, joblib parallel, seed=42)")
print("=" * 70)

t0 = time.perf_counter()
all_results = run_2x2_material_sweep(N=512, seed=42)
elapsed = time.perf_counter() - t0

print(f"\n[TIMING] Full 6-config sweep: {elapsed:.1f}s ({elapsed/60:.1f} min)")

out_dir = Path("mission_analysis_results")
out_dir.mkdir(exist_ok=True)

for config_key, res in all_results.items():
    print_mission_summary(res)

    csv_path = out_dir / f"{config_key}_sobol_indices.csv"
    export_sobol_indices_csv(res["indices"], res["problem"]["names"], csv_path)
    print(f"  Sobol CSV  -> {csv_path}")

    plot_prefix = str(out_dir / config_key)
    plot_mission_results(res, filename_prefix=plot_prefix)

    np.save(str(out_dir / f"{config_key}_feasible.npy"), res["feasible"])
    np.save(str(out_dir / f"{config_key}_samples.npy"), res["samples"])

# Cross-config comparison table
print("\n" + "=" * 80)
print("CROSS-CONFIG COMPARISON SUMMARY")
print("=" * 80)
header = f"{'Config':<25} {'Feasible%':>10} {'N_pkt':>10} {'M_kg':>10} {'stress_margin':>14}"
print(header)
print("-" * 80)
for config_key in sorted(all_results):
    res = all_results[config_key]
    f = res["feasible"]
    n = res["outputs"]["N_packets"]
    m = res["outputs"]["M_total_kg"]
    s = res["outputs"]["stress_margin"]
    row = (
        f"{config_key:<25}"
        f" {np.mean(f):>9.1%}"
        f" {np.mean(n):>10.0f}"
        f" {np.mean(m):>10.0f}"
        f" {np.nanmean(s):>14.2f}"
    )
    print(row)

print(f"\nAll results saved to: {out_dir.resolve()}/")
print("Done.")
