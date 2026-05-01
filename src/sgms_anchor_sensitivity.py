"""
Sobol sensitivity analysis for the reduced-order dynamic-anchor model.

This module evaluates analytical anchor outputs from ``sgms_anchor_v1.py`` and
uses SALib to quantify parameter importance without running the ODE solver.
That keeps paper-scale sweeps fast and deterministic.

Updated for mission-level Sobol analysis with 8 parameters and second-order indices.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample

from sgms_anchor_v1 import DEFAULT_PARAMS, analytical_metrics, mission_level_metrics


DEFAULT_PROBLEM = {
    "num_vars": 5,
    "names": ["u", "g_gain", "eps", "lam", "mp"],
    "bounds": [
        [5.0, 1600.0],    # u (m/s) - extended to operational target
        [0.02, 0.2],      # g_gain
        [0.0, 1e-3],      # eps
        [0.1, 20.0],      # lam (kg/m) - extended for operational scale
        [0.05, 8.0],      # mp (kg) - added for mass sweep
    ],
}

DEFAULT_OUTPUTS = (
    "force_per_stream_n",
    "k_eff",
    "period_s",
    "static_offset_m",
    "packet_rate_hz",
)

# Mission-level problem definition with 8 parameters
MISSION_PROBLEM = {
    "num_vars": 8,
    "names": ["u", "mp", "r", "omega", "h_km", "ms", "g_gain", "k_fp"],
    "bounds": [
        [500.0, 15000.0],     # u (m/s) - operational velocity range
        [1.0, 50.0],          # mp (kg) - packet mass
        [0.02, 0.15],         # r (m) - packet radius
        [2000.0, 6000.0],     # omega (rad/s) - spin rate (20k-60k RPM)
        [300.0, 2000.0],      # h_km (km) - orbital altitude
        [100.0, 10000.0],     # ms (kg) - station mass
        [1e-4, 1e-2],         # g_gain - control gain
        [1000.0, 15000.0],    # k_fp (N/m) - flux-pinning stiffness
    ],
}

MISSION_OUTPUTS = (
    "N_packets",
    "M_total_kg",
    "P_total_kW",
    "stress_margin",
    "thermal_margin",
    "k_eff",
    "feasible",
)


def _copy_params(base_params: dict | None = None) -> dict:
    params = DEFAULT_PARAMS.copy()
    if base_params is not None:
        params.update(base_params)
    return params


def evaluate_parameter_vector(vector: np.ndarray, base_params: dict | None = None) -> dict:
    params = _copy_params(base_params)
    u, g_gain, eps, lam, mp = [float(v) for v in vector]
    params.update({"u": u, "g_gain": g_gain, "eps": eps, "lam": lam, "mp": mp})
    return analytical_metrics(params)


def sample_anchor_problem(
    problem: dict | None = None,
    N: int = 256,
    calc_second_order: bool = False,
    seed: int | None = None,
) -> np.ndarray:
    problem = DEFAULT_PROBLEM if problem is None else problem
    return sobol_sample.sample(
        problem,
        N,
        calc_second_order=calc_second_order,
        scramble=True,
        seed=seed,
    )


def evaluate_sample_matrix(
    samples: np.ndarray,
    outputs: tuple[str, ...] = DEFAULT_OUTPUTS,
    base_params: dict | None = None,
) -> dict[str, np.ndarray]:
    values = {output: np.empty(samples.shape[0]) for output in outputs}
    for i, sample in enumerate(samples):
        metrics = evaluate_parameter_vector(sample, base_params=base_params)
        for output in outputs:
            values[output][i] = metrics[output]
    return values


def run_sobol_sensitivity(
    problem: dict | None = None,
    N: int = 256,
    outputs: tuple[str, ...] = DEFAULT_OUTPUTS,
    calc_second_order: bool = False,
    base_params: dict | None = None,
    seed: int | None = None,
) -> dict:
    problem = DEFAULT_PROBLEM if problem is None else problem
    samples = sample_anchor_problem(problem, N=N, calc_second_order=calc_second_order, seed=seed)
    values = evaluate_sample_matrix(samples, outputs=outputs, base_params=base_params)

    indices = {}
    for output in outputs:
        indices[output] = sobol_analyze.analyze(
            problem,
            values[output],
            calc_second_order=calc_second_order,
            print_to_console=False,
            seed=seed,
        )

    return {
        "problem": problem,
        "samples": samples,
        "outputs": values,
        "indices": indices,
        "calc_second_order": calc_second_order,
    }


def export_sobol_indices_csv(indices: dict, names: list[str], filename: str | Path) -> None:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for output, data in indices.items():
        st = np.asarray(data["ST"])
        s1 = np.asarray(data["S1"])
        for i, name in enumerate(names):
            rows.append(
                {
                    "output": output,
                    "parameter": name,
                    "S1": float(s1[i]),
                    "ST": float(st[i]),
                }
            )

    with filename.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["output", "parameter", "S1", "ST"])
        writer.writeheader()
        writer.writerows(rows)


def plot_sobol_indices(indices: dict, names: list[str], filename: str = "sgms_anchor_sobol.png") -> None:
    outputs = list(indices.keys())
    fig, axes = plt.subplots(len(outputs), 1, figsize=(10, 3.4 * len(outputs)), squeeze=False)

    for ax, output in zip(axes.ravel(), outputs):
        data = indices[output]
        x = np.arange(len(names))
        width = 0.38
        ax.bar(x - width / 2, data["S1"], width=width, label="S1", color="#79c0ff")
        ax.bar(x + width / 2, data["ST"], width=width, label="ST", color="#7ee787")
        ax.set_xticks(x, names)
        ax.set_ylim(bottom=min(0.0, np.min(data["S1"]) - 0.05), top=max(1.0, np.max(data["ST"]) + 0.05))
        ax.set_ylabel("Index")
        ax.set_title(f"Sobol Indices: {output}")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def print_sensitivity_summary(result: dict) -> None:
    print("=== ANCHOR SOBOL SENSITIVITY ===")
    names = result["problem"]["names"]
    for output, data in result["indices"].items():
        st = np.asarray(data["ST"])
        top_idx = int(np.nanargmax(st))
        print(f"{output}: dominant ST = {names[top_idx]} ({st[top_idx]:.4f})")


def evaluate_mission_vector(vector: np.ndarray, material_profile: str = "SmCo") -> dict:
    """
    Evaluate mission-level metrics for a single parameter vector.
    
    Args:
        vector: 8-element array [u, mp, r, omega, h_km, ms, g_gain, k_fp]
        material_profile: "SmCo" or "GdBCO"
    
    Returns:
        Dictionary with mission outputs
    """
    u, mp, r, omega, h_km, ms, g_gain, k_fp = [float(v) for v in vector]
    
    return mission_level_metrics(
        u=u,
        mp=mp,
        r=r,
        omega=omega,
        h_km=h_km,
        ms=ms,
        g_gain=g_gain,
        k_fp=k_fp,
        material_profile=material_profile,
    )


def run_mission_sobol_analysis(
    material_profile: str = "SmCo",
    N: int = 1024,
    calc_second_order: bool = True,
    seed: int = 42,
) -> dict:
    """
    Run Sobol sensitivity analysis on mission-level metrics.
    
    This function performs a comprehensive global sensitivity analysis
    across 8 design parameters with second-order interaction terms.
    
    Args:
        material_profile: Material type ("SmCo" or "GdBCO")
        N: Number of samples (>= 1024 recommended for 8 parameters)
        calc_second_order: Include S2 interaction indices
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with Sobol results including:
        - problem: Problem definition
        - samples: Sample matrix
        - outputs: Output values for each sample
        - indices: Sobol sensitivity indices
        - feasible: Boolean feasibility array
        - material_profile: Material used
    """
    print(f"Running mission-level Sobol analysis for {material_profile}...")
    print(f"  N={N}, calc_second_order={calc_second_order}, seed={seed}")
    
    # Generate samples
    samples = sobol_sample.sample(
        MISSION_PROBLEM,
        N,
        calc_second_order=calc_second_order,
        scramble=True,
        seed=seed,
    )
    
    print(f"  Generated {samples.shape[0]} samples")
    
    # Evaluate mission metrics for each sample
    n_samples = samples.shape[0]
    outputs_dict = {output: np.empty(n_samples) for output in MISSION_OUTPUTS}
    feasible_array = np.empty(n_samples, dtype=bool)
    
    for i, sample in enumerate(samples):
        metrics = evaluate_mission_vector(sample, material_profile=material_profile)
        for output in MISSION_OUTPUTS:
            if output == "feasible":
                feasible_array[i] = metrics[output]
            else:
                outputs_dict[output][i] = metrics[output]
    
    print(f"  Evaluation complete. Feasible: {np.sum(feasible_array)}/{n_samples}")
    
    # Run Sobol analysis on continuous outputs
    indices = {}
    for output in MISSION_OUTPUTS:
        if output == "feasible":
            continue  # Skip boolean output
        
        Y = outputs_dict[output]
        
        # Check for constant outputs (all same value)
        if np.std(Y) < 1e-10:
            print(f"  Warning: {output} has near-zero variance, skipping Sobol analysis")
            # Create dummy indices
            indices[output] = {
                "S1": np.zeros(MISSION_PROBLEM["num_vars"]),
                "ST": np.zeros(MISSION_PROBLEM["num_vars"]),
                "S2": np.zeros((MISSION_PROBLEM["num_vars"], MISSION_PROBLEM["num_vars"])) if calc_second_order else None,
            }
            continue
        
        try:
            indices[output] = sobol_analyze.analyze(
                MISSION_PROBLEM,
                Y,
                calc_second_order=calc_second_order,
                print_to_console=False,
                seed=seed,
            )
        except Exception as e:
            print(f"  Warning: Sobol analysis failed for {output}: {e}")
            indices[output] = {
                "S1": np.zeros(MISSION_PROBLEM["num_vars"]),
                "ST": np.zeros(MISSION_PROBLEM["num_vars"]),
                "S2": np.zeros((MISSION_PROBLEM["num_vars"], MISSION_PROBLEM["num_vars"])) if calc_second_order else None,
            }
    
    return {
        "problem": MISSION_PROBLEM,
        "samples": samples,
        "outputs": outputs_dict,
        "indices": indices,
        "feasible": feasible_array,
        "material_profile": material_profile,
        "N": N,
        "calc_second_order": calc_second_order,
        "seed": seed,
    }


def plot_mission_results(results: dict, filename_prefix: str = "mission_sobol") -> None:
    """
    Plot mission-level Sobol results.
    
    Creates bar charts of S1 and ST indices for each output,
    plus a feasibility heatmap if second-order indices are available.
    
    Args:
        results: Results from run_mission_sobol_analysis
        filename_prefix: Prefix for output filenames
    """
    import matplotlib.pyplot as plt
    
    names = results["problem"]["names"]
    indices = results["indices"]
    outputs = list(indices.keys())
    
    if not outputs:
        print("No outputs to plot")
        return
    
    # Plot first-order and total-order indices
    fig, axes = plt.subplots(len(outputs), 1, figsize=(12, 3.5 * len(outputs)), squeeze=False)
    
    for ax, output in zip(axes.ravel(), outputs):
        data = indices[output]
        x = np.arange(len(names))
        width = 0.38
        
        s1 = np.asarray(data["S1"])
        st = np.asarray(data["ST"])
        
        ax.bar(x - width / 2, s1, width=width, label="S1 (first-order)", color="#79c0ff")
        ax.bar(x + width / 2, st, width=width, label="ST (total-order)", color="#7ee787")
        ax.set_xticks(x, names, rotation=45, ha='right')
        ax.set_ylim(bottom=0, top=max(1.0, np.max(st) + 0.1))
        ax.set_ylabel("Sobol Index")
        ax.set_title(f"Sobol Indices: {output}")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
    
    fig.tight_layout()
    filename = f"{filename_prefix}_indices.png"
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {filename}")
    
    # Plot feasibility summary
    if "feasible" in results:
        feasible = results["feasible"]
        feasible_frac = np.mean(feasible)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.bar(["Feasible", "Infeasible"], 
               [np.sum(feasible), np.sum(~feasible)],
               color=["#7ee787", "#ff7b72"])
        ax.set_ylabel("Count")
        ax.set_title(f"Feasibility: {feasible_frac:.1%} feasible ({np.sum(feasible)}/{len(feasible)})")
        ax.grid(True, axis="y", alpha=0.3)
        
        filename = f"{filename_prefix}_feasibility.png"
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {filename}")


def print_mission_summary(results: dict) -> None:
    """Print summary of mission-level Sobol analysis."""
    print("\n" + "="*60)
    print("MISSION-LEVEL SOBOL SENSITIVITY ANALYSIS")
    print("="*60)
    print(f"Material: {results['material_profile']}")
    print(f"Samples: {results['N']} (second-order: {results['calc_second_order']})")
    print(f"Seed: {results['seed']}")
    print("-"*60)
    
    # Feasibility summary
    if "feasible" in results:
        feasible = results["feasible"]
        print(f"\nFeasibility: {np.mean(feasible):.1%} ({np.sum(feasible)}/{len(feasible)} designs)")
    
    # Dominant parameters for each output
    names = results["problem"]["names"]
    print("\nDominant Parameters (by total-order index ST):")
    print("-"*60)
    
    for output, data in results["indices"].items():
        st = np.asarray(data["ST"])
        if np.max(st) < 1e-6:
            continue  # Skip outputs with no variance
        
        sorted_idx = np.argsort(st)[::-1]
        top_3 = [(names[i], st[i]) for i in sorted_idx[:3]]
        
        print(f"\n{output}:")
        for name, value in top_3:
            print(f"  {name:12s}: ST = {value:.4f}")


def main() -> None:
    """Main entry point for sensitivity analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Sobol Sensitivity Analysis")
    parser.add_argument("--mission", action="store_true", 
                       help="Run mission-level analysis (8 parameters)")
    parser.add_argument("--material", choices=["SmCo", "GdBCO", "both"], default="both",
                       help="Material profile to analyze")
    parser.add_argument("--N", type=int, default=1024,
                       help="Number of Sobol samples (default: 1024)")
    parser.add_argument("--no-second-order", action="store_true",
                       help="Skip second-order interaction indices")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    args = parser.parse_args()
    
    if args.mission:
        # Mission-level analysis
        materials = ["SmCo", "GdBCO"] if args.material == "both" else [args.material]
        
        for mat in materials:
            print(f"\n{'='*60}")
            print(f"Analyzing {mat} material profile")
            print('='*60)
            
            results = run_mission_sobol_analysis(
                material_profile=mat,
                N=args.N,
                calc_second_order=not args.no_second_order,
                seed=args.seed,
            )
            
            # Save results
            output_dir = Path("mission_analysis_results")
            output_dir.mkdir(exist_ok=True)
            
            # Save to NPZ
            np.savez(
                output_dir / f"sobol_{mat.lower()}.npz",
                problem=np.array([MISSION_PROBLEM], dtype=object),
                samples=results["samples"],
                **{f"output_{k}": v for k, v in results["outputs"].items()},
                feasible=results["feasible"],
                indices=np.array([results["indices"]], dtype=object),
                material_profile=mat,
                N=args.N,
                calc_second_order=not args.no_second_order,
                seed=args.seed,
            )
            
            # Export indices to CSV
            export_sobol_indices_csv(
                results["indices"], 
                MISSION_PROBLEM["names"],
                output_dir / f"sobol_{mat.lower()}.csv"
            )
            
            # Plot results
            plot_mission_results(results, filename_prefix=f"mission_{mat.lower()}")
            
            # Print summary
            print_mission_summary(results)
        
        print(f"\nAll results saved to {output_dir}/")
    
    else:
        # Legacy anchor analysis
        result = run_sobol_sensitivity(
            N=512,
            outputs=("k_eff", "period_s", "static_offset_m", "packet_rate_hz"),
            calc_second_order=False,
            seed=11,
        )
        export_sobol_indices_csv(result["indices"], result["problem"]["names"], "sgms_anchor_sobol.csv")
        plot_sobol_indices(result["indices"], result["problem"]["names"])
        print_sensitivity_summary(result)


if __name__ == "__main__":
    main()
