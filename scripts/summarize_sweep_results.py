#!/usr/bin/env python3
"""
Aggregate and summarize sweep results across multiple profile_sweep directories.

This script analyzes multiple profile_sweep_* directories and creates a consolidated
summary of key metrics for cross-run comparison and analysis.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np


def extract_run_metrics(sweep_dir: Path) -> Dict[str, Any]:
    """Extract key metrics from a sweep directory."""
    metrics = {
        "directory": sweep_dir.name,
        "timestamp": sweep_dir.name.split("_", 3)[-1] if "_" in sweep_dir.name else "unknown",
        "profiles_found": {},
        "converged_count": 0,
        "total_profiles": 0,
        "avg_cascade_prob": 0.0,
        "max_cascade_prob": 0.0,
        "min_cascade_prob": 1.0,
        "error_count": 0,
        "errors": []
    }
    
    # Look for all JSON files in the directory
    for json_file in sweep_dir.glob("t3_sweep_*.json"):
        profile_name = json_file.stem.replace("t3_sweep_", "")
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            error_msg = f"JSON decode error in {json_file.name}: {str(e)}"
            metrics["errors"].append(error_msg)
            metrics["error_count"] += 1
            print(f"  Warning: Skipping {json_file.name} due to JSON error: {e}")
            continue
        except Exception as e:
            error_msg = f"General error in {json_file.name}: {str(e)}"
            metrics["errors"].append(error_msg)
            metrics["error_count"] += 1
            print(f"  Warning: Skipping {json_file.name} due to error: {e}")
            continue
        
        # Extract metrics for this profile
        cascade_probs = data.get("cascade_probabilities", [])
        converged_flags = data.get("converged", [])
        
        profile_metrics = {
            "profile_name": profile_name,
            "cascade_prob_avg": np.mean(cascade_probs) if cascade_probs else 0.0,
            "cascade_prob_max": np.max(cascade_probs) if cascade_probs else 0.0,
            "cascade_prob_min": np.min(cascade_probs) if cascade_probs else 0.0,
            "converged_count": sum(converged_flags) if converged_flags else 0,
            "total_points": len(converged_flags) if converged_flags else 0,
            "convergence_rate": sum(converged_flags)/len(converged_flags) if converged_flags else 0.0,
        }
        
        metrics["profiles_found"][profile_name] = profile_metrics
        metrics["converged_count"] += profile_metrics["converged_count"]
        metrics["total_profiles"] += 1
        
        # Update global cascade stats
        if cascade_probs:
            avg_cascade = np.mean(cascade_probs)
            metrics["avg_cascade_prob"] = (metrics["avg_cascade_prob"] * (metrics["total_profiles"] - 1) + avg_cascade) / metrics["total_profiles"]
            metrics["max_cascade_prob"] = max(metrics["max_cascade_prob"], np.max(cascade_probs))
            metrics["min_cascade_prob"] = min(metrics["min_cascade_prob"], np.min(cascade_probs))
    
    return metrics


def aggregate_all_sweeps(base_path: Path) -> Dict[str, Any]:
    """Aggregate metrics from all profile_sweep directories."""
    sweep_dirs = list(base_path.glob("profile_sweep_quick_*"))
    
    if not sweep_dirs:
        print("No profile_sweep directories found!")
        return {}
    
    print(f"Found {len(sweep_dirs)} sweep directories to analyze...")
    
    all_runs = []
    profile_comparison = {}
    total_errors = 0
    
    for sweep_dir in sweep_dirs:
        print(f"Processing: {sweep_dir.name}")
        run_metrics = extract_run_metrics(sweep_dir)
        total_errors += run_metrics.get("error_count", 0)
        all_runs.append(run_metrics)
        
        # Build profile comparison data
        for profile_name, profile_data in run_metrics["profiles_found"].items():
            if profile_name not in profile_comparison:
                profile_comparison[profile_name] = []
            
            profile_comparison[profile_name].append({
                "directory": run_metrics["directory"],
                "timestamp": run_metrics["timestamp"],
                "cascade_prob_avg": profile_data["cascade_prob_avg"],
                "convergence_rate": profile_data["convergence_rate"],
                "converged_count": profile_data["converged_count"]
            })
    
    # Calculate overall statistics
    valid_runs = [r for r in all_runs if not r.get("error_count", 0)]
    if valid_runs:
        total_converged = sum(r["converged_count"] for r in valid_runs)
        total_points = sum(sum(p["total_points"] for p in r["profiles_found"].values()) for r in valid_runs)
        overall_convergence_rate = total_converged / total_points if total_points > 0 else 0
    else:
        total_converged = 0
        total_points = 0
        overall_convergence_rate = 0
    
    summary = {
        "summary": {
            "total_runs_analyzed": len(all_runs),
            "valid_runs": len(valid_runs),
            "error_runs": total_errors,
            "total_converged_points": total_converged,
            "total_simulation_points": total_points,
            "overall_convergence_rate": overall_convergence_rate,
            "date_generated": "2026-05-07"  # Using current date
        },
        "runs": all_runs,
        "profile_comparison": profile_comparison
    }
    
    return summary


def main():
    """Main function to run the aggregation."""
    print("Starting sweep results aggregation...")
    
    base_path = Path(".")
    summary = aggregate_all_sweeps(base_path)
    
    if not summary:
        print("No data to process!")
        return
    
    # Save detailed summary
    output_file = "sweep_aggregated_summary.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)  # default=str handles non-serializable objects
    
    # Create a simple summary for quick review
    summary_info = summary["summary"]
    simple_summary = {
        "total_runs": summary_info["total_runs_analyzed"],
        "valid_runs": summary_info["valid_runs"],
        "error_runs": summary_info["error_runs"],
        "overall_convergence_rate": round(summary_info["overall_convergence_rate"], 4) if summary_info["total_simulation_points"] > 0 else 0,
        "runs_breakdown": [
            {
                "directory": run["directory"],
                "timestamp": run["timestamp"],
                "profiles": list(run["profiles_found"].keys()),
                "error_count": run.get("error_count", 0),
                "convergence_rate": round(
                    sum(p["converged_count"] for p in run["profiles_found"].values()) / 
                    sum(p["total_points"] for p in run["profiles_found"].values()), 4
                ) if run["profiles_found"] and sum(p["total_points"] for p in run["profiles_found"].values()) > 0 else 0
            } 
            for run in summary["runs"]
        ],
        "profile_performance": {
            profile_name: {
                "runs_count": len(data),
                "avg_cascade_prob": round(np.mean([d["cascade_prob_avg"] for d in data]), 4),
                "avg_convergence_rate": round(np.mean([d["convergence_rate"] for d in data]), 4),
            }
            for profile_name, data in summary["profile_comparison"].items()
        }
    }
    
    # Save simple summary
    simple_output_file = "sweep_summary.json"
    with open(simple_output_file, 'w') as f:
        json.dump(simple_summary, f, indent=2)
    
    print(f"\nAggregation complete!")
    print(f"Detailed results saved to: {output_file}")
    print(f"Simple summary saved to: {simple_output_file}")
    
    print(f"\nQuick Stats:")
    print(f"- Analyzed {simple_summary['total_runs']} run directories")
    print(f"- Valid runs: {simple_summary['valid_runs']}")
    print(f"- Runs with errors: {simple_summary['error_runs']}")
    print(f"- Overall convergence rate: {simple_summary['overall_convergence_rate']*100:.2f}%")
    print(f"- Profile performance:")
    for profile, perf in simple_summary["profile_performance"].items():
        print(f"  * {profile}: {perf['avg_cascade_prob']:.4f} avg cascade prob, "
              f"{perf['avg_convergence_rate']*100:.2f}% convergence")


if __name__ == "__main__":
    main()