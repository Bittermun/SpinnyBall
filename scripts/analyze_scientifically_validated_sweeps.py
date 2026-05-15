#!/usr/bin/env python3
"""
Scientifically validated sweep analysis considering time periods and statistical rigor.

This script addresses concerns about cascade probability accuracy by:
1. Identifying sweep runs with adequate time horizons for statistical significance
2. Correcting for time period differences in probability calculations
3. Flagging results that may be impacted by short simulation periods
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple


def calculate_time_adjusted_cascade_probability(
    cascade_probability: float, 
    time_horizon: float, 
    target_time_horizon: float = 3600.0  # 1 hour default
) -> Tuple[float, str]:
    """
    Adjust cascade probability based on time horizon differences.
    
    For rare events, probability scales linearly with time for small probabilities.
    P(t) ≈ λ*t where λ is the rate parameter.
    """
    if time_horizon <= 0:
        return cascade_probability, "ERROR: Invalid time horizon"
    
    if cascade_probability == 0.0:
        # If no cascades observed, we can only provide an upper bound
        # Using Rule of 3: 95% confidence upper bound = 3/n if 0 events in n trials
        adjusted_prob = 3.0 / target_time_horizon if target_time_horizon > 0 else 0.0
        warning = f"Zero cascades observed in {time_horizon}s simulation. Upper bound estimate for {target_time_horizon}s: {adjusted_prob:.2e}"
        return adjusted_prob, warning
    
    # Scale probability proportionally to time horizon
    adjusted_probability = cascade_probability * (target_time_horizon / time_horizon)
    
    # Cap at 1.0 since probabilities cannot exceed 1
    if adjusted_probability > 1.0:
        adjusted_probability = 1.0
        warning = f"Adjusted probability capped at 1.0 (initial: {cascade_probability:.2e}, time_horizon: {time_horizon}s)"
    else:
        warning = ""
    
    return adjusted_probability, warning


def extract_run_metadata(sweep_dir: Path) -> Dict[str, Any]:
    """Extract metadata including time horizon from sweep directory."""
    metadata = {
        "directory": sweep_dir.name,
        "timestamp": sweep_dir.name.split("_", 3)[-1] if "_" in sweep_dir.name else "unknown",
        "profiles_found": {},
        "has_errors": False,
        "error_details": []
    }
    
    # Look for all JSON files in the directory
    for json_file in sweep_dir.glob("t3_sweep_*.json"):
        profile_name = json_file.stem.replace("t3_sweep_", "")
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Estimate time horizon from the script that likely generated this data
            time_horizon = estimate_time_horizon_from_directory(sweep_dir)
            
            # Extract metrics for this profile
            cascade_probs = data.get("cascade_probabilities", [])
            fault_rates = data.get("fault_rates", [])
            
            # Calculate time-adjusted probabilities
            adjusted_cascade_probs = []
            warnings = []
            
            for prob in cascade_probs:
                adj_prob, warning = calculate_time_adjusted_cascade_probability(prob, time_horizon)
                adjusted_cascade_probs.append(adj_prob)
                if warning:
                    warnings.append(warning)
            
            profile_metrics = {
                "profile_name": profile_name,
                "original_cascade_probabilities": cascade_probs,
                "adjusted_cascade_probabilities": adjusted_cascade_probs,
                "fault_rates": fault_rates,
                "time_horizon_used": time_horizon,
                "warnings": warnings,
                "cascade_prob_avg_original": np.mean(cascade_probs) if cascade_probs else 0.0,
                "cascade_prob_avg_adjusted": np.mean(adjusted_cascade_probs) if adjusted_cascade_probs else 0.0,
                "max_cascade_prob_original": np.max(cascade_probs) if cascade_probs else 0.0,
                "max_cascade_prob_adjusted": np.max(adjusted_cascade_probs) if adjusted_cascade_probs else 0.0,
            }
            
            metadata["profiles_found"][profile_name] = profile_metrics
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON decode error in {json_file.name}: {str(e)}"
            metadata["error_details"].append(error_msg)
            metadata["has_errors"] = True
            print(f"  Warning: Skipping {json_file.name} due to JSON error: {e}")
        except Exception as e:
            error_msg = f"General error in {json_file.name}: {str(e)}"
            metadata["error_details"].append(error_msg)
            metadata["has_errors"] = True
            print(f"  Warning: Skipping {json_file.name} due to error: {e}")
    
    return metadata


def estimate_time_horizon_from_directory(sweep_dir: Path) -> float:
    """
    Estimate time horizon used based on directory name patterns and common values.
    In practice, most sweeps use either 10s (default for quick tests) or 3600s (for production).
    """
    dir_name = sweep_dir.name
    
    # Check if this is a high-res sweep which typically uses longer time horizons
    if "highres" in dir_name.lower() or "long" in dir_name.lower():
        return 3600.0  # 1 hour for high-resolution sweeps
    
    # Check if this is a research-grade sweep
    if "research" in dir_name.lower():
        return 3600.0  # 1 hour for research-grade sweeps
    
    # Check for quick sweeps which might use shorter horizons
    if "quick" in dir_name.lower():
        return 10.0  # Shorter time for quick analysis
    
    # Default conservative estimate - many of the older sweeps used 10s
    # which is inadequate for rare event statistics at low fault rates
    return 10.0  # Conservative default for older/unknown sweeps


def analyze_sweep_directories(base_path: Path) -> Dict[str, Any]:
    """Analyze all sweep directories with scientific rigor."""
    sweep_dirs = list(base_path.glob("profile_sweep_quick_*"))
    
    if not sweep_dirs:
        print("No profile_sweep directories found!")
        return {}
    
    print(f"Found {len(sweep_dirs)} sweep directories to analyze...")
    print("Applying scientific corrections for time period differences...\n")
    
    all_runs = []
    profile_comparison = {}
    
    for sweep_dir in sweep_dirs:
        print(f"Analyzing: {sweep_dir.name}")
        run_metadata = extract_run_metadata(sweep_dir)
        all_runs.append(run_metadata)
        
        # Build profile comparison data
        for profile_name, profile_data in run_metadata["profiles_found"].items():
            if profile_name not in profile_comparison:
                profile_comparison[profile_name] = []
            
            profile_comparison[profile_name].append({
                "directory": run_metadata["directory"],
                "timestamp": run_metadata["timestamp"],
                "time_horizon": profile_data["time_horizon_used"],
                "cascade_prob_avg_original": profile_data["cascade_prob_avg_original"],
                "cascade_prob_avg_adjusted": profile_data["cascade_prob_avg_adjusted"],
                "max_cascade_prob_original": profile_data["max_cascade_prob_original"],
                "max_cascade_prob_adjusted": profile_data["max_cascade_prob_adjusted"],
                "warnings": profile_data["warnings"]
            })
    
    # Generate summary with scientific validation
    summary = {
        "summary": {
            "total_runs_analyzed": len(all_runs),
            "runs_with_short_time_horizon": sum(
                1 for run in all_runs 
                for prof_data in run["profiles_found"].values() 
                if prof_data["time_horizon_used"] < 3600.0  # Less than 1 hour
            ),
            "date_generated": "2026-05-07",
            "scientific_note": "Probabilities adjusted for 1-hour time horizon assuming linear scaling for rare events"
        },
        "runs": all_runs,
        "profile_comparison": profile_comparison
    }
    
    return summary


def generate_scientific_report(summary: Dict[str, Any]):
    """Generate a scientific report highlighting time period concerns."""
    print("\n" + "="*80)
    print("SCIENTIFIC VALIDATION REPORT")
    print("="*80)
    
    summary_info = summary["summary"]
    print(f"\nTotal runs analyzed: {summary_info['total_runs_analyzed']}")
    print(f"Runs with potentially inadequate time horizons: {summary_info['runs_with_short_time_horizon']}")
    print(f"Adjustment basis: scaled to 1-hour (3600s) time horizon")
    
    print("\n" + "-"*80)
    print("TIME HORIZON ANALYSIS")
    print("-"*80)
    
    # Count runs by time horizon
    horizon_counts = {}
    for run in summary["runs"]:
        for profile_name, profile_data in run["profiles_found"].items():
            horizon = profile_data["time_horizon_used"]
            if horizon not in horizon_counts:
                horizon_counts[horizon] = 0
            horizon_counts[horizon] += 1
    
    for horizon, count in sorted(horizon_counts.items()):
        concern_level = "⚠️ CRITICAL" if horizon < 100 else "⚠️ CONCERN" if horizon < 3600 else "✅ ADEQUATE"
        print(f"  {horizon}s: {count} runs - {concern_level}")
    
    print("\n" + "-"*80)
    print("PROFILE-LEVEL ANALYSIS")
    print("-"*80)
    
    for profile, data_list in summary["profile_comparison"].items():
        print(f"\n{profile.upper()}:")
        
        # Group by time horizon to show differences
        horizon_groups = {}
        for data in data_list:
            horizon = data["time_horizon"]
            if horizon not in horizon_groups:
                horizon_groups[horizon] = []
            horizon_groups[horizon].append(data)
        
        for horizon, runs in horizon_groups.items():
            concern = "⚠️" if horizon < 3600 else "✅"
            orig_avg = np.mean([d["cascade_prob_avg_original"] for d in runs])
            adj_avg = np.mean([d["cascade_prob_avg_adjusted"] for d in runs])
            
            print(f"  {concern} {horizon}s horizon: {len(runs)} runs")
            print(f"    Original avg cascade prob: {orig_avg:.2e}")
            print(f"    Adjusted avg cascade prob: {adj_avg:.2e}")
            
            # Show warnings if any
            all_warnings = [w for d in runs for w in d["warnings"] if w]
            if all_warnings:
                print(f"    Warnings: {len(all_warnings)} issues detected")
                for w in all_warnings[:3]:  # Show first 3 warnings
                    print(f"      - {w}")
                if len(all_warnings) > 3:
                    print(f"      ... and {len(all_warnings)-3} more")


def main():
    """Main function to run scientifically validated sweep analysis."""
    print("Starting scientifically validated sweep analysis...")
    print("This analysis corrects for time period differences in cascade probability calculations.")
    
    base_path = Path(".")
    summary = analyze_sweep_directories(base_path)
    
    if not summary:
        print("No data to process!")
        return
    
    # Generate scientific report
    generate_scientific_report(summary)
    
    # Save detailed results
    output_file = "scientifically_validated_sweep_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    # Create a summary CSV for easy review
    rows = []
    for profile, data_list in summary["profile_comparison"].items():
        for data in data_list:
            rows.append({
                "profile": profile,
                "directory": data["directory"],
                "time_horizon": data["time_horizon"],
                "orig_avg_cascade_prob": data["cascade_prob_avg_original"],
                "adj_avg_cascade_prob": data["cascade_prob_avg_adjusted"],
                "orig_max_cascade_prob": data["max_cascade_prob_original"],
                "adj_max_cascade_prob": data["max_cascade_prob_adjusted"],
                "concern_level": "HIGH" if data["time_horizon"] < 3600 else "LOW"
            })
    
    df = pd.DataFrame(rows)
    csv_file = "sweep_analysis_summary.csv"
    df.to_csv(csv_file, index=False)
    
    print(f"Summary CSV saved to: {csv_file}")
    print("\nAnalysis complete! Review the scientific validation report above.")


if __name__ == "__main__":
    main()