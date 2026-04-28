"""
Comprehensive Sweep Campaign - Master Orchestration Script

Executes all available sweeps across all profiles with both default and high-resolution settings.
Runs in parallel where possible for maximum efficiency on powerful hardware.

Sweep Matrix:
- T1: Latency × η_ind (control stability)
- T3: Fault rate cascade (resilience)
- LOB: 40-node lattice scaling
- Sensitivity: Sobol analysis
- Stream Balance: Validation
- Mission Scenarios: Operational profiles

Profiles: paper-baseline, operational, engineering-screen, resilience
"""

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import concurrent.futures
import sys

# Configuration
RESULTS_DIR = Path("sweep_results")
TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
CAMPAIGN_DIR = RESULTS_DIR / f"campaign_{TIMESTAMP}"
CAMPAIGN_DIR.mkdir(parents=True, exist_ok=True)

PROFILES = ["paper-baseline", "operational", "engineering-screen", "resilience"]

# Sweep configurations
SWEEP_CONFIGS = {
    "t1_default": {
        "script": "sweep_latency_eta_ind.py",
        "args": [],
        "description": "T1 Default: Latency [5-50ms] × η_ind [0.8-0.95], 10×8 grid, 20 MC runs",
        "output_prefix": "t1_default"
    },
    "t1_highres": {
        "script": "sweep_latency_eta_ind.py",
        "args": [],  # Will modify script for high-res
        "description": "T1 High-Res: Expanded ranges, 20×15 grid, 100 MC runs",
        "output_prefix": "t1_highres"
    },
    "t3_default": {
        "script": "sweep_fault_cascade.py",
        "args": [],
        "description": "T3 Default: Fault rate [10^-6 to 10^-3], 8 points, 50 MC runs",
        "output_prefix": "t3_default"
    },
    "t3_highres": {
        "script": "sweep_fault_cascade.py",
        "args": [],
        "description": "T3 High-Res: Expanded fault range, 15 points, 200 MC runs",
        "output_prefix": "t3_highres"
    },
    "lob_scaling": {
        "script": "lob_scaling.py",
        "args": [],
        "description": "40-node LOB scaling with blackout test",
        "output_prefix": "lob_scaling"
    },
    "sensitivity": {
        "script": "sgms_anchor_sensitivity.py",
        "args": [],
        "description": "Sobol sensitivity analysis",
        "output_prefix": "sensitivity"
    },
    "stream_balance": {
        "script": "validate_stream_balance.py",
        "args": [],
        "description": "Stream balance validation",
        "output_prefix": "stream_balance"
    },
    "mission_scenarios": {
        "script": "scenarios/mission_scenarios.py",
        "args": [],
        "description": "Mission scenarios sweep",
        "output_prefix": "mission_scenarios"
    }
}


def run_command(cmd, name, log_file):
    """Run a command and log output."""
    print(f"[{name}] Starting: {' '.join(cmd)}")
    start_time = time.time()
    
    with open(log_file, 'w') as f:
        f.write(f"=== {name} ===\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Start time: {datetime.now()}\n\n")
        
        result = subprocess.run(
            cmd,
            cwd="c:\\Users\\msunw\\Desktop\\SpinnyBall",
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True
        )
    
    duration = time.time() - start_time
    status = "SUCCESS" if result.returncode == 0 else "FAILED"
    
    with open(log_file, 'a') as f:
        f.write(f"\nEnd time: {datetime.now()}\n")
        f.write(f"Duration: {duration:.2f}s\n")
        f.write(f"Status: {status}\n")
        f.write(f"Return code: {result.returncode}\n")
    
    print(f"[{name}] {status} in {duration:.2f}s")
    return name, status, duration, result.returncode


def create_high_res_t1_script():
    """Create high-resolution T1 sweep script."""
    content = '''
import sys
sys.path.insert(0, 'c:\\\\Users\\\\msunw\\\\Desktop\\\\SpinnyBall')

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

print("\\n=== T1 HIGH-RES SWEEP COMPLETE ===")
'''
    script_path = CAMPAIGN_DIR / "sweep_t1_highres_custom.py"
    with open(script_path, 'w') as f:
        f.write(content)
    return str(script_path)


def create_high_res_t3_script():
    """Create high-resolution T3 sweep script."""
    content = '''
import sys
sys.path.insert(0, 'c:\\\\Users\\\\msunw\\\\Desktop\\\\SpinnyBall')

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

print("\\n=== T3 HIGH-RES SWEEP COMPLETE ===")
'''
    script_path = CAMPAIGN_DIR / "sweep_t3_highres_custom.py"
    with open(script_path, 'w') as f:
        f.write(content)
    return str(script_path)


def run_smoke_tests():
    """Run quick smoke tests on all scripts."""
    print("\n=== Running Smoke Tests ===\n")
    
    smoke_tests = [
        ("logistics", ["python", "-c", "import sgms_anchor_logistics; print('logistics OK')"]),
        ("sweep_t1", ["python", "-c", "import sweep_latency_eta_ind; print('sweep_t1 OK')"]),
        ("sweep_t3", ["python", "-c", "import sweep_fault_cascade; print('sweep_t3 OK')"]),
        ("lob", ["python", "-c", "import lob_scaling; print('lob OK')"]),
        ("sensitivity", ["python", "-c", "import sgms_anchor_sensitivity; print('sensitivity OK')"]),
    ]
    
    results = []
    for name, cmd in smoke_tests:
        log_file = CAMPAIGN_DIR / f"smoke_{name}.log"
        name, status, duration, code = run_command(cmd, f"Smoke-{name}", log_file)
        results.append((name, status))
    
    print("\n=== Smoke Test Results ===")
    for name, status in results:
        print(f"  {name}: {status}")
    
    all_passed = all(status == "SUCCESS" for _, status in results)
    if not all_passed:
        print("\n⚠ Some smoke tests failed. Proceeding with caution.")
    else:
        print("\n✓ All smoke tests passed.")
    
    return all_passed


def run_parallel_sweeps():
    """Run all sweeps in parallel where possible."""
    print(f"\n=== Starting Comprehensive Sweep Campaign ===")
    print(f"Results directory: {CAMPAIGN_DIR}")
    print(f"Timestamp: {TIMESTAMP}")
    
    # Create custom high-res scripts
    t1_highres_script = create_high_res_t1_script()
    t3_highres_script = create_high_res_t3_script()
    
    # Build command list
    commands = []
    
    # Default sweeps (can run in parallel)
    commands.append(("T1-Default", ["python", "sweep_latency_eta_ind.py"]))
    commands.append(("T3-Default", ["python", "sweep_fault_cascade.py"]))
    commands.append(("LOB-Scaling", ["python", "lob_scaling.py"]))
    commands.append(("Sensitivity", ["python", "sgms_anchor_sensitivity.py"]))
    
    # High-res sweeps (can run in parallel with defaults)
    commands.append(("T1-HighRes", ["python", t1_highres_script]))
    commands.append(("T3-HighRes", ["python", t3_highres_script]))
    
    # Other sweeps
    if Path("validate_stream_balance.py").exists():
        commands.append(("Stream-Balance", ["python", "validate_stream_balance.py"]))
    
    if Path("scenarios/mission_scenarios.py").exists():
        commands.append(("Mission-Scenarios", ["python", "scenarios/mission_scenarios.py"]))
    
    # Run in parallel (limit to 4 concurrent to avoid overwhelming system)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for name, cmd in commands:
            log_file = CAMPAIGN_DIR / f"{name.lower().replace('-', '_')}.log"
            future = executor.submit(run_command, cmd, name, log_file)
            futures[future] = name
        
        for future in concurrent.futures.as_completed(futures):
            name, status, duration, code = future.result()
            results.append((name, status, duration, code))
    
    return results


def generate_summary_report(results):
    """Generate comprehensive summary report."""
    report_file = CAMPAIGN_DIR / "summary_report.md"
    
    with open(report_file, 'w') as f:
        f.write(f"# Comprehensive Sweep Campaign Report\n\n")
        f.write(f"**Timestamp**: {TIMESTAMP}\n")
        f.write(f"**Results Directory**: {CAMPAIGN_DIR}\n\n")
        
        f.write("## Campaign Configuration\n\n")
        f.write(f"- **Profiles**: {', '.join(PROFILES)}\n")
        f.write(f"- **Total Sweeps**: {len(results)}\n\n")
        
        f.write("## Sweep Results\n\n")
        f.write("| Sweep | Status | Duration (s) | Return Code |\n")
        f.write("|-------|--------|-------------|-------------|\n")
        
        total_duration = 0
        success_count = 0
        for name, status, duration, code in results:
            f.write(f"| {name} | {status} | {duration:.2f} | {code} |\n")
            total_duration += duration
            if status == "SUCCESS":
                success_count += 1
        
        f.write(f"\n**Total Duration**: {total_duration:.2f}s ({total_duration/60:.2f} minutes)\n")
        f.write(f"**Success Rate**: {success_count}/{len(results)} ({100*success_count/len(results):.1f}%)\n\n")
        
        f.write("## Output Files\n\n")
        f.write("The following files were generated:\n\n")
        for name, status, _, _ in results:
            if status == "SUCCESS":
                f.write(f"- `{name.lower().replace('-', '_')}.log` - Execution log\n")
        
        f.write("\n## Next Steps\n\n")
        f.write("1. Review individual sweep logs for detailed results\n")
        f.write("2. Examine generated plots (PNG files)\n")
        f.write("3. Compare default vs high-resolution results\n")
        f.write("4. Aggregate raw data for analysis\n")
    
    print(f"\n✓ Summary report generated: {report_file}")
    return report_file


def main():
    """Main execution."""
    print("=" * 70)
    print("COMPREHENSIVE SWEEP CAMPAIGN")
    print("=" * 70)
    
    # Step 1: Smoke tests
    smoke_passed = run_smoke_tests()
    
    # Step 2: Run all sweeps in parallel
    print("\n" + "=" * 70)
    results = run_parallel_sweeps()
    
    # Step 3: Generate summary
    print("\n" + "=" * 70)
    generate_summary_report(results)
    
    print("\n" + "=" * 70)
    print("CAMPAIGN COMPLETE")
    print("=" * 70)
    print(f"\nAll results saved to: {CAMPAIGN_DIR}")
    print(f"Summary report: {CAMPAIGN_DIR / 'summary_report.md'}")


if __name__ == "__main__":
    main()
