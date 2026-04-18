import argparse
import shutil
from pathlib import Path
from sgms_anchor_pipeline import run_experiment_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Aethelgard Experiment Suite")
    parser.add_argument("--repro", action="store_true", help="Clean output directory before running.")
    parser.add_argument("--output", type=str, default="artifacts", help="Root directory for artifacts.")
    args = parser.parse_args()

    output_root = Path(args.output)
    if args.repro and output_root.exists():
        print(f"Repro mode: cleaning {output_root}...")
        shutil.rmtree(output_root)

    manifest = run_experiment_suite("anchor_experiments.json", output_root=output_root)
    print(f"Run label: {manifest['run_label']}")
    print(f"Experiments: {len(manifest['experiments'])}")
    print(f"Report: {manifest['report_path']}")


if __name__ == "__main__":
    main()
