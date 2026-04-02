"""
Convenience entrypoint for the config-driven anchor pipeline.
"""

from sgms_anchor_pipeline import run_experiment_suite


def main() -> None:
    manifest = run_experiment_suite("anchor_experiments.json", output_root="artifacts")
    print(f"Run label: {manifest['run_label']}")
    print(f"Experiments: {len(manifest['experiments'])}")


if __name__ == "__main__":
    main()
