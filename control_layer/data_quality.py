"""
Data quality checks for synthetic failure datasets.

Implements statistical validation and quality checks for generated
synthetic failure data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import h5py
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QualityCheckResult:
    """Result of a quality check."""
    check_name: str
    passed: bool
    value: float
    threshold: float
    message: str


class DataQualityChecker:
    """
    Perform quality checks on synthetic failure datasets.

    Validates data distribution, label consistency, and statistical
    properties to ensure dataset quality.
    """

    def __init__(self, filepath: str):
        """
        Initialize data quality checker.

        Args:
            filepath: Path to HDF5 dataset file
        """
        self.filepath = filepath
        self.dataset = None
        self.load_dataset()

    def load_dataset(self):
        """Load dataset from HDF5 file."""
        logger.info(f"Loading dataset from {self.filepath}")
        try:
            with h5py.File(self.filepath, "r") as f:
                self.trajectories = f["trajectories"][:]
                self.labels = f["labels"][:]
                self.n_samples = f.attrs["n_samples"]
                self.time_horizon = f.attrs["time_horizon"]
                self.dt = f.attrs["dt"]
                self.n_packets = f.attrs["n_packets"]
            logger.info(f"Dataset loaded: {self.n_samples} samples")
        except FileNotFoundError:
            logger.error(f"Dataset file not found: {self.filepath}")
            raise
        except KeyError as e:
            logger.error(f"Missing dataset or attribute in HDF5 file: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def check_label_distribution(self, min_failure_rate: float = 0.05) -> QualityCheckResult:
        """
        Check that failure labels have reasonable distribution.

        Args:
            min_failure_rate: Minimum expected failure rate

        Returns:
            QualityCheckResult
        """
        total_timesteps = self.labels.size
        failure_timesteps = np.sum(self.labels)
        failure_rate = failure_timesteps / total_timesteps

        passed = failure_rate >= min_failure_rate
        message = (
            f"Failure rate: {failure_rate:.3f} "
            f"({'PASS' if passed else 'FAIL'} - min: {min_failure_rate:.3f})"
        )

        return QualityCheckResult(
            check_name="label_distribution",
            passed=passed,
            value=failure_rate,
            threshold=min_failure_rate,
            message=message,
        )

    def check_trajectory_continuity(self, max_jump: float = 10.0) -> QualityCheckResult:
        """
        Check that trajectories are continuous (no NaN or large jumps).

        Args:
            max_jump: Maximum allowed jump between consecutive timesteps

        Returns:
            QualityCheckResult
        """
        # Check for NaN
        has_nan = np.any(np.isnan(self.trajectories))

        # Check for large jumps across timesteps
        diffs = np.diff(self.trajectories, axis=0)
        max_diff = np.max(np.abs(diffs))
        has_large_jumps = max_diff > max_jump

        passed = not has_nan and not has_large_jumps
        message = (
            f"Continuity check: "
            f"NaN: {has_nan}, "
            f"Max jump: {max_diff:.3f} "
            f"({'PASS' if passed else 'FAIL'})"
        )

        return QualityCheckResult(
            check_name="trajectory_continuity",
            passed=passed,
            value=float(max_diff),
            threshold=max_jump,
            message=message,
        )

    def check_state_normalization(self, max_quaternion_norm: float = 1.1) -> QualityCheckResult:
        """
        Check that quaternions are properly normalized.

        Args:
            max_quaternion_norm: Maximum allowed quaternion norm

        Returns:
            QualityCheckResult
        """
        # Extract quaternions (first 4 dimensions)
        quaternions = self.trajectories[:, :, :, :4]
        quaternion_norms = np.linalg.norm(quaternions, axis=3)
        max_norm = np.max(quaternion_norms)

        passed = max_norm <= max_quaternion_norm
        message = (
            f"Quaternion normalization: max norm {max_norm:.3f} "
            f"({'PASS' if passed else 'FAIL'} - max: {max_quaternion_norm:.3f})"
        )

        return QualityCheckResult(
            check_name="state_normalization",
            passed=passed,
            value=float(max_norm),
            threshold=max_quaternion_norm,
            message=message,
        )

    def check_sample_balance(self, min_samples_per_failure_type: int = 50) -> QualityCheckResult:
        """
        Check that failure types are reasonably balanced.

        Args:
            min_samples_per_failure_type: Minimum samples per failure type

        Returns:
            QualityCheckResult
        """
        # Count failure occurrences per packet
        failure_counts = np.sum(self.labels, axis=(0, 1))
        min_count = np.min(failure_counts)

        passed = min_count >= min_samples_per_failure_type
        message = (
            f"Sample balance: min {min_count:.0f} failure timesteps per packet "
            f"({'PASS' if passed else 'FAIL'} - min: {min_samples_per_failure_type})"
        )

        return QualityCheckResult(
            check_name="sample_balance",
            passed=passed,
            value=float(min_count),
            threshold=float(min_samples_per_failure_type),
            message=message,
        )

    def check_data_range(self, max_velocity: float = 5000.0, max_angular_velocity: float = 1000.0) -> QualityCheckResult:
        """
        Check that state values are within reasonable physical ranges.

        Args:
            max_velocity: Maximum allowed velocity
            max_angular_velocity: Maximum allowed angular velocity

        Returns:
            QualityCheckResult
        """
        # Extract velocity and angular velocity
        # Note: trajectory format is [quaternion(4), angular_velocity(3)]
        angular_velocities = self.trajectories[:, :, :, 4:7]
        max_omega = np.max(np.abs(angular_velocities))

        passed = max_omega <= max_angular_velocity
        message = (
            f"Data range: max angular velocity {max_omega:.3f} rad/s "
            f"({'PASS' if passed else 'FAIL'} - max: {max_angular_velocity:.3f})"
        )

        return QualityCheckResult(
            check_name="data_range",
            passed=passed,
            value=float(max_omega),
            threshold=float(max_angular_velocity),
            message=message,
        )

    def run_all_checks(self) -> Dict[str, QualityCheckResult]:
        """
        Run all quality checks.

        Returns:
            Dictionary of check results
        """
        logger.info("Running all quality checks...")

        results = {
            "label_distribution": self.check_label_distribution(),
            "trajectory_continuity": self.check_trajectory_continuity(),
            "state_normalization": self.check_state_normalization(),
            "sample_balance": self.check_sample_balance(),
            "data_range": self.check_data_range(),
        }

        # Log results
        for check_name, result in results.items():
            logger.info(f"  {result.message}")

        # Summary
        passed_count = sum(1 for r in results.values() if r.passed)
        total_count = len(results)
        logger.info(f"Quality checks: {passed_count}/{total_count} passed")

        return results

    def generate_quality_report(self) -> str:
        """
        Generate a quality report string.

        Returns:
            Quality report string
        """
        results = self.run_all_checks()

        report_lines = [
            "=== Data Quality Report ===",
            f"Dataset: {self.filepath}",
            f"Samples: {self.n_samples}",
            f"Time horizon: {self.time_horizon}s",
            f"dt: {self.dt}s",
            f"Packets: {self.n_packets}",
            "",
            "Quality Checks:",
        ]

        for check_name, result in results.items():
            status = "✓ PASS" if result.passed else "✗ FAIL"
            report_lines.append(f"  [{status}] {result.message}")

        passed_count = sum(1 for r in results.values() if r.passed)
        total_count = len(results)
        report_lines.append(f"\nOverall: {passed_count}/{total_count} checks passed")

        return "\n".join(report_lines)


def check_dataset_quality(filepath: str) -> Dict[str, QualityCheckResult]:
    """
    Convenience function to check dataset quality.

    Args:
        filepath: Path to dataset file

    Returns:
        Dictionary of quality check results
    """
    checker = DataQualityChecker(filepath)
    return checker.run_all_checks()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    import sys

    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "control_layer/data/synthetic_failure_data.h5"

    checker = DataQualityChecker(filepath)
    report = checker.generate_quality_report()
    print(report)
