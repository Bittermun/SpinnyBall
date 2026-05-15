"""
Monte Carlo Uncertainty Quantification for Cislunar Swarms

Ensemble-based robustness analysis for shepherd control systems.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import multiprocessing as mp
from datetime import datetime

# Monte Carlo Configuration


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo uncertainty quantification."""
    
    # Ensemble
    n_samples: int = 100
    """Number of MC samples."""
    
    n_jobs: int = -1
    """Number of parallel jobs (-1 = all CPUs)."""
    
    # Uncertainty models
    sigma_ics: float = 0.001
    """Initial condition uncertainty (km)."""
    
    sigma_mascon: float = 0.05
    """Mascon coefficient uncertainty (%)."""
    
    sigma_halbach: float = 0.10
    """Halbach moment uncertainty (%)."""
    
    sigma_control: float = 0.02
    """Control gain uncertainty (%)."""
    
    # Analysis
    coherence_threshold_m: float = 50.0
    """Coherence loss threshold (meters)."""
    
    collision_threshold_m: float = 1.0
    """Collision threshold (meters)."""
    
    # Output
    save_trajectories: bool = False
    """Save full trajectories (disk-intensive)."""
    
    verbose: bool = True
    """Verbose output."""


class UncertaintyModel:
    """Sampling of uncertain parameters."""
    
    def __init__(self, config: MonteCarloConfig):
        self.config = config
    
    def sample_initial_conditions(self, nominal_state: np.ndarray) -> np.ndarray:
        """Sample perturbed initial conditions."""
        perturbed = nominal_state.copy()
        
        # Position uncertainty (km)
        perturbed[0:3] += np.random.normal(0, self.config.sigma_ics, 3)
        
        # Velocity uncertainty (km/s)
        perturbed[3:6] += np.random.normal(0, self.config.sigma_ics/1000, 3)
        
        return perturbed
    
    def sample_mascon_coefficients(self, degree_max: int) -> Dict[Tuple[int, int], Tuple[float, float]]:
        """Sample uncertain mascon coefficients."""
        coeffs = {}
        
        for degree in range(degree_max + 1):
            for order in range(degree + 1):
                # Add random perturbations (Gaussian)
                factor = 1.0 + np.random.normal(0, self.config.sigma_mascon / 100)
                coeffs[(degree, order)] = (factor, factor)
        
        return coeffs
    
    def sample_halbach_moment(self, nominal_moment: float) -> float:
        """Sample uncertain Halbach magnetic moment."""
        return nominal_moment * (1.0 + np.random.normal(0, self.config.sigma_halbach / 100))
    
    def sample_control_gains(self, nominal_kp: float, nominal_ki: float, 
                             nominal_kd: float) -> Tuple[float, float, float]:
        """Sample uncertain control gains."""
        sigma = self.config.sigma_control / 100
        
        kp = nominal_kp * (1.0 + np.random.normal(0, sigma))
        ki = nominal_ki * (1.0 + np.random.normal(0, sigma))
        kd = nominal_kd * (1.0 + np.random.normal(0, sigma))
        
        return kp, ki, kd


class MonteCarloAnalyzer:
    """Monte Carlo ensemble analysis."""
    
    def __init__(self, config: MonteCarloConfig):
        self.config = config
        self.results = []
    
    def run_sample(self, sample_id: int, propagate_func, 
                   uncertainty_model: UncertaintyModel, 
                   nominal_params: Dict) -> Dict:
        """Run single MC sample."""
        try:
            # Sample uncertainties
            perturbed_ics = uncertainty_model.sample_initial_conditions(
                nominal_params['positions'],
                nominal_params['velocities']
            )
            
            # Propagate with perturbed parameters
            sol, diag = propagate_func(perturbed_ics, nominal_params['t_eval'])
            
            # Compute metrics
            metrics = self._compute_metrics(sol, diag, nominal_params)
            
            return {
                'sample_id': sample_id,
                'success': True,
                'metrics': metrics,
                'diagnostics': diag
            }
        
        except Exception as e:
            return {
                'sample_id': sample_id,
                'success': False,
                'error': str(e)
            }
    
    def _compute_metrics(self, sol: Dict, diag: Dict, params: Dict) -> Dict:
        """Compute performance metrics from trajectory."""
        
        spacings = sol['spacings_m']
        times = sol['time']
        
        # Coherence: time when std exceeds threshold
        spacing_std = np.std(spacings, axis=1)
        coherence_idx = np.where(spacing_std > self.config.coherence_threshold_m)[0]
        
        if len(coherence_idx) > 0:
            coherence_lifetime_s = times[coherence_idx[0]]
        else:
            coherence_lifetime_s = times[-1]
        
        # Collision check
        min_spacing = np.min(spacings)
        collision_probability = 1.0 if min_spacing < self.config.collision_threshold_m else 0.0
        
        # Δv budget: integrate control accelerations
        control_effort_ms = np.sum(np.abs(diag.get('control_history', []))) * 0.1
        
        metrics = {
            'coherence_lifetime_s': coherence_lifetime_s,
            'coherence_lifetime_orbits': coherence_lifetime_s / params.get('T_orbit', 10000),
            'collision_probability': collision_probability,
            'min_spacing_m': min_spacing,
            'mean_spacing_m': np.mean(spacings),
            'spacing_std_m': np.std(spacings),
            'control_effort_ms': control_effort_ms,
            'final_spacing_m': np.mean(spacings[-1, :]) if len(spacings) > 0 else 0.0
        }
        
        return metrics
    
    def run_ensemble(self, propagate_func, uncertainty_model: UncertaintyModel,
                     nominal_params: Dict) -> Tuple[List[Dict], Dict]:
        """Run full MC ensemble."""
        
        print(f"Running {self.config.n_samples} MC samples...")
        
        # Single-threaded for now (can parallelize later)
        results = []
        
        for sample_id in range(self.config.n_samples):
            if self.config.verbose and (sample_id + 1) % 10 == 0:
                print(f"  Sample {sample_id + 1}/{self.config.n_samples}...")
            
            result = self.run_sample(sample_id, propagate_func, uncertainty_model, nominal_params)
            results.append(result)
        
        # Aggregate statistics
        stats = self._aggregate_statistics(results)
        
        return results, stats
    
    def _aggregate_statistics(self, results: List[Dict]) -> Dict:
        """Aggregate MC results."""
        
        successful = [r for r in results if r['success']]
        n_success = len(successful)
        n_fail = len(results) - n_success
        
        if n_success == 0:
            return {'n_samples': len(results), 'n_success': 0, 'n_fail': n_fail}
        
        metrics_list = [r['metrics'] for r in successful]
        
        # Compute statistics for each metric
        stats = {
            'n_samples': len(results),
            'n_success': n_success,
            'n_fail': n_fail,
            'success_rate': 100.0 * n_success / len(results),
        }
        
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list]
            
            stats[f'{key}_mean'] = np.mean(values)
            stats[f'{key}_median'] = np.median(values)
            stats[f'{key}_std'] = np.std(values)
            stats[f'{key}_min'] = np.min(values)
            stats[f'{key}_max'] = np.max(values)
            stats[f'{key}_p05'] = np.percentile(values, 5)
            stats[f'{key}_p95'] = np.percentile(values, 95)
        
        return stats


class MonteCarloReporter:
    """Format and report MC results."""
    
    @staticmethod
    def print_summary(stats: Dict):
        """Print summary statistics."""
        print("\n" + "=" * 80)
        print("MONTE CARLO ANALYSIS SUMMARY")
        print("=" * 80)
        
        print(f"\nEnsemble Statistics:")
        print(f"  Total samples: {stats['n_samples']}")
        print(f"  Successful: {stats['n_success']}/{stats['n_samples']}")
        print(f"  Success rate: {stats.get('success_rate', 0):.1f}%")
        
        print(f"\nCoherence Lifetime:")
        print(f"  Mean: {stats.get('coherence_lifetime_s_mean', 0):.1f} s")
        print(f"  Median: {stats.get('coherence_lifetime_s_median', 0):.1f} s")
        print(f"  95% CI: [{stats.get('coherence_lifetime_s_p05', 0):.1f}, {stats.get('coherence_lifetime_s_p95', 0):.1f}] s")
        
        print(f"\nMinimum Spacing:")
        print(f"  Mean: {stats.get('min_spacing_m_mean', 0):.2f} m")
        print(f"  Min: {stats.get('min_spacing_m_min', 0):.2f} m")
        print(f"  P05: {stats.get('min_spacing_m_p05', 0):.2f} m")
        
        collision_prob = np.mean([stats.get('collision_probability_mean', 0)])
        print(f"\nCollision Probability:")
        print(f"  P(collision): {collision_prob:.4f}")
        
        print(f"\nControl Effort:")
        print(f"  Mean: {stats.get('control_effort_ms_mean', 0):.2f} m/s")
        print(f"  95% CI: [{stats.get('control_effort_ms_p05', 0):.2f}, {stats.get('control_effort_ms_p95', 0):.2f}] m/s")
    
    @staticmethod
    def save_report(stats: Dict, output_path: Path):
        """Save detailed report."""
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MONTE CARLO UNCERTAINTY QUANTIFICATION REPORT\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total samples: {stats['n_samples']}\n")
            f.write(f"Successful: {stats['n_success']}/{stats['n_samples']}\n")
            f.write(f"Success rate: {stats.get('success_rate', 0):.1f}%\n\n")
            
            f.write("COHERENCE LIFETIME (seconds)\n")
            f.write("-" * 80 + "\n")
            for metric in ['coherence_lifetime_s']:
                if f'{metric}_mean' in stats:
                    f.write(f"{metric}:\n")
                    f.write(f"  Mean: {stats[f'{metric}_mean']:.1f}\n")
                    f.write(f"  Median: {stats[f'{metric}_median']:.1f}\n")
                    f.write(f"  Std: {stats[f'{metric}_std']:.1f}\n")
                    f.write(f"  Range: [{stats[f'{metric}_min']:.1f}, {stats[f'{metric}_max']:.1f}]\n")
                    f.write(f"  95% CI: [{stats[f'{metric}_p05']:.1f}, {stats[f'{metric}_p95']:.1f}]\n\n")
            
            f.write("MINIMUM SPACING (meters)\n")
            f.write("-" * 80 + "\n")
            for metric in ['min_spacing_m']:
                if f'{metric}_mean' in stats:
                    f.write(f"{metric}:\n")
                    f.write(f"  Mean: {stats[f'{metric}_mean']:.2f}\n")
                    f.write(f"  Std: {stats[f'{metric}_std']:.2f}\n")
                    f.write(f"  Min: {stats[f'{metric}_min']:.2f}\n")
                    f.write(f"  95% CI: [{stats[f'{metric}_p05']:.2f}, {stats[f'{metric}_p95']:.2f}]\n\n")


# Export
__all__ = [
    'MonteCarloConfig',
    'UncertaintyModel',
    'MonteCarloAnalyzer',
    'MonteCarloReporter',
]
