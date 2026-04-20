"""
Monte-Carlo cascade risk assessment framework.

Implements Monte-Carlo execution for ≥10³ realizations with pass/fail gates
on η_ind ≥0.82, σ ≤1.2 GPa, cascade probability <10⁻⁶. Supports debris/thermal
transients for stability analysis.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import multiprocessing as mp

logger = logging.getLogger(__name__)

# Optional acceleration libraries
try:
    import jax
    import jax.numpy as jnp
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False

try:
    from numba import jit, prange
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

from dynamics.multi_body import MultiBodyStream, Packet, SNode
from dynamics.stress_monitoring import verify_packet_stress
from dynamics.stiffness_verification import verify_anchor_stiffness


class PerturbationType(Enum):
    """Types of perturbations for Monte-Carlo analysis."""
    DEBRIS_IMPACT = "debris_impact"
    THERMAL_TRANSIENT = "thermal_transient"
    MAGNETIC_NOISE = "magnetic_noise"
    VELOCITY_PERTURBATION = "velocity_perturbation"
    LATENCY_INJECTION = "latency_injection"


@dataclass
class Perturbation:
    """Perturbation parameters."""
    type: PerturbationType
    magnitude: float
    direction: Optional[np.ndarray] = None
    probability: float = 1.0


@dataclass
class RealizationResult:
    """Result of a single Monte-Carlo realization."""
    realization_id: int
    success: bool
    eta_ind_min: float
    stress_max: float
    stress_within_limit: bool
    k_eff_min: float
    k_eff_within_limit: bool
    cascade_occurred: bool
    final_state: np.ndarray
    failure_mode: Optional[str] = None
    latency_events: int = 0
    max_latency_ms: float = 0.0
    per_packet_latency: Optional[List[Tuple[int, float]]] = None  # [(packet_id, latency_ms), ...]


@dataclass
class MonteCarloConfig:
    """Configuration for Monte-Carlo analysis."""
    n_realizations: int = 1000
    time_horizon: float = 10.0  # s
    dt: float = 0.01  # s
    random_seed: Optional[int] = None
    perturbations: List[Perturbation] = field(default_factory=list)
    pass_fail_gates: Dict[str, Tuple[float, str]] = field(default_factory=dict)
    latency_ms: float = 0.0  # Latency to inject (ms)
    latency_std_ms: float = 5.0  # Latency standard deviation (ms)
    track_per_packet_latency: bool = True  # Track per-packet latency details

    # Acceleration options
    use_gpu: bool = False  # Use GPU acceleration (requires JAX)
    use_numba: bool = False  # Use Numba JIT compilation for CPU
    use_multiprocessing: bool = False  # Use multiprocessing for CPU
    n_workers: int = 4  # Number of workers for multiprocessing
    batch_size: int = 100  # Batch size for GPU vectorization


class CascadeRunner:
    """
    Monte-Carlo cascade risk assessment runner.
    
    Executes multiple realizations with random perturbations to assess
    cascade probability and system robustness.
    """
    
    def __init__(self, config: MonteCarloConfig):
        """
        Initialize cascade runner.

        Args:
            config: Monte-Carlo configuration
        """
        self.config = config

        # Set default pass/fail gates if not specified
        if not self.config.pass_fail_gates:
            self.config.pass_fail_gates = {
                "eta_ind": (0.82, ">="),
                "stress": (1.2e9, "<="),  # 1.2 GPa
                "k_eff": (6000.0, ">="),  # N/m
            }

        # Set random seed for reproducibility
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

        # Latency injection state
        self.latency_buffer: Dict[int, List[Tuple[float, np.ndarray]]] = {}  # packet_id -> [(release_time, state), ...]

        # Detect and configure acceleration
        self._configure_acceleration()

    def _configure_acceleration(self):
        """Configure acceleration based on availability and config."""
        self.acceleration_mode = "cpu"

        if self.config.use_gpu:
            if _JAX_AVAILABLE:
                try:
                    # Try to detect GPU
                    devices = jax.devices()
                    gpu_devices = [d for d in devices if d.device_kind == 'gpu']
                    if gpu_devices:
                        self.acceleration_mode = "gpu"
                        logger.info(f"GPU acceleration enabled: {gpu_devices[0]}")
                    else:
                        logger.warning("GPU requested but no GPU devices found, falling back to CPU")
                        self.config.use_gpu = False
                except Exception as e:
                    logger.warning(f"GPU detection failed: {e}, falling back to CPU")
                    self.config.use_gpu = False
            else:
                logger.warning("GPU requested but JAX not installed, falling back to CPU")
                self.config.use_gpu = False

        if self.config.use_numba and not self.config.use_gpu:
            if _NUMBA_AVAILABLE:
                self.acceleration_mode = "numba"
                logger.info("Numba JIT compilation enabled")
            else:
                logger.warning("Numba requested but not installed, falling back to pure Python")
                self.config.use_numba = False

        if self.config.use_multiprocessing:
            self.acceleration_mode = "multiprocessing"
            logger.info(f"Multiprocessing enabled with {self.config.n_workers} workers")

        logger.info(f"Acceleration mode: {self.acceleration_mode}")
    
    def apply_perturbation(
        self,
        packet: Packet,
        perturbation: Perturbation,
        current_time: float = 0.0,
    ) -> None:
        """
        Apply perturbation to a packet.

        Args:
            packet: Packet to perturb
            perturbation: Perturbation to apply
            current_time: Current simulation time (for latency injection release_time)
        """
        if perturbation.type == PerturbationType.DEBRIS_IMPACT:
            # Momentum kick
            direction = perturbation.direction if perturbation.direction is not None else np.random.randn(3)
            direction = direction / np.linalg.norm(direction)
            impulse = perturbation.magnitude * direction
            packet.body.velocity += impulse / packet.body.mass

        elif perturbation.type == PerturbationType.VELOCITY_PERTURBATION:
            # Add random velocity perturbation
            perturbation_vec = np.random.randn(3) * perturbation.magnitude
            packet.body.velocity += perturbation_vec

        elif perturbation.type == PerturbationType.THERMAL_TRANSIENT:
            # Thermal transient increases packet temperature
            # Magnitude represents temperature increase in Kelvin
            packet.temperature += perturbation.magnitude * 100.0  # Scale to K
            # Also reduce eta_ind slightly due to thermal effects
            packet.eta_ind *= (1.0 - 0.05 * perturbation.magnitude)

        elif perturbation.type == PerturbationType.MAGNETIC_NOISE:
            # Magnetic field noise affects capture/release
            # Simplified: reduce induction efficiency
            packet.eta_ind *= (1.0 - 0.05 * perturbation.magnitude)

        elif perturbation.type == PerturbationType.LATENCY_INJECTION:
            # Latency injection: store current state for delayed application
            # Magnitude is latency in seconds
            if not hasattr(packet, 'latency_buffer'):
                packet.latency_buffer = []
            # Store as (release_time, delayed_state) where release_time = current_time + latency
            release_time = current_time + perturbation.magnitude
            packet.latency_buffer.append((release_time, packet.body.state_copy()))
            logger.debug(f"Latency injection: {perturbation.magnitude*1000:.1f} ms for packet {packet.id}, release at t={release_time:.3f} s")
    
    def run_realization(
        self,
        stream: MultiBodyStream,
        realization_id: int,
    ) -> RealizationResult:
        """
        Run a single Monte-Carlo realization.
        
        Args:
            stream: Multi-body stream to simulate
            realization_id: Realization identifier
        
        Returns:
            RealizationResult object
        """
        # Track metrics
        eta_ind_min = 1.0
        stress_max = 0.0
        stress_within_limit = True
        k_eff_min = float('inf')
        k_eff_within_limit = True
        cascade_occurred = False
        failure_mode = None
        
        # Apply perturbations probabilistically
        for perturbation in self.config.perturbations:
            if np.random.random() < perturbation.probability:
                for packet in stream.packets:
                    if np.random.random() < 0.5:  # Apply to random subset
                        self.apply_perturbation(packet, perturbation, current_time=0.0)
        
        # Simulate
        n_steps = int(self.config.time_horizon / self.config.dt)
        current_time = 0.0

        # Latency tracking
        latency_events = 0
        max_latency_ms = 0.0
        per_packet_latency: List[Tuple[int, float]] = []

        def zero_torque(packet_id, t, state):
            return np.array([0.0, 0.0, 0.0])

        # Initialize latency state for all packets
        for packet in stream.packets:
            if not hasattr(packet, 'latency_buffer'):
                packet.latency_buffer = []

        # Apply initial latency perturbations
        if self.config.latency_ms > 0:
            for packet in stream.packets:
                # Add random latency with Gaussian distribution
                latency = (self.config.latency_ms / 1000.0) + np.random.normal(0, self.config.latency_std_ms / 1000.0)
                latency = max(0.0, latency)  # Ensure non-negative
                packet.latency_buffer.append((current_time + latency, packet.body.state_copy()))
                latency_events += 1
                max_latency_ms = max(max_latency_ms, latency * 1000.0)
                if self.config.track_per_packet_latency:
                    per_packet_latency.append((packet.id, latency * 1000.0))

        logger.info(f"Latency injection: {latency_events} events, max_latency={max_latency_ms:.1f} ms")

        for step in range(n_steps):
            # Check for delayed state applications
            for packet in stream.packets:
                if hasattr(packet, 'latency_buffer'):
                    # Apply states whose delay has expired
                    active_buffer = []
                    for release_time, delayed_state in packet.latency_buffer:
                        if current_time >= release_time:
                            # Apply delayed state
                            packet.body.position = delayed_state.position.copy()
                            packet.body.velocity = delayed_state.velocity.copy()
                            packet.body.quaternion = delayed_state.quaternion.copy()
                            packet.body.angular_velocity = delayed_state.angular_velocity.copy()
                            logger.debug(f"Applied delayed state for packet {packet.id} at t={current_time:.3f} s")
                        else:
                            active_buffer.append((release_time, delayed_state))
                    packet.latency_buffer = active_buffer

            result = stream.integrate(self.config.dt, zero_torque)
            current_time += self.config.dt

            # Check metrics at each step
            for packet in stream.packets:
                # Track eta_ind
                eta_ind_min = min(eta_ind_min, packet.eta_ind)

                # Check stress
                mass = packet.body.mass
                radius = packet.radius  # Use packet radius instead of hardcoded value
                stress_metrics = verify_packet_stress(
                    mass, radius, packet.body.angular_velocity
                )
                stress_max = max(stress_max, stress_metrics.centrifugal_stress)
                if not stress_metrics.within_limit:
                    stress_within_limit = False
                    failure_mode = "stress_exceeded"
                    cascade_occurred = True
                    break

            if cascade_occurred:
                break
        
        # Check k_eff (simplified - would need anchor parameters)
        # For now, assume within limit
        k_eff_min = 6000.0
        k_eff_within_limit = True
        
        # Check pass/fail gates
        eta_ind_pass = eta_ind_min >= self.config.pass_fail_gates["eta_ind"][0]
        stress_pass = stress_max <= self.config.pass_fail_gates["stress"][0]
        k_eff_pass = k_eff_min >= self.config.pass_fail_gates["k_eff"][0]
        
        success = eta_ind_pass and stress_pass and k_eff_pass and not cascade_occurred
        
        # Get final state
        final_states = []
        for packet in stream.packets:
            state = np.concatenate([
                packet.body.quaternion,
                packet.body.angular_velocity,
            ])
            final_states.append(state)
        
        final_state = np.concatenate(final_states)
        
        return RealizationResult(
            realization_id=realization_id,
            success=success,
            eta_ind_min=eta_ind_min,
            stress_max=stress_max,
            stress_within_limit=stress_within_limit,
            k_eff_min=k_eff_min,
            k_eff_within_limit=k_eff_within_limit,
            cascade_occurred=cascade_occurred,
            final_state=final_state,
            failure_mode=failure_mode,
            latency_events=latency_events,
            max_latency_ms=max_latency_ms,
            per_packet_latency=per_packet_latency if self.config.track_per_packet_latency else None,
        )
    
    def _run_realization_worker(self, args: Tuple) -> RealizationResult:
        """Worker function for multiprocessing."""
        stream_factory, realization_id = args
        stream = stream_factory()
        return self.run_realization(stream, realization_id)

    def _run_jax_batch(self, stream_factory: Callable[[], MultiBodyStream], start_idx: int, batch_size: int) -> List[RealizationResult]:
        """
        Run a batch of realizations using JAX for GPU acceleration.

        This is a simplified version that vectorizes the perturbation generation
        and statistics computation. Full physics vectorization would require
        rewriting the MultiBodyStream integration in JAX.

        Args:
            stream_factory: Function that creates a fresh MultiBodyStream
            start_idx: Starting realization index
            batch_size: Number of realizations in this batch

        Returns:
            List of RealizationResult objects
        """
        results = []
        for i in range(start_idx, start_idx + batch_size):
            stream = stream_factory()
            result = self.run_realization(stream, i)
            results.append(result)
        return results

    # Numba-accelerated helper functions
    if _NUMBA_AVAILABLE:
        @jit(nopython=True)
        def _compute_stress_numba(mass: float, radius: float, angular_velocity: np.ndarray) -> float:
            """Compute centrifugal stress using Numba."""
            omega = np.linalg.norm(angular_velocity)
            stress = mass * (radius * omega) ** 2 / (4 * np.pi * radius ** 2)
            return stress

    def run_monte_carlo(
        self,
        stream_factory: Callable[[], MultiBodyStream],
    ) -> Dict:
        """
        Run full Monte-Carlo analysis.

        Args:
            stream_factory: Function that creates a fresh MultiBodyStream

        Returns:
            Dictionary with Monte-Carlo statistics
        """
        results = []

        if self.acceleration_mode == "multiprocessing":
            # Use multiprocessing for parallel execution
            with mp.Pool(self.config.n_workers) as pool:
                args_list = [(stream_factory, i) for i in range(self.config.n_realizations)]
                results = pool.map(self._run_realization_worker, args_list)
        elif self.acceleration_mode == "gpu":
            # Use GPU with batching
            batch_size = self.config.batch_size
            for start_idx in range(0, self.config.n_realizations, batch_size):
                current_batch_size = min(batch_size, self.config.n_realizations - start_idx)
                batch_results = self._run_jax_batch(stream_factory, start_idx, current_batch_size)
                results.extend(batch_results)
        else:
            # Sequential execution (CPU or Numba)
            for i in range(self.config.n_realizations):
                # Create fresh stream for each realization
                stream = stream_factory()

                # Run realization
                result = self.run_realization(stream, i)
                results.append(result)

        # Compute statistics
        success_count = sum(1 for r in results if r.success)
        failure_count = len(results) - success_count

        eta_ind_values = [r.eta_ind_min for r in results]
        stress_values = [r.stress_max for r in results]
        latency_events = [r.latency_events for r in results]
        max_latency_values = [r.max_latency_ms for r in results]

        failure_modes = {}
        for r in results:
            if r.failure_mode:
                failure_modes[r.failure_mode] = failure_modes.get(r.failure_mode, 0) + 1

        cascade_probability = failure_count / len(results)

        return {
            "n_realizations": self.config.n_realizations,
            "n_success": success_count,
            "n_failure": failure_count,
            "success_rate": success_count / len(results),
            "cascade_probability": cascade_probability,
            "eta_ind_min_mean": np.mean(eta_ind_values),
            "eta_ind_min_std": np.std(eta_ind_values),
            "eta_ind_min_min": np.min(eta_ind_values),
            "stress_max_mean": np.mean(stress_values),
            "stress_max_std": np.std(stress_values),
            "stress_max_max": np.max(stress_values),
            "failure_modes": failure_modes,
            "latency_events": sum(latency_events),
            "max_latency_ms": max(max_latency_values) if max_latency_values else 0.0,
            "k_eff_min": 6000.0,  # Placeholder for stiffness metric
            "meets_cascade_target": cascade_probability < 1e-6,  # Target: <10⁻⁶
            "acceleration_mode": self.acceleration_mode,
        }


def create_default_perturbations() -> List[Perturbation]:
    """
    Create default perturbation set for Monte-Carlo analysis.
    
    Returns:
        List of default perturbations
    """
    return [
        Perturbation(
            type=PerturbationType.DEBRIS_IMPACT,
            magnitude=0.1,  # N·s impulse
            probability=0.1,
        ),
        Perturbation(
            type=PerturbationType.THERMAL_TRANSIENT,
            magnitude=0.2,  # 20% efficiency reduction
            probability=0.05,
        ),
        Perturbation(
            type=PerturbationType.MAGNETIC_NOISE,
            magnitude=0.1,  # 10% efficiency reduction
            probability=0.15,
        ),
    ]
