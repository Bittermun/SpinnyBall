"""
Monte-Carlo cascade risk assessment framework.

Implements Monte-Carlo execution for ≥10³ realizations with pass/fail gates
on η_ind ≥0.82, σ ≤1.2 GPa, cascade probability <10⁻⁶. Supports debris/thermal
transients for stability analysis.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import multiprocessing as mp

# Vectorized random number generation for Monte Carlo speedup
def _vectorized_normal(loc: float, scale: float, size: int) -> np.ndarray:
    """Generate vectorized normal random samples."""
    return np.random.normal(loc, scale, size)

def _vectorized_uniform(size: int, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    """Generate vectorized uniform random samples."""
    return np.random.uniform(low, high, size)

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
    NODE_FAILURE = "node_failure"


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
    nodes_affected: int = 0  # Number of nodes that experienced failure
    containment_successful: bool = True  # True if nodes_affected <= 2
    
    # NEW: Diagnostic counters - addresses Root Cause #6 and Trust Strategy #1
    fault_events_injected: int = 0  # How many faults actually fired
    thermal_violations_count: int = 0
    quench_events: int = 0
    capture_release_events: int = 0
    max_temperature_reached: float = 0.0
    cascade_generations: int = 0  # How many cascade propagation steps occurred


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

    # Fault injection parameters for T3 sweep
    fault_rate: float = 1e-4  # Failure rate per hour (units: /hr)
    cascade_threshold: float = 1.05  # Stiffness reduction factor for cascade
    containment_threshold: int = 2  # Max nodes allowed for containment success
    
    # NEW: Fault injection mode - addresses Root Cause #1
    fault_injection_mode: str = "rate"  # "rate", "guaranteed", or "poisson"
    n_guaranteed_faults: int = 0  # Inject exactly N faults per realization (for "guaranteed" mode)
    
    # NEW: Cascade propagation - addresses Root Cause #2
    enable_cascade_propagation: bool = False  # Enable neighbor load redistribution
    cascade_propagation_factor: float = 0.1  # Fraction of failed load transferred to neighbors
    max_cascade_generations: int = 5  # Maximum cascade propagation depth
    
    # NEW: Thermal/quench integration - addresses Root Cause #4
    enable_thermal_quench: bool = False  # Enable thermal-quench coupling
    quench_detection_enabled: bool = False  # Enable quench detector monitoring

    # Acceleration options
    use_gpu: bool = False  # Use GPU acceleration (requires JAX)
    use_numba: bool = False  # Use Numba JIT compilation for CPU
    use_multiprocessing: bool = False  # Use multiprocessing for CPU
    n_workers: int = 4  # Number of workers for multiprocessing
    batch_size: int = 100  # Batch size for GPU vectorization

    # Early termination options
    enable_early_termination: bool = False  # Stop early when CI converges
    ci_width_threshold: float = 0.05  # Stop when CI width < this fraction (e.g., 0.05 = 5%)
    min_realizations: int = 20  # Minimum realizations before checking convergence

    # Numba acceleration
    use_numba_rk4: bool = True  # Use Numba-compiled RK4 integrator
    use_zero_torque_numba: bool = False  # Use zero-torque Numba RK4 (fastest, no callback)


class CascadeRunner:
    """
    Monte-Carlo cascade risk assessment runner.
    
    Executes multiple realizations with random perturbations to assess
    cascade probability and system robustness.
    """
    
    def __init__(self, config: MonteCarloConfig):
        """Initialize cascade runner with configuration."""
        self.config = config

        # Detect and configure acceleration
        self._configure_acceleration()
        
        # Initialize Wilson CI method
        self._wilson_ci = self._create_wilson_ci()
    
    def _create_wilson_ci(self):
        """Create Wilson CI function."""
        def _wilson_ci(k, n, z=1.96):
            if n == 0:
                return (0.0, 1.0)
            p = k / n
            denominator = 1 + z**2 / n
            centre = (p + z**2 / (2*n)) / denominator
            half_width = z * np.sqrt((p * (1-p) + z**2 / (4*n)) / n) / denominator
            return (max(0.0, centre - half_width), min(1.0, centre + half_width))
        return _wilson_ci

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

        elif perturbation.type == PerturbationType.NODE_FAILURE:
            # Node failure: reduce stiffness to simulate quench/degradation
            # Magnitude is the node_id to fail (or -1 for random)
            # This is applied to nodes, not packets - handled in run_realization
            logger.debug(f"Node failure perturbation requested for node {perturbation.magnitude}")
    
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
        nodes_affected = set()  # Track which nodes experienced failure
        
        # NEW: Diagnostic counters - Trust Strategy #1
        fault_events_injected = 0
        thermal_violations_count = 0
        quench_events = 0
        max_temperature_reached = 0.0
        cascade_generations = 0
        
        # Initialize quench detector if enabled (Root Cause #4)
        quench_detector = None
        if self.config.quench_detection_enabled or self.config.enable_thermal_quench:
            try:
                from dynamics.quench_detector import QuenchDetector
                quench_detector = QuenchDetector()
            except ImportError:
                logger.warning("QuenchDetector not available, disabling quench detection")
                self.config.quench_detection_enabled = False
                self.config.enable_thermal_quench = False
        
        # Apply perturbations probabilistically
        for perturbation in self.config.perturbations:
            if np.random.random() < perturbation.probability:
                for packet in stream.packets:
                    if np.random.random() < 0.5:  # Apply to random subset
                        self.apply_perturbation(packet, perturbation, current_time=0.0)
        
        # Simulate
        n_steps = int(self.config.time_horizon / self.config.dt)
        current_time = 0.0

        # Save initial node stiffnesses so fault degradation doesn't
        # permanently mutate the stream between realizations.
        initial_k_fp = {node.id: node.k_fp for node in stream.nodes if hasattr(node, 'k_fp')}

        # Latency tracking
        latency_events = 0
        max_latency_ms = 0.0
        per_packet_latency: List[Tuple[int, float]] = []

        # Integration method
        use_numba = self.config.use_numba_rk4 and _NUMBA_AVAILABLE
        use_zero_torque_numba = self.config.use_zero_torque_numba and _NUMBA_AVAILABLE

        def zero_torque(packet_id, t, state):
            return np.array([0.0, 0.0, 0.0])

        # Initialize latency state for all packets
        for packet in stream.packets:
            if not hasattr(packet, 'latency_buffer'):
                packet.latency_buffer = []

        # Apply initial latency perturbations (vectorized for speed)
        if self.config.latency_ms > 0:
            n_packets = len(stream.packets)
            # Generate all latencies at once using vectorized operations
            latencies = _vectorized_normal(
                loc=self.config.latency_ms / 1000.0,
                scale=self.config.latency_std_ms / 1000.0,
                size=n_packets
            )
            latencies = np.maximum(0.0, latencies)  # Ensure non-negative
            
            for idx, packet in enumerate(stream.packets):
                latency = latencies[idx]
                packet.latency_buffer.append((current_time + latency, packet.body.state_copy()))
                latency_events += 1
                max_latency_ms = max(max_latency_ms, latency * 1000.0)
                if self.config.track_per_packet_latency:
                    per_packet_latency.append((packet.id, latency * 1000.0))

        # fault_rate is per hour, convert to per-step probability
        fault_prob_per_step = self.config.fault_rate * self.config.dt / 3600.0
        
        # NEW: Guaranteed fault injection (Root Cause #1)
        guaranteed_fault_times = []
        guaranteed_fault_nodes = []
        if self.config.fault_injection_mode == "guaranteed" and self.config.n_guaranteed_faults > 0:
            # Pre-sample N fault times uniformly in [0, time_horizon]
            guaranteed_fault_times = np.random.uniform(0, self.config.time_horizon, self.config.n_guaranteed_faults)
            # Pre-sample N node IDs uniformly from [0, n_nodes)
            n_nodes_available = len(stream.nodes) if stream.nodes else 1
            guaranteed_fault_nodes = np.random.randint(0, n_nodes_available, self.config.n_guaranteed_faults)
            logger.info(f"Guaranteed fault injection: {self.config.n_guaranteed_faults} faults at times {guaranteed_fault_times}")
        
        # NEW: Poisson fault injection (Root Cause #1 alternative)
        poisson_n_faults = 0
        poisson_fault_times = []
        poisson_fault_nodes = []
        if self.config.fault_injection_mode == "poisson":
            lambda_rate = self.config.fault_rate * self.config.time_horizon * len(stream.nodes) / 3600.0 if stream.nodes else 0.0
            poisson_n_faults = np.random.poisson(lambda_rate)
            # Pre-sample fault times and nodes (like guaranteed faults)
            if poisson_n_faults > 0:
                poisson_fault_times = np.random.uniform(0, self.config.time_horizon, poisson_n_faults)
                n_nodes_available = len(stream.nodes) if stream.nodes else 1
                poisson_fault_nodes = np.random.randint(0, n_nodes_available, poisson_n_faults)
            logger.info(f"Poisson fault injection: lambda={lambda_rate:.4f}, sampled n_faults={poisson_n_faults}")

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

            # Apply fault injection at each step (continuous rate-based)
            if fault_prob_per_step > 0 and stream.nodes:
                for node in stream.nodes:
                    if np.random.random() < fault_prob_per_step:
                        if hasattr(node, 'k_fp'):
                            original_k = node.k_fp
                            node.k_fp /= self.config.cascade_threshold
                            nodes_affected.add(node.id)
                            fault_events_injected += 1
                            logger.debug(f"Step {step}: Node {node.id} failed: k_fp {original_k:.1f} -> {node.k_fp:.1f}")
                            
                            # NEW: Cascade propagation (Root Cause #2)
                            if self.config.enable_cascade_propagation:
                                cascade_generations = self._propagate_cascade(
                                    stream, node, nodes_affected, current_time
                                )

            # NEW: Guaranteed fault injection (Root Cause #1)
            if self.config.fault_injection_mode == "guaranteed" and len(guaranteed_fault_times) > 0:
                for i, (fault_time, fault_node_idx) in enumerate(zip(guaranteed_fault_times, guaranteed_fault_nodes)):
                    # Check if this fault should fire at this step
                    step_time = current_time + self.config.dt  # Will be time after this integration
                    if fault_time <= step_time and fault_time > current_time:
                        if stream.nodes and fault_node_idx < len(stream.nodes):
                            node = stream.nodes[fault_node_idx]
                            if hasattr(node, 'k_fp'):
                                original_k = node.k_fp
                                node.k_fp /= self.config.cascade_threshold
                                nodes_affected.add(node.id)
                                fault_events_injected += 1
                                logger.debug(f"Guaranteed fault #{i+1}: Node {node.id} failed at t={current_time:.3f}s: k_fp {original_k:.1f} -> {node.k_fp:.1f}")
                                
                                # Cascade propagation for guaranteed faults
                                if self.config.enable_cascade_propagation:
                                    gen = self._propagate_cascade(stream, node, nodes_affected, current_time)
                                    cascade_generations = max(cascade_generations, gen)

            # NEW: Poisson fault injection (Root Cause #1)
            if self.config.fault_injection_mode == "poisson" and poisson_n_faults > 0:
                # Use pre-sampled fault times and nodes
                for i, (fault_time, fault_node_idx) in enumerate(zip(poisson_fault_times, poisson_fault_nodes)):
                    step_time = current_time + self.config.dt
                    if fault_time <= step_time and fault_time > current_time:
                        if stream.nodes and fault_node_idx < len(stream.nodes):
                            node = stream.nodes[fault_node_idx]
                            if hasattr(node, 'k_fp'):
                                original_k = node.k_fp
                                node.k_fp /= self.config.cascade_threshold
                                nodes_affected.add(node.id)
                                fault_events_injected += 1
                                logger.debug(f"Poisson fault #{i+1}: Node {node.id} failed at t={current_time:.3f}s")
                                
                                if self.config.enable_cascade_propagation:
                                    gen = self._propagate_cascade(stream, node, nodes_affected, current_time)
                                    cascade_generations = max(cascade_generations, gen)

            result = stream.integrate(
                self.config.dt,
                zero_torque,
                use_numba_rk4=use_numba,
                use_zero_torque_numba=use_zero_torque_numba,
            )
            current_time += self.config.dt
            
            # NEW: Thermal/quench monitoring (Root Cause #4)
            if quench_detector is not None or self.config.enable_thermal_quench:
                for packet in stream.packets:
                    # Track max temperature
                    max_temperature_reached = max(max_temperature_reached, packet.temperature)
                    
                    # Check for thermal violations
                    if hasattr(packet, 'temperature') and hasattr(packet, 'material'):
                        try:
                            critical_temp = packet.material.properties.Tc
                            if packet.temperature >= critical_temp:
                                thermal_violations_count += 1
                                logger.debug(f"Thermal violation: packet {packet.id} T={packet.temperature:.1f}K >= Tc={critical_temp:.1f}K")
                                
                                # Trigger quench event
                                quench_events += 1
                                
                                # If quench detector enabled, check for emergency shutdown
                                if quench_detector is not None:
                                    quench_detected = quench_detector.check_quench(
                                        packet.temperature, 
                                        critical_temp,
                                        current_time
                                    )
                                    if quench_detected:
                                        logger.warning(f"Quench detected for packet {packet.id}!")
                                        # Reduce k_fp of nearby nodes to simulate loss of superconducting pinning
                                        for node in stream.nodes:
                                            if hasattr(node, 'k_fp'):
                                                node.k_fp *= 0.01  # Near-zero stiffness during quench
                                                nodes_affected.add(node.id)
                                                fault_events_injected += 1
                        except (AttributeError, TypeError):
                            pass  # Material properties not available

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
        
        # Compute k_eff_min from the minimum stiffness reached during the run
        if stream.nodes:
            # Each node's actual stiffness at the end is its initial_k_fp reduced by the failures.
            # We compute this BEFORE restoring the nodes.
            k_eff_min = min(n.k_fp for n in stream.nodes if hasattr(n, 'k_fp'))
        else:
            k_eff_min = 6000.0  # fallback if no nodes

        # Restore node stiffnesses to initial values (undo in-place fault mutations)
        for node in stream.nodes:
            if node.id in initial_k_fp:
                node.k_fp = initial_k_fp[node.id]
        k_eff_within_limit = k_eff_min >= self.config.pass_fail_gates.get("k_eff", (6000.0,))[0]
        
        # Determine containment success and cascade from node failures
        containment_successful = len(nodes_affected) <= self.config.containment_threshold

        # Cascade also occurs if too many nodes fail (cascade propagation)
        if len(nodes_affected) > self.config.containment_threshold and not cascade_occurred:
            cascade_occurred = True
            failure_mode = "cascade_propagation"

        # Check pass/fail gates with safe defaults
        eta_ind_pass = eta_ind_min >= self.config.pass_fail_gates.get("eta_ind", (0.82,))[0]
        stress_pass = stress_max <= self.config.pass_fail_gates.get("stress", (1.2e9,))[0]
        k_eff_pass = k_eff_min >= self.config.pass_fail_gates.get("k_eff", (6000.0,))[0]

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
            nodes_affected=len(nodes_affected),
            containment_successful=containment_successful,
            # NEW: Diagnostic counters - Trust Strategy #1
            fault_events_injected=fault_events_injected,
            thermal_violations_count=thermal_violations_count,
            quench_events=quench_events,
            max_temperature_reached=max_temperature_reached,
            cascade_generations=cascade_generations,
        )
    
    def _run_realization_worker(self, args: Tuple) -> RealizationResult:
        """Worker function for multiprocessing."""
        stream_factory, realization_id = args
        stream = stream_factory()
        return self.run_realization(stream, realization_id)
    
    def _propagate_cascade(
        self, 
        stream: MultiBodyStream, 
        failed_node, 
        nodes_affected: set,
        current_time: float,
        generation: int = 0
    ) -> int:
        """
        Propagate cascade failure to neighboring nodes (Root Cause #2).
        
        When a node fails, transfer load to adjacent nodes, increasing their
        failure probability. This creates the positive feedback loop that
        defines a true cascade.
        
        Args:
            stream: Multi-body stream containing nodes
            failed_node: The node that just failed
            nodes_affected: Set of already-affected node IDs (modified in place)
            current_time: Current simulation time
            generation: Current cascade generation depth
            
        Returns:
            Maximum cascade generation reached
        """
        if generation >= self.config.max_cascade_generations:
            return generation
        
        if not stream.nodes:
            return generation
        
        # Find neighbors (simple distance-based adjacency)
        failed_pos = failed_node.position
        neighbors = []
        for node in stream.nodes:
            if node.id != failed_node.id and node.id not in nodes_affected:
                distance = np.linalg.norm(node.position - failed_pos)
                # Consider nodes within 20m as neighbors (adjustable)
                if distance < 20.0:
                    neighbors.append(node)
        
        if not neighbors:
            return generation
        
        # Transfer load to neighbors
        load_factor = 1.0 + self.config.cascade_propagation_factor / len(neighbors)
        
        for neighbor in neighbors:
            if hasattr(neighbor, 'k_fp'):
                # Reduce neighbor's stiffness to model increased stress
                original_k = neighbor.k_fp
                neighbor.k_fp /= load_factor
                
                # Check if neighbor also fails (cascades further)
                # Use a simplified criterion: if k_fp drops below threshold, it fails too
                k_fp_threshold = self.config.pass_fail_gates.get("k_eff", (6000.0,))[0]
                if neighbor.k_fp < k_fp_threshold * 0.5:  # 50% of minimum required
                    nodes_affected.add(neighbor.id)
                    logger.debug(
                        f"Cascade gen {generation+1}: Node {neighbor.id} failed due to load transfer: "
                        f"k_fp {original_k:.1f} -> {neighbor.k_fp:.1f}"
                    )
                    
                    # Recursively propagate
                    sub_gen = self._propagate_cascade(
                        stream, neighbor, nodes_affected, current_time, generation + 1
                    )
                    generation = max(generation, sub_gen)
                else:
                    logger.debug(
                        f"Load transferred to Node {neighbor.id}: k_fp {original_k:.1f} -> {neighbor.k_fp:.1f}"
                    )
        
        return generation

    def _run_jax_batch(self, stream_factory: Callable[[], MultiBodyStream], start_idx: int, batch_size: int) -> List[RealizationResult]:
        """
        Run a batch of realizations using JAX for GPU acceleration.

        NOTE: This is currently a sequential loop wrapper - it does NOT vectorize
        the physics integration. Full physics vectorization would require rewriting
        the MultiBodyStream integration in JAX (major refactoring task).
        
        Current implementation:
        - Sequential loop over realizations (no GPU vectorization)
        - JAX is only used for perturbation generation if enabled elsewhere
        - TODO: Rewrite MultiBodyStream.integrate() in JAX for true GPU acceleration

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

        # Early termination: run in batches and check CI convergence
        if self.config.enable_early_termination and self.acceleration_mode != "multiprocessing":
            batch_size = 10
            for start_idx in range(0, self.config.n_realizations, batch_size):
                current_batch_size = min(batch_size, self.config.n_realizations - start_idx)
                for i in range(start_idx, start_idx + current_batch_size):
                    stream = stream_factory()
                    result = self.run_realization(stream, i)
                    results.append(result)

                # Check convergence after min_realizations
                if len(results) >= self.config.min_realizations:
                    success_count = sum(1 for r in results if r.success)
                    ci_lower, ci_upper = self._wilson_ci(success_count, len(results))
                    ci_width = ci_upper - ci_lower

                    if ci_width < self.config.ci_width_threshold:
                        logger.info(f"Early termination: CI width {ci_width:.3f} < threshold {self.config.ci_width_threshold} after {len(results)} realizations")
                        break
        elif self.acceleration_mode == "multiprocessing":
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
        nodes_affected_values = [r.nodes_affected for r in results]
        containment_successful_values = [r.containment_successful for r in results]

        failure_modes = {}
        for r in results:
            if r.failure_mode:
                failure_modes[r.failure_mode] = failure_modes.get(r.failure_mode, 0) + 1

        cascade_count = sum(1 for r in results if r.cascade_occurred)
        cascade_probability = cascade_count / len(results)
        containment_rate = sum(containment_successful_values) / len(results) if results else 1.0
        n = len(results)

        # Normal CI for means (95%)
        def mean_ci(values, z=1.96):
            if len(values) == 0:
                return (0.0, 0.0)
            mean = np.mean(values)
            sem = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
            return (mean - z * sem, mean + z * sem)

        # NEW: Diagnostic counter aggregation - Trust Strategy #1 & #4
        fault_events_total = sum(r.fault_events_injected for r in results)
        thermal_violations_total = sum(r.thermal_violations_count for r in results)
        quench_events_total = sum(r.quench_events for r in results)
        max_temperature_global = max(r.max_temperature_reached for r in results) if results else 0.0
        cascade_generations_max = max(r.cascade_generations for r in results) if results else 0
        
        # Provenance metadata - Trust Strategy #4
        # Create ONE reference stream for provenance inspection
        _ref_stream = stream_factory()
        provenance = {
            "expected_faults_per_realization": (
                self.config.fault_rate * self.config.time_horizon * len(_ref_stream.nodes) / 3600.0
                if _ref_stream.nodes and self.config.fault_injection_mode == "rate"
                else (self.config.n_guaranteed_faults if self.config.fault_injection_mode == "guaranteed" else 0)
            ),
            "actual_faults_total": fault_events_total,
            "actual_faults_per_realization_mean": fault_events_total / n if n > 0 else 0.0,
            "thermal_model_active": bool(_ref_stream.packets and hasattr(_ref_stream.packets[0], 'temperature')),
            "quench_detector_active": self.config.quench_detection_enabled,
            "mpc_controller_active": False,  # Not used in MC loop
            "n_packets": len(_ref_stream.packets) if _ref_stream.packets else 0,
            "n_nodes": len(_ref_stream.nodes) if _ref_stream.nodes else 0,
            "stream_topology": "linear",  # Default topology
            "fault_injection_mode": self.config.fault_injection_mode,
            "cascade_propagation_enabled": self.config.enable_cascade_propagation,
        }

        return {
            "n_realizations": n,
            "n_success": success_count,
            "n_failure": failure_count,
            "n_cascade": cascade_count,
            "success_rate": success_count / n,
            "success_rate_ci": self._wilson_ci(success_count, n),
            "cascade_probability": cascade_probability,
            "cascade_probability_ci": self._wilson_ci(cascade_count, n),

            "eta_ind_min_mean": np.mean(eta_ind_values),
            "eta_ind_min_std": np.std(eta_ind_values),
            "eta_ind_min_min": np.min(eta_ind_values),
            "eta_ind_min_ci": mean_ci(eta_ind_values),
            "stress_max_mean": np.mean(stress_values),
            "stress_max_std": np.std(stress_values),
            "stress_max_max": np.max(stress_values),
            "stress_max_ci": mean_ci(stress_values),
            "failure_modes": failure_modes,
            "latency_events": sum(latency_events),
            "max_latency_ms": max(max_latency_values) if max_latency_values else 0.0,
            "k_eff_min": float(np.min([r.k_eff_min for r in results])) if results else 6000.0,
            "meets_cascade_target": cascade_probability < 1e-6,  # Target: <10⁻⁶
            "acceleration_mode": self.acceleration_mode,
            "nodes_affected_mean": np.mean(nodes_affected_values) if nodes_affected_values else 0.0,
            "nodes_affected_std": np.std(nodes_affected_values) if nodes_affected_values else 0.0,
            "nodes_affected_max": max(nodes_affected_values) if nodes_affected_values else 0,
            "nodes_affected_ci": mean_ci(nodes_affected_values),
            "containment_rate": containment_rate,
            "containment_rate_ci": self._wilson_ci(sum(containment_successful_values), n),
            "delay_margin_ms": None,  # Not calculated in Monte-Carlo - requires MPC controller
            
            # NEW: Diagnostic counters - Trust Strategy #1
            "fault_events_total": fault_events_total,
            "fault_events_per_realization_mean": fault_events_total / n if n > 0 else 0.0,
            "thermal_violations_total": thermal_violations_total,
            "quench_events_total": quench_events_total,
            "max_temperature_global": max_temperature_global,
            "cascade_generations_max": cascade_generations_max,
            
            # NEW: Provenance metadata - Trust Strategy #4
            "provenance": provenance,
            
            # NEW: Sanity flag - Trust Strategy #2
            "sanity_check_passed": fault_events_total > 0 or self.config.fault_rate == 0 or self.config.fault_injection_mode != "rate",
            "sanity_warning": "" if (fault_events_total > 0 or self.config.fault_rate == 0 or self.config.fault_injection_mode != "rate") 
                              else "NO FAULTS INJECTED - results may not reflect cascade behavior",
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
