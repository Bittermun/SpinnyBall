"""
Velocity Optimization Engine - Infrastructure Cost Minimizer

Optimizes stream velocity to minimize infrastructure requirements.
Key insight: For constant momentum flux, ball count scales as N ~ 1/v²
Therefore, increasing velocity exponentially reduces infrastructure cost.

Physics Constraints:
- Gyroscopic stability requires minimum spin rate
- Flux-pinning has velocity-dependent capture efficiency
- Structural limits on maximum acceleration
- Thermal limits on friction/drag heating

Reference: SGMS momentum flux physics + Bean-London dynamics
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from enum import Enum


class OptimizationStrategy(Enum):
    """Optimization strategies for velocity selection."""
    MIN_BALL_COUNT = "min_balls"
    MIN_INFRASTRUCTURE_MASS = "min_mass"
    MAX_EFFICIENCY = "max_eff"
    BALANCED = "balanced"


@dataclass
class VelocityConstraint:
    """Velocity constraint from physics or engineering limits."""
    name: str
    min_value: float
    max_value: float
    weight: float = 1.0

    def is_feasible(self, v: float) -> bool:
        return self.min_value <= v <= self.max_value

    def penalty(self, v: float) -> float:
        if v < self.min_value:
            return self.weight * (self.min_value - v) / self.min_value
        elif v > self.max_value:
            return self.weight * (v - self.max_value) / self.max_value
        return 0.0


@dataclass
class OptimizationResult:
    """Result of velocity optimization."""
    optimal_velocity: float
    min_balls_required: int
    infrastructure_cost: float
    efficiency_score: float
    stability_margin: float
    slingshot_recommended: bool
    velocity_gain_potential: float
    total_momentum_capacity: float
    power_requirement: float


class VelocityOptimizer:
    """
    Optimizes stream velocity to minimize infrastructure cost.
    N ~ 1/v² for constant momentum capacity
    """

    DEFAULT_BALL_MASS = 35.0
    DEFAULT_CAPTURE_EFFICIENCY = 0.85
    DEFAULT_STREAM_DENSITY = 0.5
    DEFAULT_TARGET_FORCE = 10000.0
    DEFAULT_STREAM_LENGTH = 4.8  # meters

    def __init__(
        self,
        ball_mass: float = DEFAULT_BALL_MASS,
        capture_efficiency: float = DEFAULT_CAPTURE_EFFICIENCY,
        stream_density: float = DEFAULT_STREAM_DENSITY,
        target_force: float = DEFAULT_TARGET_FORCE,
        stream_length: float = DEFAULT_STREAM_LENGTH,
        use_slingshot: bool = True,
        use_flux_gyro: bool = True
    ):
        self.ball_mass = ball_mass
        self.capture_efficiency = capture_efficiency
        self.stream_density = stream_density
        self.target_force = target_force
        self.stream_length = stream_length
        self.use_slingshot = use_slingshot
        self.use_flux_gyro = use_flux_gyro
        self._setup_default_constraints()

    def _setup_default_constraints(self):
        """Setup default velocity constraints."""
        self.constraints = [
            VelocityConstraint("gyroscopic_stability", 500.0, 15000.0, 1.0),
            VelocityConstraint("flux_capture", 100.0, 8000.0, 0.9),
            VelocityConstraint("structural", 0.0, 12000.0, 0.8),
            VelocityConstraint("thermal", 0.0, 10000.0, 0.7),
        ]
        if self.use_flux_gyro:
            self.constraints[0].max_value = 20000.0
            self.constraints[2].max_value = 18000.0

    def compute_ball_count(self, velocity: float, include_slingshot: bool = False) -> int:
        """Compute number of balls required for target force.
        
        Formula from TECHNICAL_SPEC.md: N = F * L / (m * v² * η)
        where F is force, L is stream length, m is ball mass, v is velocity, η is efficiency.
        """
        if velocity < 1.0:
            return 999999

        v_eff = velocity
        if include_slingshot:
            v_eff = velocity * 1.2

        # Correct dimensional formula: N = F * L / (m * v² * η)
        N = int(np.ceil(self.target_force * self.stream_length / 
                        (self.ball_mass * v_eff**2 * self.capture_efficiency)))
        return max(N, 1)

    def compute_infrastructure_cost(self, velocity: float, ball_count: int) -> float:
        """Compute total infrastructure cost relative to baseline."""
        v_base = 1000.0
        n_base = self.compute_ball_count(v_base)

        ball_cost = (ball_count / n_base) ** 0.9
        mass_cost = ball_count / n_base
        tracking_cost = (ball_count / n_base) ** 0.5
        complexity_factor = 1.0 + 0.1 * (velocity / v_base - 1.0)

        total_cost = 0.4 * ball_cost + 0.3 * mass_cost + 0.2 * tracking_cost + 0.1 * complexity_factor
        return total_cost

    def compute_efficiency_score(self, velocity: float, ball_count: int) -> float:
        """Compute momentum transfer efficiency score (0-1)."""
        v_capture_max = 8000.0
        eta_capture = 1.0 / (1.0 + (velocity / v_capture_max) ** 2)
        v_gyro_char = 2000.0
        eta_gyro = np.tanh(velocity / v_gyro_char)
        eta_energy = 0.7 + 0.3 * np.exp(-velocity / 5000.0)
        return 0.4 * eta_capture + 0.3 * eta_gyro + 0.3 * eta_energy

    def compute_stability_margin(self, velocity: float) -> float:
        """Compute stability margin (1.0 = nominal)."""
        v_nominal = 5000.0
        stability_gyro = velocity / v_nominal
        if self.use_flux_gyro:
            stability_gyro *= 2.5
        v_capture_max = self.constraints[1].max_value
        capture_penalty = max(0, (velocity - 0.8 * v_capture_max) / (0.2 * v_capture_max))
        return stability_gyro * (1.0 - 0.3 * capture_penalty)

    def objective_function(self, velocity: float, strategy: OptimizationStrategy) -> float:
        """Compute optimization objective (lower is better)."""
        constraint_penalty = sum(c.penalty(velocity) for c in self.constraints)
        if constraint_penalty > 1.0:
            return 1e10 * constraint_penalty

        ball_count = self.compute_ball_count(velocity)
        cost = self.compute_infrastructure_cost(velocity, ball_count)
        efficiency = self.compute_efficiency_score(velocity, ball_count)
        stability = self.compute_stability_margin(velocity)

        if strategy == OptimizationStrategy.MIN_BALL_COUNT:
            obj = ball_count / 10.0 + 0.1 * cost - 0.5 * efficiency
        elif strategy == OptimizationStrategy.MIN_INFRASTRUCTURE_MASS:
            mass = ball_count * self.ball_mass
            obj = mass / 100.0 + 0.2 * cost
        elif strategy == OptimizationStrategy.MAX_EFFICIENCY:
            obj = 1.0 - efficiency + 0.1 * ball_count / 100.0
        else:
            obj = 0.3 * ball_count / 100.0 + 0.3 * cost + 0.2 * (1.0 - efficiency) + 0.2 * max(0, 1.0 - stability)

        return obj + constraint_penalty

    def optimize(
        self,
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
        velocity_range: Tuple[float, float] = (500.0, 15000.0),
        n_samples: int = 200
    ) -> OptimizationResult:
        """Optimize velocity for given strategy."""
        velocities = np.linspace(velocity_range[0], velocity_range[1], n_samples)
        objectives = [self.objective_function(v, strategy) for v in velocities]
        best_idx = np.argmin(objectives)
        optimal_v = velocities[best_idx]

        ball_count = self.compute_ball_count(optimal_v)
        cost = self.compute_infrastructure_cost(optimal_v, ball_count)
        efficiency = self.compute_efficiency_score(optimal_v, ball_count)
        stability = self.compute_stability_margin(optimal_v)

        slingshot_recommended = self.use_slingshot and optimal_v < 8000.0 and stability > 0.8
        v_gain = optimal_v * 0.2 if slingshot_recommended else 0.0
        power = self.target_force * optimal_v / 1000.0

        return OptimizationResult(
            optimal_velocity=optimal_v,
            min_balls_required=ball_count,
            infrastructure_cost=cost,
            efficiency_score=efficiency,
            stability_margin=stability,
            slingshot_recommended=slingshot_recommended,
            velocity_gain_potential=v_gain,
            total_momentum_capacity=ball_count * self.ball_mass * optimal_v,
            power_requirement=power
        )

    def compare_strategies(self) -> Dict[str, OptimizationResult]:
        """Compare all optimization strategies."""
        results = {}
        for strategy in OptimizationStrategy:
            result = self.optimize(strategy)
            results[strategy.value] = result
        return results


def demo_velocity_optimizer():
    """Demonstrate velocity optimization."""
    print("=" * 70)
    print("VELOCITY OPTIMIZATION ENGINE - Infrastructure Cost Minimizer")
    print("=" * 70)

    optimizer = VelocityOptimizer(
        ball_mass=35.0,
        capture_efficiency=0.85,
        target_force=10000.0,
        use_slingshot=True,
        use_flux_gyro=True
    )

    print("\n--- Strategy Comparison ---")
    results = optimizer.compare_strategies()

    for name, result in results.items():
        print(f"\n{name.upper()}:")
        print(f"  Optimal velocity: {result.optimal_velocity/1000:.1f} km/s")
        print(f"  Balls required: {result.min_balls_required}")
        print(f"  Infrastructure cost: {result.infrastructure_cost:.2f}x")
        print(f"  Efficiency: {result.efficiency_score:.2f}")
        print(f"  Stability margin: {result.stability_margin:.2f}")
        if result.slingshot_recommended:
            print(f"  Slingshot recommended: +{result.velocity_gain_potential/1000:.1f} km/s gain")

    best = results["balanced"]
    print(f"\n--- Recommended Configuration ---")
    print(f"Optimal velocity: {best.optimal_velocity/1000:.1f} km/s")
    print(f"Ball count reduction: {(1 - best.infrastructure_cost)*100:.0f}% vs baseline")
    print(f"Key insight: N ~ 1/v², so doubling velocity = 4x fewer balls")


if __name__ == "__main__":
    demo_velocity_optimizer()
