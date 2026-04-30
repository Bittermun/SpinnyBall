"""
Earth-Moon Active Pumping - Simplified Conceptual Demonstration

Demonstrates the physics concept of active flux-gyro enabled velocity pumping
without full orbital mechanics complexity.

Key Insight: Active control enables repeated lunar encounters,
achieving 11-13 km/s velocities through 5-10 controlled cycles.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum


@dataclass
class PumpingCycle:
    """Results from one pumping cycle."""
    cycle_number: int
    perigee_velocity_start: float  # km/s
    perigee_velocity_end: float  # km/s
    lunar_dv_gain: float  # km/s from gravity assist
    flux_dv_applied: float  # km/s from active thrust
    energy_efficiency: float  # Fraction of flux energy converted to orbital energy


@dataclass
class PumpingMission:
    """Complete pumping mission results."""
    cycles: List[PumpingCycle]
    final_velocity: float  # km/s at perigee
    velocity_gain: float  # km/s total increase
    total_lunar_assists: int
    total_flux_dv: float  # km/s
    flux_energy_consumed: float  # MJ
    ball_reduction_factor: float


class EarthMoonPumpingConcept:
    """
    Conceptual demonstration of active Earth-Moon velocity pumping.

    Physics Model:
    - Each lunar encounter adds ~0.3 km/s via gravity assist
    - Active flux thrust enables repeated encounters (unlike passive orbits)
    - 5-10 cycles achieve 11-13 km/s final velocity
    """

    def __init__(
        self,
        ball_mass: float = 35.0,
        flux_force: float = 15.0,  # N
        initial_velocity: float = 10.9,  # km/s at perigee
        lunar_dv_per_encounter: float = 0.3,  # km/s gain per lunar assist
        flux_energy_efficiency: float = 0.15  # Energy conversion efficiency
    ):
        self.ball_mass = ball_mass
        self.flux_force = flux_force
        self.v_initial = initial_velocity
        self.lunar_dv = lunar_dv_per_encounter
        self.energy_efficiency = flux_energy_efficiency

        # Physics constants
        self.v_escape_earth = 11.2  # km/s
        self.v_circular_lunar = 1.0  # km/s

    def simulate_single_cycle(
        self,
        cycle_number: int,
        current_velocity: float,  # km/s at perigee
        target_apogee: float  # km (distance to lunar encounter)
    ) -> PumpingCycle:
        """
        Simulate one pumping cycle.

        Process:
        1. Start at perigee with current velocity
        2. Coast to apogee (lunar distance)
        3. Lunar gravity assist adds ~1 km/s
        4. Active flux thrust for orbital tuning
        5. Return to perigee with higher velocity
        """

        # Lunar encounter - gravity assist
        # Simplified: assume we can get the Moon at the right phase
        lunar_dv_gain = self.lunar_dv

        # Active flux thrust between encounters
        # Need to adjust orbit to intersect Moon again
        # This requires ~0.05 km/s dV per cycle for orbital tuning
        tuning_dv = 0.05  # km/s (simplified)

        # Energy required for tuning thrust
        # E = (1/2) * m * v^2 for the tuning burn
        tuning_energy = 0.5 * self.ball_mass * (tuning_dv * 1000)**2  # Joules
        tuning_energy_mj = tuning_energy / 1e6  # MJ

        # Final perigee velocity after lunar assist
        # Simplified model: lunar assist increases velocity, tuning adjusts orbit
        velocity_after_lunar = current_velocity + lunar_dv_gain
        final_velocity = velocity_after_lunar - tuning_dv * 0.1  # Some energy loss to orbital tuning

        # Energy efficiency: how much flux energy becomes useful orbital energy
        orbital_energy_gain = 0.5 * self.ball_mass * ((final_velocity - current_velocity) * 1000)**2
        energy_efficiency = orbital_energy_gain / (tuning_energy * self.energy_efficiency) if tuning_energy > 0 else 1.0

        return PumpingCycle(
            cycle_number=cycle_number,
            perigee_velocity_start=current_velocity,
            perigee_velocity_end=final_velocity,
            lunar_dv_gain=lunar_dv_gain,
            flux_dv_applied=tuning_dv,
            energy_efficiency=min(energy_efficiency, 1.0)
        )

    def run_pumping_mission(
        self,
        num_cycles: int = 10,
        target_force: float = 10000.0  # N (momentum flux requirement)
    ) -> PumpingMission:
        """
        Run complete pumping mission with specified number of cycles.
        """

        print("\n" + "="*70)
        print("EARTH-MOON ACTIVE PUMPING - CONCEPTUAL DEMONSTRATION")
        print("="*70)
        print(f"\nMission Parameters:")
        print(f"  Ball mass: {self.ball_mass} kg")
        print(f"  Flux force: {self.flux_force} N")
        print(f"  Lunar dV per encounter: {self.lunar_dv} km/s")
        print(f"  Energy efficiency: {self.energy_efficiency*100:.0f}%")
        print(f"  Cycles: {num_cycles}")

        # Simulate cycles
        cycles = []
        current_velocity = self.v_initial

        print(f"\nInitial velocity: {current_velocity:.2f} km/s")

        for cycle in range(1, num_cycles + 1):
            cycle_result = self.simulate_single_cycle(
                cycle_number=cycle,
                current_velocity=current_velocity,
                target_apogee=384400.0  # Moon distance in km
            )

            cycles.append(cycle_result)
            current_velocity = cycle_result.perigee_velocity_end

            print(f"\nCycle {cycle}:")
            print(f"  Start velocity: {cycle_result.perigee_velocity_start:.2f} km/s")
            print(f"  Lunar dV gain: +{cycle_result.lunar_dv_gain:.3f} km/s")
            print(f"  Flux tuning dV: -{cycle_result.flux_dv_applied:.3f} km/s")
            print(f"  End velocity: {cycle_result.perigee_velocity_end:.2f} km/s")
            print(f"  Net gain: +{cycle_result.perigee_velocity_end - cycle_result.perigee_velocity_start:.3f} km/s")

            # Check for target achievement
            if current_velocity >= 11.0 and cycle >= 5:
                print(f"\n*** TARGET ACHIEVED: {current_velocity:.1f} km/s at cycle {cycle} ***")
                break

        # Mission summary
        final_velocity = current_velocity
        velocity_gain = final_velocity - self.v_initial
        total_lunar_assists = len(cycles)
        total_flux_dv = sum(c.flux_dv_applied for c in cycles)
        total_energy = sum(0.5 * self.ball_mass * (c.flux_dv_applied * 1000)**2 for c in cycles) / 1e6  # MJ

        # Infrastructure impact
        # N ~ F / (m * v^2 * efficiency)
        capture_eff = 0.85  # Ball capture efficiency
        n_baseline = int(np.ceil(target_force / (self.ball_mass * self.v_initial**2 * 1000 * capture_eff)))
        n_final = int(np.ceil(target_force / (self.ball_mass * final_velocity**2 * 1000 * capture_eff)))
        ball_reduction = n_baseline / n_final if n_final > 0 else 1.0

        mission = PumpingMission(
            cycles=cycles,
            final_velocity=final_velocity,
            velocity_gain=velocity_gain,
            total_lunar_assists=total_lunar_assists,
            total_flux_dv=total_flux_dv,
            flux_energy_consumed=total_energy,
            ball_reduction_factor=ball_reduction
        )

        self._print_mission_summary(mission, n_baseline, n_final)
        return mission

    def _print_mission_summary(self, mission: PumpingMission, n_baseline: int, n_final: int):
        """Print formatted mission summary."""
        print("\n" + "="*70)
        print("MISSION SUMMARY")
        print("="*70)

        print(f"\nVelocity Evolution:")
        print(f"  Initial: {self.v_initial:.2f} km/s")
        print(f"  Final:   {mission.final_velocity:.2f} km/s")
        print(f"  Gain:    +{mission.velocity_gain:.2f} km/s")
        print(f"  Ratio:   {mission.final_velocity/self.v_initial:.2f}x")

        print(f"\nCycle Details:")
        print(f"{'Cycle':<8} {'Start v':<10} {'Lunar dV':<10} {'Flux dV':<10} {'End v':<10}")
        print("-"*60)
        for c in mission.cycles:
            print(f"{c.cycle_number:<8} {c.perigee_velocity_start:<10.2f} {c.lunar_dv_gain:<10.3f} {c.flux_dv_applied:<10.3f} {c.perigee_velocity_end:<10.2f}")

        print(f"\nEnergy & Work:")
        print(f"  Lunar assists: {mission.total_lunar_assists}")
        print(f"  Total flux dV: {mission.total_flux_dv:.3f} km/s")
        print(f"  Energy consumed: {mission.flux_energy_consumed:.2f} MJ")
        print(f"  Average efficiency: {np.mean([c.energy_efficiency for c in mission.cycles]):.1%}")

        print(f"\nInfrastructure Impact:")
        print(f"  Baseline ball count: {n_baseline}")
        print(f"  Final ball count: {n_final}")
        print(f"  Ball reduction: {(1 - 1/mission.ball_reduction_factor)*100:.0f}%")
        print(f"  Reduction factor: {mission.ball_reduction_factor:.1f}x")

        # Success metrics
        if mission.final_velocity >= 11.0:
            print(f"\n*** SUCCESS: 11+ km/s achieved for skyhook capture ***")
        if mission.final_velocity >= 13.0:
            print(f"*** EXCELLENT: 13+ km/s for high-energy transfer ***")
        if mission.ball_reduction_factor >= 5.0:
            print(f"*** MAJOR INFRASTRUCTURE SAVINGS: 5x fewer balls needed ***")

        print("\n" + "="*70)


def demonstrate_concept():
    """Demonstrate the Earth-Moon pumping concept."""
    print("\n" + "#"*70)
    print("#" + " EARTH-MOON ACTIVE PUMPING CONCEPT DEMONSTRATION ".center(68) + "#")
    print("#"*70)

    # Realistic parameters
    pumping = EarthMoonPumpingConcept(
        ball_mass=35.0,
        flux_force=15.0,  # N
        initial_velocity=10.9,  # km/s (just below Earth escape)
        lunar_dv_per_encounter=0.3,  # km/s per lunar assist
        flux_energy_efficiency=0.15  # 15% energy conversion
    )

    mission = pumping.run_pumping_mission(
        num_cycles=10,
        target_force=10000.0  # N momentum flux
    )

    # Compare with passive approach
    print(f"\nComparison with Passive Gravity Assists:")
    print(f"Passive: Single lunar assist -> {10.9 + 1.0:.1f} km/s (1 encounter)")
    print(f"Active:  {len(mission.cycles)} lunar assists -> {mission.final_velocity:.1f} km/s")
    print(f"Improvement: {mission.final_velocity / (10.9 + 1.0):.1f}x velocity via active control")

    print(f"\nKey Physics Insight:")
    print(f"- Passive gravity assists are one-time events")
    print(f"- Active flux-gyro control enables repeated encounters")
    print(f"- Each cycle adds energy through lunar gravity + active thrust")
    print(f"- Result: {mission.ball_reduction_factor:.1f}x infrastructure cost reduction")

    return mission


if __name__ == "__main__":
    demonstrate_concept()
