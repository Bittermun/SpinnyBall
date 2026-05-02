# dynamics/alternatives_comparison.py

"""
Comparison of SGMS packet stream to conventional station-keeping alternatives.

Provides quantitative metrics for:
1. Ion thrusters (Hall effect, gridded ion)
2. Chemical thrusters (hydrazine, bipropellant)
3. Electrodynamic tethers (EDT)
4. Solar sails
5. SGMS packet stream (this project)

This module answers the reviewer question: "why not just use ion thrusters?"
by providing a quantitative comparison across key metrics.
"""

from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class StationKeepingSystem:
    name: str
    force_N: float              # Available force
    stiffness_N_m: float        # Effective stiffness (0 for thrusters)
    power_kW: float             # Power consumption
    mass_kg: float              # System mass (propellant + hardware)
    isp_s: float                # Specific impulse (inf for propellantless)
    lifetime_years: float       # Operational lifetime
    propellant_mass_kg: float   # Propellant mass for lifetime
    precision_m: float          # Positioning precision
    trl: int                    # Technology Readiness Level
    notes: str


def compare_alternatives(
    F_required_N: float = 4.2,
    mission_years: float = 15.0,
    station_mass_kg: float = 1000.0,
) -> List[StationKeepingSystem]:
    """
    Compare station-keeping alternatives for given requirements.
    
    Args:
        F_required_N: Required station-keeping force (N)
        mission_years: Mission duration
        station_mass_kg: Station mass (kg)
    
    Returns:
        List of StationKeepingSystem objects for comparison
    """
    systems = []
    
    # 1. Hall-effect thruster
    isp_hall = 1500  # s
    m_dot_hall = F_required_N / (isp_hall * 9.81)  # kg/s
    prop_mass_hall = m_dot_hall * mission_years * 365.25 * 86400
    systems.append(StationKeepingSystem(
        name="Hall-Effect Thruster",
        force_N=F_required_N,
        stiffness_N_m=0.0,  # No passive stiffness
        power_kW=F_required_N / 0.06,  # ~60 mN/kW typical
        mass_kg=50 + prop_mass_hall,  # 50 kg hardware
        isp_s=isp_hall,
        lifetime_years=mission_years,
        propellant_mass_kg=prop_mass_hall,
        precision_m=0.01,  # ~1 cm with feedback
        trl=9,
        notes="Mature technology. Requires propellant resupply."
    ))
    
    # 2. Hydrazine monopropellant
    isp_hydrazine = 220  # s
    m_dot_hyd = F_required_N / (isp_hydrazine * 9.81)
    prop_mass_hyd = m_dot_hyd * mission_years * 365.25 * 86400
    systems.append(StationKeepingSystem(
        name="Hydrazine Monopropellant",
        force_N=F_required_N,
        stiffness_N_m=0.0,
        power_kW=0.01,  # Minimal
        mass_kg=20 + prop_mass_hyd,
        isp_s=isp_hydrazine,
        lifetime_years=min(mission_years, prop_mass_hyd / (m_dot_hyd * 86400 * 365)),
        propellant_mass_kg=prop_mass_hyd,
        precision_m=0.1,
        trl=9,
        notes="Simple but massive propellant requirement."
    ))
    
    # 3. EDT (from archived_edt module)
    # Lorentz force: F = I * L * B, typical: 0.1-1 N for 10 km tether
    systems.append(StationKeepingSystem(
        name="Electrodynamic Tether (10 km)",
        force_N=0.5,  # Typical for 10 km, 1A
        stiffness_N_m=0.0,  # No passive stiffness
        power_kW=1.0,  # For electron emitter
        mass_kg=50,  # 10 km of aluminum wire + deployer
        isp_s=float('inf'),  # Propellantless
        lifetime_years=5,  # Debris impact risk
        propellant_mass_kg=0,
        precision_m=10.0,  # Poor - tether dynamics
        trl=5,
        notes="Propellantless but only works in LEO (needs ionosphere). Cross-track only."
    ))
    
    # 4. Solar sail
    # F = 2 * P_solar * A / c, where P_solar = 1361 W/m², c = 3e8 m/s
    # For 4.2 N: A = 4.2 * 3e8 / (2 * 1361) = 462,000 m² = 680m × 680m
    sail_area = F_required_N * 3e8 / (2 * 1361)
    sail_side = np.sqrt(sail_area)
    systems.append(StationKeepingSystem(
        name=f"Solar Sail ({sail_side:.0f}m × {sail_side:.0f}m)",
        force_N=F_required_N,
        stiffness_N_m=0.0,
        power_kW=0.0,
        mass_kg=sail_area * 0.003,  # 3 g/m² areal density
        isp_s=float('inf'),
        lifetime_years=mission_years,
        propellant_mass_kg=0,
        precision_m=100.0,  # Very poor - solar pressure variations
        trl=6,
        notes=f"Requires {sail_area/1e6:.1f} km² sail. Force direction limited by Sun angle."
    ))
    
    # 5. SGMS Packet Stream (this project)
    # Use mission_level_metrics defaults
    systems.append(StationKeepingSystem(
        name="SGMS Packet Stream (15 km/s SmCo)",
        force_N=F_required_N,
        stiffness_N_m=2.3e6,  # From operational profile
        power_kW=0.0,  # SmCo needs no cryocooler
        mass_kg=560,  # From mission Sobol optimal
        isp_s=float('inf'),  # Propellantless (with slingshot)
        lifetime_years=mission_years,
        propellant_mass_kg=0,
        precision_m=1e-4,  # Sub-mm from k_eff = 2.3 MN/m
        trl=2,
        notes="Propellantless with slingshot replenishment. Extreme stiffness for precision."
    ))
    
    return systems


def format_comparison_table(systems: List[StationKeepingSystem]) -> str:
    """Format comparison as markdown table."""
    header = "| System | Force (N) | k_eff (N/m) | Power (kW) | Mass (kg) | Precision (m) | TRL |\n"
    header += "|--------|-----------|-------------|------------|-----------|---------------|-----|\n"
    rows = ""
    for s in systems:
        k_str = f"{s.stiffness_N_m:.0e}" if s.stiffness_N_m > 0 else "0 (active)"
        rows += f"| {s.name} | {s.force_N:.1f} | {k_str} | {s.power_kW:.1f} | {s.mass_kg:.0f} | {s.precision_m:.1e} | {s.trl} |\n"
    return header + rows


def generate_comparison_report(
    F_required_N: float = 4.2,
    mission_years: float = 15.0,
) -> str:
    """
    Generate a full comparison report with analysis.
    
    Args:
        F_required_N: Required force (N)
        mission_years: Mission duration
    
    Returns:
        Markdown-formatted comparison report
    """
    systems = compare_alternatives(F_required_N, mission_years)
    
    report = "# Station-Keeping System Comparison\n\n"
    report += f"## Requirements\n"
    report += f"- Force: {F_required_N:.1f} N\n"
    report += f"- Mission Duration: {mission_years:.1f} years\n\n"
    
    report += "## Comparison Table\n\n"
    report += format_comparison_table(systems)
    report += "\n\n"
    
    report += "## Key Findings\n\n"
    
    # Find the best in each category
    min_mass = min(systems, key=lambda s: s.mass_kg)
    max_stiffness = max(systems, key=lambda s: s.stiffness_N_m)
    best_precision = min(systems, key=lambda s: s.precision_m)
    
    report += f"- **Lowest Mass**: {min_mass.name} ({min_mass.mass_kg:.0f} kg)\n"
    report += f"- **Highest Stiffness**: {max_stiffness.name} ({max_stiffness.stiffness_N_m:.0e} N/m)\n"
    report += f"- **Best Precision**: {best_precision.name} ({best_precision.precision_m:.1e} m)\n\n"
    
    report += "## Detailed Analysis\n\n"
    for s in systems:
        report += f"### {s.name}\n\n"
        report += f"- **Force**: {s.force_N:.1f} N\n"
        report += f"- **Stiffness**: {s.stiffness_N_m:.0e} N/m\n"
        report += f"- **Power**: {s.power_kW:.1f} kW\n"
        report += f"- **Mass**: {s.mass_kg:.0f} kg"
        if s.propellant_mass_kg > 0:
            report += f" (including {s.propellant_mass_kg:.0f} kg propellant)"
        report += "\n"
        report += f"- **Precision**: {s.precision_m:.1e} m\n"
        report += f"- **TRL**: {s.trl}\n"
        report += f"- **Notes**: {s.notes}\n\n"
    
    return report
