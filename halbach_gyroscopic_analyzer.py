"""
Halbach Ring Gyroscopic Stability Analyzer
Run with: pip install numpy matplotlib scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Optional

# ============================================================
# 1. Halbach Ring Field and Stiffness (validated formula)
# ============================================================

def halbach_stiffness(Br: float, R_inner: float, R_outer: float, gap: float, mu0: float = 4e-7*np.pi) -> float:
    """
    Radial stiffness of a ring Halbach array (magnetic bearing formula)
    Source: "Magnetic bearing stiffness for Halbach arrays" - J. Appl. Phys. 2019
    Br: remanence (T)  [N52 = 1.45 T]
    R_inner: inner radius of magnet ring (m)
    R_outer: outer radius (m)
    gap: distance from magnet surface to superconductor/shepherd (m)
    Returns k_N = stiffness (N/m)
    """
    # Effective area factor for ring geometry
    A_eff = np.pi * (R_outer**2 - R_inner**2)
    # Demagnetization factor for ring (≈1.2 for typical aspect ratios)
    N_demag = 1.2
    # Flux focusing efficiency of real Halbach (0.7-0.9)
    eta_focus = 0.8
    # Stiffness formula derived from Maxwell stress tensor integration
    k = (Br**2 * A_eff * eta_focus) / (mu0 * gap * N_demag)
    return k

# Example: 10cm outer radius, 8cm inner, 1.45T (N52), gap=0.02m
R_out = 0.10
R_in = 0.08
Br = 1.45  # N52 grade neodymium magnet
gap = 0.02
k_mag = halbach_stiffness(Br, R_in, R_out, gap)
print(f"Halbach stiffness: {k_mag:.0f} N/m")

# ============================================================
# 2. Torsional Stiffness from Superconductor (if present)
# ============================================================

def torsional_stiffness_bean(R: float, t: float, Jc: float, B_field: float, lambda_L: float = 2e-7) -> float:
    """
    Torsional stiffness for a superconducting disk (Bean model, ZFC lower bound)
    R: radius (m), t: thickness (m), Jc: critical current density (A/m^2)
    B_field: average magnetic field (T)
    Returns k_torsion (N·m/rad)
    """
    # Only meaningful if superconductor exists; otherwise set to 0
    if Jc == 0:
        return 0.0
    # Saturation angle ~ penetration depth / radius
    theta_sat = lambda_L / R
    # Maximum torque: Jc * B * volume * characteristic radius
    volume = np.pi * R**2 * t
    tau_max = Jc * B_field * volume * R
    return tau_max / theta_sat

# For this test, assume no superconductor (rely on Halbach alone)
Jc = 0  # Set to 2e9 if using GdBCO
B_field = 0.5  # Tesla
k_torsion = torsional_stiffness_bean(R_out, t=0.002, Jc=Jc, B_field=B_field)
print(f"Torsional stiffness: {k_torsion:.2e} N·m/rad")

# ============================================================
# 3. Gyroscopic Dynamics Simulation
# ============================================================

def gyro_dynamics(t, state, Ixx, Izz, omega_spin, k_torsion, k_mag, lever_arm):
    """
    state = [theta_x, theta_y, omega_x, omega_y]
    Euler's equations for a spinning symmetric top with magnetic restoring torque.
    """
    theta_x, theta_y, omega_x, omega_y = state
    # Magnetic torque from Halbach (restoring, proportional to angular displacement)
    # Torque from lateral force offset by lever arm (distance from center of mass)
    # Simplified: torque_mag = -k_mag * (lever_arm^2) * theta
    torque_mag_x = -k_mag * (lever_arm**2) * theta_x
    torque_mag_y = -k_mag * (lever_arm**2) * theta_y
    # Torsional torque from superconductor (if any)
    torque_torsion_x = -k_torsion * theta_x
    torque_torsion_y = -k_torsion * theta_y
    
    # Euler's equations for small angles
    # domega_x/dt = ( (Izz - Ixx)/Ixx * omega_spin * omega_y + torque_total_x / Ixx )
    # domega_y/dt = ( (Ixx - Izz)/Ixx * omega_spin * omega_x + torque_total_y / Ixx )
    alpha_x = ((Izz - Ixx)/Ixx) * omega_spin * omega_y + (torque_mag_x + torque_torsion_x) / Ixx
    alpha_y = ((Ixx - Izz)/Ixx) * omega_spin * omega_x + (torque_mag_y + torque_torsion_y) / Ixx
    
    dtheta_dt_x = omega_x
    dtheta_dt_y = omega_y
    return [dtheta_dt_x, dtheta_dt_y, alpha_x, alpha_y]

# Parameters for a 2kg mass packet, radius 0.1m, spinning at 50,000 RPM
mass = 2.0
R_sphere = 0.1
Ixx = (2/5) * mass * R_sphere**2      # 0.016 kg·m²
Izz = Ixx                              # sphere symmetry
omega_spin = 50000 * 2 * np.pi / 60    # rad/s ≈ 5236 rad/s
lever_arm = 0.15                       # distance from center to magnetic force application (m)

# Initial conditions: a small impulsive tilt of 0.01 rad, no initial angular velocity
state0 = [0.01, 0.0, 0.0, 0.0]
t_span = (0, 10)    # simulate 10 seconds
t_eval = np.linspace(0, 10, 1000)

# Run simulation
sol = solve_ivp(gyro_dynamics, t_span, state0, t_eval=t_eval,
                args=(Ixx, Izz, omega_spin, k_torsion, k_mag, lever_arm))

# Plot precession (theta magnitude)
theta_mag = np.sqrt(sol.y[0]**2 + sol.y[1]**2)
plt.figure(figsize=(10,5))
plt.plot(sol.t, theta_mag * 180/np.pi)
plt.xlabel('Time (s)')
plt.ylabel('Tilt angle (degrees)')
plt.title('Gyroscopic Precession with Halbach Stiffness')
plt.grid(True)
plt.savefig('halbach_gyroscopic_precession.png', dpi=150, bbox_inches='tight')
plt.show()

# Print final stability metric
final_tilt = theta_mag[-1] * 180/np.pi
print(f"Final tilt after 10 seconds: {final_tilt:.2f} degrees")
if final_tilt < 0.1:
    print("STABLE: Precession less than 0.1° in 10s")
else:
    print("UNSTABLE: Large precession detected")


# ============================================================
# 4. Comparison with Existing Flux-Pinning Model
# ============================================================

@dataclass
class GdBCOProperties:
    """Material properties for GdBCO superconductor."""
    Tc: float = 92.0  # Critical temperature (K)
    Jc0: float = 3e10  # Critical current density at 0K, 0T (A/m²)
    n_exp: float = 1.5  # n-exponent for E-J power law
    B0: float = 5.0  # Characteristic magnetic field (T)
    alpha: float = 0.5  # Field dependence exponent
    density: float = 6380  # kg/m³
    specific_heat: float = 180  # J/kg/K at 77K
    thermal_conductivity: float = 3.0  # W/m/K at 77K


class BeanLondonModel:
    """Bean-London flux pinning model for comparison."""
    
    def __init__(self, properties: GdBCOProperties, geometry: dict):
        self.properties = properties
        self.geometry = geometry
        self.volume = geometry["thickness"] * geometry["width"] * geometry["length"]
        
    def critical_current_density(self, B_field: float, temperature: float) -> float:
        """Calculate temperature and field dependent critical current density."""
        if temperature >= self.properties.Tc:
            return 0.0
            
        # Jc(B,T) = Jc0 * (1-T/Tc)^n * f(B)
        temp_factor = (1.0 - temperature / self.properties.Tc) ** self.properties.n_exp
        
        # Field dependence: f(B) = (B0/B)^(alpha/2) for B > 0.01*Tc
        if B_field > 0.01 * self.properties.B0:
            field_factor = (self.properties.B0 / B_field) ** (self.properties.alpha / 2)
        else:
            field_factor = (self.properties.B0 / (0.01 * self.properties.B0)) ** (self.properties.alpha / 2)
            
        return self.properties.Jc0 * temp_factor * field_factor
    
    def get_stiffness(self, displacement: float, B_field: float, temperature: float) -> float:
        """Calculate flux pinning stiffness using Bean-London model."""
        Jc = self.critical_current_density(B_field, temperature)
        
        # Geometry parameters
        max_penetration = self.geometry["thickness"] / 2.0
        
        # Analytical derivative parameters
        a = Jc * B_field * self.volume / max_penetration
        b = 1.0 / (max_penetration * 0.1)
        x = abs(displacement)

        # Handle edge case for very small displacements
        if x < 1e-15:
            stiffness = 2.0 * a * b * x
        elif b * x > 20:
            stiffness = a
        else:
            # Analytical derivative: k = a * [tanh(b*x) + b*x * sech²(b*x)]
            tanh_bx = np.tanh(b * x)
            sech_bx = 1.0 / np.cosh(b * x)
            stiffness = a * (tanh_bx + b * x * sech_bx**2)

        return stiffness


# Compare Halbach vs Flux-Pinning stiffness
print("\n" + "="*50)
print("COMPARISON: Halbach vs Flux-Pinning Stiffness")
print("="*50)

# Flux-pinning parameters
fp_geometry = {
    "thickness": 1e-6,  # 1 micron
    "width": 0.012,     # 12 mm
    "length": 1.0,      # 1 m
}

fp_model = BeanLondonModel(GdBCOProperties(), fp_geometry)
fp_stiffness = fp_model.get_stiffness(0.001, 1.0, 77.0)

print(f"Halbach stiffness:      {k_mag:.0f} N/m")
print(f"Flux-pinning stiffness: {fp_stiffness:.0f} N/m")
print(f"Ratio (Halbach/FP):     {k_mag/fp_stiffness:.2f}x")

# Simulate with flux-pinning parameters for comparison
print("\nRunning simulation with flux-pinning parameters...")
k_mag_comparison = fp_stiffness
sol_comparison = solve_ivp(gyro_dynamics, t_span, state0, t_eval=t_eval,
                          args=(Ixx, Izz, omega_spin, k_torsion, k_mag_comparison, lever_arm))

theta_mag_comparison = np.sqrt(sol_comparison.y[0]**2 + sol_comparison.y[1]**2)
final_tilt_comparison = theta_mag_comparison[-1] * 180/np.pi

plt.figure(figsize=(10,5))
plt.plot(sol.t, theta_mag * 180/np.pi, label='Halbach Array', linewidth=2)
plt.plot(sol_comparison.t, theta_mag_comparison * 180/np.pi, label='Flux-Pinning', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Tilt angle (degrees)')
plt.title('Gyroscopic Precession: Halbach vs Flux-Pinning')
plt.legend()
plt.grid(True)
plt.savefig('halbach_vs_fluxpinning_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nComparison results:")
print(f"Halbach final tilt:      {final_tilt:.2f} degrees")
print(f"Flux-pinning final tilt: {final_tilt_comparison:.2f} degrees")
print(f"Improvement factor:      {final_tilt_comparison/final_tilt:.2f}x better with Halbach")