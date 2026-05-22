import numpy as np

# Physical constants
mu_0 = 4.0 * np.pi * 1e-7  # H/m
eps_0 = 8.854e-12          # F/m
sigma_sb = 5.670374e-8     # W/(m^2 K^4) Stefan-Boltzmann
c = 299792458.0            # m/s

print("--- DOMAIN 2: Faraday Sleeve AC Eddy Current ---")
# Parameters
f = 31250.0  # Hz
omega = 2.0 * np.pi * f  # rad/s
sigma = 1.5e8  # S/m (copper at 120 K)
B_tilde = 0.05  # T
d = 1e-3  # m (sleeve thickness)
L = 1500.0  # m (channel length)
r_s = 0.1  # m (sleeve radius, typical stator bore is 0.1m, let's verify if there is an area or radius)

# Skin depth
delta = np.sqrt(2.0 / (omega * mu_0 * sigma))
print(f"Skin depth delta: {delta * 1e6:.4f} micrometers")

# If delta > d, the magnetic field fully penetrates, and we can use the thin sheet approximation
# The electric field induced in a cylinder or thin sheet:
# Let's derive the eddy current density.
# From Faraday's law, curl E = -dB/dt.
# For a thin-walled cylinder of radius r_s and thickness d, subjected to axial or transverse magnetic field:
# Let's assume a transverse magnetic field B(t) = B_tilde * sin(omega * t).
# The induced electric field in the sleeve:
# E = r_s * omega * B_tilde * cos(omega * t) / 2
# Volumetric heating power density: p_eddy = sigma * <E^2> = sigma * (r_s * omega * B_tilde)^2 / 8
# Total power P_eddy = p_eddy * Volume = [sigma * (r_s * omega * B_tilde)^2 / 8] * (2 * pi * r_s * d * L)
# Let's check this or another standard formulation:
# For a thin plate of thickness d in a magnetic field parallel to the plate or perpendicular:
# If the magnetic field fluctuation is perpendicular to the sleeve surface:
# For a thin sheet under perpendicular magnetic fluctuation:
# P_eddy_per_volume = (sigma * d^2 * omega^2 * B_tilde^2) / 24
# Let's calculate both and see what is physically appropriate.
# The guide-wire has a Faraday sleeve. YBCO guide-wire is protected from magnetic pulses.
# A Faraday sleeve is a cylinder surrounding the guide-wire. The guide-wire radius is small, say r_w = 2.0 mm.
# The Faraday sleeve radius r_s is also small, say r_s = 5.0 mm.
# Let's assume r_s = 5.0 mm (0.005 m).
r_s = 0.005  # 5 mm
volume_sleeve = 2.0 * np.pi * r_s * d * L

# Perpendicular thin sheet approximation:
# P_eddy = (sigma * d**2 * omega**2 * B_tilde**2 / 24) * volume_sleeve (for a flat plate, but let's check cylinder)
# For a cylinder in a transverse fluctuating magnetic field:
# P_eddy = (pi / 4) * sigma * d * r_s**3 * omega**2 * B_tilde**2 * L (let's verify the thin cylinder formula)
# Let's compute with thin plate parallel/perpendicular formulas.
p_vol_perp = (sigma * d**2 * omega**2 * B_tilde**2) / 24
P_eddy_perp = p_vol_perp * volume_sleeve

p_vol_cyl_trans = (sigma * r_s**2 * omega**2 * B_tilde**2) / 8
P_eddy_cyl_trans = p_vol_cyl_trans * volume_sleeve

print(f"Volume of Faraday sleeve: {volume_sleeve:.6f} m^3")
print(f"Thin sheet (d=1mm) perpendicular power dissipation: {P_eddy_perp:.4f} W")
print(f"Thin cylinder (r=5mm) transverse power dissipation: {P_eddy_cyl_trans:.4f} W")

print("\n--- DOMAIN 3: Dynamic Tension Winching ---")
# Parameters
T_0 = 1.0e6  # N (mean tension, say 1 MN)
T_tilde = 1.0e5  # N (dynamic tension fluctuation, say 100 kN)
f_d = 10.0  # Hz
omega_d = 2.0 * np.pi * f_d
L_tether = 10000.0  # m (10 km)
# Carbon Nanotube (CNT) properties:
# Density rho_cnt = 1400 kg/m^3
# Young's modulus E_cnt = 1e12 Pa (1 TPa)
# Area A_cnt: for a tension of 1 MN, let's say the operating stress is 10 GPa (CNT tensile strength is ~60 GPa).
# So A_cnt = T_0 / stress = 1e6 / 1e10 = 1e-4 m^2 (1 cm^2).
# Fiber mass: m_tether = rho_cnt * A_cnt * L_tether = 1400 * 1e-4 * 10000 = 1400 kg.
# Elastic strain: epsilon(t) = T(t) / (E_cnt * A_cnt)
# Mechanical power for modulation:
# T(t) = T_0 + T_tilde * sin(omega_d * t)
# Displacement: x(t) = L_tether * epsilon(t) = (L_tether / (E_cnt * A_cnt)) * T(t)
# Velocity: v(t) = dx/dt = (L_tether / (E_cnt * A_cnt)) * T_tilde * omega_d * cos(omega_d * t)
# Instantaneous mechanical power: P_mech(t) = T(t) * v(t)
# P_mech(t) = [T_0 + T_tilde * sin(omega_d * t)] * (L_tether / (E_cnt * A_cnt)) * T_tilde * omega_d * cos(omega_d * t)
# Over a cycle, the average of sin*cos is 0, and average of cos is 0. So conservative power is 0.
# But the peak mechanical power is:
# P_mech_peak: since T_tilde << T_0, P_mech_peak occurs when cos(omega_d * t) is max, which is:
# P_mech_peak = T_0 * (L_tether / (E_cnt * A_cnt)) * T_tilde * omega_d
# Let's compute this peak power:
E_cnt = 1.0e12  # Pa
A_cnt = 1.0e-4  # m^2
k_tether = (E_cnt * A_cnt) / L_tether  # N/m
print(f"Tether stiffness k_tether: {k_tether:.4e} N/m")
x_tilde = T_tilde / k_tether
print(f"Tension modulation stroke x_tilde: {x_tilde:.6f} m ({x_tilde*1000:.2f} mm)")
v_tilde = x_tilde * omega_d
print(f"Peak velocity v_tilde: {v_tilde:.6f} m/s")
P_mech_peak = T_0 * v_tilde
print(f"Peak mechanical power: {P_mech_peak/1e6:.4f} MW")

# Hysteretic loss: for CNTs, the loss tangent or specific damping capacity is psi.
# Let's say loss tangent tan_delta = 0.015 (typical for CNT fibers due to inter-tube sliding).
# The energy dissipated per cycle is: E_diss = pi * tan_delta * k_tether * x_tilde^2
# The continuous power dissipation is: P_diss = f_d * E_diss = pi * f_d * tan_delta * T_tilde^2 / k_tether
tan_delta = 0.015
P_diss = np.pi * f_d * tan_delta * (T_tilde**2) / k_tether
print(f"Hysteretic damping power loss: {P_diss:.4f} W")

# Winch electromechanical efficiency: let's say 85% motoring, 80% generating (or vice versa).
# The exergy loss in the winch per cycle:
# In motoring phase (T(t) increasing): winch does work.
# In generating phase (T(t) decreasing): winch regenerates, but with loss.
# Let's calculate the continuous electrical power overhead from winch round-trip inefficiency.
# Electrical power input: P_in = P_motoring / eta_motoring
# Regenerated power: P_out = P_generating * eta_generating
# Let's calculate the exact integral of P_winch_loss over one period.
# Power flow to tether: P(t) = T(t) * dx/dt = (T_0 + T_tilde * sin(w t)) * (T_tilde * w / k_tether) * cos(w t)
# Let's integrate P_loss(t):
# If P(t) > 0, winch is motoring: P_elec(t) = P(t) / eta_m
# If P(t) < 0, winch is generating: P_elec(t) = P(t) * eta_g (representing negative power returned)
# The average electrical power consumption is:
# P_elec_avg = (1 / T) * \int_0^T P_elec(t) dt
# Let's integrate this numerically!
eta_m = 0.85
eta_g = 0.80
t_pts = np.linspace(0, 2*np.pi/omega_d, 1000)
P_t = (T_0 + T_tilde * np.sin(omega_d * t_pts)) * (T_tilde * omega_d / k_tether) * np.cos(omega_d * t_pts)
P_elec_t = np.where(P_t > 0, P_t / eta_m, P_t * eta_g)
P_elec_avg = np.mean(P_elec_t)
print(f"Winch average electrical power overhead: {P_elec_avg/1e3:.4f} kW")

print("\n--- DOMAIN 4: Radiative Heat Transfer ---")
# Parameters
T_shield = 300.0  # K
T_cryo = 77.0  # K
A_stator = 9420.0  # m^2 (1.5 km stator)
# Emissivities: Beryllium/Tungsten composite shield has epsilon_1, HTS cryostat has epsilon_2.
# Let's use standard values or highly polished vacuum surfaces:
# For polished beryllium and tungsten, let's say epsilon_1 = 0.15.
# For polished gold or aluminum MLI (Multi-Layer Insulation) outer layer on the cryostat, let's say epsilon_2 = 0.05.
# The effective emissivity for concentric long cylinders is:
# epsilon_eff = 1 / (1/epsilon_1 + (A_1/A_2)*(1/epsilon_2 - 1))
# Since A_1 approx A_2, epsilon_eff = 1 / (1/epsilon_1 + 1/epsilon_2 - 1)
# Let's compute for epsilon_1 = 0.15, epsilon_2 = 0.05:
epsilon_1 = 0.15
epsilon_2 = 0.05
epsilon_eff = 1.0 / (1.0/epsilon_1 + 1.0/epsilon_2 - 1.0)
print(f"Effective emissivity (unshielded cryostat): {epsilon_eff:.4f}")

# Radiative heat leak: Q_leak = epsilon_eff * sigma_sb * A_stator * (T_shield^4 - T_cryo^4)
Q_leak = epsilon_eff * sigma_sb * A_stator * (T_shield**4 - T_cryo**4)
print(f"Radiative heat leak (no MLI): {Q_leak/1e3:.4f} kW")

# Let's check with an MLI blanket (say 30 layers).
# A 30-layer MLI blanket reduces the radiative heat transfer by a factor of ~ (N_layers + 1) = 31.
# Or let's assume a realistic effective emissivity with MLI: epsilon_eff = 0.002.
# Let's calculate both so we have a completely thorough, transparent analysis.
# For a pure unshielded model:
epsilon_eff_mli = 0.002
Q_leak_mli = epsilon_eff_mli * sigma_sb * A_stator * (T_shield**4 - T_cryo**4)
print(f"Radiative heat leak with MLI (eps_eff=0.002): {Q_leak_mli/1e3:.4f} kW")

# Wall-plug power calculation:
# Carnot COP: COP_C = T_cryo / (T_ambient - T_cryo) = 77 / (300 - 77) = 77 / 223 = 0.3453
# Actual COP = 15% of Carnot = 0.15 * 0.3453 = 0.0518
cop_actual = 0.15 * (77.0 / (300.0 - 77.0))
P_wp_no_mli = Q_leak / cop_actual
P_wp_mli = Q_leak_mli / cop_actual
print(f"Carnot COP at 77 K: {77.0 / (300.0 - 77.0):.4f}")
print(f"Actual COP at 77 K (15% Carnot): {cop_actual:.4f}")
print(f"Wall-plug power (no MLI): {P_wp_no_mli/1e6:.4f} MW")
print(f"Wall-plug power (with MLI): {P_wp_mli/1e6:.4f} MW")
