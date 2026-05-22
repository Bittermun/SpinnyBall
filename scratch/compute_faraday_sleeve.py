import numpy as np

# Constants
mu_0 = 4.0 * np.pi * 1e-7
sigma = 1.5e8  # S/m
f = 31250.0
omega = 2.0 * np.pi * f
B_tilde = 0.05  # T
L = 1500.0  # m
r_s = 0.003  # m (sleeve radius, 3mm)
d_sleeve = 0.001  # m (1.0 mm)

delta = np.sqrt(2.0 / (omega * mu_0 * sigma))
R_s = 1.0 / (sigma * delta)
H_tilde = B_tilde / mu_0

# Power per unit area using skin effect:
P_per_area = 0.5 * R_s * (H_tilde**2)
A_sleeve = 2.0 * np.pi * r_s * L
P_eddy_skin = P_per_area * A_sleeve

print(f"Skin depth delta: {delta * 1e6:.4f} micrometers")
print(f"Surface resistance R_s: {R_s:.6e} Ohms")
print(f"Peak magnetic field intensity H_tilde: {H_tilde:.2f} A/m")
print(f"Power per unit area: {P_per_area:.4f} W/m^2")
print(f"Faraday sleeve area: {A_sleeve:.4f} m^2")
print(f"Total skin-effect eddy current power: {P_eddy_skin/1e3:.4f} kW")

# Radiator area calculation at 120 K
# We reject heat passively to space at 120 K.
# Stator is in deep space where background temperature is T_space = 2.7 K or let's say T_sink = 4 K.
# The radiative heat flux rejected by a blackbody radiator (or a high-emissivity coating with epsilon = 0.9) at T_rad = 120 K is:
# q_rad = epsilon * sigma_sb * (T_rad^4 - T_sink^4)
# Let's compute this:
sigma_sb = 5.670374e-8
epsilon_rad = 0.9
T_rad = 120.0
T_sink = 4.0
q_rad = epsilon_rad * sigma_sb * (T_rad**4 - T_sink**4)
A_rad = P_eddy_skin / q_rad
print(f"Radiatively rejected flux at 120 K: {q_rad:.4f} W/m^2")
print(f"Required passive radiator area: {A_rad:.4f} m^2")

# Active cooling exergy penalty (cryocooler wall-plug power):
# Operating temperature T_cold = 120 K, rejection temperature T_hot = 300 K.
# Carnot COP: COP_C = 120 / (300 - 120) = 120 / 180 = 0.6667
# Space cryocooler at 120 K operates at ~20% of Carnot efficiency:
# COP_actual = 0.20 * 0.6667 = 0.1333
# Wall-plug power: P_wp = P_eddy_skin / COP_actual
cop_c_120 = 120.0 / (300.0 - 120.0)
cop_actual_120 = 0.20 * cop_c_120
P_wp_120 = P_eddy_skin / cop_actual_120
print(f"Carnot COP at 120 K: {cop_c_120:.4f}")
print(f"Actual COP at 120 K: {cop_actual_120:.4f}")
print(f"Wall-plug power for active cooling: {P_wp_120/1e6:.4f} MW")
