import numpy as np
from scipy.optimize import fsolve

# Parameters
m_p = 0.1  # kg
k_u = 3000.0  # N/m (unstable stiffness)
# Let's find K_p and K_d that are standard, or find if there is a specific design.
# If K_p = 6000 N/m and K_d = 20 N-s/m, let's solve:
# m_p^2 * w^4 + (2 * m_p * k_u - K_d^2) * w^2 + (k_u^2 - K_p^2) = 0
K_p = 6000.0
K_d = 20.0

a = m_p**2
b = 2.0 * m_p * k_u - K_d**2
c = k_u**2 - K_p**2

# Solve for w^2:
roots = np.roots([a, b, c])
print("Roots for w^2:", roots)
for r in roots:
    if r > 0:
        w_c = np.sqrt(r)
        tau_c = (1.0 / w_c) * np.arctan2(K_d * w_c, K_p)
        print(f"Crossover w_c: {w_c:.4f} rad/s ({w_c/(2*np.pi):.2f} Hz)")
        print(f"Critical delay tau_c: {tau_c * 1e6:.4f} microseconds")
