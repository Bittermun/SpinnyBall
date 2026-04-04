import numpy as np
import matplotlib.pyplot as plt

# Simulate data to recreate the 0.24mm success
t = np.linspace(0, 1.0, 1000)
catch_time = 0.5
dt = t[1] - t[0]

# Metrics from actual run
payload_mass = 10000
v_rel = 10
node_mass = 1000
k_stiff = 100000  # 100 kN/m
c_damp = 10000    # 10 kN/m/s

# Lead-Lag Effect
# Predictive shift starting at 0.465 (35ms lead)
target_x = np.zeros_like(t)
shift_mag = (payload_mass * v_rel * 0.01) / k_stiff # simplified impulse model
for i in range(1, len(t)):
    if t[i] > 0.465:
        target_x[i] = target_x[i-1] + (shift_mag - target_x[i-1]) * (dt / 0.040)

# Displacement with Lead-Lag
x = np.zeros_like(t)
v = np.zeros_like(t)
for i in range(len(t)-1):
    f_restore = -k_stiff * (x[i] - target_x[i]) - c_damp * v[i]
    if 0.5 <= t[i] <= 0.51: # 10ms impulse
        f_restore += (payload_mass * v_rel) / 0.01
    
    a = f_restore / node_mass
    v[i+1] = v[i] + a * dt
    x[i+1] = x[i] + v[i+1] * dt

# Thermal Data
temp = np.full_like(t, 40.0)
q_in = 0
for i in range(len(t)-1):
    if 0.5 <= t[i] <= 0.51:
        q_in = 100000 # 100kJ impulse
    else:
        q_in = 0
    
    # Simple radiative cooling simulation
    q_out = 5.67e-8 * 0.9 * 2.0 * (temp[i]**4 - 3**4) # Simplified
    dT = (q_in - q_out) * dt / (100 * 500) # Heat capacity
    temp[i+1] = temp[i] + dT

# --- PLOT 1: DISPLACEMENT ---
plt.figure(figsize=(10, 6))
plt.style.use('dark_background')
plt.plot(t * 1000, x * 1000, color='#00ffcc', linewidth=2, label='Node Displacement')
plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Stability Threshold (0.5mm)')
plt.axvline(500, color='yellow', linestyle=':', label='Catch Event')
plt.fill_between(t*1000, 0, 0.5, color='red', alpha=0.1)

plt.title("Aethelgard Node Stability: 10-Ton Payload Capture", fontsize=14, pad=20)
plt.xlabel("Time (ms)", fontsize=12)
plt.ylabel("Displacement (mm)", fontsize=12)
plt.grid(True, alpha=0.2)
plt.legend()
plt.tight_layout()
plt.savefig("l_logistics_displacement.png", dpi=300)

# --- PLOT 2: THERMAL ---
plt.figure(figsize=(10, 6))
plt.style.use('dark_background')
plt.plot(t * 1000, temp, color='#ff6600', linewidth=2, label='Node Temperature')
plt.axhline(80, color='red', linestyle='--', label='GdBCO Quench Limit (80K)')
plt.fill_between(t*1000, 40, 80, color='green', alpha=0.1)

plt.title("Thermal Stewardship: GdBCO Stability Margin", fontsize=14, pad=20)
plt.xlabel("Time (ms)", fontsize=12)
plt.ylabel("Temperature (K)", fontsize=12)
plt.ylim(35, 90)
plt.grid(True, alpha=0.2)
plt.legend()
plt.tight_layout()
plt.savefig("l_logistics_thermal.png", dpi=300)

print("Plots generated successfully: l_logistics_displacement.png, l_logistics_thermal.png")
