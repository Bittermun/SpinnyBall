import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import argparse

# --- LOB CONFIGURATION ---
R_LUNAR = 1737.1e3
H_ORB = 100e3
C_BELT = 2 * np.pi * (R_LUNAR + H_ORB)

# Physical Constants (Aethelgard Hardened Profile)
U_STREAM = 1600.0  
LAMBDA_STREAM = 0.5 
K_PINNING_BASE = 100000.0 
C_DAMPING = 10000.0  
NODE_MASS = 1000.0   

# --- DYNAMIC SIMULATOR ---
class LOBNetwork:
    def __init__(self, n, fail_node=None):
        self.n = n
        self.L_SPAN = C_BELT / n
        self.fail_node = fail_node
        
    def get_derivatives(self, t, state, p_mag):
        dydt = np.zeros_like(state)
        
        # Impulse Profile (Node 0)
        impulse_start = 0.1
        impulse_end = 0.12
        f_payload_active = 0
        if impulse_start <= t <= impulse_end:
            width = impulse_end - impulse_start
            amp = (p_mag * np.pi) / (2 * width)
            f_payload_active = amp * np.sin(np.pi * (t - impulse_start) / width)
            
        for j in range(self.n):
            idx = j * 2
            xj, vj = state[idx], state[idx+1]
            
            # --- ACTIVE CONTROL (Fails if Blackout) ---
            k_local = K_PINNING_BASE
            if j == self.fail_node and t > 0.1: # Fail at 100ms
                k_local = 0.0 # Total GdBCO quench
                f_ff = 0.0
            else:
                f_ff = -f_payload_active if j == 0 else 0.0
                
            # Local Restoring Force
            f_local = -k_local * xj - C_DAMPING * vj + f_ff
            
            # Lattice Coupling (TENSION ONLY)
            prev_node = (j - 1) % self.n
            next_node = (j + 1) % self.n
            x_prev = state[prev_node * 2]
            x_next = state[next_node * 2]
            
            tension = LAMBDA_STREAM * (U_STREAM**2)
            f_lattice = (tension / self.L_SPAN) * (x_prev + x_next - 2*xj)
            
            # Application
            f_total = f_local + f_lattice
            if j == 0:
                f_total += f_payload_active
                
            dydt[idx] = vj
            dydt[idx+1] = f_total / NODE_MASS
            
        return dydt

def run_lob_simulation(nodes, fail_node=None):
    network = LOBNetwork(n=nodes, fail_node=fail_node)
    p_mag = 10000.0 * 10.0 # 100k N-s
    
    t_span = (0, 1.0) # Longer for failure drift
    t_eval = np.linspace(0, 1.0, 1000)
    y0 = np.zeros(nodes * 2)
    
    sol = solve_ivp(network.get_derivatives, t_span, y0, t_eval=t_eval, 
                   args=(p_mag,), rtol=1e-6, atol=1e-9, max_step=0.001)
    return sol

# --- AUDIT MODES ---
def perform_audit():
    print("--- LOB PHASE 18: SOVEREIGN AUDIT ---")
    
    # 1. N=20 Baseline
    sol20 = run_lob_simulation(20)
    max20 = np.max(np.abs(sol20.y[::2,:]))*1000
    
    # 2. N=40 Density Check
    sol40 = run_lob_simulation(40)
    max40 = np.max(np.abs(sol40.y[::2,:]))*1000
    
    # 3. Node 0 Blackout Fail
    sol_fail = run_lob_simulation(20, fail_node=0)
    max_fail = np.max(np.abs(sol_fail.y[::2,:]))*1000

    print(f"N=20 Stiffness: {max20:.6f} mm peak jitter")
    print(f"N=40 Stiffness: {max40:.6f} mm peak jitter")
    print(f"Node Blackout:  {max_fail:.2f} mm drift @ 1.0s")
    
    # Visualization: Failure Drift
    plt.figure(figsize=(10, 6))
    plt.style.use('dark_background')
    plt.plot(sol_fail.t * 1000, sol_fail.y[0]*1000, color='red', label='Node 0 (FAILED)')
    plt.plot(sol_fail.t * 1000, sol_fail.y[2]*1000, color='#00ffcc', linestyle='--', label='Node 1 (Lattice Tension Support)')
    plt.axhline(5, color='orange', linestyle=':', label='Fail-Safe Threshold (5mm)')
    
    plt.title("Aethelgard Survivability: Node 0 Blackout Event", fontsize=14)
    plt.xlabel("Time (ms)", fontsize=12)
    plt.ylabel("Displacement (mm)", fontsize=12)
    plt.grid(True, alpha=0.1)
    plt.legend()
    plt.tight_layout()
    plt.savefig("lob_survivability_blackout.png", dpi=300)
    print("Audit Artifact Generated: lob_survivability_blackout.png")

if __name__ == "__main__":
    perform_audit()
