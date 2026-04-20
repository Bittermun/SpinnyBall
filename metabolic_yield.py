"""
Metabolic Yield (ROI Mapping)
Maps Kinetic Logistics Flux to Sovereign Cognition Points (CP).
"""

# Parameters from lob_scaling.py
N_nodes = 40
payload_mass = 10000  # 10 tons (kg)
stream_velocity = 10  # m/s
cadence_hz = 83.3     # 1 packet every 12ms

# Metabolic Constants
# 1 CP = 1e6 kg*m/s total delivered flux
CP_CONVERSION = 1e6 

def calculate_yield():
    # Momentum per "hit" (kg*m/s)
    dp_per_hit = payload_mass * stream_velocity
    
    # Global flux (N-node aggregate hits per second)
    total_flux_per_node = dp_per_hit * cadence_hz
    global_system_flux = total_flux_per_node * N_nodes
    
    # CP Yield per hour
    cp_per_hour = (global_system_flux * 3600) / CP_CONVERSION
    
    print(f"--- METABOLIC YIELD (N={N_nodes}) ---")
    print(f"Payload: {payload_mass/1000:.1f} Tons @ {stream_velocity} m/s")
    print(f"Global Flux: {global_system_flux:,.0f} kg*m/s^2")
    print(f"Metabolic Yield: {cp_per_hour:,.2f} CP / hour")
    print("------------------------------------------")

if __name__ == "__main__":
    calculate_yield()
