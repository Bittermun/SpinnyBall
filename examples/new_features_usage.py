"""
Usage examples for new SpinnyBall features.

This file demonstrates how to use the newly implemented features:
1. Eddy-current drag and thermal-energy closure
2. Discrete-time MPC delay compensation
3. Multi-pass Δvx accumulation analysis
"""

import numpy as np
from dynamics.thermal_model import update_temperature_euler, eddy_heating_power
from dynamics.lumped_thermal import LumpedThermalModel, LumpedThermalParams
from dynamics.cryocooler_model import CryocoolerModel, DEFAULT_CRYOCOOLER_SPECS
from control_layer.mpc_controller import create_mpc_controller, ConfigurationMode
from sgms_v1 import simulate_multi_pass_accumulation, DEFAULT_PARAMS


# ============================================================================
# Example 1: Eddy-Current Drag and Thermal-Energy Closure
# ============================================================================

def example_eddy_heating():
    """Calculate eddy-current heating from velocity-dependent drag."""
    # Parameters
    k_drag = 0.01  # N·s/m (eddy-current drag coefficient)
    velocity = 1600.0  # m/s (operational velocity)
    radius = 0.1  # m (packet radius)
    
    # Calculate eddy heating power
    power = eddy_heating_power(velocity, k_drag, radius)
    print(f"Eddy heating power at {velocity} m/s: {power:.2f} W")
    
    # P = k_drag * v^2, so at 1600 m/s: P = 0.01 * 1600^2 = 25.6 kW
    assert abs(power - 25600.0) < 1.0


def example_thermal_with_eddy_heating():
    """Update packet temperature with eddy-current heating."""
    # Initial conditions
    temperature = 77.0  # K (operating temperature)
    mass = 0.05  # kg
    radius = 0.1  # m
    emissivity = 0.1
    specific_heat = 500.0  # J/kg/K
    dt = 0.01  # s
    
    # Eddy heating at 1600 m/s
    k_drag = 0.01
    velocity = 1600.0
    eddy_power = eddy_heating_power(velocity, k_drag, radius)
    
    # Update temperature with eddy heating
    new_temp = update_temperature_euler(
        temperature=temperature,
        mass=mass,
        radius=radius,
        emissivity=emissivity,
        specific_heat=specific_heat,
        dt=dt,
        eddy_heating_power=eddy_power,
    )
    
    print(f"Temperature increase due to eddy heating: {new_temp - temperature:.4f} K")


def example_lumped_thermal_with_cryocooler():
    """Use lumped thermal model with cryocooler cooling."""
    # Create cryocooler model for temperature-dependent cooling
    cryo_model = CryocoolerModel(DEFAULT_CRYOCOOLER_SPECS)
    
    # Configure thermal parameters with cryocooler
    thermal_params = LumpedThermalParams(
        enable_cryocooler=True,
        cryocooler_model=cryo_model,  # Use temperature-dependent cooling
        initial_temp=77.0,
    )
    
    # Create thermal model
    thermal_model = LumpedThermalModel(thermal_params, dt=0.01)
    
    # Apply heat input (e.g., from eddy heating)
    heat_input = 10.0  # W
    result = thermal_model.step({'stator': heat_input, 'rotor': 0.0})
    
    print(f"Stator temperature after heating: {result['T_stator']:.2f} K")
    print(f"Rotor temperature: {result['T_rotor']:.2f} K")


def example_lumped_thermal_constant_cryocooler():
    """Use lumped thermal model with constant cryocooler power (fallback)."""
    # Configure with constant cooling power (simpler, no CryocoolerModel needed)
    thermal_params = LumpedThermalParams(
        enable_cryocooler=True,
        cryocooler_cooling_power=5.0,  # W (constant)
        cryocooler_model=None,  # Use fallback
        initial_temp=77.0,
    )
    
    thermal_model = LumpedThermalModel(thermal_params, dt=0.01)
    result = thermal_model.step({'stator': 10.0, 'rotor': 0.0})
    
    print(f"Stator temperature (constant cryocooler): {result['T_stator']:.2f} K")


# ============================================================================
# Example 2: Discrete-Time MPC Delay Compensation
# ============================================================================

def example_mpc_discrete_time_delay():
    """Use MPC with discrete-time sampling and communication delay."""
    # Create MPC controller with discrete-time delay
    controller = create_mpc_controller(
        configuration_mode=ConfigurationMode.TEST,
        sampling_period=0.01,  # 10 ms sampling
        communication_delay=0.005,  # 5 ms communication delay
        enable_discrete_time=True,
        delay_compensation_mode='discrete_time',
    )
    
    # Initial state [qx, qy, qz, qw, ωx, ωy, ωz]
    x0 = np.array([0.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0])
    x_target = np.zeros(7)
    
    # Solve MPC problem
    u_opt, info = controller.solve(x0, x_target)
    
    print(f"MPC solve time: {info['solve_time']*1000:.2f} ms")
    print(f"Success: {info['success']}")


def example_mpc_smith_predictor():
    """Use MPC with Smith predictor for delay compensation."""
    controller = create_mpc_controller(
        configuration_mode=ConfigurationMode.TEST,
        delay_steps=5,  # Compensate for 5 control cycles
        dt_delay=0.01,
        enable_delay_compensation=True,
        delay_compensation_mode='smith',
    )
    
    x0 = np.array([0.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0])
    x_target = np.zeros(7)
    
    u_opt, info = controller.solve(x0, x_target)
    
    print(f"Smith predictor delay steps: {info['delay_steps']}")


def example_mpc_both_compensation():
    """Use MPC with both discrete-time and Smith predictor (additive)."""
    controller = create_mpc_controller(
        configuration_mode=ConfigurationMode.TEST,
        sampling_period=0.01,
        communication_delay=0.005,
        enable_discrete_time=True,
        delay_steps=5,
        enable_delay_compensation=True,
        delay_compensation_mode='both',  # Apply both in sequence
    )
    
    x0 = np.array([0.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0])
    x_target = np.zeros(7)
    
    u_opt, info = controller.solve(x0, x_target)
    
    print(f"Both compensation modes enabled")


# ============================================================================
# Example 3: Multi-Pass Δvx Accumulation Analysis
# ============================================================================

def example_multi_pass_accumulation_small():
    """Analyze Δvx accumulation over small number of passes (fast)."""
    # Use small n_passes for quick testing
    results = simulate_multi_pass_accumulation(
        n_passes=100,
        params=DEFAULT_PARAMS,
        verbose=True,
    )
    
    print(f"\nMulti-pass accumulation results:")
    print(f"  Mean Δvx per pass: {results['mean_delta_vx']:.6f} m/s")
    print(f"  Std Δvx per pass: {results['std_delta_vx']:.6f} m/s")
    print(f"  Final cumulative Δvx: {results['final_cumulative']:.6f} m/s")
    print(f"  Drift rate: {results['drift_rate']:.9f} m/s/pass")
    print(f"  Error type: {results['error_type']}")
    print(f"  Failed passes: {results['failed_passes']}/{results['n_passes']}")


def example_multi_pass_accumulation_large():
    """Analyze Δvx accumulation over large number of passes (slow)."""
    # WARNING: This will take significant time
    # Use for production analysis, not quick testing
    results = simulate_multi_pass_accumulation(
        n_passes=10000,  # 10k passes (adjust as needed)
        params=DEFAULT_PARAMS,
        verbose=True,
    )
    
    print(f"\nLarge-scale accumulation results:")
    print(f"  Error type: {results['error_type']}")
    print(f"  Drift rate: {results['drift_rate']:.9f} m/s/pass")
    
    # Interpret error type
    if results['error_type'] == 'random_walk':
        print("  → Errors compound over time (random walk)")
    elif results['error_type'] == 'mean_reverting':
        print("  → Errors cancel over time (mean-reverting)")
    else:
        print("  → Insufficient variance to classify")


# ============================================================================
# Example 4: Combined Thermal + Drag + Control
# ============================================================================

def example_combined_system():
    """Combine thermal closure, eddy drag, and control delay."""
    # 1. Set up thermal model with cryocooler
    cryo_model = CryocoolerModel(DEFAULT_CRYOCOOLER_SPECS)
    thermal_params = LumpedThermalParams(
        enable_cryocooler=True,
        cryocooler_model=cryo_model,
        initial_temp=77.0,
    )
    thermal_model = LumpedThermalModel(thermal_params, dt=0.01)
    
    # 2. Calculate eddy heating at operational velocity
    k_drag = 0.01
    velocity = 1600.0
    eddy_power = eddy_heating_power(velocity, k_drag, radius=0.1)
    
    # 3. Apply eddy heating to thermal model
    result = thermal_model.step({'stator': eddy_power, 'rotor': 0.0})
    
    # 4. Set up MPC with delay compensation
    controller = create_mpc_controller(
        configuration_mode=ConfigurationMode.TEST,
        sampling_period=0.01,
        communication_delay=0.005,
        enable_discrete_time=True,
        delay_compensation_mode='both',
    )
    
    print(f"\nCombined system:")
    print(f"  Eddy heating power: {eddy_power:.2f} W")
    print(f"  Stator temperature: {result['T_stator']:.2f} K")
    print(f"  MPC delay compensation: both (discrete-time + Smith)")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SpinnyBall New Features Usage Examples")
    print("=" * 60)
    
    print("\n--- Example 1: Eddy-Current Drag and Thermal Closure ---")
    example_eddy_heating()
    example_thermal_with_eddy_heating()
    example_lumped_thermal_with_cryocooler()
    example_lumped_thermal_constant_cryocooler()
    
    print("\n--- Example 2: Discrete-Time MPC Delay Compensation ---")
    example_mpc_discrete_time_delay()
    example_mpc_smith_predictor()
    example_mpc_both_compensation()
    
    print("\n--- Example 3: Multi-Pass Δvx Accumulation ---")
    example_multi_pass_accumulation_small()
    # Uncomment for large-scale analysis (slow):
    # example_multi_pass_accumulation_large()
    
    print("\n--- Example 4: Combined System ---")
    example_combined_system()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
