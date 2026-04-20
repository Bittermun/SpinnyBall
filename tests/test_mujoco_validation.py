"""
MuJoCo 6-DoF validation tests.

Cross-validates rigid-body dynamics against MuJoCo physics oracle
using trajectory evolution instead of instantaneous force comparison.
"""

import numpy as np
import pytest
from dynamics.rigid_body import RigidBody
from dynamics.gyro_matrix import skew_symmetric
import logging

logger = logging.getLogger(__name__)

try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    logger.warning("MuJoCo not available, skipping validation tests")

@pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not installed")
def test_trajectory_cross_validation():
    """
    Cross-validate trajectory evolution between our implementation and MuJoCo.
    
    Simulates same initial conditions in both systems and compares final states.
    """
    # Initial conditions
    mass = 10.0
    I = np.diag([0.1, 0.1, 0.2])
    q0 = np.array([0.0, 0.0, 0.0, 1.0])  # Quaternion (identity)
    omega0 = np.array([0.0, 0.0, 50.0])  # Spin about z-axis (rad/s)
    
    # Our implementation
    body = RigidBody(mass=mass, I=I, quaternion=q0, angular_velocity=omega0)
    
    # Simulate our implementation
    t_span = (0.0, 0.1)
    dt = 0.001
    n_steps = int(t_span[1] / dt)
    
    def torques(t, state):
        return np.zeros(3)  # Zero external torques
    
    result_our = body.integrate(
        t_span=t_span,
        torques=torques,
        method='RK45',
        rtol=1e-8,
        atol=1e-10,
        max_step=dt,
    )
    
    # MuJoCo model (same parameters)
    model = mujoco.MjModel.from_xml_string(f"""
        <mujoco>
            <worldbody>
                <body name="rotor" pos="0 0 0">
                    <inertial mass="{mass}" diaginertia="0.1 0.1 0.2"/>
                </body>
            </worldbody>
        </mujoco>
    """)
    data = mujoco.MjData(model)
    
    # Set initial conditions
    data.qpos[3:7] = q0  # Quaternion
    data.qvel[:3] = omega0  # Angular velocity
    
    # Simulate MuJoCo
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
    
    # Compare final states
    q_our = result_our['state'][-1, :4]
    omega_our = result_our['state'][-1, 4:]
    q_mujoco = data.qpos[3:7]
    omega_mujoco = data.qvel[:3]
    
    # Quaternion comparison (angular distance)
    def quaternion_angular_distance(q1, q2):
        """Compute angular distance between quaternions."""
        return 2 * np.arccos(np.clip(np.abs(np.dot(q1, q2)), 0, 1))
    
    angular_dist = quaternion_angular_distance(q_our, q_mujoco)
    omega_diff = np.linalg.norm(omega_our - omega_mujoco)
    
    # Tolerances
    assert angular_dist < 1e-2, f"Quaternion mismatch: {angular_dist}"
    assert omega_diff < 1e-1, f"Angular velocity mismatch: {omega_diff}"
    
    logger.info(f"Trajectory cross-validation: PASS (angular_dist={angular_dist:.2e}, omega_diff={omega_diff:.2e})")

@pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not installed")
def test_angular_momentum_conservation():
    """
    Test angular momentum conservation in both systems.
    
    Verifies both implementations conserve angular momentum
    to within numerical tolerance.
    """
    # Initial conditions
    mass = 10.0
    I = np.diag([0.1, 0.1, 0.2])
    omega0 = np.array([10.0, 5.0, 50.0])
    
    # Our implementation
    body = RigidBody(mass=mass, I=I, angular_velocity=omega0)
    L0_our = I @ omega0
    
    # Simulate our implementation
    t_span = (0.0, 1.0)
    def torques(t, state):
        return np.zeros(3)
    
    result_our = body.integrate(
        t_span=t_span,
        torques=torques,
        method='RK45',
        rtol=1e-8,
        atol=1e-10,
    )
    
    omega_final_our = result_our['state'][-1, 4:]
    L_final_our = I @ omega_final_our
    
    # Check conservation in our implementation
    assert np.allclose(L0_our, L_final_our, rtol=1e-6, atol=1e-9), \
        f"Our implementation: L not conserved: L0={L0_our}, L_final={L_final_our}"
    
    # MuJoCo model
    model = mujoco.MjModel.from_xml_string(f"""
        <mujoco>
            <worldbody>
                <body name="rotor">
                    <inertial mass="{mass}" diaginertia="0.1 0.1 0.2"/>
                </body>
            </worldbody>
        </mujoco>
    """)
    data = mujoco.MjData(model)
    data.qvel[:3] = omega0
    
    # Simulate MuJoCo
    dt = 0.001
    n_steps = int(t_span[1] / dt)
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
    
    omega_final_mujoco = data.qvel[:3]
    L_final_mujoco = I @ omega_final_mujoco
    
    # Check conservation in MuJoCo
    assert np.allclose(L0_our, L_final_mujoco, rtol=1e-6, atol=1e-9), \
        f"MuJoCo: L not conserved: L0={L0_our}, L_final={L_final_mujoco}"
    
    # Compare final states
    omega_diff = np.linalg.norm(omega_final_our - omega_final_mujoco)
    assert omega_diff < 1e-1, f"Final omega mismatch: {omega_diff}"
    
    logger.info("Angular momentum conservation validation: PASS")

def test_rigid_body_dynamics_standalone():
    """
    Test rigid-body dynamics without MuJoCo (always runs).
    
    Verifies basic properties of our implementation.
    """
    body = RigidBody(
        mass=10.0,
        I=np.diag([0.1, 0.1, 0.2]),
    )
    
    # Test inertia symmetry
    assert np.allclose(body.I, body.I.T), "Inertia matrix not symmetric"
    
    # Test positive definiteness
    eigenvals = np.linalg.eigvals(body.I)
    assert np.all(eigenvals > 0), "Inertia matrix not positive definite"
    
    # Test skew-symmetric matrix property
    omega = np.array([1.0, 2.0, 3.0])
    skew = skew_symmetric(omega)
    assert np.allclose(skew, -skew.T), "Skew-symmetric matrix property violated"
    
    # Test gyroscopic coupling produces torque (cross product property)
    I_omega = body.I @ omega
    gyro_torque = skew_symmetric(omega) @ I_omega
    # Torque should be perpendicular to both omega and I_omega
    assert np.abs(np.dot(gyro_torque, omega)) < 1e-10, "Torque not perpendicular to omega"
    assert np.abs(np.dot(gyro_torque, I_omega)) < 1e-10, "Torque not perpendicular to I_omega"
    
    logger.info("Rigid-body dynamics standalone tests: PASS")
