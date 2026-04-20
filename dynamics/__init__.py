"""
Dynamics core for gyroscopic mass-stream simulation.

This module implements full 3D rigid-body dynamics with explicit
gyroscopic coupling for Sovereign Bean (spin-stabilized magnetic packet)
simulation.

Conventions:
- Quaternion: scalar-last [x, y, z, w] for scipy Rotation compatibility
- Angular velocity: [ωx, ωy, ωz] in body frame (rad/s)
- State vector: [qx, qy, qz, qw, ωx, ωy, ωz] (7 elements)
"""

from .rigid_body import RigidBody, euler_equations, quaternion_derivative
from .gyro_matrix import skew_symmetric, gyroscopic_coupling
from .multi_body import MultiBodyStream, Packet, SNode, PacketState, EventQueue
from .stress_monitoring import (
    calculate_centrifugal_stress,
    verify_stress_constraint,
    verify_packet_stress,
    StressMetrics,
    get_stress_alert_level,
)
from .stiffness_verification import (
    calculate_effective_stiffness,
    verify_stiffness_constraint,
    verify_anchor_stiffness,
    StiffnessMetrics,
    get_stiffness_alert_level,
    sweep_stiffness_velocity,
)

try:
    from .jax_thermal import JAXThermalModel
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False
    JAXThermalModel = None

# EDT (Electrodynamic Tethers) archived - see archived_edt/ directory

__all__ = [
    "RigidBody",
    "euler_equations",
    "quaternion_derivative",
    "skew_symmetric",
    "gyroscopic_coupling",
    "MultiBodyStream",
    "Packet",
    "SNode",
    "PacketState",
    "EventQueue",
    "calculate_centrifugal_stress",
    "verify_stress_constraint",
    "verify_packet_stress",
    "StressMetrics",
    "get_stress_alert_level",
    "calculate_effective_stiffness",
    "verify_stiffness_constraint",
    "verify_anchor_stiffness",
    "StiffnessMetrics",
    "get_stiffness_alert_level",
    "sweep_stiffness_velocity",
    "JAXThermalModel",
]
