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

from .alternatives_comparison import (
    StationKeepingSystem,
    compare_alternatives,
    format_comparison_table,
    generate_comparison_report,
)
from .gyro_matrix import gyroscopic_coupling, skew_symmetric
from .heritage_scaling import HeritageScalingConfig
from .mobile_station import (
    MobileStationConfig,
    MobileStationState,
    compute_energy_exchange,
    compute_mobile_station_force,
    simulate_mobile_station_trajectory,
)
from .multi_body import EventQueue, MultiBodyStream, Packet, PacketState, SNode
from .packet_budget import (
    PacketBudget,
    compute_packet_budget,
    compute_replacement_schedule,
    estimate_slingshot_pipeline_capacity,
)
from .rigid_body import RigidBody, euler_equations, quaternion_derivative
from .stiffness_verification import (
    StiffnessMetrics,
    calculate_effective_stiffness,
    get_stiffness_alert_level,
    sweep_stiffness_velocity,
    verify_anchor_stiffness,
    verify_stiffness_constraint,
)
from .stream_energy_model import (
    StreamEnergyBudget,
    analytical_lunar_slingshot_dv,
    compute_multi_cycle_slingshot_dv,
    compute_stream_energy_budget,
)
from .stress_monitoring import (
    StressMetrics,
    calculate_centrifugal_stress,
    get_stress_alert_level,
    verify_packet_stress,
    verify_stress_constraint,
)

try:
    from .coil_switching import (
        DEFAULT_COIL_SPECS,
        CoilSpecs,
        CoilSwitchingModel,
        SwitchingEvent,
        create_pulsed_switching_event,
    )
    _COIL_SWITCHING_AVAILABLE = True
except ImportError:
    _COIL_SWITCHING_AVAILABLE = False
    CoilSpecs = None
    SwitchingEvent = None
    CoilSwitchingModel = None
    DEFAULT_COIL_SPECS = None
    create_pulsed_switching_event = None

try:
    from .mutual_inductance import (
        CoilGeometry,
        MutualInductanceModel,
        MutualInductanceResult,
        create_circular_coil,
    )
    _MUTUAL_INDUCTANCE_AVAILABLE = True
except ImportError:
    _MUTUAL_INDUCTANCE_AVAILABLE = False
    CoilGeometry = None
    MutualInductanceResult = None
    MutualInductanceModel = None
    create_circular_coil = None

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
    "HeritageScalingConfig",
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
    # Stream sustainability models
    "StreamEnergyBudget",
    "compute_stream_energy_budget",
    "analytical_lunar_slingshot_dv",
    "compute_multi_cycle_slingshot_dv",
    "PacketBudget",
    "compute_packet_budget",
    "compute_replacement_schedule",
    "estimate_slingshot_pipeline_capacity",
    "MobileStationState",
    "MobileStationConfig",
    "compute_mobile_station_force",
    "simulate_mobile_station_trajectory",
    "compute_energy_exchange",
    "StationKeepingSystem",
    "compare_alternatives",
    "format_comparison_table",
    "generate_comparison_report",
    # Optional modules
    "JAXThermalModel",
    "CoilSpecs",
    "SwitchingEvent",
    "CoilSwitchingModel",
    "DEFAULT_COIL_SPECS",
    "create_pulsed_switching_event",
    "CoilGeometry",
    "MutualInductanceResult",
    "MutualInductanceModel",
    "create_circular_coil",
]
