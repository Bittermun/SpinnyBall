"""
Multi-body packet stream dynamics for closed-loop mass-stream simulation.

Implements N-body packet dynamics with event-driven magnetic capture/release
at sparse S-Nodes. This is the foundation for the closed-loop mass-stream
architecture described in the ideal blueprint.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Callable, Optional
from enum import Enum

from .rigid_body import RigidBody
from .thermal_model import update_temperature_euler, check_thermal_limits, ThermalLimits

# Optional flux-pinning model
try:
    from .bean_london_model import BeanLondonModel
    from .gdBCO_material import GdBCOMaterial, GdBCOProperties
    FLUX_PINNING_AVAILABLE = True
except ImportError:
    FLUX_PINNING_AVAILABLE = False
    BeanLondonModel = None
    GdBCOMaterial = None
    GdBCOProperties = None

# Optional orbital dynamics
try:
    from .orbital_coupling import OrbitalState, OrbitalPropagator, create_circular_orbit, compute_eclipse
    ORBITAL_DYNAMICS_AVAILABLE = True
except ImportError:
    ORBITAL_DYNAMICS_AVAILABLE = False
    OrbitalState = None
    OrbitalPropagator = None
    create_circular_orbit = None
    compute_eclipse = None


class PacketState(Enum):
    """State of a packet in the stream."""
    FREE = "free"  # Free-flying between nodes
    CAPTURED = "captured"  # Captured by S-Node
    TRANSIT = "transit"  # In transit between nodes


@dataclass
class SNode:
    """
    Sparse S-Node (Shepherding Node) for magnetic capture/release.

    Attributes:
        id: Node identifier
        position: 3D position [x, y, z] (m)
        capture_radius: Magnetic capture radius (m)
        release_radius: Magnetic release radius (m)
        max_packets: Maximum number of packets that can be held
        eta_ind_min: Minimum induction efficiency constraint (η_ind ≥ 0.82)
        held_packets: List of packet IDs currently held
        k_fp: Flux-pinning stiffness (N/m)
    """
    id: int
    position: np.ndarray
    capture_radius: float = 10.0  # m
    release_radius: float = 5.0  # m
    max_packets: int = 10
    eta_ind_min: float = 0.82
    held_packets: List[int] = field(default_factory=list)
    k_fp: float = 4500.0  # N/m, default flux-pinning stiffness
    
    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)
        if self.position.shape != (3,):
            raise ValueError(f"S-Node position must be 3-element vector, got shape {self.position.shape}")
    
    def can_capture(self, eta_ind: float) -> bool:
        """Check if node can capture a packet."""
        return (len(self.held_packets) < self.max_packets) and (eta_ind >= self.eta_ind_min)
    
    def distance_to(self, position: np.ndarray) -> float:
        """Compute distance from node to a position."""
        return np.linalg.norm(position - self.position)


@dataclass
class Packet:
    """
    Mass packet with rigid-body dynamics and orbital state.
    
    Attributes:
        id: Unique identifier
        body: RigidBody with 6DOF dynamics
        state: Current packet state
        current_node: ID of S-Node holding this packet (if captured)
        eta_ind: Current induction efficiency
        radius: Packet radius for stress calculations (m)
        temperature: Packet temperature (K)
        emissivity: Thermal emissivity for radiation cooling
        specific_heat: Specific heat capacity (J/kg·K)
        orbital_state: Orbital state vector (position/velocity in ECI frame)
        in_eclipse: Whether packet is in Earth's shadow
    """
    id: int
    body: RigidBody
    state: PacketState = PacketState.FREE
    current_node: Optional[int] = None
    eta_ind: float = 1.0  # Default induction efficiency
    radius: float = 0.02  # Default 2cm radius for stress calculations
    temperature: float = 300.0  # Initial temperature (K)
    emissivity: float = 0.8  # Al/BFRP emissivity
    specific_heat: float = 900.0  # J/kg·K for Al
    orbital_state: Optional[OrbitalState] = None  # Orbital state in ECI frame
    in_eclipse: bool = False  # Eclipse state
    
    def compute_flux_pinning_torque(self, B_field: np.ndarray) -> np.ndarray:
        """Compute flux-pinning torque from body's flux model.
        
        Args:
            B_field: Magnetic field vector [Bx, By, Bz] (T)
            
        Returns:
            Torque vector [τx, τy, τz] in body frame (N·m)
        """
        if self.body.flux_model is None:
            return np.zeros(3)
        
        # Get 6-DoF force/torque from Bean-London model
        force_torque = self.body.compute_flux_pinning_force(
            B_field=B_field,
            superconductor_temp=self.temperature,
        )
        
        # Return only torque component [τx, τy, τz]
        return force_torque[3:6]
    
    @property
    def position(self) -> np.ndarray:
        """Get packet position."""
        return self.body.position
    
    @property
    def velocity(self) -> np.ndarray:
        """Get packet velocity."""
        return self.body.velocity
    
    @property
    def angular_velocity(self) -> np.ndarray:
        """Get packet angular velocity."""
        return self.body.angular_velocity


@dataclass
class CaptureEvent:
    """Event for magnetic capture at S-Node."""
    time: float
    packet_id: int
    node_id: int
    eta_ind: float


@dataclass
class ReleaseEvent:
    """Event for magnetic release from S-Node."""
    time: float
    packet_id: int
    node_id: int
    target_velocity: np.ndarray


class EventQueue:
    """Event queue for managing capture/release events."""
    
    def __init__(self):
        self.events: List[CaptureEvent | ReleaseEvent] = []
        self.current_time: float = 0.0
    
    def add_capture(self, time: float, packet_id: int, node_id: int, eta_ind: float):
        """Add capture event to queue."""
        event = CaptureEvent(time=time, packet_id=packet_id, node_id=node_id, eta_ind=eta_ind)
        self.events.append(event)
        self.events.sort(key=lambda e: e.time)
    
    def add_release(self, time: float, packet_id: int, node_id: int, target_velocity: np.ndarray):
        """Add release event to queue."""
        event = ReleaseEvent(time=time, packet_id=packet_id, node_id=node_id, target_velocity=target_velocity)
        self.events.append(event)
        self.events.sort(key=lambda e: e.time)
    
    def get_events_at(self, time: float) -> List[CaptureEvent | ReleaseEvent]:
        """Get all events at or before given time."""
        return [e for e in self.events if e.time <= time]
    
    def remove_processed(self, time: float):
        """Remove events that have been processed."""
        self.events = [e for e in self.events if e.time > time]


class MultiBodyStream:
    """
    Multi-body packet stream with event-driven magnetic handoff.
    
    Manages N=5–20 packets with event-driven capture/release at sparse S-Nodes.
    Implements the closed-loop stream architecture from the ideal blueprint.
    
    Attributes:
        packets: List of Packet objects
        nodes: List of S-Node objects
        stream_velocity: Target stream velocity (m/s)
        B_field: Magnetic field vector [Bx, By, Bz] (T) for flux-pinning calculations
    """
    
    def __init__(
        self,
        packets: List[Packet],
        nodes: List[SNode],
        stream_velocity: float = 1600.0,  # m/s
        B_field: Optional[np.ndarray] = None,  # Magnetic field (T)
        enable_orbital_dynamics: bool = False,
        initial_altitude: float = 400.0,  # km
        initial_inclination: float = 0.0,  # deg
        enable_j2_perturbation: bool = True,
        enable_srp_perturbation: bool = True,
        enable_drag_perturbation: bool = False,
        drag_coefficient: float = 2.2,
        cross_sectional_area: float = 1.0,
        srp_coefficient: float = 1.8,
    ):
        """
        Initialize multi-body stream.
        
        Args:
            packets: List of Packet objects
            nodes: List of S-Node objects
            stream_velocity: Target stream velocity (m/s)
            B_field: Magnetic field vector [Bx, By, Bz] (T) for flux-pinning.
                     If None, defaults to [0, 0, 0.1] (100 mT axial field).
            enable_orbital_dynamics: Enable orbital dynamics integration
            initial_altitude: Initial orbital altitude (km)
            initial_inclination: Initial orbital inclination (deg)
            enable_j2_perturbation: Enable J2 (Earth oblateness) perturbation
            enable_srp_perturbation: Enable Solar Radiation Pressure perturbation
            enable_drag_perturbation: Enable atmospheric drag perturbation
            drag_coefficient: Drag coefficient for atmospheric drag
            cross_sectional_area: Cross-sectional area (m²) for drag/SRP
            srp_coefficient: Reflectivity coefficient for SRP
        """
        self.packets = packets
        self.nodes = nodes
        self.stream_velocity = stream_velocity
        self.event_queue = EventQueue()
        self.time: float = 0.0
        self.enable_orbital_dynamics = enable_orbital_dynamics
        
        # Magnetic field for flux-pinning (default 100 mT axial)
        if B_field is None:
            B_field = np.array([0.0, 0.0, 0.1])  # 100 mT in z-direction
        self.B_field = np.asarray(B_field, dtype=float)
        
        # Build node lookup by ID
        self.node_map = {i: node for i, node in enumerate(nodes)}
        
        # Orbital perturbation settings
        self.enable_j2_perturbation = enable_j2_perturbation
        self.enable_srp_perturbation = enable_srp_perturbation
        self.enable_drag_perturbation = enable_drag_perturbation
        self.drag_coefficient = drag_coefficient
        self.cross_sectional_area = cross_sectional_area
        self.srp_coefficient = srp_coefficient
        
        # Initialize orbital dynamics if enabled
        if enable_orbital_dynamics and ORBITAL_DYNAMICS_AVAILABLE:
            self.orbital_propagator = OrbitalPropagator()
            self._initialize_orbital_states(initial_altitude, initial_inclination)
            self._configure_perturbations()
        else:
            self.orbital_propagator = None
    
    def _initialize_orbital_states(self, altitude: float, inclination: float):
        """Initialize orbital states for all packets.
        
        Args:
            altitude: Initial orbital altitude (km)
            inclination: Initial orbital inclination (deg)
        """
        initial_orbit = create_circular_orbit(altitude, inclination)
        
        for packet in self.packets:
            # Assign same initial orbit to all packets
            # In reality, packets would be distributed along the orbit
            packet.orbital_state = OrbitalState(
                r=initial_orbit.r.copy(),
                v=initial_orbit.v.copy(),
                epoch=self.time
            )
            # Initialize eclipse state
            if compute_eclipse is not None:
                packet.in_eclipse = compute_eclipse(packet.orbital_state.r)
    
    def _configure_perturbations(self):
        """Configure orbital perturbations based on settings."""
        if not ORBITAL_DYNAMICS_AVAILABLE or self.orbital_propagator is None:
            return
        
        # Add J2 perturbation (Earth oblateness)
        if self.enable_j2_perturbation:
            self.orbital_propagator.add_j2_perturbation()
        
        # Add SRP perturbation (Solar Radiation Pressure)
        if self.enable_srp_perturbation:
            # Estimate total mass from packets for SRP calculation
            total_mass = sum(p.body.mass for p in self.packets) if self.packets else 100.0
            self.orbital_propagator.add_srp_perturbation(
                C_r=self.srp_coefficient,
                A=self.cross_sectional_area,
                m=total_mass
            )
        
        # Add atmospheric drag perturbation
        if self.enable_drag_perturbation:
            total_mass = sum(p.body.mass for p in self.packets) if self.packets else 100.0
            self.orbital_propagator.add_drag_perturbation(
                C_d=self.drag_coefficient,
                A=self.cross_sectional_area,
                m=total_mass
            )
    
    def propagate_orbital_dynamics(self, dt: float):
        """Propagate orbital state for all packets.
        
        Args:
            dt: Time step (s)
        """
        if not self.enable_orbital_dynamics or not ORBITAL_DYNAMICS_AVAILABLE:
            return
        
        for packet in self.packets:
            if packet.orbital_state is not None and packet.state == PacketState.FREE:
                # Propagate orbital state
                self.orbital_propagator.from_state_vector(packet.orbital_state)
                packet.orbital_state = self.orbital_propagator.propagate(dt)
                
                # Update eclipse state
                if compute_eclipse is not None:
                    packet.in_eclipse = compute_eclipse(packet.orbital_state.r)
    
    def check_capture_conditions(self, packet: Packet, node: SNode) -> tuple[bool, float]:
        """
        Check if packet can be captured by node.
        
        Args:
            packet: Packet to check
            node: S-Node to check against
        
        Returns:
            (can_capture, distance)
        """
        distance = node.distance_to(packet.position)
        in_capture_radius = distance <= node.capture_radius
        can_capture = in_capture_radius and node.can_capture(packet.eta_ind)
        return can_capture, distance
    
    def process_capture_event(self, event: CaptureEvent):
        """Process a capture event."""
        packet = self.packets[event.packet_id]
        node = self.node_map[event.node_id]
        
        if node.can_capture(event.eta_ind):
            packet.state = PacketState.CAPTURED
            packet.current_node = event.node_id
            packet.eta_ind = event.eta_ind
            node.held_packets.append(packet.id)
            # Stop packet motion when captured
            packet.body.velocity = np.zeros(3)
            packet.body.angular_velocity = np.zeros(3)
    
    def process_release_event(self, event: ReleaseEvent):
        """Process a release event."""
        packet = self.packets[event.packet_id]
        node = self.node_map[event.node_id]
        
        if packet.id in node.held_packets:
            packet.state = PacketState.FREE
            packet.current_node = None
            node.held_packets.remove(packet.id)
            # Set target velocity for release
            packet.body.velocity = event.target_velocity.copy()
    
    def update_events(self, dt: float):
        """
        Update and process events for current time step.
        
        Args:
            dt: Time step (s)
        """
        self.time += dt
        events = self.event_queue.get_events_at(self.time)
        
        for event in events:
            if isinstance(event, CaptureEvent):
                self.process_capture_event(event)
            elif isinstance(event, ReleaseEvent):
                self.process_release_event(event)
        
        self.event_queue.remove_processed(self.time)
    
    def detect_auto_capture(self, packet: Packet) -> Optional[int]:
        """
        Detect if packet should be automatically captured by any node.
        
        Args:
            packet: Packet to check
        
        Returns:
            Node ID if capture should occur, None otherwise
        """
        for node_id, node in self.node_map.items():
            can_capture, distance = self.check_capture_conditions(packet, node)
            if can_capture and packet.state == PacketState.FREE:
                return node_id
        return None
    
    def integrate(
        self,
        dt: float,
        torques: Callable[[int, float, np.ndarray], np.ndarray],
        max_steps: int = 1000,
        thermal_limits: ThermalLimits = None,
        use_numba_rk4: bool = True,
        use_zero_torque_numba: bool = False,
    ) -> dict:
        """
        Integrate multi-body dynamics over time step.

        Args:
            dt: Time step (s)
            torques: Function torques(packet_id, t, state) returning torque
            max_steps: Maximum integration steps per packet
            thermal_limits: ThermalLimits object for temperature constraints
            use_numba_rk4: Use Numba-compiled RK4 integrator (faster)
            use_zero_torque_numba: Use zero-torque Numba RK4 (fastest, no callback)

        Returns:
            Dictionary with integration results
        """
        results = {
            "time": self.time + dt,
            "packets": [],
            "events_processed": 0,
            "thermal_violations": [],
        }
        
        if thermal_limits is None:
            thermal_limits = ThermalLimits()
        
        # Process events first
        initial_events = len(self.event_queue.events)
        self.update_events(dt)
        results["events_processed"] = initial_events - len(self.event_queue.events)
        
        # Propagate orbital dynamics (if enabled)
        self.propagate_orbital_dynamics(dt)
        
        # Integrate each free packet
        for packet in self.packets:
            if packet.state == PacketState.FREE:
                # Check for auto-capture
                capture_node_id = self.detect_auto_capture(packet)
                if capture_node_id is not None:
                    # Schedule capture event
                    self.event_queue.add_capture(
                        time=self.time,
                        packet_id=packet.id,
                        node_id=capture_node_id,
                        eta_ind=packet.eta_ind,
                    )
                    # Process immediately
                    self.update_events(0.0)
                    results["events_processed"] += 1
                
                # Integrate if still free after capture check
                if packet.state == PacketState.FREE:
                    def packet_torques(t, state):
                        # Base torque from control layer
                        tau_control = torques(packet.id, t, state)
                        
                        # Add flux-pinning torque if available
                        tau_pin = packet.compute_flux_pinning_torque(self.B_field)
                        
                        # Total torque: control + flux-pinning
                        return tau_control + tau_pin
                    
                    packet_result = packet.body.integrate(
                        t_span=(self.time, self.time + dt),
                        torques=packet_torques,
                        method="RK45",
                        rtol=1e-8,
                        atol=1e-10,
                        max_step=dt / max_steps,
                        use_numba_rk4=use_numba_rk4,
                        use_zero_torque_numba=use_zero_torque_numba,
                    )
                    
                    # Thermal update (radiation cooling + solar heating)
                    solar_flux = 0.0
                    if self.enable_orbital_dynamics and ORBITAL_DYNAMICS_AVAILABLE:
                        # Solar heating when not in eclipse
                        if not packet.in_eclipse:
                            # Solar constant ~1361 W/m^2 at 1 AU
                            solar_flux = 1361.0  # W/m^2
                            # Reduce by albedo and view factor (simplified)
                            solar_flux *= 0.3  # Effective absorption
                    
                    packet.temperature = update_temperature_euler(
                        temperature=packet.temperature,
                        mass=packet.body.mass,
                        radius=packet.radius,
                        emissivity=packet.emissivity,
                        specific_heat=packet.specific_heat,
                        dt=dt,
                        solar_flux=solar_flux,
                        eddy_heating_power=0.0,  # No eddy heating in multi-body dynamics
                    )
                    
                    # Check thermal limits
                    within_limits, violation_type = check_thermal_limits(
                        packet.temperature, thermal_limits
                    )
                    if not within_limits:
                        results["thermal_violations"].append({
                            "packet_id": packet.id,
                            "temperature": packet.temperature,
                            "violation_type": violation_type,
                        })
                    
                    results["packets"].append({
                        "id": packet.id,
                        "position": packet.position.copy(),
                        "velocity": packet.velocity.copy(),
                        "angular_velocity": packet.angular_velocity.copy(),
                        "temperature": packet.temperature,
                    })
            else:
                # Captured packets don't move but still cool thermally
                solar_flux = 0.0
                if self.enable_orbital_dynamics and ORBITAL_DYNAMICS_AVAILABLE:
                    # Solar heating when not in eclipse
                    if not packet.in_eclipse:
                        # Solar constant ~1361 W/m^2 at 1 AU
                        solar_flux = 1361.0  # W/m^2
                        # Reduce by albedo and view factor (simplified)
                        solar_flux *= 0.3  # Effective absorption
                
                packet.temperature = update_temperature_euler(
                    temperature=packet.temperature,
                    mass=packet.body.mass,
                    radius=packet.radius,
                    emissivity=packet.emissivity,
                    specific_heat=packet.specific_heat,
                    dt=dt,
                    solar_flux=solar_flux,
                    eddy_heating_power=0.0,  # No eddy heating in multi-body dynamics
                )
                
                # Check thermal limits
                within_limits, violation_type = check_thermal_limits(
                    packet.temperature, thermal_limits
                )
                if not within_limits:
                    results["thermal_violations"].append({
                        "packet_id": packet.id,
                        "temperature": packet.temperature,
                        "violation_type": violation_type,
                    })
                
                results["packets"].append({
                    "id": packet.id,
                    "position": packet.position.copy(),
                    "velocity": packet.velocity.copy(),
                    "angular_velocity": packet.angular_velocity.copy(),
                    "temperature": packet.temperature,
                })
        
        return results
    
    def get_stream_metrics(self) -> dict:
        """
        Get current stream metrics.
        
        Returns:
            Dictionary with stream metrics
        """
        free_packets = sum(1 for p in self.packets if p.state == PacketState.FREE)
        captured_packets = sum(1 for p in self.packets if p.state == PacketState.CAPTURED)
        
        avg_eta_ind = np.mean([p.eta_ind for p in self.packets]) if self.packets else 0.0
        
        return {
            "total_packets": len(self.packets),
            "free_packets": free_packets,
            "captured_packets": captured_packets,
            "avg_eta_ind": avg_eta_ind,
            "time": self.time,
        }
