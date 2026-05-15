"""
Multi-body packet stream dynamics for closed-loop mass-stream simulation.

Implements N-body packet dynamics with event-driven magnetic capture/release
at sparse S-Nodes using Halbach array self-confinement.

This is the canonical Shepherded Gyroscopic Mass Stream (SGMS) implementation
with:
- Spherical Halbach arrays for magnetic dipole-dipole confinement
- Hoop tension from momentum flux
- Shepherd stations with quadrupole lenses

References:
- Leupold & Potenziani (1988) - Halbach cylinders
- Jackson, Classical Electrodynamics - Dipole-dipole interactions
- Hoyt & Forward (1999) - Momentum-exchange tethers
"""

from __future__ import annotations

import heapq
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .rigid_body import RigidBody
from .thermal_model import ThermalLimits, check_thermal_limits, update_temperature_euler

# Halbach array model (new canonical implementation)
try:
    from .halbach_array import HalbachArray, HalbachConfig, create_standard_halbach
    from .interball_magnetic import InterBallMagneticInteraction, dipole_dipole_force
    from .hoop_tension import HoopTensionModel, StreamGeometry
    from .shepherd_station import ShepherdStation, StationType
    HALBACH_AVAILABLE = True
except ImportError:
    HALBACH_AVAILABLE = False
    HalbachArray = None
    HalbachConfig = None
    create_standard_halbach = None
    InterBallMagneticInteraction = None
    dipole_dipole_force = None
    HoopTensionModel = None
    StreamGeometry = None
    ShepherdStation = None
    StationType = None

# Optional orbital dynamics
try:
    from .orbital_coupling import (
        OrbitalPropagator,
        OrbitalState,
        compute_eclipse,
        create_circular_orbit,
    )
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

    This is the legacy S-Node class. For new code, use ShepherdStation
    from shepherd_station module which provides Halbach quadrupole lens
    functionality.

    Attributes:
        id: Node identifier
        position: 3D position [x, y, z] (m)
        capture_radius: Magnetic capture radius (m)
        release_radius: Magnetic release radius (m)
        max_packets: Maximum number of packets that can be held
        eta_ind_min: Minimum induction efficiency constraint (η_ind ≥ 0.82)
        held_packets: List of packet IDs currently held
        k_fp: Flux-pinning stiffness (N/m) - legacy, kept for compatibility
        shepherd_station: Optional ShepherdStation for Halbach-based guidance
    """
    id: int
    position: np.ndarray
    capture_radius: float = 10.0  # m
    release_radius: float = 5.0  # m
    max_packets: int = 10
    eta_ind_min: float = 0.82
    held_packets: list[int] = field(default_factory=list)
    k_fp: float = 6000.0  # N/m, legacy flux-pinning stiffness
    shepherd_station: 'ShepherdStation' = None  # New Halbach-based guidance

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)
        if self.position.shape != (3,):
            raise ValueError(f"S-Node position must be 3-element vector, got shape {self.position.shape}")
        
        # Create default shepherd station if not provided
        if self.shepherd_station is None and HALBACH_AVAILABLE:
            from .shepherd_station import create_passive_shepherd
            self.shepherd_station = create_passive_shepherd(
                position=self.position,
                station_id=self.id,
                capture_radius=self.capture_radius
            )

    def can_capture(self, eta_ind: float) -> bool:
        """Check if node can capture a packet."""
        return (len(self.held_packets) < self.max_packets) and (eta_ind >= self.eta_ind_min)

    def distance_to(self, position: np.ndarray) -> float:
        """Compute distance from node to a position."""
        return np.linalg.norm(position - self.position)
    
    def compute_halbach_force(
        self,
        ball_position: np.ndarray,
        ball_velocity: np.ndarray,
        dipole_moment: float = 580.0
    ) -> np.ndarray:
        """Compute Halbach-based guidance force on a ball.
        
        Args:
            ball_position: Ball position [x, y, z] (m)
            ball_velocity: Ball velocity [vx, vy, vz] (m/s)
            dipole_moment: Ball dipole moment (A·m²)
        
        Returns:
            Force vector [Fx, Fy, Fz] (N)
        """
        if self.shepherd_station is not None:
            return self.shepherd_station.compute_total_force(
                ball_position, ball_velocity, dipole_moment
            )
        return np.zeros(3)


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
        halbach_array: Spherical Halbach array for magnetic interactions
    """
    id: int
    body: RigidBody
    state: PacketState = PacketState.FREE
    current_node: int | None = None
    eta_ind: float = 1.0  # Default induction efficiency
    radius: float = 0.05  # Default 5cm radius for Halbach sphere
    temperature: float = 293.0  # Initial temperature (K) - room temp for Halbach
    emissivity: float = 0.8  # Al/BFRP emissivity
    specific_heat: float = 900.0  # J/kg·K for Al
    orbital_state: OrbitalState | None = None  # Orbital state in ECI frame
    in_eclipse: bool = False  # Eclipse state
    halbach_array: 'HalbachArray' = None  # Spherical Halbach array

    def __post_init__(self):
        """Initialize Halbach array if not provided."""
        if self.halbach_array is None and HALBACH_AVAILABLE:
            from .halbach_array import create_standard_halbach
            self.halbach_array = create_standard_halbach(
                radius=self.radius,
                material='NdFeB',
                temperature=self.temperature
            )

    def compute_halbach_force_from_neighbors(
        self,
        neighbor_packets: list['Packet']
    ) -> np.ndarray:
        """Compute magnetic force from neighboring Halbach arrays.

        Args:
            neighbor_packets: List of neighboring packets

        Returns:
            Force vector [Fx, Fy, Fz] (N)
        """
        if self.halbach_array is None or not HALBACH_AVAILABLE:
            return np.zeros(3)

        total_force = np.zeros(3)
        my_pos = self.position
        my_dipole = self.halbach_array.dipole_moment

        for neighbor in neighbor_packets:
            if neighbor.halbach_array is None or neighbor.id == self.id:
                continue

            neighbor_pos = neighbor.position
            neighbor_dipole = neighbor.halbach_array.dipole_moment

            # Position vector from self to neighbor
            r_vec = neighbor_pos - my_pos

            # Force from neighbor
            F = dipole_dipole_force(my_dipole, neighbor_dipole, r_vec)
            total_force += F

        return total_force

    def compute_halbach_torque_from_neighbors(
        self,
        neighbor_packets: list['Packet']
    ) -> np.ndarray:
        """Compute magnetic torque from neighboring Halbach arrays.

        Args:
            neighbor_packets: List of neighboring packets

        Returns:
            Torque vector [τx, τy, τz] (N·m)
        """
        if self.halbach_array is None or not HALBACH_AVAILABLE:
            return np.zeros(3)

        total_torque = np.zeros(3)
        my_pos = self.position
        my_dipole = self.halbach_array.dipole_moment

        for neighbor in neighbor_packets:
            if neighbor.halbach_array is None or neighbor.id == self.id:
                continue

            neighbor_pos = neighbor.position
            neighbor_dipole = neighbor.halbach_array.dipole_moment

            # Position vector from self to neighbor
            r_vec = neighbor_pos - my_pos

            # Torque = m × B from neighbor
            B = neighbor.halbach_array.magnetic_field(-r_vec)  # Field at my position
            tau = np.cross(my_dipole, B)
            total_torque += tau

        return total_torque

    def compute_flux_pinning_torque(self, B_field: np.ndarray, node_position: np.ndarray) -> np.ndarray:
        """Compute flux-pinning torque from body's flux model.

        DEPRECATED: This method is kept for backward compatibility.
        Use compute_halbach_torque_from_neighbors for new code.

        Args:
            B_field: Magnetic field vector [Bx, By, Bz] (T)
            node_position: Position of the S-Node for relative displacement calculation

        Returns:
            Torque vector [τx, τy, τz] in body frame (N·m)
        """
        # For Halbach arrays, use magnetic torque instead
        if self.halbach_array is not None:
            # Simplified: torque from shepherd station field
            m = self.halbach_array.dipole_moment
            return np.cross(m, B_field)

        # Legacy flux-pinning code
        if self.body.flux_model is None:
            return np.zeros(3)

        # Relative displacement from S-Node
        displacement = self.body.position - node_position

        # Get 6-DoF force/torque from Bean-London model
        force_torque = self.body.compute_flux_pinning_force(
            B_field=B_field,
            superconductor_temp=self.temperature,
            displacement=displacement,
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
    
    @property
    def dipole_moment(self) -> np.ndarray:
        """Get magnetic dipole moment from Halbach array."""
        if self.halbach_array is not None:
            return self.halbach_array.dipole_moment
        return np.zeros(3)


@dataclass(order=True)
class CaptureEvent:
    """Event for magnetic capture at S-Node."""
    time: float
    packet_id: int
    node_id: int
    eta_ind: float


@dataclass(order=True)
class ReleaseEvent:
    """Event for magnetic release from S-Node."""
    time: float
    packet_id: int
    node_id: int
    target_velocity: np.ndarray = field(compare=False)


class EventQueue:
    """Event queue for managing capture/release events using heapq for O(log n) operations."""

    def __init__(self):
        self.events: list[tuple[float, CaptureEvent | ReleaseEvent]] = []  # (time, event) heap
        self.current_time: float = 0.0

    def add_capture(self, time: float, packet_id: int, node_id: int, eta_ind: float):
        """Add capture event to queue using heapq for O(log n) insertion."""
        event = CaptureEvent(time=time, packet_id=packet_id, node_id=node_id, eta_ind=eta_ind)
        heapq.heappush(self.events, (time, event))

    def add_release(self, time: float, packet_id: int, node_id: int, target_velocity: np.ndarray):
        """Add release event to queue using heapq for O(log n) insertion."""
        event = ReleaseEvent(time=time, packet_id=packet_id, node_id=node_id, target_velocity=target_velocity)
        heapq.heappush(self.events, (time, event))

    def get_events_at(self, time: float) -> list[CaptureEvent | ReleaseEvent]:
        """Get all events at or before given time."""
        result = []
        while self.events and self.events[0][0] <= time:
            _, event = heapq.heappop(self.events)
            result.append(event)
        return result

    def remove_processed(self, time: float):
        """Remove events that have been processed (no-op with heapq approach)."""
        # Events are already removed during get_events_at via heappop
        pass


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
        topology: Stream topology ('linear', 'ring', 'orbital_ring')
        counter_propagating: Whether two streams propagate in opposite directions
    """

    def __init__(
        self,
        packets: list[Packet],
        nodes: list[SNode],
        stream_velocity: float = 1600.0,  # m/s
        B_field: np.ndarray | None = None,  # Magnetic field (T)
        enable_orbital_dynamics: bool = False,
        initial_altitude: float = 400.0,  # km
        initial_inclination: float = 0.0,  # deg
        enable_j2_perturbation: bool = True,
        enable_srp_perturbation: bool = True,
        enable_drag_perturbation: bool = False,
        drag_coefficient: float = 2.2,
        cross_sectional_area: float = 1.0,
        srp_coefficient: float = 1.8,
        topology: str = 'linear',  # 'linear', 'ring', 'orbital_ring'
        counter_propagating: bool = True,  # Two streams in opposite directions
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
            topology: Stream topology. One of:
                - 'linear': Linear array (original behavior)
                - 'ring': Closed ring where node[N-1] is adjacent to node[0]
                - 'orbital_ring': Ring with orbital mechanics (stream_length = 2*pi*(R_earth + altitude))
            counter_propagating: If True, two streams propagate in opposite directions
        """
        self.packets = packets
        self.nodes = nodes
        self.stream_velocity = stream_velocity
        self.event_queue = EventQueue()
        self.time: float = 0.0
        self.enable_orbital_dynamics = enable_orbital_dynamics
        self.topology = topology
        self.counter_propagating = counter_propagating

        # Magnetic field for flux-pinning (default 100 mT axial)
        if B_field is None:
            B_field = np.array([0.0, 0.0, 0.1])  # 100 mT in z-direction
        self.B_field = np.asarray(B_field, dtype=float)

        # Build node lookup by ID
        self.node_map = {node.id: node for node in nodes}

        # Configure topology-specific parameters
        self._configure_topology(initial_altitude)

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

        # Initialize Halbach arrays for packets (new canonical implementation)
        if HALBACH_AVAILABLE:
            for packet in self.packets:
                if packet.halbach_array is None:
                    from .halbach_array import create_standard_halbach
                    packet.halbach_array = create_standard_halbach(
                        radius=packet.radius,
                        material='NdFeB',
                        temperature=packet.temperature
                    )
        
        # Initialize hoop tension model for stream confinement
        if HALBACH_AVAILABLE and len(self.packets) > 0:
            self.hoop_model = self._create_hoop_model()
            self.interball_interaction = self._create_interball_interaction()
        else:
            self.hoop_model = None
            self.interball_interaction = None

    def _create_hoop_model(self) -> 'HoopTensionModel':
        """Create hoop tension model for stream confinement.
        
        Returns:
            HoopTensionModel configured for this stream
        """
        if not HALBACH_AVAILABLE or len(self.packets) == 0:
            return None
        
        from .hoop_tension import StreamGeometry
        
        # Estimate stream geometry from packets
        avg_mass = np.mean([p.body.mass for p in self.packets])
        
        # For orbital ring, use stream_length
        if hasattr(self, 'stream_length') and self.stream_length is not None:
            radius = self.stream_length / (2.0 * np.pi)
        else:
            # Estimate from packet positions
            positions = [p.position for p in self.packets]
            if len(positions) > 1:
                center = np.mean(positions, axis=0)
                avg_radius = np.mean([np.linalg.norm(p - center) for p in positions])
                radius = max(avg_radius, 1000.0)  # Minimum 1km
            else:
                radius = 1000.0
        
        geometry = StreamGeometry(
            radius=radius,
            n_balls=len(self.packets),
            ball_mass=avg_mass,
            stream_velocity=self.stream_velocity
        )
        
        from .hoop_tension import HoopTensionModel
        return HoopTensionModel(geometry)
    
    def _create_interball_interaction(self) -> 'InterBallMagneticInteraction':
        """Create inter-ball magnetic interaction model.
        
        Returns:
            InterBallMagneticInteraction for this stream
        """
        if not HALBACH_AVAILABLE or len(self.packets) == 0:
            return None
        
        halbach_arrays = [p.halbach_array for p in self.packets if p.halbach_array is not None]
        
        if len(halbach_arrays) == 0:
            return None
        
        return InterBallMagneticInteraction(halbach_arrays, neighbor_range=2)

    def _configure_topology(self, initial_altitude: float):
        """Configure topology-specific parameters.

        Args:
            initial_altitude: Orbital altitude (km) for orbital_ring topology
        """
        n_nodes = len(self.nodes)

        if self.topology == 'linear':
            # Original linear array behavior
            self.stream_length = None  # Not applicable
            self.node_spacing = None
            self.is_closed_loop = False

        elif self.topology == 'ring':
            # Closed ring - node[N-1] is adjacent to node[0]
            # Estimate stream length from node positions
            if n_nodes > 1:
                # Sum distances between consecutive nodes
                total_length = 0.0
                for i in range(n_nodes):
                    next_i = (i + 1) % n_nodes  # Wrap around
                    pos_curr = self.nodes[i].position
                    pos_next = self.nodes[next_i].position
                    total_length += np.linalg.norm(pos_next - pos_curr)
                self.stream_length = total_length
                self.node_spacing = total_length / n_nodes
            else:
                self.stream_length = 0.0
                self.node_spacing = 0.0
            self.is_closed_loop = True

        elif self.topology == 'orbital_ring':
            # Orbital ring - stream follows circular orbit
            R_earth = 6371.0  # km
            orbital_radius_km = R_earth + initial_altitude
            self.stream_length = 2 * np.pi * orbital_radius_km * 1000  # Convert to meters
            self.node_spacing = self.stream_length / n_nodes if n_nodes > 0 else 0.0
            self.is_closed_loop = True

            # For orbital_ring, counter-propagating doubles packet count
            if self.counter_propagating:
                # Two streams in opposite directions
                self.n_effective_packets = len(self.packets) * 2
            else:
                self.n_effective_packets = len(self.packets)
        else:
            raise ValueError(f"Unknown topology: {self.topology}. "
                           f"Available: 'linear', 'ring', 'orbital_ring'")

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
            # Use average packet mass for SRP calculation (propagator is shared)
            avg_mass = sum(p.body.mass for p in self.packets) / len(self.packets) if self.packets else 100.0
            self.orbital_propagator.add_srp_perturbation(
                C_r=self.srp_coefficient,
                A=self.cross_sectional_area,
                m=avg_mass
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

                # SYNC: Update RigidBody position/velocity from orbital state
                packet.body.position = packet.orbital_state.r.copy()
                packet.body.velocity = packet.orbital_state.v.copy()

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

    def detect_auto_capture(self, packet: Packet) -> int | None:
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
                    # Find nearest node for shepherd station guidance
                    nearest_node_pos = None
                    nearest_node = None
                    if self.nodes:
                        distances = [node.distance_to(packet.position) for node in self.nodes]
                        nearest_idx = np.argmin(distances)
                        nearest_node = self.nodes[nearest_idx]
                        nearest_node_pos = nearest_node.position

                    # Compute inter-ball magnetic forces (new canonical Halbach physics)
                    F_interball = np.zeros(3)
                    if self.interball_interaction is not None:
                        neighbor_indices = self._get_neighbor_indices(packet.id)
                        neighbors = [self.packets[i] for i in neighbor_indices]
                        F_interball = packet.compute_halbach_force_from_neighbors(neighbors)
                    
                    # Compute shepherd station guidance force (new canonical)
                    F_shepherd = np.zeros(3)
                    if nearest_node is not None:
                        F_shepherd = nearest_node.compute_halbach_force(
                            packet.position, packet.velocity,
                            np.linalg.norm(packet.dipole_moment)
                        )
                    
                    # Apply inter-ball and shepherd forces to body
                    # F = ma, so acceleration = F/m
                    total_force = F_interball + F_shepherd
                    packet.body.velocity += (total_force / packet.body.mass) * dt

                    # Define control torque function outside lambda for Numba compatibility
                    def control_torque_func(packet_id: int, t: float, state: np.ndarray) -> np.ndarray:
                        return torques(packet_id, t, state)

                    # Define Halbach magnetic torque function (new canonical)
                    def halbach_torque_func(packet_id: int, t: float, state: np.ndarray) -> np.ndarray:
                        torque = np.zeros(3)
                        
                        # Add torque from neighboring packets (inter-ball magnetic)
                        if self.interball_interaction is not None:
                            neighbor_indices = self._get_neighbor_indices(packet.id)
                            neighbors = [self.packets[i] for i in neighbor_indices]
                            torque += packet.compute_halbach_torque_from_neighbors(neighbors)
                        
                        # Add torque from shepherd station (if near node)
                        if nearest_node_pos is not None:
                            torque += packet.compute_flux_pinning_torque(self.B_field, nearest_node_pos)
                        
                        return torque

                    # Combined torque function for Numba
                    def packet_torques_func(t: float, state: np.ndarray) -> np.ndarray:
                        tau_control = control_torque_func(packet.id, t, state)
                        tau_halbach = halbach_torque_func(packet.id, t, state)
                        return tau_control + tau_halbach

                    packet.body.integrate(
                        t_span=(self.time, self.time + dt),
                        torques=packet_torques_func,
                        method="RK45",
                        rtol=1e-8,
                        atol=1e-10,
                        max_step=dt / max_steps,
                        use_numba_rk4=False,  # Disable Numba to avoid lambda issues
                        use_zero_torque_numba=use_zero_torque_numba,
                    )

                    # Thermal update (radiation cooling + solar heating + eddy heating)
                    solar_flux = 0.0
                    if self.enable_orbital_dynamics and ORBITAL_DYNAMICS_AVAILABLE:
                        # Solar heating when not in eclipse
                        if not packet.in_eclipse:
                            from dynamics.thermal_model import (
                                SOLAR_ABSORPTION_FACTOR,
                                SOLAR_CONSTANT,
                            )
                            solar_flux = SOLAR_CONSTANT * SOLAR_ABSORPTION_FACTOR

                    # Calculate eddy heating power for FREE packets
                    eddy_power = 0.0
                    if packet.state == PacketState.FREE:
                        from .thermal_model import eddy_heating_power
                        velocity_mag = np.linalg.norm(packet.velocity)
                        eddy_power = eddy_heating_power(
                            velocity=velocity_mag,
                            k_drag=0.01,  # Match sgms_v1.py k_drag
                            radius=packet.radius
                        )

                    packet.temperature = update_temperature_euler(
                        temperature=packet.temperature,
                        mass=packet.body.mass,
                        radius=packet.radius,
                        emissivity=packet.emissivity,
                        specific_heat=packet.specific_heat,
                        dt=dt,
                        position_eci=packet.orbital_state.r if packet.orbital_state else None,
                        enable_eclipse=packet.orbital_state is not None,
                        solar_flux=solar_flux,
                        eddy_heating_power=eddy_power,
                        shape="prolate_spheroid",  # Use prolate spheroid for packets
                        aspect_ratio=1.2  # Standard aspect ratio
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

                # Captured packets have no eddy heating (velocity = 0)
                eddy_power = 0.0

                packet.temperature = update_temperature_euler(
                    temperature=packet.temperature,
                    mass=packet.body.mass,
                    radius=packet.radius,
                    emissivity=packet.emissivity,
                    specific_heat=packet.specific_heat,
                    dt=dt,
                    position_eci=packet.orbital_state.r if packet.orbital_state else None,
                    enable_eclipse=packet.orbital_state is not None,
                    solar_flux=solar_flux,
                    eddy_heating_power=eddy_power,
                    shape="prolate_spheroid",  # Use prolate spheroid for packets
                    aspect_ratio=1.2  # Standard aspect ratio
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

    def _get_neighbor_indices(self, packet_id: int, n_neighbors: int = 2) -> list[int]:
        """Get indices of neighboring packets in the stream.
        
        Args:
            packet_id: ID of the packet to find neighbors for
            n_neighbors: Number of neighbors on each side
        
        Returns:
            List of neighbor packet indices
        """
        if not hasattr(self, 'packets') or len(self.packets) <= 1:
            return []
        
        # Find index of packet with given ID
        try:
            idx = next(i for i, p in enumerate(self.packets) if p.id == packet_id)
        except StopIteration:
            return []
        
        n = len(self.packets)
        neighbors = []
        
        # Get neighbors on both sides
        for offset in range(1, n_neighbors + 1):
            # Previous neighbor (with wrap-around for closed loop)
            if self.is_closed_loop:
                prev_idx = (idx - offset) % n
                neighbors.append(prev_idx)
            elif idx - offset >= 0:
                neighbors.append(idx - offset)
            
            # Next neighbor (with wrap-around for closed loop)
            if self.is_closed_loop:
                next_idx = (idx + offset) % n
                neighbors.append(next_idx)
            elif idx + offset < n:
                neighbors.append(idx + offset)
        
        return neighbors

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
