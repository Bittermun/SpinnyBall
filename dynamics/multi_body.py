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
        position: 3D position [x, y, z] (m)
        capture_radius: Magnetic capture radius (m)
        release_radius: Magnetic release radius (m)
        max_packets: Maximum number of packets that can be held
        eta_ind_min: Minimum induction efficiency constraint (η_ind ≥ 0.82)
        held_packets: List of packet IDs currently held
    """
    position: np.ndarray
    capture_radius: float = 10.0  # m
    release_radius: float = 5.0  # m
    max_packets: int = 10
    eta_ind_min: float = 0.82
    held_packets: List[int] = field(default_factory=list)
    
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
    Sovereign Bean (spin-stabilized magnetic packet).
    
    Attributes:
        id: Unique packet identifier
        body: RigidBody dynamics object
        state: Current packet state
        current_node: ID of S-Node holding this packet (if captured)
        eta_ind: Current induction efficiency
        radius: Packet radius for stress calculations (m)
        temperature: Packet temperature (K)
        emissivity: Thermal emissivity for radiation cooling
        specific_heat: Specific heat capacity (J/kg·K)
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
    """
    
    def __init__(
        self,
        packets: List[Packet],
        nodes: List[SNode],
        stream_velocity: float = 1600.0,  # m/s
    ):
        """
        Initialize multi-body stream.
        
        Args:
            packets: List of Packet objects
            nodes: List of S-Node objects
            stream_velocity: Target stream velocity (m/s)
        """
        self.packets = packets
        self.nodes = nodes
        self.stream_velocity = stream_velocity
        self.event_queue = EventQueue()
        self.time: float = 0.0
        
        # Build node lookup by ID
        self.node_map = {i: node for i, node in enumerate(nodes)}
    
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
    ) -> dict:
        """
        Integrate multi-body dynamics over time step.
        
        Args:
            dt: Time step (s)
            torques: Function torques(packet_id, t, state) returning torque
            max_steps: Maximum integration steps per packet
            thermal_limits: ThermalLimits object for temperature constraints
        
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
                        return torques(packet.id, t, state)
                    
                    packet_result = packet.body.integrate(
                        t_span=(self.time, self.time + dt),
                        torques=packet_torques,
                        method="RK45",
                        rtol=1e-8,
                        atol=1e-10,
                        max_step=dt / max_steps,
                    )
                    
                    # Thermal update (radiation cooling)
                    packet.temperature = update_temperature_euler(
                        temperature=packet.temperature,
                        mass=packet.body.mass,
                        radius=packet.radius,
                        emissivity=packet.emissivity,
                        specific_heat=packet.specific_heat,
                        dt=dt,
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
                packet.temperature = update_temperature_euler(
                    temperature=packet.temperature,
                    mass=packet.body.mass,
                    radius=packet.radius,
                    emissivity=packet.emissivity,
                    specific_heat=packet.specific_heat,
                    dt=dt,
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
