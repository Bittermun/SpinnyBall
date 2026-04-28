"""
Multi-Lane Stream Dynamics for Orbital Ring Skyhook.

This module implements the multi-lane mass stream concept where packets
from lunar injection can couple to different velocity lanes based on
their target relative velocity.

Lane Structure:
- SLOW: 1-2 km/s relative (precision cargo, delicate payloads)
- STANDARD: 3-5 km/s relative (general logistics)
- FAST: 5-15 km/s relative (high-priority, time-critical)

Each lane has distinct momentum flux and energy transfer characteristics.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

# Import from orbital_coupling for reference frame handling
try:
    from .orbital_coupling import (
        StreamReferenceFrame, 
        StationState, 
        CouplingResult,
        ORBITAL_DYNAMICS_AVAILABLE
    )
except ImportError:
    # Fallback for direct execution
    from dynamics.orbital_coupling import (
        StreamReferenceFrame, 
        StationState, 
        CouplingResult,
        ORBITAL_DYNAMICS_AVAILABLE
    )

# Try to import lunar injection result type
try:
    from scenarios.lunar_injection import LunarInjectionResult
    LUNAR_INJECTION_AVAILABLE = True
except ImportError:
    LUNAR_INJECTION_AVAILABLE = False
    LunarInjectionResult = None


class LaneType(Enum):
    """Operational lane classification."""
    SLOW = "slow"
    STANDARD = "standard"
    FAST = "fast"
    NONE = "none"


@dataclass
class StreamLane:
    """
    Definition of a single velocity lane in the mass stream.
    
    Attributes:
        lane_type: Lane classification (SLOW/STANDARD/FAST)
        v_min_km_s: Minimum relative velocity (km/s)
        v_max_km_s: Maximum relative velocity (km/s)
        mass_flow_rate: Design mass flow rate (kg/s)
        packet_count: Current number of packets in lane
        accumulated_momentum: Total momentum stored from packets (kg·m/s)
        lane_efficiency: Operational efficiency (0-1)
    """
    lane_type: LaneType
    v_min_km_s: float
    v_max_km_s: float
    mass_flow_rate: float  # kg/s
    packet_count: int = 0
    accumulated_momentum: float = 0.0  # kg·m/s (NOT flux - that's a rate)
    lane_efficiency: float = 1.0
    
    def contains_velocity(self, v_rel_km_s: float) -> bool:
        """Check if a relative velocity falls within this lane."""
        return self.v_min_km_s <= v_rel_km_s <= self.v_max_km_s
    
    def calculate_design_momentum_flux(self) -> float:
        """
        Calculate design momentum flux for this lane.
        
        F = dm/dt * v_average
        
        Returns:
            Momentum flux (N)
        """
        v_avg = (self.v_min_km_s + self.v_max_km_s) / 2 * 1000  # Convert to m/s
        return self.mass_flow_rate * v_avg
    
    def add_packet(self, packet_mass: float, v_relative: np.ndarray):
        """
        Add a packet to this lane and update accumulated momentum.
        
        Args:
            packet_mass: Mass of the packet (kg)
            v_relative: Relative velocity vector (km/s)
        """
        self.packet_count += 1
        # Update accumulated momentum: p_new = p_old + m * v
        # Note: This is total momentum (kg·m/s), NOT flux (N)
        v_mag_ms = np.linalg.norm(v_relative) * 1000  # km/s -> m/s
        self.accumulated_momentum += packet_mass * v_mag_ms
    
    def remove_packet(self, packet_mass: float, v_relative: np.ndarray):
        """Remove a packet from this lane."""
        if self.packet_count > 0:
            self.packet_count -= 1
            v_mag_ms = np.linalg.norm(v_relative) * 1000
            self.accumulated_momentum -= packet_mass * v_mag_ms
    
    def get_true_momentum_flux(self, time_window_s: float) -> float:
        """
        Calculate true momentum flux (force) over a time window.
        
        Flux = accumulated_momentum / time_window
        
        Args:
            time_window_s: Time window in seconds
            
        Returns:
            True momentum flux (N)
        """
        if time_window_s <= 0:
            return 0.0
        return self.accumulated_momentum / time_window_s


@dataclass
class PacketStreamEvent:
    """
    Event representing a packet coupling/decoupling from the stream.
    
    Attributes:
        packet_id: Unique packet identifier
        event_type: 'COUPLE' or 'DECOUPLE'
        lane_type: Which lane the event occurs in
        v_relative_km_s: Relative velocity at event (km/s)
        timestamp: Simulation time of event (s)
        energy_transfer_J: Energy transferred during event (J)
        momentum_change_Ns: Momentum change (N·s)
    """
    packet_id: str
    event_type: str
    lane_type: LaneType
    v_relative_km_s: float
    timestamp: float
    energy_transfer_J: float
    momentum_change_Ns: float


class MultiLaneStream:
    """
    Manages multiple velocity lanes in the circulating mass stream.
    
    The multi-lane architecture allows simultaneous operations at different
    relative velocities, optimizing throughput and enabling specialized
    handling for different payload types.
    
    Key Physics:
    - Momentum Flux: F = dm/dt * v_relative (determines station loading)
    - Energy Transfer: E = 0.5 * m * (v_f² - v_i²) (coupling energy cost)
    - Lane Assignment: Based on packet's target_relative_velocity
    
    Integration with Lunar Injection:
    - Accepts LunarInjectionResult as incoming packet source
    - Determines lane based on target_relative_velocity field
    - Converts arrival_eci_vector to station-relative frame for coupling
    
    Example Usage:
        >>> stream = MultiLaneStream()
        >>> station = StationState(
        ...     position_eci=np.array([6871, 0, 0]),
        ...     velocity_eci=np.array([0, 7.6, 0]),
        ...     stream_velocity_mag=7.6
        ... )
        >>> # Simulate lunar injection result
        >>> from scenarios.lunar_injection import LunarInjectionResult
        >>> injection = LunarInjectionResult(
        ...     departure_dv=2800,
        ...     transfer_time_days=4.0,
        ...     arrival_eci_vector=np.array([0, 12.6, 0]),  # 5 km/s relative
        ...     hyperbolic_excess_velocity=1000,
        ...     arrival_altitude_km=500,
        ...     target_relative_velocity=5000,  # 5 km/s in m/s
        ...     energy_budget_warning=False,
        ...     spin_rate_rpm=133
        ... )
        >>> lane = stream.find_matching_lane(injection)
        >>> print(lane.lane_type)  # Should be STANDARD
    """
    
    # Default lane configuration
    DEFAULT_LANES = [
        StreamLane(LaneType.SLOW, 0.5, 2.0, mass_flow_rate=10.0),
        StreamLane(LaneType.STANDARD, 2.0, 5.0, mass_flow_rate=50.0),
        StreamLane(LaneType.FAST, 5.0, 15.0, mass_flow_rate=25.0),
    ]
    
    def __init__(self, custom_lanes: Optional[List[StreamLane]] = None):
        """
        Initialize multi-lane stream.
        
        Args:
            custom_lanes: Optional custom lane configuration.
                         If None, uses DEFAULT_LANES.
        """
        self.lanes = custom_lanes if custom_lanes else self.DEFAULT_LANES.copy()
        self.reference_frame = StreamReferenceFrame()
        self.event_log: List[PacketStreamEvent] = []
        self._packet_counter = 0
    
    def find_matching_lane(
        self, 
        packet_source: object,
        v_relative_override: Optional[float] = None
    ) -> Optional[StreamLane]:
        """
        Find the appropriate lane for an incoming packet.
        
        This is the primary integration point with lunar_injection.py.
        Accepts either:
        - LunarInjectionResult: Uses target_relative_velocity field
        - Any object with target_relative_velocity attribute (m/s)
        - Custom v_relative_override (km/s)
        
        Args:
            packet_source: Source of packet (e.g., LunarInjectionResult)
            v_relative_override: Override relative velocity (km/s)
            
        Returns:
            Matching StreamLane or None if no match
        """
        # Extract target relative velocity
        if v_relative_override is not None:
            v_rel_km_s = v_relative_override
        elif hasattr(packet_source, 'target_relative_velocity'):
            # LunarInjectionResult provides this in m/s
            v_rel_km_s = packet_source.target_relative_velocity / 1000.0
        else:
            raise ValueError(
                "packet_source must have target_relative_velocity attribute "
                "or v_relative_override must be provided"
            )
        
        # Find matching lane
        for lane in self.lanes:
            if lane.contains_velocity(v_rel_km_s):
                return lane
        
        return None  # No matching lane
    
    def process_lunar_injection(
        self,
        injection_result: object,  # LunarInjectionResult
        station_state: StationState,
        packet_mass: float = 100.0,  # kg
        timestamp: float = 0.0
    ) -> Tuple[Optional[StreamLane], CouplingResult, Optional[PacketStreamEvent]]:
        """
        Process a packet arriving from lunar injection.
        
        This method:
        1. Converts arrival_eci_vector to station-relative frame
        2. Finds the matching lane based on target_relative_velocity
        3. Calculates energy transfer and momentum flux
        4. Logs the coupling event
        
        Args:
            injection_result: Result from LunarInjectionCalculator
            station_state: Current state of target station
            packet_mass: Mass of the incoming packet (kg)
            timestamp: Simulation timestamp (s)
            
        Returns:
            Tuple of (matched_lane, coupling_result, event)
            Returns (None, coupling_result, None) if no lane matches
        """
        if not LUNAR_INJECTION_AVAILABLE:
            # Create minimal mock if LunarInjectionResult not available
            class MockResult:
                def __init__(self, arr_vec, target_rel_vel):
                    self.arrival_eci_vector = arr_vec
                    self.target_relative_velocity = target_rel_vel
            if hasattr(injection_result, 'arrival_eci_vector'):
                pass  # Use as-is
            else:
                raise ImportError("LunarInjectionResult not available")
        
        # Step 1: Convert ECI arrival vector to station frame
        # Note: injection_result.arrival_eci_vector is in m/s, convert to km/s
        v_arrival_eci_km_s = injection_result.arrival_eci_vector / 1000.0
        
        coupling_result = self.reference_frame.eci_to_station_frame(
            v_arrival_eci_km_s,
            station_state
        )
        
        # Step 2: Find matching lane
        matched_lane = self.find_matching_lane(injection_result)
        
        if matched_lane is None:
            return (None, coupling_result, None)
        
        # Step 3: Calculate energy transfer
        # Assume packet couples to stream velocity (decelerates to stream speed)
        v_stream_rel = np.array([station_state.stream_velocity_mag, 0.0, 0.0])
        v_initial_rel = coupling_result.v_packet_relative
        v_final_rel = v_stream_rel  # After coupling, moves with stream
        
        energy_transfer = self.reference_frame.calculate_energy_transfer(
            packet_mass,
            v_initial_rel,
            v_final_rel
        )
        
        # Step 4: Calculate momentum change
        momentum_change = packet_mass * (
            np.linalg.norm(v_final_rel) - np.linalg.norm(v_initial_rel)
        ) * 1000  # Convert to N·s
        
        # Step 5: Add packet to lane
        matched_lane.add_packet(packet_mass, v_initial_rel)
        
        # Step 6: Log event
        self._packet_counter += 1
        event = PacketStreamEvent(
            packet_id=f"PKT-{self._packet_counter:04d}",
            event_type="COUPLE",
            lane_type=matched_lane.lane_type,
            v_relative_km_s=coupling_result.relative_speed_magnitude,
            timestamp=timestamp,
            energy_transfer_J=energy_transfer,
            momentum_change_Ns=momentum_change
        )
        self.event_log.append(event)
        
        return (matched_lane, coupling_result, event)
    
    def process_skyhook_launch(
        self,
        station_state: StationState,
        target_relative_velocity_ms: float,  # m/s
        packet_mass: float = 100.0,  # kg
        timestamp: float = 0.0,
        launch_direction: np.ndarray = None  # Optional direction override
    ) -> Tuple[Optional[StreamLane], CouplingResult, Optional[PacketStreamEvent], np.ndarray]:
        """
        Process a payload being launched FROM the skyhook station.
        
        This is the inverse of process_lunar_injection: instead of capturing
        an incoming packet, we accelerate a payload from station speed to
        a target relative velocity, extracting momentum from the stream lane.
        
        Physics:
        - Payload starts at station velocity (v_station_eci)
        - Accelerates to v_target_eci = v_station_eci + v_target_relative
        - Momentum is REMOVED from the lane (decelerates the stream)
        - Energy must be supplied to accelerate the payload
        
        Args:
            station_state: Current state of the launching station
            target_relative_velocity_ms: Target relative velocity after launch (m/s)
            packet_mass: Mass of the payload being launched (kg)
            timestamp: Simulation timestamp (s)
            launch_direction: Optional direction vector for launch (ECI frame).
                            If None, uses station velocity direction.
        
        Returns:
            Tuple of (lane, coupling_result, event, v_launch_eci)
            - lane: The lane that provided momentum
            - coupling_result: Frame conversion results
            - event: DECOUPLE event log entry
            - v_launch_eci: Final launch velocity in ECI frame (km/s)
        """
        # Convert target relative velocity to km/s
        target_rel_km_s = target_relative_velocity_ms / 1000.0
        
        # Determine launch direction
        if launch_direction is None:
            # Default: launch in direction of station orbital motion
            v_station_mag = np.linalg.norm(station_state.velocity_eci)
            if v_station_mag < 1e-10:
                launch_direction = np.array([1.0, 0.0, 0.0])
            else:
                launch_direction = station_state.velocity_eci / v_station_mag
        
        launch_direction = np.asarray(launch_direction, dtype=float)
        if np.linalg.norm(launch_direction) < 1e-10:
            raise ValueError("launch_direction cannot be zero vector")
        launch_direction = launch_direction / np.linalg.norm(launch_direction)
        
        # Construct target relative velocity vector (km/s)
        v_target_relative = launch_direction * target_rel_km_s
        
        # Calculate ECI launch velocity: v_eci = v_station + v_relative
        v_launch_eci = station_state.velocity_eci + v_target_relative
        
        # Create a mock coupling result for the launch
        # (pretend we're analyzing the reverse process)
        coupling_result = CouplingResult(
            v_packet_eci=v_launch_eci,
            v_packet_relative=v_target_relative,
            v_stream_relative=np.array([0.0, 0.0, 0.0]),  # Stream at rest in station frame
            relative_speed_magnitude=target_rel_km_s,
            coupling_feasible=True,
            lane_classification=self.reference_frame._classify_lane(target_rel_km_s),
            kinetic_energy_eci=0.5 * np.linalg.norm(v_launch_eci)**2 * 1e6,
            kinetic_energy_relative=0.5 * target_rel_km_s**2 * 1e6
        )
        
        # Find matching lane based on target velocity
        class MockPacketSource:
            def __init__(self, v_rel_ms):
                self.target_relative_velocity = v_rel_ms
        
        mock_source = MockPacketSource(target_relative_velocity_ms)
        matched_lane = self.find_matching_lane(mock_source)
        
        if matched_lane is None:
            return (None, coupling_result, None, v_launch_eci)
        
        # Calculate energy required to accelerate payload
        # Payload starts at station speed (v_rel=0) and accelerates to v_target_relative
        v_initial_rel = np.array([0.0, 0.0, 0.0])  # At rest relative to station
        v_final_rel = v_target_relative
        
        energy_required = self.reference_frame.calculate_energy_transfer(
            packet_mass,
            v_initial_rel,
            v_final_rel
        )
        # Positive energy means we need to ADD energy to accelerate
        
        # Calculate momentum extracted from lane
        # Lane loses momentum equal to what payload gains
        momentum_extracted = packet_mass * np.linalg.norm(v_target_relative) * 1000  # N·s
        
        # Remove momentum from lane (decelerates the stream)
        matched_lane.remove_packet(packet_mass, v_target_relative)
        
        # Log decouple event
        self._packet_counter += 1
        event = PacketStreamEvent(
            packet_id=f"LAUNCH-{self._packet_counter:04d}",
            event_type="DECOUPLE",
            lane_type=matched_lane.lane_type,
            v_relative_km_s=target_rel_km_s,
            timestamp=timestamp,
            energy_transfer_J=energy_required,  # Positive = energy consumed
            momentum_change_Ns=-momentum_extracted  # Negative = momentum removed from lane
        )
        self.event_log.append(event)
        
        return (matched_lane, coupling_result, event, v_launch_eci)
    
    def calculate_total_momentum_flux(self, time_window_s: float = 1.0) -> Dict[LaneType, float]:
        """
        Calculate total momentum flux across all lanes.
        
        Args:
            time_window_s: Time window for flux calculation (seconds).
                          Flux = accumulated_momentum / time_window_s
        
        Returns:
            Dictionary mapping lane type to momentum flux (N)
        """
        flux_by_lane = {}
        for lane in self.lanes:
            flux_by_lane[lane.lane_type] = lane.get_true_momentum_flux(time_window_s)
        return flux_by_lane
    
    def calculate_total_energy_budget(
        self,
        time_window_s: float = 3600.0  # 1 hour default
    ) -> Dict[str, float]:
        """
        Calculate energy budget for coupling operations.
        
        Args:
            time_window_s: Time window for calculation (seconds)
            
        Returns:
            Dictionary with energy statistics (Joules)
        """
        # Filter events in time window
        recent_events = [
            e for e in self.event_log 
            if e.timestamp >= (max(ev.timestamp for ev in self.event_log) - time_window_s)
        ] if self.event_log else []
        
        total_energy_in = sum(
            abs(e.energy_transfer_J) for e in recent_events 
            if e.energy_transfer_J > 0
        )
        total_energy_out = sum(
            abs(e.energy_transfer_J) for e in recent_events 
            if e.energy_transfer_J < 0
        )
        
        return {
            'total_energy_input_J': total_energy_in,
            'total_energy_recovered_J': total_energy_out,
            'net_energy_J': total_energy_in - total_energy_out,
            'event_count': len(recent_events)
        }
    
    def get_lane_statistics(self, time_window_s: float = 1.0) -> List[Dict]:
        """
        Get statistics for all lanes.
        
        Args:
            time_window_s: Time window for flux calculation (seconds)
        
        Returns:
            List of dictionaries with lane statistics
        """
        stats = []
        for lane in self.lanes:
            stats.append({
                'lane_type': lane.lane_type.value,
                'v_min_km_s': lane.v_min_km_s,
                'v_max_km_s': lane.v_max_km_s,
                'mass_flow_rate_kg_s': lane.mass_flow_rate,
                'packet_count': lane.packet_count,
                'accumulated_momentum_kgms': lane.accumulated_momentum,
                'true_momentum_flux_N': lane.get_true_momentum_flux(time_window_s),
                'efficiency': lane.lane_efficiency,
                'design_momentum_flux_N': lane.calculate_design_momentum_flux()
            })
        return stats
    
    def reset(self):
        """Reset all lanes and clear event log."""
        for lane in self.lanes:
            lane.packet_count = 0
            lane.accumulated_momentum = 0.0
            lane.lane_efficiency = 1.0
        self.event_log.clear()
        self._packet_counter = 0


def create_default_multi_lane_stream() -> MultiLaneStream:
    """
    Factory function to create a standard multi-lane stream configuration.
    
    Returns:
        MultiLaneStream with SLOW/STANDARD/FAST lanes
    """
    return MultiLaneStream()


def validate_lane_configuration(lanes: List[StreamLane]) -> bool:
    """
    Validate that lane configuration is physically consistent.
    
    Checks:
    - No overlapping velocity ranges
    - All velocities are positive
    - Mass flow rates are positive
    
    Args:
        lanes: List of StreamLane to validate
        
    Returns:
        True if configuration is valid
    """
    # Check for overlaps
    sorted_lanes = sorted(lanes, key=lambda l: l.v_min_km_s)
    for i in range(len(sorted_lanes) - 1):
        if sorted_lanes[i].v_max_km_s > sorted_lanes[i+1].v_min_km_s:
            return False
    
    # Check positive values
    for lane in lanes:
        if lane.v_min_km_s <= 0 or lane.v_max_km_s <= 0:
            return False
        if lane.mass_flow_rate <= 0:
            return False
    
    return True
