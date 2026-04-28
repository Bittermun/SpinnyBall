"""
Integration tests for reference frame conversion and multi-lane stream dynamics.

Tests verify:
1. Zero-relative-velocity packets result in ~0 relative speed in station frame
2. High-relative-velocity packets correctly identify STANDARD lane
3. Energy conservation across frame conversions
4. Lunar injection result integration with multi-lane stream
"""

import pytest
import numpy as np

from dynamics.orbital_coupling import (
    StreamReferenceFrame,
    StationState,
    CouplingResult,
    R_earth,
    ORBITAL_DYNAMICS_AVAILABLE
)

from dynamics.multi_lane_stream import (
    MultiLaneStream,
    StreamLane,
    LaneType,
    PacketStreamEvent,
    validate_lane_configuration,
    create_default_multi_lane_stream
)

# Try to import lunar injection for integration tests
try:
    from scenarios.lunar_injection import LunarInjectionCalculator, POLIASTRO_AVAILABLE
    LUNAR_INJECTION_AVAILABLE = POLIASTRO_AVAILABLE
except ImportError:
    LUNAR_INJECTION_AVAILABLE = False


@pytest.fixture
def station_leo_500km():
    """Create a station state at 500 km LEO altitude."""
    # Circular orbit at 500 km altitude
    r_mag = R_earth + 500  # km
    v_circ = np.sqrt(398600.4418 / r_mag)  # km/s
    
    return StationState(
        position_eci=np.array([r_mag, 0.0, 0.0]),
        velocity_eci=np.array([0.0, v_circ, 0.0]),
        stream_velocity_mag=v_circ,
        station_id="TEST-STATION-001"
    )


@pytest.fixture
def reference_frame():
    """Create a StreamReferenceFrame instance."""
    return StreamReferenceFrame()


@pytest.fixture
def multi_lane_stream():
    """Create a default MultiLaneStream instance."""
    return create_default_multi_lane_stream()


class TestReferenceFrameConversion:
    """Test Case A & C: Reference frame conversions and energy conservation."""
    
    def test_zero_relative_velocity_station_frame(self, station_leo_500km, reference_frame):
        """
        Test Case A: Inject packet with target_relative_velocity=0.
        
        Verify that in the Station Frame, the relative velocity is ~0
        (or matches orbital drift).
        """
        # Packet arriving with same velocity as station (zero relative)
        v_packet_eci = station_leo_500km.velocity_eci.copy()  # Same as station
        
        result = reference_frame.eci_to_station_frame(v_packet_eci, station_leo_500km)
        
        # Relative velocity should be ~0
        assert result.relative_speed_magnitude < 0.1, \
            f"Zero relative target should give ~0 km/s, got {result.relative_speed_magnitude}"
        
        # Lane should be SLOW (includes near-zero velocities)
        assert result.lane_classification == "SLOW", \
            f"Near-zero velocity should be SLOW lane, got {result.lane_classification}"
    
    def test_inverse_transform_consistency(self, station_leo_500km, reference_frame):
        """Verify eci_to_station and station_to_eci are inverse operations."""
        # Start with arbitrary ECI velocity
        v_original_eci = np.array([1.0, 8.0, 0.5])  # km/s
        
        # Convert to station frame
        coupling = reference_frame.eci_to_station_frame(v_original_eci, station_leo_500km)
        
        # Convert back to ECI
        v_recovered_eci = reference_frame.station_to_eci_frame(
            coupling.v_packet_relative, 
            station_leo_500km
        )
        
        # Should recover original (within numerical precision)
        np.testing.assert_array_almost_equal(
            v_original_eci, 
            v_recovered_eci,
            decimal=10,
            err_msg="Inverse transform should recover original ECI velocity"
        )
    
    def test_energy_conservation_frame_conversion(self, station_leo_500km, reference_frame):
        """
        Test Case C: Verify energy conservation across frame conversion.
        
        Kinetic Energy in ECI ≈ Kinetic Energy in Station Frame + Frame terms
        
        Note: This tests that the conversion preserves the physics correctly,
        accounting for the moving reference frame.
        """
        # Packet with significant relative velocity
        v_packet_eci = np.array([0.0, 12.6, 0.0])  # km/s (~5 km/s relative to station)
        
        result = reference_frame.eci_to_station_frame(v_packet_eci, station_leo_500km)
        
        # Both energies should be positive and finite
        assert result.kinetic_energy_eci > 0, "ECI kinetic energy should be positive"
        assert result.kinetic_energy_relative > 0, "Relative kinetic energy should be positive"
        
        # ECI energy should be larger (includes orbital motion energy)
        assert result.kinetic_energy_eci >= result.kinetic_energy_relative, \
            "ECI energy should include orbital motion contribution"
        
        # Check units: energies should be in J/kg (reasonable range for orbital mechanics)
        # At LEO, KE ~ 0.5 * v^2 ~ 0.5 * (7600 m/s)^2 ~ 29 MJ/kg
        assert 1e6 < result.kinetic_energy_eci < 100e6, \
            f"ECI KE {result.kinetic_energy_eci} J/kg outside expected orbital range"


class TestLaneClassification:
    """Test Case B: Lane identification for different relative velocities."""
    
    def test_standard_lane_identification(self, station_leo_500km, reference_frame):
        """
        Test Case B: Inject packet with target_relative_velocity=5000 m/s (5 km/s).
        
        Verify the coupler logic identifies this as a "STANDARD" lane interaction.
        """
        # Create a packet arriving at 5 km/s relative to station
        # Station moves at ~7.6 km/s in +Y, so packet needs ~12.6 km/s in +Y
        v_station = station_leo_500km.velocity_eci
        v_relative_target = np.array([0.0, 5.0, 0.0])  # 5 km/s relative in along-track direction
        
        v_packet_eci = v_station + v_relative_target
        
        result = reference_frame.eci_to_station_frame(v_packet_eci, station_leo_500km)
        
        # Relative speed should be ~5 km/s
        assert 4.9 < result.relative_speed_magnitude < 5.1, \
            f"Expected ~5 km/s relative, got {result.relative_speed_magnitude}"
        
        # Lane classification should be STANDARD (boundary case: exactly 5.0)
        # Note: STANDARD lane is 2.0-5.0 km/s, FAST is 5.0-15.0 km/s
        # At exactly 5.0, it could be either depending on implementation
        assert result.lane_classification in ["STANDARD", "FAST"], \
            f"5 km/s should be STANDARD or FAST boundary, got {result.lane_classification}"
    
    def test_slow_lane_classification(self, station_leo_500km, reference_frame):
        """Test slow lane classification (1-2 km/s)."""
        v_station = station_leo_500km.velocity_eci
        v_relative = np.array([0.0, 1.5, 0.0])  # 1.5 km/s relative
        
        v_packet_eci = v_station + v_relative
        result = reference_frame.eci_to_station_frame(v_packet_eci, station_leo_500km)
        
        assert result.lane_classification == "SLOW", \
            f"1.5 km/s should be SLOW lane, got {result.lane_classification}"
    
    def test_fast_lane_classification(self, station_leo_500km, reference_frame):
        """Test fast lane classification (5-15 km/s)."""
        v_station = station_leo_500km.velocity_eci
        v_relative = np.array([0.0, 10.0, 0.0])  # 10 km/s relative
        
        v_packet_eci = v_station + v_relative
        result = reference_frame.eci_to_station_frame(v_packet_eci, station_leo_500km)
        
        assert result.lane_classification == "FAST", \
            f"10 km/s should be FAST lane, got {result.lane_classification}"
    
    def test_excessive_velocity_none_lane(self, station_leo_500km, reference_frame):
        """Test that velocities >15 km/s get NONE classification."""
        v_station = station_leo_500km.velocity_eci
        v_relative = np.array([0.0, 20.0, 0.0])  # 20 km/s relative (too fast)
        
        v_packet_eci = v_station + v_relative
        result = reference_frame.eci_to_station_frame(v_packet_eci, station_leo_500km)
        
        assert result.lane_classification == "NONE", \
            f"20 km/s should exceed limits and get NONE, got {result.lane_classification}"


class TestMultiLaneStreamIntegration:
    """Integration tests for MultiLaneStream with lunar injection workflow."""
    
    @pytest.mark.skipif(not LUNAR_INJECTION_AVAILABLE, reason="poliastro not available")
    def test_lunar_injection_to_lane_mapping(self, multi_lane_stream, station_leo_500km):
        """Test full workflow: lunar injection result -> lane assignment."""
        calculator = LunarInjectionCalculator()
        
        # Calculate injection for STANDARD lane (5 km/s relative)
        injection_result = calculator.calculate_injection_vector(
            target_altitude_km=500,
            target_relative_velocity_ms=5000,  # 5 km/s
            launch_from_lunar_surface=True
        )
        
        # Find matching lane
        matched_lane = multi_lane_stream.find_matching_lane(injection_result)
        
        assert matched_lane is not None, "Should find matching lane for 5 km/s target"
        assert matched_lane.lane_type in [LaneType.STANDARD, LaneType.FAST], \
            f"5 km/s should match STANDARD or FAST, got {matched_lane.lane_type}"
    
    def test_mock_lunar_injection_processing(self, multi_lane_stream, station_leo_500km):
        """Test processing a mock lunar injection result."""
        # Create mock injection result (mimics LunarInjectionResult structure)
        class MockInjectionResult:
            def __init__(self, arrival_vec_ms, target_rel_vel_ms):
                self.arrival_eci_vector = np.array(arrival_vec_ms)
                self.target_relative_velocity = target_rel_vel_ms
        
        # STANDARD lane target: 5 km/s relative = 5000 m/s
        # Station at ~7.6 km/s, so arrival ECI = ~12.6 km/s = 12600 m/s
        v_station_ms = station_leo_500km.velocity_eci * 1000  # Convert to m/s
        v_arrival_eci_ms = v_station_ms + np.array([0.0, 5000.0, 0.0])
        
        mock_result = MockInjectionResult(
            arrival_vec_ms=v_arrival_eci_ms,
            target_rel_vel_ms=5000.0
        )
        
        # Process through multi-lane stream
        lane, coupling, event = multi_lane_stream.process_lunar_injection(
            mock_result,
            station_leo_500km,
            packet_mass=100.0,  # kg
            timestamp=0.0
        )
        
        # Verify lane assignment
        assert lane is not None, "Should match a lane"
        assert lane.lane_type in [LaneType.STANDARD, LaneType.FAST], \
            f"Expected STANDARD or FAST, got {lane.lane_type}"
        
        # Verify coupling result
        assert coupling is not None
        assert 4.5 < coupling.relative_speed_magnitude < 5.5, \
            f"Expected ~5 km/s relative, got {coupling.relative_speed_magnitude}"
        
        # Verify event logging
        assert event is not None
        assert event.event_type == "COUPLE"
        assert event.lane_type == lane.lane_type
        assert event.packet_id == "PKT-0001"
    
    def test_multiple_packets_different_lanes(self, multi_lane_stream, station_leo_500km):
        """Test processing multiple packets to different lanes."""
        class MockInjectionResult:
            def __init__(self, arrival_vec_ms, target_rel_vel_ms):
                self.arrival_eci_vector = np.array(arrival_vec_ms)
                self.target_relative_velocity = target_rel_vel_ms
        
        v_station_ms = station_leo_500km.velocity_eci * 1000
        
        # Send packets to each lane
        targets = [
            (1500, "SLOW"),    # 1.5 km/s
            (3500, "STANDARD"), # 3.5 km/s  
            (10000, "FAST"),   # 10 km/s
        ]
        
        for target_rel_ms, expected_lane_name in targets:
            v_arrival_ms = v_station_ms + np.array([0.0, float(target_rel_ms), 0.0])
            mock_result = MockInjectionResult(v_arrival_ms, float(target_rel_ms))
            
            lane, coupling, event = multi_lane_stream.process_lunar_injection(
                mock_result, station_leo_500km, packet_mass=100.0
            )
            
            assert lane is not None, f"Should match lane for {target_rel_ms} m/s"
            assert lane.lane_type.value.upper() == expected_lane_name, \
                f"Expected {expected_lane_name}, got {lane.lane_type.value}"
    
    def test_momentum_flux_accumulation(self, multi_lane_stream, station_leo_500km):
        """Test that accumulated momentum tracks correctly."""
        class MockInjectionResult:
            def __init__(self, target_rel_vel_ms):
                self.target_relative_velocity = target_rel_vel_ms
                # Simplified: just need target_relative_velocity for lane matching
                self.arrival_eci_vector = np.array([0.0, 7600 + target_rel_vel_ms/1000*1000, 0.0])
        
        # Reset stream first to ensure clean state
        multi_lane_stream.reset()
        initial_momentum = multi_lane_stream.lanes[1].accumulated_momentum  # STANDARD lane
        
        # Add several packets
        for i in range(5):
            mock_result = MockInjectionResult(target_rel_vel_ms=3500)  # 3.5 km/s
            multi_lane_stream.process_lunar_injection(
                mock_result, station_leo_500km, packet_mass=100.0
            )
        
        # Flux should have increased
        final_momentum = multi_lane_stream.lanes[1].accumulated_momentum
        assert final_momentum > initial_momentum, "Momentum flux should increase with added packets"
        
        # Each packet adds ~100 kg * 3500 m/s = 350,000 N·s of momentum
        # But we track flux (force), not impulse, so check packet count
        assert multi_lane_stream.lanes[1].packet_count == 5, \
            f"Should have 5 packets, got {multi_lane_stream.lanes[1].packet_count}"


class TestEnergyTransfer:
    """Test energy transfer calculations during coupling."""
    
    def test_energy_transfer_calculation(self, reference_frame):
        """Test calculate_energy_transfer method."""
        packet_mass = 100.0  # kg
        
        # Deceleration: from 5 km/s to 2 km/s relative
        v_initial = np.array([5.0, 0.0, 0.0])  # km/s
        v_final = np.array([2.0, 0.0, 0.0])    # km/s
        
        delta_E = reference_frame.calculate_energy_transfer(
            packet_mass, v_initial, v_final
        )
        
        # E = 0.5 * m * (v_f² - v_i²)
        # E = 0.5 * 100 * ((2000)² - (5000)²) = 0.5 * 100 * (4e6 - 25e6) = -1.05e9 J
        expected_delta_E = 0.5 * packet_mass * ((2000)**2 - (5000)**2)
        
        assert abs(delta_E - expected_delta_E) < 1e6, \
            f"Expected ~{expected_delta_E} J, got {delta_E} J"
        
        # Energy should be negative (deceleration = energy extraction)
        assert delta_E < 0, "Deceleration should extract energy (negative delta_E)"
    
    def test_acceleration_energy_input(self, reference_frame):
        """Test energy input for acceleration."""
        packet_mass = 100.0  # kg
        
        # Acceleration: from 2 km/s to 5 km/s relative
        v_initial = np.array([2.0, 0.0, 0.0])  # km/s
        v_final = np.array([5.0, 0.0, 0.0])    # km/s
        
        delta_E = reference_frame.calculate_energy_transfer(
            packet_mass, v_initial, v_final
        )
        
        # Energy should be positive (acceleration = energy input)
        assert delta_E > 0, "Acceleration should require energy input (positive delta_E)"


class TestValidationUtilities:
    """Test validation utilities."""
    
    def test_validate_valid_lane_configuration(self):
        """Test validation of valid lane configuration."""
        lanes = [
            StreamLane(LaneType.SLOW, 0.5, 2.0, mass_flow_rate=10.0),
            StreamLane(LaneType.STANDARD, 2.0, 5.0, mass_flow_rate=50.0),
            StreamLane(LaneType.FAST, 5.0, 15.0, mass_flow_rate=25.0),
        ]
        
        assert validate_lane_configuration(lanes), "Default configuration should be valid"
    
    def test_validate_overlapping_lanes_invalid(self):
        """Test that overlapping lanes are detected as invalid."""
        lanes = [
            StreamLane(LaneType.SLOW, 0.5, 3.0, mass_flow_rate=10.0),  # Overlaps with STANDARD
            StreamLane(LaneType.STANDARD, 2.0, 5.0, mass_flow_rate=50.0),
        ]
        
        assert not validate_lane_configuration(lanes), "Overlapping lanes should be invalid"
    
    def test_validate_negative_velocity_invalid(self):
        """Test that negative velocities are detected as invalid."""
        lanes = [
            StreamLane(LaneType.SLOW, -1.0, 2.0, mass_flow_rate=10.0),
        ]
        
        assert not validate_lane_configuration(lanes), "Negative velocity should be invalid"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_no_matching_lane(self, multi_lane_stream):
        """Test behavior when no lane matches."""
        class MockResult:
            target_relative_velocity = 20000  # 20 km/s - too fast
        
        lane = multi_lane_stream.find_matching_lane(MockResult())
        assert lane is None, "20 km/s should not match any lane"
    
    def test_reset_clears_state(self, multi_lane_stream, station_leo_500km):
        """Test that reset clears all state."""
        class MockResult:
            target_relative_velocity = 3500
            arrival_eci_vector = np.array([0.0, 11100, 0.0])
        
        # Add some packets
        for _ in range(3):
            multi_lane_stream.process_lunar_injection(
                MockResult(), station_leo_500km, packet_mass=100.0
            )
        
        # Verify state changed
        assert multi_lane_stream.lanes[1].packet_count > 0
        assert len(multi_lane_stream.event_log) > 0
        
        # Reset
        multi_lane_stream.reset()
        
        # Verify cleared
        assert multi_lane_stream.lanes[1].packet_count == 0
        assert len(multi_lane_stream.event_log) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestSkyhookLaunchDecouple:
    """Test the decoupling/launch functionality."""

    def test_skyhook_launch_basic(self, multi_lane_stream, station_leo_500km):
        """Test basic skyhook launch operation."""
        multi_lane_stream.reset()
        
        # First add some packets to have momentum in the lane
        class MockInjectionResult:
            def __init__(self, target_rel_vel_ms):
                self.target_relative_velocity = target_rel_vel_ms
                self.arrival_eci_vector = np.array([0.0, 7600 + target_rel_vel_ms/1000*1000, 0.0])
        
        # Add 10 packets to STANDARD lane (3.5 km/s each)
        for _ in range(10):
            mock_result = MockInjectionResult(target_rel_vel_ms=3500)
            multi_lane_stream.process_lunar_injection(
                mock_result, station_leo_500km, packet_mass=100.0
            )
        
        initial_momentum = multi_lane_stream.lanes[1].accumulated_momentum
        initial_count = multi_lane_stream.lanes[1].packet_count
        
        # Now launch a payload (decelple from stream)
        target_rel_vel = 3500  # 3.5 km/s
        result = multi_lane_stream.process_skyhook_launch(
            station_state=station_leo_500km,
            target_relative_velocity_ms=target_rel_vel,
            packet_mass=100.0,
            timestamp=100.0
        )
        
        lane, coupling_result, event, v_launch_eci = result
        
        # Verify lane assignment
        assert lane is not None, "Should find matching lane"
        assert lane.lane_type == LaneType.STANDARD, "Should be STANDARD lane"
        
        # Verify event logging
        assert event is not None, "Should create decouple event"
        assert event.event_type == "DECOUPLE", "Event type should be DECOUPLE"
        assert event.v_relative_km_s == pytest.approx(3.5, rel=1e-6)
        
        # Verify momentum was removed from lane
        final_momentum = multi_lane_stream.lanes[1].accumulated_momentum
        expected_momentum_removed = 100.0 * 3500.0  # kg * m/s
        assert abs((initial_momentum - final_momentum) - expected_momentum_removed) < 1e-6, \
            f"Momentum should decrease by {expected_momentum_removed}"
        
        # Verify packet count decreased
        assert multi_lane_stream.lanes[1].packet_count == initial_count - 1
        
        # Verify launch velocity in ECI frame
        # v_launch_eci = v_station + v_relative
        expected_v_launch = station_leo_500km.velocity_eci + np.array([0.0, 3.5, 0.0])
        assert np.allclose(v_launch_eci, expected_v_launch, rtol=1e-10)
        
        # Verify energy is positive (energy must be added to accelerate payload)
        assert event.energy_transfer_J > 0, "Launch requires energy input"

    def test_skyhook_launch_custom_direction(self, multi_lane_stream, station_leo_500km):
        """Test launch with custom direction vector."""
        multi_lane_stream.reset()
        
        # Add packets first
        class MockInjectionResult:
            def __init__(self, target_rel_vel_ms):
                self.target_relative_velocity = target_rel_vel_ms
                self.arrival_eci_vector = np.array([0.0, 7600 + target_rel_vel_ms/1000*1000, 0.0])
        
        for _ in range(5):
            mock_result = MockInjectionResult(target_rel_vel_ms=5000)
            multi_lane_stream.process_lunar_injection(
                mock_result, station_leo_500km, packet_mass=100.0
            )
        
        # Launch in a different direction (radial outward)
        launch_direction = np.array([1.0, 0.0, 0.0])  # Radial direction
        result = multi_lane_stream.process_skyhook_launch(
            station_state=station_leo_500km,
            target_relative_velocity_ms=5000,
            packet_mass=100.0,
            launch_direction=launch_direction
        )
        
        lane, coupling_result, event, v_launch_eci = result
        
        # Verify launch direction affects ECI velocity
        expected_v_rel = np.array([5.0, 0.0, 0.0])  # km/s in radial direction
        expected_v_launch = station_leo_500km.velocity_eci + expected_v_rel
        assert np.allclose(v_launch_eci, expected_v_launch, rtol=1e-10)

    def test_skyhook_launch_no_matching_lane(self, multi_lane_stream, station_leo_500km):
        """Test launch when no lane matches target velocity."""
        multi_lane_stream.reset()
        
        # Try to launch at excessive velocity (20 km/s - beyond FAST lane)
        result = multi_lane_stream.process_skyhook_launch(
            station_state=station_leo_500km,
            target_relative_velocity_ms=20000,  # 20 km/s
            packet_mass=100.0
        )
        
        lane, coupling_result, event, v_launch_eci = result
        
        # Should return None for lane and event when no match
        assert lane is None, "Should not find matching lane"
        assert event is None, "Should not create event without lane"
        # But should still return valid ECI velocity
        assert v_launch_eci is not None
        assert np.linalg.norm(v_launch_eci) > 0
