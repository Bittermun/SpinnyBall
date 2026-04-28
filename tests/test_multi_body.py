"""
Unit tests for multi-body packet stream dynamics.
"""

from __future__ import annotations

import numpy as np
import pytest

from dynamics.multi_body import (
    MultiBodyStream,
    Packet,
    SNode,
    PacketState,
    EventQueue,
    CaptureEvent,
    ReleaseEvent,
)
from dynamics.rigid_body import RigidBody


class TestSNode:
    """Test S-Node functionality."""
    
    def test_initialization(self):
        """S-Node initializes correctly."""
        position = np.array([100.0, 0.0, 0.0])
        node = SNode(id=0, position=position)
        
        assert np.allclose(node.position, position)
        assert node.capture_radius == 10.0
        assert node.max_packets == 10
        assert node.eta_ind_min == 0.82
        assert len(node.held_packets) == 0
    
    def test_position_validation(self):
        """S-Node validates position shape."""
        with pytest.raises(ValueError):
            SNode(id=0, position=np.array([1.0, 2.0]))  # Wrong shape
    
    def test_can_capture(self):
        """Test capture condition."""
        node = SNode(id=0, position=np.array([0.0, 0.0, 0.0]), max_packets=5)
        
        # Can capture when below max and eta_ind sufficient
        assert node.can_capture(0.9)
        assert not node.can_capture(0.8)  # Below eta_ind_min
        
        # Cannot capture when at max capacity
        node.held_packets = [0, 1, 2, 3, 4]
        assert not node.can_capture(0.9)
    
    def test_distance_to(self):
        """Test distance calculation."""
        node = SNode(id=0, position=np.array([0.0, 0.0, 0.0]))
        position = np.array([3.0, 4.0, 0.0])
        
        distance = node.distance_to(position)
        assert np.abs(distance - 5.0) < 1e-9


class TestPacket:
    """Test Packet functionality."""
    
    def test_initialization(self):
        """Packet initializes correctly."""
        mass = 0.05
        I = np.diag([0.0001, 0.00011, 0.00009])
        body = RigidBody(mass, I)
        
        packet = Packet(id=0, body=body)
        
        assert packet.id == 0
        assert packet.body == body
        assert packet.state == PacketState.FREE
        assert packet.current_node is None
        assert packet.eta_ind == 1.0
    
    def test_position_property(self):
        """Packet position property delegates to body."""
        mass = 0.05
        I = np.diag([0.0001, 0.00011, 0.00009])
        position = np.array([1.0, 2.0, 3.0])
        body = RigidBody(mass, I, position=position)
        
        packet = Packet(id=0, body=body)
        
        assert np.allclose(packet.position, position)


class TestEventQueue:
    """Test EventQueue functionality."""
    
    def test_initialization(self):
        """EventQueue initializes correctly."""
        queue = EventQueue()
        
        assert queue.events == []
        assert queue.current_time == 0.0
    
    def test_add_capture(self):
        """Test adding capture event."""
        queue = EventQueue()
        queue.add_capture(1.0, 0, 1, 0.9)
        
        assert len(queue.events) == 1
        assert isinstance(queue.events[0], CaptureEvent)
        assert queue.events[0].time == 1.0
        assert queue.events[0].packet_id == 0
        assert queue.events[0].node_id == 1
    
    def test_add_release(self):
        """Test adding release event."""
        queue = EventQueue()
        target_velocity = np.array([1.0, 0.0, 0.0])
        queue.add_release(2.0, 0, 1, target_velocity)
        
        assert len(queue.events) == 1
        assert isinstance(queue.events[0], ReleaseEvent)
        assert queue.events[0].time == 2.0
        assert np.allclose(queue.events[0].target_velocity, target_velocity)
    
    def test_events_sorted(self):
        """Test events are sorted by time."""
        queue = EventQueue()
        queue.add_capture(3.0, 0, 1, 0.9)
        queue.add_capture(1.0, 1, 2, 0.9)
        queue.add_capture(2.0, 2, 3, 0.9)
        
        assert queue.events[0].time == 1.0
        assert queue.events[1].time == 2.0
        assert queue.events[2].time == 3.0
    
    def test_get_events_at(self):
        """Test getting events at or before time."""
        queue = EventQueue()
        queue.add_capture(1.0, 0, 1, 0.9)
        queue.add_capture(2.0, 1, 2, 0.9)
        queue.add_capture(3.0, 2, 3, 0.9)
        
        events = queue.get_events_at(2.0)
        assert len(events) == 2
    
    def test_remove_processed(self):
        """Test removing processed events."""
        queue = EventQueue()
        queue.add_capture(1.0, 0, 1, 0.9)
        queue.add_capture(2.0, 1, 2, 0.9)
        queue.add_capture(3.0, 2, 3, 0.9)
        
        queue.remove_processed(2.0)
        
        assert len(queue.events) == 1
        assert queue.events[0].time == 3.0


class TestMultiBodyStream:
    """Test MultiBodyStream functionality."""
    
    @pytest.fixture
    def simple_stream(self):
        """Create a simple multi-body stream for testing."""
        # Create 3 packets
        mass = 0.05
        I = np.diag([0.0001, 0.00011, 0.00009])
        
        packets = []
        for i in range(3):
            # Position packets far from nodes to avoid auto-capture
            position = np.array([i * 100.0, 0.0, 0.0])
            velocity = np.array([100.0, 0.0, 0.0])
            body = RigidBody(mass, I, position=position, velocity=velocity)
            packets.append(Packet(id=i, body=body))
        
        # Create 2 S-Nodes
        nodes = [
            SNode(id=0, position=np.array([15.0, 0.0, 0.0]), capture_radius=10.0),
            SNode(id=1, position=np.array([35.0, 0.0, 0.0]), capture_radius=10.0),
        ]
        
        return MultiBodyStream(packets=packets, nodes=nodes, stream_velocity=100.0)
    
    def test_initialization(self, simple_stream):
        """MultiBodyStream initializes correctly."""
        assert len(simple_stream.packets) == 3
        assert len(simple_stream.nodes) == 2
        assert simple_stream.stream_velocity == 100.0
        assert simple_stream.time == 0.0
    
    def test_check_capture_conditions(self, simple_stream):
        """Test capture condition checking."""
        packet = simple_stream.packets[0]
        node = simple_stream.nodes[0]
        
        can_capture, distance = simple_stream.check_capture_conditions(packet, node)
        
        assert isinstance(can_capture, (bool, np.bool_))
        assert isinstance(distance, (float, np.floating))
        assert distance >= 0
    
    def test_process_capture_event(self, simple_stream):
        """Test processing capture event."""
        packet = simple_stream.packets[0]
        node = simple_stream.nodes[0]
        
        event = CaptureEvent(time=0.0, packet_id=0, node_id=0, eta_ind=0.9)
        simple_stream.process_capture_event(event)
        
        assert packet.state == PacketState.CAPTURED
        assert packet.current_node == 0
        assert packet.id in node.held_packets
    
    def test_process_release_event(self, simple_stream):
        """Test processing release event."""
        packet = simple_stream.packets[0]
        node = simple_stream.nodes[0]
        
        # First capture
        node.held_packets.append(packet.id)
        packet.state = PacketState.CAPTURED
        packet.current_node = 0
        
        # Then release
        target_velocity = np.array([100.0, 0.0, 0.0])
        event = ReleaseEvent(time=0.0, packet_id=0, node_id=0, target_velocity=target_velocity)
        simple_stream.process_release_event(event)
        
        assert packet.state == PacketState.FREE
        assert packet.current_node is None
        assert packet.id not in node.held_packets
        assert np.allclose(packet.body.velocity, target_velocity)
    
    def test_get_stream_metrics(self, simple_stream):
        """Test getting stream metrics."""
        metrics = simple_stream.get_stream_metrics()
        
        assert metrics["total_packets"] == 3
        assert metrics["free_packets"] == 3
        assert metrics["captured_packets"] == 0
        assert "avg_eta_ind" in metrics
        assert "time" in metrics
    
    def test_integrate_free_packets(self, simple_stream):
        """Test integrating free packets."""
        def zero_torque(packet_id, t, state):
            return np.array([0.0, 0.0, 0.0])
        
        result = simple_stream.integrate(dt=0.01, torques=zero_torque)
        
        assert result["time"] == 0.01
        assert len(result["packets"]) == 3
        assert "events_processed" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
