"""Test that sweep factories create non-trivial stream topology."""
import numpy as np
import pytest

MIN_PACKETS = 3  # Minimum for meaningful cascade dynamics

def test_quick_profile_sweep_topology():
    from scripts.quick_profile_sweep import _make_stream_factory, PROFILES
    for name, params in PROFILES.items():
        factory = _make_stream_factory(params)
        stream = factory()
        assert len(stream.packets) >= MIN_PACKETS, \
            f"Profile '{name}': {len(stream.packets)} packets < {MIN_PACKETS} minimum"

def test_extended_velocity_sweep_topology():
    from scripts.extended_velocity_sweep import create_stream_factory
    params = {"u": 1600.0, "mp": 8.0, "radius": 0.1, "k_fp": 6000.0}
    stream = create_stream_factory(params)()
    assert len(stream.packets) >= MIN_PACKETS

def test_sweep_fault_cascade_topology():
    from scripts.sweep_fault_cascade import create_stream_factory_with_nodes
    stream = create_stream_factory_with_nodes(10)()
    assert len(stream.packets) >= MIN_PACKETS

def test_packets_have_spatial_distribution():
    """Packets must not all be at the origin."""
    from scripts.quick_profile_sweep import _make_stream_factory, PROFILES
    params = PROFILES['operational']
    stream = _make_stream_factory(params)()
    positions = [p.body.position for p in stream.packets]
    if len(positions) > 1:
        dists = [np.linalg.norm(positions[i] - positions[0]) for i in range(1, len(positions))]
        assert max(dists) > 0, "All packets at same position — no spatial distribution"
