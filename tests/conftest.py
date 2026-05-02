"""Pytest configuration - add repo paths for imports."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "monte_carlo"))
sys.path.insert(0, str(REPO_ROOT / "dynamics"))
sys.path.insert(0, str(REPO_ROOT / "control_layer"))

import pytest
import numpy as np
from params.canonical_values import MATERIAL_PROPERTIES, SIMULATION_PARAMS

@pytest.fixture
def canonical_gdbco():
    return MATERIAL_PROPERTIES['GdBCO']

@pytest.fixture
def canonical_ybco():
    return MATERIAL_PROPERTIES['YBCO']

@pytest.fixture
def realistic_stream():
    """A stream with 5+ packets and 10 nodes — not trivial topology."""
    from dynamics.multi_body import MultiBodyStream, Packet, SNode
    from dynamics.rigid_body import RigidBody
    mass, I = 8.0, np.diag([0.0001, 0.00011, 0.00009])
    omega = np.array([0.0, 0.0, 5236.0])
    packets = [
        Packet(id=i, body=RigidBody(mass, I,
            position=np.array([i*10.0, 0.0, 0.0]),
            velocity=np.array([1600.0, 0.0, 0.0]),
            angular_velocity=omega), radius=0.1, eta_ind=0.9)
        for i in range(5)
    ]
    nodes = [
        SNode(id=i, position=np.array([i*20.0, 0.0, 0.0]), k_fp=6000.0)
        for i in range(10)
    ]
    return MultiBodyStream(packets=packets, nodes=nodes, stream_velocity=1600.0)

