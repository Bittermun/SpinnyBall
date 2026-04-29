"""
Test script for digital twin backend API endpoints.

This script verifies the FastAPI backend endpoints work correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.app import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint."""
    print("Testing root endpoint...")
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "SGMS MRT Digital Twin API"
    assert data["status"] == "running"
    print("  ✓ Root endpoint working")


def test_health_endpoint():
    """Test health check endpoint."""
    print("Testing health endpoint...")
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    print("  ✓ Health endpoint working")


def test_simulation_init():
    """Test simulation initialization."""
    print("Testing simulation initialization...")
    params = {
        "n_packets": 5,
        "n_nodes": 2,
        "velocity": 1600.0,
        "dt": 0.01,
    }
    response = client.post("/simulation/init", json=params)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "initialized"
    assert data["n_packets"] == 5
    assert data["n_nodes"] == 2
    print("  ✓ Simulation initialization working")


def test_simulation_start_stop():
    """Test simulation start/stop."""
    print("Testing simulation start/stop...")
    
    # Initialize first
    params = {"n_packets": 3, "n_nodes": 1, "velocity": 1000.0, "dt": 0.01}
    client.post("/simulation/init", json=params)
    
    # Start
    response = client.post("/simulation/start")
    assert response.status_code == 200
    assert response.json()["status"] == "running"
    
    # Stop
    response = client.post("/simulation/stop")
    assert response.status_code == 200
    assert response.json()["status"] == "stopped"
    
    print("  ✓ Simulation start/stop working")


def test_simulation_step():
    """Test simulation stepping."""
    print("Testing simulation step...")
    
    # Initialize
    params = {"n_packets": 3, "n_nodes": 1, "velocity": 1000.0, "dt": 0.01}
    client.post("/simulation/init", json=params)
    
    # Step
    response = client.post("/simulation/step", json=params)
    assert response.status_code == 200
    data = response.json()
    assert "time" in data
    assert data["time"] > 0
    assert "events_processed" in data
    
    print("  ✓ Simulation step working")


def test_simulation_state():
    """Test getting simulation state."""
    print("Testing simulation state retrieval...")
    
    # Initialize
    params = {"n_packets": 3, "n_nodes": 1, "velocity": 1000.0, "dt": 0.01}
    client.post("/simulation/init", json=params)
    
    # Get state
    response = client.get("/simulation/state")
    assert response.status_code == 200
    data = response.json()
    assert "time" in data
    assert "packets" in data
    assert "nodes" in data
    assert "metrics" in data
    assert len(data["packets"]) == 3
    assert len(data["nodes"]) == 1
    
    print("  ✓ Simulation state retrieval working")


def test_monte_carlo_endpoint():
    """Test Monte Carlo endpoint (small sample)."""
    print("Testing Monte Carlo endpoint...")
    
    request = {
        "n_realizations": 5,  # Small sample for quick test
        "n_packets": 3,
        "n_nodes": 1,
        "velocity": 1600.0,
        "time_horizon": 1.0,  # Short horizon
        "dt": 0.01,
    }
    
    response = client.post("/monte-carlo/run", json=request)
    assert response.status_code == 200
    data = response.json()
    assert data["n_realizations"] == 5
    assert "success_rate" in data
    assert "cascade_probability" in data
    assert "eta_ind_min_mean" in data
    assert "stress_max_mean" in data
    
    print("  ✓ Monte Carlo endpoint working")


if __name__ == "__main__":
    print("=" * 60)
    print("Digital Twin Backend API Test")
    print("=" * 60)
    
    try:
        test_root_endpoint()
        test_health_endpoint()
        test_simulation_init()
        test_simulation_start_stop()
        test_simulation_step()
        test_simulation_state()
        test_monte_carlo_endpoint()
        
        print("\n" + "=" * 60)
        print("All backend API tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
