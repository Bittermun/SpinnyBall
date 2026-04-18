"""
FastAPI backend for Digital Twin dashboard.

Provides REST API endpoints for the digital twin visualization,
connecting the Python dynamics simulation to the HTML frontend.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np

from dynamics.rigid_body import RigidBody
from dynamics.multi_body import MultiBodyStream, Packet, SNode
from dynamics.stress_monitoring import verify_packet_stress
from dynamics.stiffness_verification import verify_anchor_stiffness
from monte_carlo.cascade_runner import CascadeRunner, MonteCarloConfig, Perturbation, PerturbationType
from monte_carlo.pass_fail_gates import create_default_gate_set, evaluate_monte_carlo_gates

app = FastAPI(title="SGMS MRT Digital Twin API", version="0.1.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Pydantic Models
# ============================================================

class SimulationParams(BaseModel):
    n_packets: int = 10
    n_nodes: int = 3
    velocity: float = 1600.0
    dt: float = 0.01
    eta_gate: float = 0.82
    stress_gate: float = 800.0


class PacketState(BaseModel):
    id: int
    position: List[float]
    velocity: List[float]
    quaternion: List[float]
    angular_velocity: List[float]
    eta_ind: float
    state: str


class NodeState(BaseModel):
    id: int
    position: List[float]
    captured_packets: List[int]


class SimulationState(BaseModel):
    time: float
    packets: List[PacketState]
    nodes: List[NodeState]
    metrics: Dict[str, float]


class MonteCarloRequest(BaseModel):
    n_realizations: int = 100
    n_packets: int = 10
    n_nodes: int = 3
    velocity: float = 1600.0
    time_horizon: float = 10.0
    dt: float = 0.01


class MonteCarloResponse(BaseModel):
    n_realizations: int
    n_success: int
    n_failure: int
    success_rate: float
    cascade_probability: float
    eta_ind_min_mean: float
    stress_max_mean: float
    meets_cascade_target: bool


# ============================================================
# Global State
# ============================================================

simulation_stream: Optional[MultiBodyStream] = None
simulation_running: bool = False
simulation_time: float = 0.0


# ============================================================
# API Endpoints
# ============================================================

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "SGMS MRT Digital Twin API",
        "version": "0.1.0",
        "status": "running",
    }


@app.post("/simulation/init")
async def init_simulation(params: SimulationParams):
    """Initialize simulation with given parameters."""
    global simulation_stream, simulation_running, simulation_time
    
    # Create packets
    mass = 0.05
    I = np.diag([0.0001, 0.00011, 0.00009])
    
    packets = []
    for i in range(params.n_packets):
        position = np.array([i * 10.0, 0.0, 0.0])
        velocity = np.array([params.velocity, 0.0, 0.0])
        body = RigidBody(mass, I, position=position, velocity=velocity)
        packets.append(Packet(id=i, body=body))
    
    # Create S-Nodes
    nodes = []
    for i in range(params.n_nodes):
        position = np.array([i * 20.0, 0.0, 0.0])
        nodes.append(SNode(position))
    
    # Create stream
    simulation_stream = MultiBodyStream(packets=packets, nodes=nodes, stream_velocity=params.velocity)
    simulation_running = False
    simulation_time = 0.0
    
    return {"status": "initialized", "n_packets": params.n_packets, "n_nodes": params.n_nodes}


@app.post("/simulation/start")
async def start_simulation():
    """Start simulation."""
    global simulation_running
    if simulation_stream is None:
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    simulation_running = True
    return {"status": "running"}


@app.post("/simulation/stop")
async def stop_simulation():
    """Stop simulation."""
    global simulation_running
    simulation_running = False
    return {"status": "stopped"}


@app.post("/simulation/reset")
async def reset_simulation():
    """Reset simulation to initial state."""
    global simulation_running, simulation_time
    simulation_running = False
    simulation_time = 0.0
    
    # Re-initialize would go here
    return {"status": "reset"}


@app.post("/simulation/step")
async def step_simulation(params: SimulationParams):
    """Advance simulation by one time step."""
    global simulation_stream, simulation_time
    
    if simulation_stream is None:
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    def zero_torque(packet_id, t, state):
        return np.array([0.0, 0.0, 0.0])
    
    result = simulation_stream.integrate(params.dt, zero_torque)
    simulation_time += params.dt
    
    return {
        "time": simulation_time,
        "events_processed": result["events_processed"],
    }


@app.get("/simulation/state")
async def get_simulation_state():
    """Get current simulation state."""
    if simulation_stream is None:
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    
    # Convert packets to JSON-serializable format
    packet_states = []
    for packet in simulation_stream.packets:
        packet_states.append(PacketState(
            id=packet.id,
            position=packet.position.tolist(),
            velocity=packet.velocity.tolist(),
            quaternion=packet.body.quaternion.tolist(),
            angular_velocity=packet.angular_velocity.tolist(),
            eta_ind=packet.eta_ind,
            state=packet.state.value,
        ))
    
    # Convert nodes to JSON-serializable format
    node_states = []
    for i, node in enumerate(simulation_stream.nodes):
        node_states.append(NodeState(
            id=i,
            position=node.position.tolist(),
            captured_packets=node.held_packets,
        ))
    
    # Get metrics
    metrics = simulation_stream.get_stream_metrics()
    
    return SimulationState(
        time=simulation_time,
        packets=packet_states,
        nodes=node_states,
        metrics=metrics,
    )


@app.post("/monte-carlo/run")
async def run_monte_carlo(request: MonteCarloRequest):
    """Run Monte-Carlo analysis."""
    
    def stream_factory():
        """Factory function to create fresh stream for each realization."""
        mass = 0.05
        I = np.diag([0.0001, 0.00011, 0.00009])
        
        packets = []
        for i in range(request.n_packets):
            position = np.array([i * 100.0, 0.0, 0.0])
            velocity = np.array([request.velocity, 0.0, 0.0])
            body = RigidBody(mass, I, position=position, velocity=velocity)
            packets.append(Packet(id=i, body=body))
        
        nodes = []
        for i in range(request.n_nodes):
            position = np.array([i * 200.0, 0.0, 0.0])
            nodes.append(SNode(position))
        
        return MultiBodyStream(packets=packets, nodes=nodes, stream_velocity=request.velocity)
    
    # Create config
    config = MonteCarloConfig(
        n_realizations=request.n_realizations,
        time_horizon=request.time_horizon,
        dt=request.dt,
    )
    
    # Create runner
    runner = CascadeRunner(config)
    
    # Run Monte-Carlo
    results = runner.run_monte_carlo(stream_factory)
    
    return MonteCarloResponse(
        n_realizations=results["n_realizations"],
        n_success=results["n_success"],
        n_failure=results["n_failure"],
        success_rate=results["success_rate"],
        cascade_probability=results["cascade_probability"],
        eta_ind_min_mean=results["eta_ind_min_mean"],
        stress_max_mean=results["stress_max_mean"],
        meets_cascade_target=results["meets_cascade_target"],
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "simulation_initialized": simulation_stream is not None}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
