"""
FastAPI backend for Digital Twin dashboard.

Provides REST API endpoints for the digital twin visualization,
connecting the Python dynamics simulation to the HTML frontend.
"""

from __future__ import annotations

import asyncio
import logging
import os

import numpy as np  # noqa: F401

from backend.logging_config import setup_logging

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, field_validator  # noqa: E402

from dynamics.multi_body import MultiBodyStream, Packet, SNode  # noqa: E402
from dynamics.rigid_body import RigidBody  # noqa: E402
from monte_carlo.cascade_runner import (  # noqa: E402
    CascadeRunner,
    MonteCarloConfig,
)
from monte_carlo.pass_fail_gates import evaluate_monte_carlo_gates  # noqa: E402

# Logger will be configured at startup
logger = logging.getLogger(__name__)

try:
    from ml_integration import MLIntegrationLayer
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    MLIntegrationLayer = None

app = FastAPI(title="SGMS MRT Digital Twin API", version="0.1.0")


@app.on_event("startup")
async def startup_event():
    """Configure logging at application startup."""
    setup_logging(level=logging.INFO)
    logger.info("Application started")


# Enable CORS for frontend
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000")
origins_list = [origin.strip() for origin in allowed_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins_list,
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
    enable_delay_compensation: bool = True
    delay_steps: int = 5
    inject_latency_ms: float = 0.0


class PacketState(BaseModel):
    id: int
    position: list[float]
    velocity: list[float]
    quaternion: list[float]
    angular_velocity: list[float]
    eta_ind: float
    state: str


class NodeState(BaseModel):
    id: int
    position: list[float]
    captured_packets: list[int]


class SimulationState(BaseModel):
    time: float
    packets: list[PacketState]
    nodes: list[NodeState]
    metrics: dict[str, float]
    delay_compensation_enabled: bool = False
    delay_steps: int = 0
    mpc_latency_ms: float = 0.0


class MonteCarloRequest(BaseModel):
    n_realizations: int = 100
    n_packets: int = 10
    n_nodes: int = 3
    velocity: float = 1600.0
    time_horizon: float = 10.0
    dt: float = 0.01
    inject_latency_ms: float = 0.0
    latency_std_ms: float = 5.0
    track_per_packet_latency: bool = True


class MonteCarloResponse(BaseModel):
    n_realizations: int
    n_success: int
    n_failure: int
    success_rate: float
    cascade_probability: float
    eta_ind_min_mean: float
    stress_max_mean: float
    meets_cascade_target: bool
    latency_events: int = 0
    max_latency_ms: float = 0.0
    latency_gate_status: str = "unknown"
    delay_margin_ms: float | None = None
    nodes_affected_mean: float = 0.0
    containment_rate: float = 1.0


class WobbleDetectionRequest(BaseModel):
    signals: list[list[float]]
    threshold: float = 0.1

    @field_validator("signals")
    @classmethod
    def validate_signals(cls, v: list[list[float]]) -> list[list[float]]:
        """Validate signals array."""
        if not v:
            raise ValueError("signals cannot be empty")
        for i, signal in enumerate(v):
            if len(signal) < 100:
                raise ValueError(f"signal {i} must have at least 100 samples, got {len(signal)}")
            if not all(-1000.0 <= x <= 1000.0 for x in signal):
                raise ValueError(f"signal {i} contains values outside valid range [-1000, 1000]")
        return v

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Validate threshold value."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        return v


class ThermalPredictionRequest(BaseModel):
    T_initial: list[float]
    Q_in: list[list[float]]
    T_amb: float = 293.15

    @field_validator("T_initial")
    @classmethod
    def validate_temperatures(cls, v: list[float]) -> list[float]:
        """Validate initial temperatures."""
        if not v:
            raise ValueError("T_initial cannot be empty")
        if not all(0.0 <= x <= 1000.0 for x in v):
            raise ValueError("temperatures must be between 0K and 1000K")
        return v

    @field_validator("Q_in")
    @classmethod
    def validate_heat_input(cls, v: list[list[float]]) -> list[list[float]]:
        """Validate heat input rates."""
        if not v:
            raise ValueError("Q_in cannot be empty")
        for i, q_row in enumerate(v):
            if len(q_row) != len(v):
                raise ValueError(f"Q_in row {i} must have same length as T_initial")
            if not all(0.0 <= x <= 10000.0 for x in q_row):
                raise ValueError(f"Q_in row {i} contains values outside valid range [0, 10000] W")
        return v

    @field_validator("T_amb")
    @classmethod
    def validate_ambient_temperature(cls, v: float) -> float:
        """Validate ambient temperature."""
        if not 0.0 <= v <= 1000.0:
            raise ValueError("T_amb must be between 0K and 1000K")
        return v


# ============================================================
# Simulation State with Thread Safety
# ============================================================

class SimulationState:
    """Thread-safe simulation state management."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self.stream: MultiBodyStream | None = None
        self.running: bool = False
        self.time: float = 0.0
        self.delay_config: dict = {}

    async def initialize(self, stream: MultiBodyStream, delay_config: dict) -> None:
        """Initialize simulation with new stream and config."""
        async with self._lock:
            self.stream = stream
            self.running = False
            self.time = 0.0
            self.delay_config = delay_config

    async def set_running(self, running: bool) -> None:
        """Set simulation running state."""
        async with self._lock:
            self.running = running

    async def reset(self) -> None:
        """Reset simulation to initial state."""
        async with self._lock:
            self.running = False
            self.time = 0.0

    async def step(self, dt: float) -> dict:
        """Advance simulation by one time step."""
        async with self._lock:
            if self.stream is None:
                raise HTTPException(status_code=400, detail="Simulation not initialized")

            def zero_torque(packet_id: int, t: float, state: np.ndarray) -> np.ndarray:
                return np.array([0.0, 0.0, 0.0])

            result = self.stream.integrate(dt, zero_torque)
            self.time += dt
            return result

    async def get_state(self) -> tuple[MultiBodyStream, float, dict]:
        """Get current simulation state."""
        async with self._lock:
            if self.stream is None:
                raise HTTPException(status_code=400, detail="Simulation not initialized")
            return self.stream, self.time, self.delay_config


# Global simulation state instance
simulation_state = SimulationState()

# ML integration
if ML_AVAILABLE:
    try:
        ml_integration = MLIntegrationLayer()
    except Exception as e:
        logger.error(f"Failed to initialize ML integration: {e}")
        ml_integration = None
else:
    ml_integration = None
    logger.warning("ML integration layer unavailable - ML endpoints will return errors")


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

    # Create packets
    mass = 0.05
    I = np.diag([0.0001, 0.00011, 0.00009])  # noqa: N806, E741

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
        nodes.append(SNode(id=i, position=position))

    # Create stream
    stream = MultiBodyStream(packets=packets, nodes=nodes, stream_velocity=params.velocity)

    # Store latency config for use in MPC
    delay_config = {
        'enable_delay_compensation': params.enable_delay_compensation,
        'delay_steps': params.delay_steps,
        'inject_latency_ms': params.inject_latency_ms,
    }

    await simulation_state.initialize(stream, delay_config)

    return {"status": "initialized", "n_packets": params.n_packets, "n_nodes": params.n_nodes}


@app.post("/simulation/start")
async def start_simulation():
    """Start simulation."""
    await simulation_state.set_running(True)
    return {"status": "running"}


@app.post("/simulation/stop")
async def stop_simulation():
    """Stop simulation."""
    await simulation_state.set_running(False)
    return {"status": "stopped"}


@app.post("/simulation/reset")
async def reset_simulation():
    """Reset simulation to initial state."""
    await simulation_state.reset()
    return {"status": "reset"}


@app.post("/simulation/step")
async def step_simulation(params: SimulationParams):
    """Advance simulation by one time step."""
    result = await simulation_state.step(params.dt)

    return {
        "time": simulation_state.time,
        "events_processed": result["events_processed"],
    }


@app.get("/simulation/state")
async def get_simulation_state():
    """Get current simulation state."""
    stream, time, delay_config = await simulation_state.get_state()

    # Convert packets to JSON-serializable format
    packet_states = []
    for packet in stream.packets:
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
    for i, node in enumerate(stream.nodes):
        node_states.append(NodeState(
            id=i,
            position=node.position.tolist(),
            captured_packets=node.held_packets,
        ))

    # Get metrics
    metrics = stream.get_stream_metrics()

    # Add latency metrics if available
    if delay_config:
        metrics['delay_compensation_enabled'] = delay_config.get('enable_delay_compensation', False)
        metrics['delay_steps'] = delay_config.get('delay_steps', 0)
    else:
        metrics['delay_compensation_enabled'] = False
        metrics['delay_steps'] = 0

    return SimulationState(
        time=time,
        packets=packet_states,
        nodes=node_states,
        metrics=metrics,
        delay_compensation_enabled=metrics.get('delay_compensation_enabled', False),
        delay_steps=metrics.get('delay_steps', 0),
        mpc_latency_ms=0.0,  # Will be populated by MPC controller
    )


@app.post("/monte-carlo/run")
async def run_monte_carlo(request: MonteCarloRequest):
    """Run Monte-Carlo analysis."""

    def stream_factory():
        """Factory function to create fresh stream for each realization."""
        mass = 0.05
        I = np.diag([0.0001, 0.00011, 0.00009])  # noqa: N806, E741

        packets = []
        for i in range(request.n_packets):
            position = np.array([i * 100.0, 0.0, 0.0])
            velocity = np.array([request.velocity, 0.0, 0.0])
            body = RigidBody(mass, I, position=position, velocity=velocity)
            packets.append(Packet(id=i, body=body))

        nodes = []
        for i in range(request.n_nodes):
            position = np.array([i * 200.0, 0.0, 0.0])
            nodes.append(SNode(id=i, position=position))

        return MultiBodyStream(packets=packets, nodes=nodes, stream_velocity=request.velocity)

    # Create config
    config = MonteCarloConfig(
        n_realizations=request.n_realizations,
        time_horizon=request.time_horizon,
        dt=request.dt,
        latency_ms=request.inject_latency_ms,
        latency_std_ms=request.latency_std_ms,
        track_per_packet_latency=request.track_per_packet_latency,
    )

    # Create runner
    runner = CascadeRunner(config)

    # Run Monte-Carlo
    results = runner.run_monte_carlo(stream_factory)

    # Evaluate gates including latency
    gate_results = evaluate_monte_carlo_gates(results)
    latency_gate_status = "unknown"
    for result in gate_results.get("results", []):
        if result.gate_name == "max_latency_ms":
            latency_gate_status = result.status.value
            break

    return MonteCarloResponse(
        n_realizations=results["n_realizations"],
        n_success=results["n_success"],
        n_failure=results["n_failure"],
        success_rate=results["success_rate"],
        cascade_probability=results["cascade_probability"],
        eta_ind_min_mean=results["eta_ind_min_mean"],
        stress_max_mean=results["stress_max_mean"],
        meets_cascade_target=results["meets_cascade_target"],
        latency_events=results.get("latency_events", 0),
        max_latency_ms=results.get("max_latency_ms", 0.0),
        latency_gate_status=latency_gate_status,
        delay_margin_ms=results.get("delay_margin_ms", 0.0),
        nodes_affected_mean=results.get("nodes_affected_mean", 0.0),
        containment_rate=results.get("containment_rate", 1.0),
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    stream, _, _ = await simulation_state.get_state()
    return {"status": "healthy", "simulation_initialized": stream is not None}


# ============================================================
# EDT endpoints archived - see archived_edt/ directory
# ============================================================

# ============================================================
# ML Endpoints
# ============================================================

@app.post("/ml/wobble-detect")
async def detect_wobble(request: WobbleDetectionRequest):
    """Detect wobble in packet signals."""
    if ml_integration is None:
        raise HTTPException(status_code=503, detail="ML integration layer unavailable")

    try:
        # Convert list signals to numpy arrays
        signals_np = [np.array(s) for s in request.signals]
        results = ml_integration.detect_wobble_batch(
            signals_np,
            request.threshold,
        )
        return {"results": results}
    except ValueError as e:
        logger.error(f"Invalid input for wobble detection: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Invalid input data") from e
    except Exception as e:
        logger.error(f"Wobble detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.post("/ml/thermal-predict")
async def predict_thermal(request: ThermalPredictionRequest):
    """Predict thermal evolution."""
    if ml_integration is None:
        raise HTTPException(status_code=503, detail="ML integration layer unavailable")

    try:
        # Convert lists to numpy arrays
        T_initial_np = np.array(request.T_initial)  # noqa: N806
        Q_in_np = np.array(request.Q_in)  # noqa: N806
        result = ml_integration.predict_thermal_batch(
            T_initial_np,
            Q_in_np,
            request.T_amb,
        )
        return result
    except ValueError as e:
        logger.error(f"Invalid input for thermal prediction: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Invalid input data") from e
    except Exception as e:
        logger.error(f"Thermal prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.get("/ml/status")
async def get_ml_status():
    """Get ML model status."""
    if ml_integration is None:
        return {
            "wobble_detector": {"available": False, "info": None},
            "thermal_model": {"available": False, "info": None},
            "integration_available": False,
        }
    return ml_integration.get_model_status()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
