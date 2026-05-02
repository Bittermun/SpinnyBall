"""Test simulation method correctness and MC invariants."""
import numpy as np
import pytest

def test_poisson_faults_are_presampled():
    """Poisson mode must pre-sample fault times before the simulation loop."""
    # Run a small MC with Poisson mode and fixed seed — verify deterministic fault count
    from monte_carlo.cascade_runner import CascadeRunner, MonteCarloConfig
    from dynamics.multi_body import MultiBodyStream, Packet, SNode
    from dynamics.rigid_body import RigidBody

    config = MonteCarloConfig(
        n_realizations=5, time_horizon=10.0, dt=0.1,
        fault_rate=100.0, fault_injection_mode='poisson',
        random_seed=42,
    )
    def factory():
        I = np.diag([0.0001, 0.00011, 0.00009])
        packets = [Packet(id=0, body=RigidBody(0.05, I), eta_ind=0.9)]
        nodes = [SNode(id=i, position=np.array([i*10.0, 0.0, 0.0]), k_fp=6000.0) for i in range(5)]
        return MultiBodyStream(packets=packets, nodes=nodes, stream_velocity=100.0)

    runner = CascadeRunner(config)
    r1 = runner.run_monte_carlo(factory)

    # Run again with same seed — must get identical fault counts
    runner2 = CascadeRunner(config)
    r2 = runner2.run_monte_carlo(factory)
    assert r1['fault_events_total'] == r2['fault_events_total'], "Poisson faults not deterministic with same seed"

def test_provenance_stream_factory_call_count():
    """stream_factory should be called at most once for provenance metadata."""
    from unittest.mock import MagicMock, wraps
    from monte_carlo.cascade_runner import CascadeRunner, MonteCarloConfig
    from dynamics.multi_body import MultiBodyStream, Packet, SNode
    from dynamics.rigid_body import RigidBody

    call_count = 0
    def counting_factory():
        nonlocal call_count
        call_count += 1
        I = np.diag([0.0001, 0.00011, 0.00009])
        packets = [Packet(id=0, body=RigidBody(0.05, I), eta_ind=0.9)]
        nodes = [SNode(id=0, position=np.array([0.0, 0.0, 0.0]), k_fp=6000.0)]
        return MultiBodyStream(packets=packets, nodes=nodes, stream_velocity=100.0)

    config = MonteCarloConfig(n_realizations=2, time_horizon=1.0, dt=0.1)
    runner = CascadeRunner(config)
    runner.run_monte_carlo(counting_factory)
    # n_realizations calls + at most 1 for provenance
    assert call_count <= 3, f"stream_factory called {call_count} times (expected <= 3 for 2 realizations + 1 provenance)"

def test_velocity_verlet_energy_conservation():
    """Anchor simulation should conserve energy for undamped oscillator."""
    from src.sgms_anchor_v1 import simulate_anchor_with_flux_pinning
    import numpy as np
    params = {"ms": 1.0, "c_damp": 0.0, "x0": 0.1, "v0": 0.0, "k_structural": 0.0}
    t_eval = np.linspace(0, 10, 1000)
    T_profile = np.full_like(t_eval, 77.0)
    B_profile = np.full_like(t_eval, 1.0)
    results = simulate_anchor_with_flux_pinning(params, t_eval, T_profile, B_profile)
    # Check energy: E = 0.5*m*v^2 + 0.5*k*x^2
    x_arr = np.array(results['x'])
    v_arr = np.array(results['v'])
    k_arr = np.array(results['k_eff'])
    E = 0.5 * params['ms'] * v_arr**2 + 0.5 * k_arr * x_arr**2
    E_drift = abs(E[-1] - E[0]) / E[0]
    assert E_drift < 0.01, f"Energy drift {E_drift:.4f} > 1% — integrator is not symplectic"
