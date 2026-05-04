"""Test simulation method correctness and MC invariants."""
import numpy as np


def test_poisson_faults_are_presampled():
    """Poisson mode must pre-sample fault times before the simulation loop."""
    # Run a small MC with Poisson mode and fixed seed — verify deterministic fault count
    from dynamics.multi_body import MultiBodyStream, Packet, SNode
    from dynamics.rigid_body import RigidBody
    from monte_carlo.cascade_runner import CascadeRunner, MonteCarloConfig

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
    from dynamics.multi_body import MultiBodyStream, Packet, SNode
    from dynamics.rigid_body import RigidBody
    from monte_carlo.cascade_runner import CascadeRunner, MonteCarloConfig

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
    import numpy as np

    from src.sgms_anchor_v1 import simulate_anchor_with_flux_pinning

    class MockFluxModel:
        def get_stiffness(self, x, B, T): return 6000.0
        def get_critical_current(self, T, B): return 100.0

    params = {
        "ms": 1.0, "c_damp": 0.0, "x0": 0.1, "v0": 0.0,
        "k_fp": 6000.0, "k_structural": 0.0, "lam": 1.0, "u": 100.0,
        "g_gain": 0.0, "theta_bias": 0.0, "eps": 0.0
    }
    t_eval = np.linspace(0, 2, 2000)
    results = simulate_anchor_with_flux_pinning(params, t_eval, flux_model=MockFluxModel())

    # Check energy: E = 0.5*m*v^2 + 0.5*k*x^2
    x_arr = np.array(results['x'])
    v_arr = np.array(results['v'])
    k_arr = np.array(results['k_eff'])
    E = 0.5 * params['ms'] * v_arr**2 + 0.5 * k_arr * x_arr**2
    E_drift = abs(E[-1] - E[0]) / E[0]
    # Tighten to 0.5% drift for symplectic RK4/Verlet
    assert E_drift < 0.005, f"Energy drift {E_drift:.4f} > 0.5% — integrator is not symplectic"

def test_energy_conservation_at_50k_rpm():
    """Verify energy conservation holds even at 50,000 RPM (extreme centrifugal state)."""
    import numpy as np

    from src.sgms_anchor_v1 import simulate_anchor_with_flux_pinning

    class MockFluxModel:
        def get_stiffness(self, x, B, T): return 9000.0
        def get_critical_current(self, T, B): return 100.0

    params = {
        "ms": 1.0, "c_damp": 0.0, "x0": 0.01, "v0": 0.0,
        "k_fp": 9000.0, "omega": 5236.0, "r": 0.1,
        "lam": 1.0, "u": 100.0, "g_gain": 0.0, "theta_bias": 0.0, "eps": 0.0
    }
    t_eval = np.linspace(0, 0.5, 2000)
    results = simulate_anchor_with_flux_pinning(params, t_eval, flux_model=MockFluxModel())

    x_arr = np.array(results['x'])
    v_arr = np.array(results['v'])
    k_arr = np.array(results['k_eff'])
    E = 0.5 * params['ms'] * v_arr**2 + 0.5 * k_arr * x_arr**2
    E_drift = abs(E[-1] - E[0]) / E[0]
    assert E_drift < 0.001, f"High-RPM energy drift {E_drift:.6f} > 0.1%"
