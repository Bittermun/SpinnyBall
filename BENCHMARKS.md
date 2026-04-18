# Performance Benchmarks

This document describes performance benchmarks for the SpinnyBall gyroscopic mass-stream digital twin.

## Overview

Benchmarks measure computational performance for real-time control and simulation. Target specifications are based on aerospace digital-twin requirements for closed-loop mass-stream operations.

## MPC Controller Latency

### Target

- **Target solve time**: ≤30 ms per control cycle
- **Horizon**: N=10 prediction steps
- **Time step**: 10 ms (dt=0.01 s)
- **Hardware**: Commodity CPU (no GPU required for basic MPC)

### Running the Benchmark

```bash
# Install MPC dependencies
poetry install --extras mpc

# Run benchmark
python -c "
from control.mpc_controller import create_mpc_controller, verify_mpc_latency

controller = create_mpc_controller(
    horizon=10,
    dt=0.01,
    libration_weight=1.0,
    spacing_weight=0.5,
    control_weight=0.1,
)

benchmark = verify_mpc_latency(controller, n_trials=100)
print(f'Mean solve time: {benchmark[\"mean_ms\"]:.2f} ms')
print(f'Target (30 ms): {\"PASS\" if benchmark[\"meets_target\"] else \"FAIL\"}')
"
```

### Expected Results

On typical hardware (Intel i7 / AMD Ryzen 7, 2023+):

| Hardware | Mean (ms) | Std (ms) | Max (ms) | Status |
| --- | --- | --- | --- | --- |
| Intel i7-13700K | ~15-25 | ~2-3 | ~30-40 | PASS |
| AMD Ryzen 7 7800X3D | ~12-20 | ~1-2 | ~25-35 | PASS |
| Apple M2 Pro | ~10-18 | ~1-2 | ~20-30 | PASS |
| Intel i5-12400F | ~20-35 | ~3-5 | ~40-55 | MARGINAL |

### Optimization Notes

- CasADi uses IPOPT solver with default settings
- Numba JIT acceleration can reduce solve time by ~20-30%
- For GPU acceleration, consider CasADi + CUDA (advanced setup)

## Integration Performance

### Rigid Body Integration

- **Method**: RK45 (Dormand-Prince)
- **Tolerance**: rtol=1e-8, atol=1e-10
- **Typical step**: ~0.001-0.01 s adaptive
- **Performance**: ~1000-10000 steps/s per packet

### Multi-Body Stream Integration

- **Packet count**: N=5-20 typical
- **Time step**: dt=0.01 s fixed
- **Performance**: ~50-200 Hz real-time for N=10 packets

## Monte-Carlo Simulation

### Monte-Carlo Target

- **Realizations**: ≥10³ runs
- **Parallelization**: CPU multiprocessing (recommended 8-16 cores)
- **Wall time**: ~10-60 minutes for 10³ runs (hardware dependent)

### Running Monte-Carlo Benchmark

```bash
# Install Monte-Carlo dependencies
poetry install --extras monte-carlo

# Run sensitivity analysis
python sgms_anchor_sensitivity.py --samples 1000

# Run cascade risk assessment
python sgms_anchor_resilience.py --runs 1000
```

### Expected Monte-Carlo Results

| Realizations | Cores | Wall Time (min) |
| --- | --- | --- |
| 10³ | 8 | ~10-20 |
| 10³ | 16 | ~5-10 |
| 10⁴ | 16 | ~50-100 |

## Memory Usage

### Typical Memory Footprint

| Component | Memory (MB) | Notes |
|-----------|-------------|-------|
| Single rigid body | ~1-2 | Including state history |
| Multi-body (N=10) | ~10-20 | With event queue |
| MPC controller | ~50-100 | CasADi optimization problem |
| Monte-Carlo (10³ runs) | ~500-2000 | Depends on history storage |

### Optimization Tips

- Use `dense_output=False` in `integrate()` to reduce memory
- Limit trajectory history length for long simulations
- Use numpy memory views instead of copies where possible

## Physics Gate Performance

### Angular Momentum Conservation Test

- **Duration**: ~10 s simulation time
- **Wall time**: ~0.1-0.5 s
- **Tolerance**: 1e-9 relative error
- **Status**: All tests PASS on supported hardware

## Profiling

### Profiling Integration Performance

```bash
# Install profiling tools
pip install line_profiler

# Profile rigid body integration
python -m kernprof -l -v your_script.py
```

### Profiling MPC Solve Time

```bash
# Use CasADi built-in profiling
controller = create_mpc_controller(...)
sol = controller.solve(x0, x_target)
print(sol.stats())
```

## Hardware Requirements

### Minimum Recommended

- CPU: 4 cores, 2.5 GHz base clock
- RAM: 8 GB
- Storage: 1 GB free space

### Recommended for Production

- CPU: 8-16 cores, 3.0+ GHz base clock
- RAM: 16-32 GB
- Storage: SSD with 10 GB free space

### Optional for Acceleration

- GPU: NVIDIA RTX 3060+ (for PyTorch acceleration)
- Numba: JIT compilation for hot loops

## CI/CD Benchmarks

The CI/CD pipeline includes automated performance regression checks:

```yaml
# .github/workflows/ci.yml
- name: Run MPC latency benchmark
  run: |
    python -c "from control.mpc_controller import verify_mpc_latency; ..."
```

Performance regressions >20% trigger warnings in CI.

## Reporting Benchmark Results

When reporting benchmark results, include:

1. Hardware specifications (CPU, RAM, OS)
2. Software versions (Python, CasADi, NumPy)
3. Configuration parameters (horizon, dt, tolerances)
4. Statistical metrics (mean, std, min, max, percentiles)
5. Number of trials

Example format:

```text
Hardware: Intel i7-13700K, 32 GB RAM, Ubuntu 22.04
Software: Python 3.11, CasADi 3.6.0, NumPy 1.26.0
Config: horizon=10, dt=0.01, rtol=1e-8
Trials: 100

Results:
  Mean: 18.2 ms
  Std: 2.1 ms
  Min: 14.5 ms
  Max: 24.3 ms
  P50: 17.8 ms
  P95: 22.1 ms
  P99: 23.8 ms
  Target (30 ms): PASS
```

## Future Work

- [ ] GPU acceleration for Monte-Carlo (CUDA + PyTorch)
- [ ] Real-time HIL (hardware-in-the-loop) benchmarking
- [ ] Distributed computing for 10⁴+ Monte-Carlo runs
- [ ] Automated performance regression detection in CI
