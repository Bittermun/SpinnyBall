# Benchmarks

JAX XLA: 256k realizations in 0.96s (~3,751x CPU).

Thermal: 2.9-63x speedup vs NumPy depending on batch size.

Cascade Monte Carlo: stable up to 65ms latency at η=0.90.

Tests: `pytest tests/test_simulation_invariants.py -v`

Sobol (9 params, N=1024): seconds to evaluate.