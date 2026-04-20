# SpinnyBall Dependencies

This document describes all dependencies required for SpinnyBall, including optional dependencies for specific features.

## Core Dependencies

These are required for basic functionality:

- **python** >= 3.11 - Python interpreter
- **numpy** >= 1.26.0 - Numerical computing
- **scipy** >= 1.11.0 - Scientific computing (integration, optimization)
- **sympy** >= 1.12 - Symbolic mathematics
- **matplotlib** >= 3.8.0 - Plotting and visualization

## Optional Dependencies

Optional dependencies enable specific features. Install them using the extras shown below.

### MPC & Optimization

**Extra**: `mpc`

Dependencies:
- **casadi** >= 3.6.0 - Nonlinear optimization for MPC control
- **numba** >= 0.60.0 - JIT compilation for CPU acceleration

**Features enabled**:
- Model Predictive Control (MPC) for advanced packet steering
- Real-time trajectory optimization
- Numba-accelerated Monte Carlo simulations

**Install**:
```bash
poetry install --extras mpc
# or
pip install casadi numba
```

### ML & Training

**Extra**: `ml`

Dependencies:
- **torch** >= 2.5.0 - PyTorch for neural network training

**Features enabled**:
- VMD-IRCNN model training
- Deep learning-based prediction
- GPU acceleration for training (requires CUDA)

**Install**:
```bash
poetry install --extras ml
# or
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### JAX Acceleration

**Extra**: `jax`

Dependencies:
- **jax** >= 0.4.0 - JAX for automatic differentiation
- **jaxlib** >= 0.4.0 - JAX XLA library

**Features enabled**:
- JAX-accelerated thermal modeling
- GPU acceleration for thermal simulations
- Automatic differentiation for sensitivity analysis

**Install**:
```bash
poetry install --extras jax
# or
pip install jax jaxlib
```

### Monte Carlo Analysis

**Extra**: `monte-carlo`

Dependencies:
- **SALib** >= 1.4.0 - Sensitivity analysis library
- **torch** >= 2.5.0 - PyTorch (included from ml extra)

**Features enabled**:
- Sobol sensitivity analysis
- Parameter sweep validation
- Monte Carlo statistical analysis

**Install**:
```bash
poetry install --extras monte-carlo
# or
pip install SALib torch
```

### Physics Validation

**Extra**: `validation`

Dependencies:
- **mujoco** >= 3.1.0 - MuJoCo physics engine

**Features enabled**:
- 6-DOF physics validation
- High-fidelity oracle comparison
- MuJoCo-based trajectory verification

**Install**:
```bash
poetry install --extras validation
# or
pip install mujoco
```

### Backend API

**Extra**: `backend`

Dependencies:
- **fastapi** >= 0.115.0 - FastAPI web framework
- **uvicorn** >= 0.32.0 - ASGI server
- **pydantic** >= 2.10.0 - Data validation

**Features enabled**:
- REST API for digital twin
- Real-time simulation control
- Web-based dashboard

**Install**:
```bash
poetry install --extras backend
# or
pip install fastapi uvicorn pydantic
```

### EDT (Electrodynamic Tethers) - ARCHIVED

The EDT module has been archived to `archived_edt/` directory.

## Development Dependencies

These are installed automatically with `poetry install`:

- **pytest** >= 7.4.0 - Testing framework
- **pytest-cov** >= 4.1.0 - Coverage reporting
- **pytest-watch** >= 0.8.0 - Watch mode for development
- **black** >= 23.12.0 - Code formatting
- **ruff** >= 0.1.0 - Linting
- **mypy** >= 1.7.0 - Type checking

## Installation

### Install All Extras

To install all optional dependencies at once:

```bash
poetry install --extras all
```

### Selective Installation

Install only the extras you need:

```bash
poetry install --extras mpc --extras ml --extras jax
```

### Minimal Installation

For basic functionality without optional features:

```bash
poetry install
```

## Configuration

### Debug vs Operational Mode

Set the `SPINNYBALL_MODE` environment variable to control debug vs operational values:

```bash
# Debug mode (default) - uses faster debug values
export SPINNYBALL_MODE=debug

# Operational mode - uses realistic operational values
export SPINNYBALL_MODE=operational
```

**PowerShell**:
```powershell
$env:SPINNYBALL_MODE="operational"
```

This affects parameters like stream velocity:
- Debug: 10.0 m/s (faster iteration)
- Operational: 1600.0 m/s (realistic)

## GPU Requirements

For GPU acceleration, ensure you have:
- NVIDIA GPU with CUDA support (RTX 4070 or similar recommended)
- CUDA 12.1 or later
- PyTorch with CUDA enabled
- JAX with GPU support

**GPU Installation**:
```bash
# PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# JAX with GPU
pip install jax[cuda12] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Troubleshooting

### Import Errors

If you see `ImportError` for optional dependencies:
1. Install the required extra using the commands above
2. Or install the dependency directly with pip

### GPU Not Detected

If GPU acceleration is not working:
1. Check CUDA installation: `nvidia-smi`
2. Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. Verify JAX GPU: `python -c "import jax; print(jax.devices())"`

### MuJoCo Installation

MuJoCo may require additional setup on some platforms:
- Linux: Follow MuJoCo installation guide
- macOS: Use Homebrew: `brew install mujoco`
- Windows: Download from MuJoCo GitHub releases

## Dependency Security

All dependencies are sourced from PyPI with verified checksums. For production deployments:
1. Pin specific versions in `pyproject.toml`
2. Use `poetry lock` to generate lock file
3. Audit dependencies with `poetry audit`
4. Consider using `pip-audit` for vulnerability scanning
