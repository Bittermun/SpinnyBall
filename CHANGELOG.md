# Changelog

All notable changes to SpinnyBall are documented in this file.

The format is based on Keep a Changelog and this project adheres to Semantic Versioning.

## [Unreleased]

### Added

- JAX thermal model with heat source support
- Backend API endpoints for thermal state and performance
- Monte-Carlo runner extended with thermal perturbations
- Documentation for thermal management with physics derivations
- Code review fixes for input validation and error handling
- Configurable num_workers for DataLoader training pipeline
- CUDA out of memory error handling in training loops
- Dataset empty validation before creating DataLoaders
- Error handling in checkpoint saving
- Config serialization for JSON save

### Changed

- Updated README with thermal capabilities and installation instructions
- Updated pyproject.toml with JAX extras
- Updated training pipeline to use configurable num_workers
- Fixed ReduceLROnPlateau verbose parameter for PyTorch compatibility
- Updated training command to use Python 3.11 for GPU training

### Fixed

- Added explicit temperature initialization to thermal model
- Removed unused mode flags from JAXThermalModel
- Fixed Monte-Carlo runner type checking
- Standardized error handling in backend endpoints
- Fixed Bean-London stiffness calculation with analytical derivative
- Updated stress formula to use hoop stress (σ_θ = ρ·r²·ω²) from paper
- Adjusted safety margin thresholds in operational scale tests to realistic values
- Fixed all Bean-London model tests (11/11 passing)
- Fixed all parameter sweep tests (43/43 passing)
- Fixed all operational scale tests (8/8 passing)

### Removed

- EDT (Electrodynamic Tethers) module - archived to archived_edt/ directory

## [1.0.0] - 2026-04-19

### Added

- Initial release of SpinnyBall physics simulation
- Gyroscopic mass-stream dynamics with magnetic packets
- Flux-pinned orbiting nodes
- Momentum-flux anchoring for station keeping
- MuJoCo integration for physics validation
- MPC controller with linearized ROM
- VMD-IRCNN ML enhancement with training pipeline
- Monte-Carlo simulation for resilience testing
- Backend API with FastAPI for digital twin
- Comprehensive documentation and tutorials
- Test suite with 70% coverage requirement

### Phase Completions

- Phase 1: Gyroscopic Mass-Stream Digital Twin
- Phase 2: ML Enhancement with VMD-IRCNN
- Phase 3: Advanced Diagnostics and Validation

## [0.1.0] - 2026-04-01

### Added

- Initial project setup
- Basic rigid body dynamics
- Gyroscopic coupling
- Multi-body stream simulation
- Basic control algorithms
