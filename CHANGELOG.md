# Changelog

All notable changes to SpinnyBall are documented in this file.

The format is based on Keep a Changelog and this project adheres to Semantic Versioning.

## [Unreleased]

### Added

- 3-profile system for interchangeable parameter sweeps (material, geometry, environment profiles)
- Material profile catalog (paper_model/gdbco_apc_catalog.json) with flux-pinning stiffness ranges
- Geometry profile catalog (geometry_profiles.json) with shape, mass, radius parameters
- Environment profile catalog (environment_profiles.json) with temperature, B_field, radiation, gravity
- Profile validation functions with type and range checking
- Skipped experiments tracking in pipeline manifest
- File existence checks in validation scripts
- Simulation error handling in validate_profile.py
- Type and range validation tests for all profile types
- Documentation for profile system in README
- Operational profile with paper targets (8.0 kg mass, 1600 m/s velocity, 6000 N/m flux-pinning stiffness)
- Mass sweep capability in sensitivity analysis (mp parameter added to Sobol analysis)
- Extended parameter bounds for operational scale (velocity: 5-1600 m/s, linear density: 0.1-20.0 kg/m)
- Operational profile validation tests (8 tests for paper target validation)
- FMECA JSON export for risk matrix and kill criteria analysis
- JAX thermal model with heat source support
- Backend API endpoints for thermal state and performance
- Monte-Carlo runner extended with thermal perturbations
- Documentation for thermal management with physics derivations
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

- Fixed environment profile override order to allow experiment params to override environment values
- Added warning when both legacy k_fp and new material_profile are present
- Added broader exception handling in pipeline (FileNotFoundError, OSError, json.JSONDecodeError)
- Fixed profile validation to check experiment overrides before applying environment values
- Fixed PIDParameters dataclass field order (added default values to kp, ki, kd for backward compatibility)
- Fixed simulate_multi_pass_accumulation undefined rhs bug (replaced with lambda calling eom)
- Fixed simulate_multi_pass_accumulation state vector length (6→9 elements to include spin axis)
- Removed unnecessary global P declaration from simulate_multi_pass_accumulation
- Fixed mutual inductance B_amp calculation in Bx_field (now uses correct segment amplitude)
- Fixed multi_body node_map to use node.id as key instead of enumerate index
- Fixed flux pinning displacement to use relative position from S-Node instead of absolute position
- Fixed orbital/RigidBody state desync by syncing orbital_state to body.position/velocity after propagation
- Fixed RigidBody.flux_model initialization in MultiBodyStream (now initializes BeanLondonModel when available)
- Updated test_flux_pinning_integration.py to pass node_position parameter to compute_flux_pinning_torque
- Fixed numpy bool to Python bool casting in ML integration (API contract compliance)
- Fixed division by zero edge case in FMECA export (abs(omega_initial) > 1e-10 check)
- Added TYPE_CHECKING for BeanLondonModel type hint in rigid_body.py
- Added explicit temperature initialization to thermal model
- Removed unused mode flags from JAXThermalModel
- Fixed Monte-Carlo runner type checking
- Standardized error handling in backend endpoints
- Fixed Bean-London stiffness calculation with analytical derivative
- Updated stress formula to use hoop stress (σ_θ = ρ·r²·ω²) from paper
- Adjusted safety margin thresholds in operational scale tests to realistic values
- Fixed ThermalLimits max_packet_temp from 450K to 90K for GdBCO superconductors (below Tc=92K)
- Fixed multi_body.py thermal integration to pass position_eci for eclipse detection
- Fixed multi_body.py to calculate eddy heating power for FREE packets using velocity
- Fixed multi_body.py to use prolate spheroid surface area instead of sphere
- Fixed JAX thermal model to use radiative cooling instead of invalid convection for space
- Added prolate spheroid surface area calculation to thermal_model.py with aspect_ratio support
- Updated TemperatureGate default to 90K for superconducting operation
- Code review fixes: Moved import outside loop in MultiBodyStream for efficiency
- Code review fixes: Added __del__ cleanup to JAXThermalModel to prevent memory leaks
- Code review fixes: Fixed null reference checks in thermal model eclipse detection
- Code review fixes: Added comprehensive parameter validation to update_temperature_euler
- Code review fixes: Defined magic numbers as module-level constants (SOLAR_CONSTANT, etc.)
- Code review fixes: Added type hints to torque functions in multi_body.py
- Fixed all Bean-London model tests (11/11 passing)
- Fixed all parameter sweep tests (43/43 passing)
- Fixed all operational scale tests (8/8 passing)
- Documentation: Salvaged Phase 3 Integration Report template from origin/science branch
- Documentation: Salvaged BENCHMARKS.md from origin/science branch with performance metrics
- Documentation: Created comprehensive thermal model validation report using salvaged templates
- Documentation: Updated README.md with thermal model validation framework

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
