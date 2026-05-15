# Flux-Pinning Legacy Code Archive

This directory contains the obsolete flux-pinning superconductor code that has been
replaced by the canonical Halbach array SGMS implementation.

## Why This Code Was Archived

The SpinnyBall project has transitioned from a flux-pinning-based architecture to a
pure Halbach array (permanent magnet) architecture for the Shepherded Gyroscopic
Mass Stream (SGMS) concept. This change was made because:

1. **No cryogenics required**: Halbach arrays operate at ambient temperature
2. **No quench risk**: Permanent magnets don't have catastrophic failure modes like superconductors
3. **No AC losses**: Simplified thermal management
4. **Self-confinement**: Dipole-dipole repulsion + hoop tension eliminates need for physical tracks
5. **Higher TRL**: Permanent magnet technology is more mature for space applications

## Files in This Archive

### Core Flux-Pinning Modules
- `bean_london_model.py` - Bean-London critical-state model for GdBCO superconductors
- `gdBCO_material.py` - GdBCO material properties and J_c(B,T) calculations
- `quench_detector.py` - Quench detection and emergency shutdown logic

### Tests
- `test_bean_london.py` - Unit tests for Bean-London model
- `test_flux_pinning.py` - Tests for flux-pinning forces
- `test_flux_pinning_integration.py` - Integration tests
- `test_quench_detection.py` - Quench detector tests
- `test_gdbco_enhancements.py` - GdBCO enhancement tests

## New Canonical Implementation

The Halbach-based SGMS implementation is in:
- `dynamics/halbach_array.py` - Spherical Halbach array physics
- `dynamics/interball_magnetic.py` - Dipole-dipole interactions
- `dynamics/hoop_tension.py` - Hoop tension restoring forces
- `dynamics/shepherd_station.py` - Quadrupole lens guidance

## References

- Leupold & Potenziani (1988) - Halbach cylinders & spherical extensions
- Yonnet (1981) - Permanent magnet bearings
- Jackson, Classical Electrodynamics - Dipole-dipole interactions
- Hoyt & Forward (1999) - Momentum-exchange tethers
