# Centralized Parameter Registry

## Purpose

This module provides a single source of truth for all physical constants and simulation parameters in the SGMS anchor project. It resolves the following discrepancies identified during code review:

### Resolved Discrepancies

| Parameter | Previous Values | Canonical Value | Notes |
|-----------|----------------|-----------------|-------|
| `Jc0` (critical current density) | 3e10 A/m² (sgms_anchor_v1.py), 2e9 A/m² (earth_moon_pumping.py) | 3e10 A/m² | 15x discrepancy resolved |
| `thickness` (superconductor layer) | 1e-6 m (explicit), bulk geometry (implicit) | 1e-6 m | Thin-film tape geometry |
| `temperature` (operating) | 77K default, 300K in some fixtures | 77K | Liquid nitrogen operating point |
| `k_fp` (flux pinning stiffness) | 0, 4500, 6000, 8000-220000 N/m across modules | Context-dependent | See geometry scaling below |

## Usage

```python
from params import get_parameter, MATERIAL_PROPERTIES, SIMULATION_PARAMS

# Get canonical values
Jc0 = get_parameter('MATERIAL_PROPERTIES', 'GdBCO', 'Jc0')
thickness = get_parameter('SIMULATION_PARAMS', 'flux_pinning', 'thickness')
T_op = get_parameter('SIMULATION_PARAMS', 'operating_conditions', 'temperature')

# Access full parameter dict with metadata
jc0_dict = MATERIAL_PROPERTIES['GdBCO']['Jc0']
print(f"Value: {jc0_dict['value']}")
print(f"Source: {jc0_dict['source']}")
print(f"Note: {jc0_dict['note']}")
```

## Geometry Scaling for k_fp

The catalog (`paper_model/gdbco_apc_catalog.json`) contains k_fp values measured from **bulk samples** (10mm × 10mm × 1mm). To compute effective stiffness for the thin-film tape geometry used in simulation (1μm × 12mm × 1m), apply the volume scaling factor:

```
k_fp_sim = k_fp_catalog × (V_tape / V_bulk)
         = k_fp_catalog × (1e-6 × 0.012 × 1.0) / (0.01 × 0.01 × 0.001)
         = k_fp_catalog × 0.12
```

Alternatively, use the Bean-London model with material-specific Jc0 values and actual tape cross-section for first-principles calculation.

## Migration Guide

To migrate existing code to use the centralized registry:

### Before
```python
# In sgms_anchor_v1.py
Jc0 = 3e10  # Hardcoded
thickness = 1e-6
```

### After
```python
from params import get_parameter

Jc0 = get_parameter('MATERIAL_PROPERTIES', 'GdBCO', 'Jc0')
thickness = get_parameter('SIMULATION_PARAMS', 'flux_pinning', 'thickness')
```

## Validation

Run self-test:
```bash
python3 params/canonical_values.py
```

Expected output:
```
Canonical Parameters Validation
==================================================
✓ All parameters validated successfully

Key Values:
  Jc0 = 30000000000.0 A/m²
  Tc = 92.0 K
  Operating T = 77.0 K
  Thickness = 1e-06 m
  Geometry scale = 0.12
```

## Adding New Parameters

1. Add to appropriate category in `canonical_values.py`
2. Include provenance: `value`, `uncertainty` (if known), `source`, and `note`
3. Update this README if resolving a new discrepancy
4. Run validation to ensure consistency

## Files

- `__init__.py` - Package initialization and exports
- `canonical_values.py` - Authoritative parameter definitions
- `README.md` - This documentation
