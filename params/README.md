# Centralized Parameter Registry

## Purpose

Single source of truth for physical constants and simulation parameters. Resolves cross-module discrepancies.

## Resolved Discrepancies

| Parameter | Previous | Canonical | Notes |
|-----------|----------|-----------|-------|
| Jc0 | 3e10 / 2e9 A/m² | 3e10 A/m² | 15x discrepancy resolved |
| thickness | 1e-6 / bulk | 1e-6 m | Thin-film tape geometry |
| temperature | 77K / 300K | 77K | Liquid nitrogen operating |
| k_fp | 0–220k N/m | Context-dependent | See geometry scaling |

## Usage

```python
from params import get_parameter, MATERIAL_PROPERTIES, SIMULATION_PARAMS

Jc0 = get_parameter('MATERIAL_PROPERTIES', 'GdBCO', 'Jc0')
thickness = get_parameter('SIMULATION_PARAMS', 'flux_pinning', 'thickness')
```

## Geometry Scaling for k_fp

Catalog k_fp values are from bulk samples (10mm × 10mm × 1mm). For thin-film tape (1μm × 12mm × 1m):

```
k_fp_sim = k_fp_catalog × (V_tape / V_bulk) = k_fp_catalog × 0.12
```

Alternatively, use Bean-London model with material Jc0 and actual tape cross-section.

## Validation

```bash
python3 params/canonical_values.py
```

Expected: all parameters validated successfully.
