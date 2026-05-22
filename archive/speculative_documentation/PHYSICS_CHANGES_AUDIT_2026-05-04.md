# Physics Changes Audit - May 4, 2026

## Summary
After pulling latest from main (21 commits behind), significant physics changes were identified that affect simulation outcomes. This audit documents the changes and validates their impact through fresh data runs.

## Critical Physics Changes Identified

### 1. dynamics/gdBCO_material.py - B0 Parameter (MAJOR)
**Change:** `B0` (characteristic magnetic field) changed from **0.1 T → 5.0 T** (50x increase)

**Impact:** The critical current density formula uses:
```
field_factor = 1.0 / (1.0 + (B / B0) ** alpha)
J_c(B, T) = J_c0 * (1 - T/T_c)^n * field_factor
```

With B0=5.0 T vs 0.1 T:
- At B=1.0T: field_factor increases from ~0.24 → ~0.69 (**2.9x increase**)
- This increases pinning force and stiffness estimates by ~3x

### 2. dynamics/gdBCO_material.py - Thermal Properties (MODERATE)
| Property | Old Value | New Value | Change |
|----------|-----------|-----------|--------|
| specific_heat | 500 J/kg/K | 180 J/kg/K | -64% |
| thermal_conductivity | 10.0 W/m/K | 3.0 W/m/K | -70% |
| density | 6300 kg/m³ | 6380 kg/m³ | +1.3% |

**Impact:** Thermal dynamics have different time constants; heating/cooling rates will change.

### 3. monte_carlo/cascade_runner.py - Stress Formula (MAJOR)
**Change:** Centrifugal stress formula modified

**Old:** `stress = mass * omega^2 / (4 * pi * radius)`
**New:** `stress = mass * (radius * omega)^2 / (4 * pi * radius^2)` = `mass * omega^2 / (4 * pi)`

**Impact:** Removes radius dependency from stress calculation, changing stress estimates for all packet sizes.

## Outcome Comparison: Mission-Level Analysis

### Material Configuration Feasibility Rates

| Configuration | Old Feasibility | New Feasibility | Delta |
|--------------|-----------------|-----------------|-------|
| GdBCO_BFRP | 16.76% | 17.7% | +0.9% |
| GdBCO_CFRP | 26.88% | 27.4% | +0.5% |
| GdBCO_CNT_yarn | 28.34% | 28.8% | +0.5% |
| SmCo_BFRP | 0.34% | 0.4% | +0.1% |
| SmCo_CFRP | 1.54% | 1.0% | -0.5% |
| SmCo_CNT_yarn | 1.92% | 1.2% | -0.7% |

**Conclusion:** Physics changes produced measurable differences in outcomes (±0.5-1.0%), confirming the need for data re-runs.

## Files Updated with New Data
- `mission_analysis_results/` - Fresh Sobol analysis results for all 6 material configurations
- `sweep_t3_fault_cascade.png` - Updated T3 sweep visualization
- `sweep_t3_highres_results.json` - Updated high-resolution sweep data

## Recommendations
1. ✅ **Data re-run completed** - Mission-level analysis has been re-run with new physics
2. ⚠️ **T3 fault cascade sweeps** still show statistical limitations (0 faults injected at low rates due to insufficient realization count × time horizon)
3. Consider increasing `n_realizations_per_point` or `time_horizon` for T3 sweeps, or use `fault_injection_mode='guaranteed'`

## Verification
Current code confirms:
- B0 = 5.0 T (canonical value from params.canonical_values)
- specific_heat = 180.0 J/kg/K (77K value, NOT room temp)
- thermal_conductivity = 3.0 W/m/K (77K value)
- density = 6380.0 kg/m³

All changes are now active in the codebase and producing quantifiably different outcomes.
