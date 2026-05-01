"""
Canonical Parameter Values for SGMS Anchor Simulation

This module defines the authoritative values for all physical constants
and simulation parameters. Each value includes provenance information
(source, uncertainty, and usage notes).

DISCREPANCY RESOLUTION:
- Jc0: Unified to 3e10 A/m² (was 3e10 in sgms_anchor_v1.py, 2e9 in earth_moon_pumping.py)
- thickness: Unified to 1e-6 m (thin-film tape geometry)
- temperature: Unified to 77K default (liquid nitrogen operating point)
- k_fp: Context-dependent - see MATERIAL_PROPERTIES for scaling relationships
"""

from typing import Dict, Any

# =============================================================================
# PHYSICAL CONSTANTS (Universal, from CODATA/NIST where applicable)
# =============================================================================

PHYSICAL_CONSTANTS: Dict[str, Any] = {
    # Stefan-Boltzmann constant
    'stefan_boltzmann': {
        'value': 5.670374419e-8,  # W/(m²·K⁴)
        'uncertainty': 0.0,  # Exact by definition (2019 SI redefinition)
        'source': 'CODATA 2018',
    },
    
    # Boltzmann constant
    'boltzmann': {
        'value': 1.380649e-23,  # J/K
        'uncertainty': 0.0,  # Exact by definition (2019 SI redefinition)
        'source': 'CODATA 2018',
    },
    
    # Elementary charge
    'elementary_charge': {
        'value': 1.602176634e-19,  # C
        'uncertainty': 0.0,  # Exact by definition
        'source': 'CODATA 2018',
    },
    
    # Vacuum permeability
    'mu0': {
        'value': 1.25663706212e-6,  # H/m (N/A²)
        'uncertainty': 1.9e-16,
        'source': 'CODATA 2018',
    },
    
    # Vacuum permittivity
    'eps0': {
        'value': 8.8541878128e-12,  # F/m
        'uncertainty': 1.3e-21,
        'source': 'CODATA 2018',
    },
}

# =============================================================================
# MATERIAL PROPERTIES (GdBCO Superconductor)
# =============================================================================

MATERIAL_PROPERTIES: Dict[str, Any] = {
    'GdBCO': {
        # Critical temperature
        'Tc': {
            'value': 92.0,  # K
            'uncertainty': 2.0,
            'source': 'Standard REBCO properties',
            'note': 'Gadolinium Barium Copper Oxide critical temperature',
        },
        
        # Critical current density at 0K, 0T
        'Jc0': {
            'value': 3e10,  # A/m²
            'uncertainty': 1e10,
            'source': 'Bean-London model calibration (2024)',
            'note': 'RESOLVED: Previously had 15x discrepancy between modules. '
                    'Unified to 3e10 A/m² based on thin-film tape measurements.',
        },
        
        # Flux pinning stiffness range (bulk samples)
        'k_fp_bulk_range': {
            'value': [80000, 120000],  # N/m
            'uncertainty': 20000,
            'source': 'gdbco_apc_catalog.json - baseline_gdbco profile',
            'note': 'Measured from bulk samples (10mm x 10mm x 1mm). '
                    'Apply geometry scaling for thin-film tape.',
        },
        
        # Geometry scaling factor for thin-film tape
        'geometry_scaling_factor': {
            'value': 0.12,  # dimensionless
            'calculation': '(1e-6 × 0.012 × 1.0) / (0.01 × 0.01 × 0.001)',
            'note': 'Volume ratio: V_tape / V_bulk. Use to scale k_fp from bulk to tape geometry.',
        },
        
        # n-exponent for E-J power law
        'n_exponent': {
            'value': 1.5,
            'uncertainty': 0.5,
            'source': 'HTS flux creep literature',
        },
        
        # Characteristic magnetic field
        'B0': {
            'value': 5.0,  # T
            'uncertainty': 1.0,
            'source': 'Bean-London model parameter',
        },
        
        # Field dependence exponent
        'alpha': {
            'value': 0.5,
            'uncertainty': 0.1,
            'source': 'Kim-Anderson model',
        },
        
        # Physical properties
        'density': {
            'value': 6380,  # kg/m³
            'source': 'REBCO material data',
        },
        'specific_heat': {
            'value': 180,  # J/kg/K at 77K (NOT room temperature)
            'source': 'Low-temperature specific heat data',
            'note': 'At 77K, NOT room temperature. Room temp value is ~500 J/kg/K.',
        },
        'thermal_conductivity': {
            'value': 3.0,  # W/m/K at 77K
            'source': 'Low-temperature thermal conductivity data',
            'note': 'At 77K',
        },
    },
    
    'YBCO': {
        # Critical temperature
        'Tc': {
            'value': 92.0,  # K
            'source': 'Standard REBCO',
        },
        
        # Critical current density at 0K, 0T
        'Jc0': {
            'value': 3e10,  # A/m²
            'source': 'Comparable to GdBCO',
        },
        
        # n-exponent for E-J power law
        'n_exponent': {
            'value': 1.5,
        },
        
        # Characteristic magnetic field (lower irreversibility field than GdBCO)
        'B0': {
            'value': 3.0,  # T
            'note': 'Lower irreversibility field than GdBCO',
        },
        
        # Field dependence exponent
        'alpha': {
            'value': 0.5,
        },
        
        # Physical properties
        'density': {
            'value': 6380,  # kg/m³
        },
        'specific_heat': {
            'value': 180,  # J/kg/K at 77K
            'note': 'At 77K, NOT room temperature',
        },
        'thermal_conductivity': {
            'value': 3.0,  # W/m/K at 77K
            'note': 'At 77K',
        },
        
        # Flux pinning stiffness range (bulk samples) - lower than GdBCO
        'k_fp_bulk_range': {
            'value': [30000, 60000],  # N/m
            'note': 'Lower than GdBCO',
        },
    },
    
    'BFRP': {
        # Density
        'density': {
            'value': 2500,  # kg/m³
            'uncertainty': 100,
            'source': 'Composite materials handbook',
        },
        
        # Tensile strength limit
        'tensile_strength': {
            'value': 1.2e9,  # Pa (1.2 GPa)
            'uncertainty': 0.2e9,
            'source': 'BFRP manufacturer specs',
            'safety_factor': 1.5,
            'allowable_stress': 8e8,  # 800 MPa with SF=1.5
        },
        
        # Thermal emissivity
        'emissivity': {
            'value': 0.85,
            'uncertainty': 0.05,
            'source': 'Surface coating data',
        },
    },
    
    'CFRP': {
        # Density
        'density': {
            'value': 1580,  # kg/m³
        },
        
        # Tensile strength (T700 grade)
        'tensile_strength': {
            'value': 3.0e9,  # Pa (3.0 GPa)
            'note': 'T700 grade',
        },
        
        # Safety factor
        'safety_factor': {
            'value': 1.5,
        },
        
        # Allowable stress
        'allowable_stress': {
            'value': 2.0e9,  # Pa (2.0 GPa)
        },
        
        # Thermal emissivity
        'emissivity': {
            'value': 0.88,
        },
        
        # Maximum operating temperature (epoxy limited)
        'max_operating_temp': {
            'value': 423,  # K (150C)
            'note': 'K (150C, epoxy limited)',
        },
    },
    
    'CNT_yarn': {
        # Density
        'density': {
            'value': 1400,  # kg/m³
        },
        
        # Tensile strength (state-of-art CNT yarn)
        'tensile_strength': {
            'value': 5.0e9,  # Pa (5.0 GPa)
            'note': 'State-of-art CNT yarn',
        },
        
        # Safety factor (higher for less mature technology)
        'safety_factor': {
            'value': 2.0,
        },
        
        # Allowable stress
        'allowable_stress': {
            'value': 2.5e9,  # Pa (2.5 GPa)
        },
        
        # Thermal emissivity (near-blackbody)
        'emissivity': {
            'value': 0.98,
            'note': 'Near-blackbody',
        },
        
        # Maximum operating temperature
        'max_operating_temp': {
            'value': 873,  # K (600C in vacuum)
            'note': 'K (600C in vacuum)',
        },
    },
    
    'NdFeB': {
        # Remanence (N52 grade)
        'remanence': {
            'value': 1.45,  # T
            'source': 'N52 grade',
        },
        
        # Coercivity
        'coercivity': {
            'value': 875e3,  # A/m
            'note': 'A/m',
        },
        
        # Maximum operating temperature
        'max_operating_temp': {
            'value': 353,  # K (80C for N52)
            'note': 'K (80C for N52)',
        },
        
        # Curie temperature
        'curie_temp': {
            'value': 583,  # K
            'note': 'K',
        },
        
        # Density
        'density': {
            'value': 7500,  # kg/m³
        },
        
        # Thermal coefficient of remanence (4x more sensitive than SmCo)
        'alpha_Br': {
            'value': -0.0012,  # /K (-0.12%/K)
            'source': 'N52 grade NdFeB thermal data',
            'note': '4x more sensitive than SmCo (-0.03%/K)',
        },
    },
    
    'SmCo': {
        # Note: SmCo is an alternative magnet material (not superconductor)
        # See TECHNICAL_SPEC.md lines 116-120 for qualitative trade study
        
        # Remanence
        'remanence': {
            'value': 1.1,  # T
            'source': 'Magnet manufacturer catalog',
        },
        
        # Coercivity
        'coercivity': {
            'value': 700e3,  # A/m
            'source': 'Magnet manufacturer catalog',
        },
        
        # Maximum operating temperature
        'max_operating_temp': {
            'value': 573,  # K (300C)
            'source': 'SmCo thermal stability data',
        },
        
        # Curie temperature
        'curie_temp': {
            'value': 1023,  # K (750C)
            'source': 'SmCo thermal stability data',
        },
        
        # Density
        'density': {
            'value': 8400,  # kg/m³
            'source': 'Permanent magnet material data',
        },
        
        # Thermal coefficient of remanence (very stable)
        'alpha_Br': {
            'value': -0.0003,  # /K (-0.03%/K)
            'source': 'SmCo thermal stability data',
            'note': 'Very stable, 4x better than NdFeB',
        },
    },
}

# =============================================================================
# SIMULATION PARAMETERS (Default operational values)
# =============================================================================

SIMULATION_PARAMS: Dict[str, Any] = {
    # Flux-pinning geometry (thin-film tape)
    'flux_pinning': {
        'thickness': {
            'value': 1e-6,  # m (1 μm)
            'note': 'Superconductor layer thickness',
        },
        'width': {
            'value': 0.012,  # m (12 mm)
            'note': 'Tape width',
        },
        'length': {
            'value': 1.0,  # m
            'note': 'Active length of tape in simulation',
        },
    },
    
    # Operating conditions
    'operating_conditions': {
        'temperature': {
            'value': 77.0,  # K
            'note': 'Liquid nitrogen boiling point - nominal operating temperature',
            'uncertainty': 5.0,
        },
        'max_temperature': {
            'value': 450.0,  # K
            'note': 'Thermal runaway threshold',
        },
    },
    
    # Control system defaults
    'control': {
        'g_gain_default': {
            'value': 0.000140,
            'note': 'Tuned for k_eff ≈ 6000 N/m in operational profile',
        },
        'damping_ratio': {
            'value': 0.05,
            'note': 'Baseline structural damping',
        },
    },
    
    # Monte Carlo settings
    'monte_carlo': {
        'n_converged': {
            'value': 100,
            'note': 'Minimum samples per point for converged CI (3.7% width)',
        },
        'n_preliminary': {
            'value': 20,
            'note': 'Preliminary runs only - CI width ~15%',
        },
    },
}

# =============================================================================
# GEOMETRY PARAMETERS (System configuration)
# =============================================================================

GEOMETRY_PARAMS: Dict[str, Any] = {
    # Prolate spheroid packet geometry
    'packet': {
        'radius': {
            'value': 0.1,  # m
            'note': 'BFRP sleeve radius',
        },
        'spacing': {
            'value': 0.48,  # m
            'note': 'Packet spacing along stream',
        },
    },
    
    # Anchor station
    'station': {
        'mass_default': {
            'value': 1000.0,  # kg
            'note': 'Suspended node baseline mass',
        },
    },
    
    # Spin dynamics
    'spin': {
        'rate_operational': {
            'value': 5236,  # rad/s (50,000 RPM)
            'note': 'Nominal operational spin rate',
        },
        'rate_recommended': {
            'value': 4189,  # rad/s (40,000 RPM)
            'note': 'Recommended for improved stress margin (50%)',
        },
        'stress_limit': {
            'value': 8e8,  # Pa (800 MPa)
            'note': 'Allowable stress with SF=1.5',
        },
    },
}


def get_parameter(category: str, subcategory: str, key: str) -> float:
    """
    Retrieve a canonical parameter value.
    
    Args:
        category: One of 'PHYSICAL_CONSTANTS', 'MATERIAL_PROPERTIES', 
                  'SIMULATION_PARAMS', 'GEOMETRY_PARAMS'
        subcategory: Material name or subsystem (e.g., 'GdBCO', 'flux_pinning')
        key: Parameter name (e.g., 'Jc0', 'thickness')
        
    Returns:
        The parameter value
        
    Raises:
        KeyError: If parameter not found
    """
    registry = {
        'PHYSICAL_CONSTANTS': PHYSICAL_CONSTANTS,
        'MATERIAL_PROPERTIES': MATERIAL_PROPERTIES,
        'SIMULATION_PARAMS': SIMULATION_PARAMS,
        'GEOMETRY_PARAMS': GEOMETRY_PARAMS,
    }
    
    if category not in registry:
        raise KeyError(f"Unknown category: {category}")
    
    cat_dict = registry[category]
    if subcategory not in cat_dict:
        raise KeyError(f"Unknown subcategory: {subcategory}")
    
    param = cat_dict[subcategory].get(key)
    if param is None:
        raise KeyError(f"Unknown parameter: {category}.{subcategory}.{key}")
    
    if isinstance(param, dict):
        return param.get('value', param)
    return param


def validate_parameters() -> list:
    """
    Validate that all canonical parameters are within reasonable ranges.
    
    Returns:
        List of validation warnings (empty if all OK)
    """
    warnings = []
    
    # Check Jc0 is positive and in expected range
    jc0 = MATERIAL_PROPERTIES['GdBCO']['Jc0']['value']
    if jc0 <= 0:
        warnings.append(f"Jc0 must be positive, got {jc0}")
    if jc0 < 1e8 or jc0 > 1e12:
        warnings.append(f"Jc0={jc0} A/m² outside typical HTS range [1e8, 1e12]")
    
    # Check temperature is below Tc
    T_op = SIMULATION_PARAMS['operating_conditions']['temperature']['value']
    Tc = MATERIAL_PROPERTIES['GdBCO']['Tc']['value']
    if T_op >= Tc:
        warnings.append(f"Operating temp {T_op}K >= Tc {Tc}K - superconductor would quench")
    
    # Check geometry scaling factor
    scale = MATERIAL_PROPERTIES['GdBCO']['geometry_scaling_factor']['value']
    if scale <= 0 or scale > 1:
        warnings.append(f"Geometry scaling factor {scale} should be in (0, 1]")
    
    return warnings


if __name__ == '__main__':
    # Self-test
    print("Canonical Parameters Validation")
    print("=" * 50)
    
    warnings = validate_parameters()
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  ⚠ {w}")
    else:
        print("✓ All parameters validated successfully")
    
    print("\nKey Values:")
    print(f"  Jc0 = {MATERIAL_PROPERTIES['GdBCO']['Jc0']['value']} A/m²")
    print(f"  Tc = {MATERIAL_PROPERTIES['GdBCO']['Tc']['value']} K")
    print(f"  Operating T = {SIMULATION_PARAMS['operating_conditions']['temperature']['value']} K")
    print(f"  Thickness = {SIMULATION_PARAMS['flux_pinning']['thickness']['value']} m")
    print(f"  Geometry scale = {MATERIAL_PROPERTIES['GdBCO']['geometry_scaling_factor']['value']}")
