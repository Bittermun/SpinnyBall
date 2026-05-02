"""Test that all modules use canonical parameter values."""
import pytest
import ast
import re
from pathlib import Path
from params.canonical_values import MATERIAL_PROPERTIES

CANONICAL_JC0 = MATERIAL_PROPERTIES['GdBCO']['Jc0']['value']  # 3e10
CANONICAL_B0 = MATERIAL_PROPERTIES['GdBCO']['B0']['value']    # 5.0

# Directories to scan for hardcoded values
SCAN_DIRS = ['dynamics', 'src', 'monte_carlo', 'scripts']

def test_no_hardcoded_jc0_2e9():
    """No module should hardcode Jc0 = 2e9 (the known-wrong value)."""
    root = Path(__file__).parent.parent
    violations = []
    for d in SCAN_DIRS:
        for f in (root / d).rglob('*.py'):
            text = f.read_text()
            if re.search(r'2e9(?!\d)', text) and 'Jc' in text:
                violations.append(str(f.relative_to(root)))
    assert not violations, f"Files with hardcoded Jc=2e9: {violations}"

def test_gdbco_b0_not_0_1():
    """GdBCO B0 default must not be 0.1 T."""
    from dynamics.gdBCO_material import GdBCOProperties
    props = GdBCOProperties()
    assert props.B0 == CANONICAL_B0, f"GdBCOProperties.B0={props.B0}, expected {CANONICAL_B0}"

def test_default_gdbco_props_match_canonical():
    """DEFAULT_GDBCO_PROPS in sgms_anchor_v1 must match canonical registry."""
    from src.sgms_anchor_v1 import DEFAULT_GDBCO_PROPS
    assert DEFAULT_GDBCO_PROPS.Jc0 == CANONICAL_JC0
    assert DEFAULT_GDBCO_PROPS.B0 == CANONICAL_B0

def test_ybco_differs_from_gdbco():
    """YBCO must have distinct B0 and k_fp_bulk_range from GdBCO."""
    gdbco = MATERIAL_PROPERTIES['GdBCO']
    ybco = MATERIAL_PROPERTIES['YBCO']
    assert gdbco['B0']['value'] != ybco['B0']['value'], "YBCO B0 must differ from GdBCO"
    assert gdbco['k_fp_bulk_range']['value'] != ybco['k_fp_bulk_range']['value']

def test_mission_level_metrics_differentiates_materials():
    """mission_level_metrics must produce different results for YBCO vs GdBCO."""
    from src.sgms_anchor_v1 import mission_level_metrics
    gdbco_result = mission_level_metrics(magnet_material="GdBCO", jacket_material="BFRP")
    ybco_result = mission_level_metrics(magnet_material="YBCO", jacket_material="BFRP")
    # At least one metric must differ
    assert gdbco_result != ybco_result, "YBCO and GdBCO must produce different results"

def test_snode_default_above_feasibility_gate():
    """SNode default k_fp must be >= 6000 N/m feasibility gate."""
    from dynamics.multi_body import SNode
    import numpy as np
    node = SNode(id=0, position=np.array([0.0, 0.0, 0.0]))
    assert node.k_fp >= 6000.0, f"SNode default k_fp={node.k_fp} < 6000 feasibility gate"
