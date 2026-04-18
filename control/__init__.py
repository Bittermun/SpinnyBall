"""
Control layer for gyroscopic mass-stream simulation.

This module implements model-predictive control (MPC) and reduced-order
model (ROM) predictors for the closed-loop mass-stream system.
"""

from .mpc_controller import (
    MPCController,
    StubMPCController,
    create_mpc_controller,
    verify_mpc_latency,
)
from .rom_predictor import (
    LinearizedROM,
    ROMParameters,
    create_rom,
)
from .vmd_ircnn_stub import (
    SimplifiedPredictorCascade,
    SimplifiedFrequencyDecomposer,
    SimplifiedLinearPredictor,
    SimplifiedDecompositionParameters,
    create_simplified_predictor_cascade,
    # Deprecated legacy names (for backward compatibility)
    VMDIRCNNCascade as _VMDIRCNNCascade,
    StubVMD as _StubVMD,
    StubIRCNN as _StubIRCNN,
    VMDParameters as _VMDParameters,
    create_vmd_ircnn_cascade as _create_vmd_ircnn_cascade,
)

__all__ = [
    "MPCController",
    "StubMPCController",
    "create_mpc_controller",
    "verify_mpc_latency",
    "LinearizedROM",
    "ROMParameters",
    "create_rom",
    # New accurate names
    "SimplifiedPredictorCascade",
    "SimplifiedFrequencyDecomposer",
    "SimplifiedLinearPredictor",
    "SimplifiedDecompositionParameters",
    "create_simplified_predictor_cascade",
    # Deprecated legacy names (for backward compatibility)
    "VMDIRCNNCascade",
    "StubVMD",
    "StubIRCNN",
    "VMDParameters",
    "create_vmd_ircnn_cascade",
]

# Provide backward compatibility aliases with deprecation warnings
VMDIRCNNCascade = _VMDIRCNNCascade
StubVMD = _StubVMD
StubIRCNN = _StubIRCNN
VMDParameters = _VMDParameters
create_vmd_ircnn_cascade = _create_vmd_ircnn_cascade
