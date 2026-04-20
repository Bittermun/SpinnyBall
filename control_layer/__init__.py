"""
Control layer for gyroscopic mass-stream simulation.

This module implements model-predictive control (MPC) and reduced-order
model (ROM) predictors for the closed-loop mass-stream system.
"""

from .mpc_controller import (
    StubMPCController,
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
    VMDIRCNNDetector,
    # Deprecated legacy names (for backward compatibility)
    VMDIRCNNCascade as _VMDIRCNNCascade,
    StubVMD as _StubVMD,
    StubIRCNN as _StubIRCNN,
    VMDParameters as _VMDParameters,
    create_vmd_ircnn_cascade as _create_vmd_ircnn_cascade,
)

try:
    from .vmd_decomposition import VMDDecomposer, VMDParameters as TrueVMDParameters
    VMD_AVAILABLE = True
except ImportError:
    VMD_AVAILABLE = False
    VMDDecomposer = None
    TrueVMDParameters = None

try:
    from .ircnn_predictor import IRCNNPredictor, IRCNNParameters
    IRCNN_AVAILABLE = True
except ImportError:
    IRCNN_AVAILABLE = False
    IRCNNPredictor = None
    IRCNNParameters = None

from .state_converter import StateConverter

from .ml_integration import MLIntegrationLayer, get_ml_integration

try:
    from .training_pipeline import TrainingPipeline, TrainingConfig
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False
    TrainingPipeline = None
    TrainingConfig = None

__all__ = [
    "StubMPCController",
    "LinearizedROM",
    "ROMParameters",
    "create_rom",
    # New accurate names
    "SimplifiedPredictorCascade",
    "SimplifiedFrequencyDecomposer",
    "SimplifiedLinearPredictor",
    "SimplifiedDecompositionParameters",
    "create_simplified_predictor_cascade",
    "VMDIRCNNDetector",
    # True VMD decomposition
    "VMDDecomposer",
    "TrueVMDParameters",
    "VMD_AVAILABLE",
    # True IRCNN predictor
    "IRCNNPredictor",
    "IRCNNParameters",
    "IRCNN_AVAILABLE",
    # State converter
    "StateConverter",
    # ML integration layer
    "MLIntegrationLayer",
    "get_ml_integration",
    # Training pipeline
    "TrainingPipeline",
    "TrainingConfig",
    "TRAINING_AVAILABLE",
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
