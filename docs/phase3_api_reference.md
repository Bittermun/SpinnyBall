# Phase 3 API Reference

This document provides API reference for Phase 3 Advanced Diagnostics & Radiation Hardening components.

## Enhanced VMD-IRCNN Stub

### `control_layer.vmd_enhanced_stub`

#### `EnhancedDecompositionParameters`

Parameters for adaptive frequency decomposition.

**Attributes:**
- `num_modes` (int): Number of frequency modes (default: 6)
- `adaptive_bands` (bool): Enable adaptive band selection (default: True)
- `min_freq` (float): Minimum frequency in Hz (default: 0.05)
- `max_freq` (float): Maximum frequency in Hz (default: 0.8)
- `overlap_ratio` (float): Overlap between bands (default: 0.2)

#### `AdaptiveFrequencyDecomposer`

Adaptive frequency-based signal decomposer.

**Methods:**
- `__init__(params: EnhancedDecompositionParameters)`: Initialize decomposer
- `decompose(signal: np.ndarray) -> np.ndarray`: Decompose signal into frequency bands

**Parameters:**
- `signal`: Input signal [n_samples]

**Returns:**
- Decomposed frequency bands [num_modes × n_samples]

#### `DeepResidualPredictor`

Deep residual network with skip connections.

**Methods:**
- `__init__(input_dim, output_dim, hidden_dim, num_blocks, dropout_rate)`: Initialize predictor
- `forward(x: torch.Tensor) -> torch.Tensor`: Forward pass

**Parameters:**
- `input_dim`: Input dimension (default: 7)
- `output_dim`: Output dimension (default: 7)
- `hidden_dim`: Hidden dimension (default: 128)
- `num_blocks`: Number of residual blocks (default: 4)
- `dropout_rate`: Dropout rate (default: 0.1)

#### `EnhancedPredictorCascade`

Enhanced predictor cascade combining decomposition and deep residual network.

**Methods:**
- `__init__(decomp_params, predictor_input_dim, predictor_output_dim, predictor_hidden_dim, predictor_num_blocks, is_trained)`: Initialize cascade
- `predict(trajectory: np.ndarray, horizon: int) -> np.ndarray`: Predict future trajectory

**Parameters:**
- `trajectory`: Past trajectory [n_timesteps × state_dim]
- `horizon`: Prediction horizon (default: 10)

**Returns:**
- Predicted trajectory [horizon × state_dim]

## Synthetic Failure Data Pipeline

### `control_layer.failure_modes`

#### `FailureType`

Enum of failure types:
- `DEBRIS_IMPACT`: Momentum kick from debris
- `THERMAL_RUNAWAY`: Temperature increase
- `MAGNETIC_QUENCH`: Magnetic efficiency drop
- `SENSOR_FAILURE`: Sensor noise/bias
- `ACTUATOR_FAILURE`: Control signal loss
- `PACKET_CAPTURE_FAILURE`: Missed capture
- `PACKET_RELEASE_FAILURE`: Stuck packet
- `VELOCITY_PERTURBATION`: Velocity kick
- `SPIN_RATE_PERTURBATION`: Angular velocity change
- `POSITION_PERTURBATION`: Position offset

#### `FailureModeLibrary`

Library of failure modes with realistic physics.

**Methods:**
- `__init__(random_seed: int)`: Initialize library
- `apply_failure(failure_event: FailureEvent, packet, current_time: float)`: Apply failure to packet
- `generate_failure_sequence(time_horizon, max_failures, failure_types) -> List[FailureEvent]`: Generate failure sequence

### `control_layer.data_generator`

#### Module Constants

- `FAILURE_DURATION_TIMESTEPS` (int): Number of timesteps to label as failure after event (default: 50)

#### `DataGenerationConfig`

Configuration for synthetic data generation.

**Attributes:**
- `n_samples` (int): Number of samples (default: 10000)
- `time_horizon` (float): Simulation time horizon (default: 10.0)
- `dt` (float): Time step (default: 0.01)
- `n_packets` (int): Number of packets (default: 5)
- `max_failures_per_sample` (int): Max failures per sample (default: 3)
- `random_seed` (int): Random seed (default: 42)
- `output_dir` (str): Output directory (default: "control_layer/data")
- `dataset_name` (str): Dataset name (default: "synthetic_failure_data")

#### `SyntheticDataGenerator`

Generate synthetic failure data for ML training.

**Methods:**
- `__init__(config, failure_library)`: Initialize generator
- `generate_sample(sample_id) -> Tuple[np.ndarray, List[FailureEvent], np.ndarray]`: Generate single sample
- `generate_dataset() -> dict`: Generate full dataset
- `save_dataset_hdf5(dataset, filepath) -> str`: Save dataset to HDF5

### `control_layer.data_quality`

#### `DataQualityChecker`

Perform quality checks on synthetic failure datasets.

**Methods:**
- `__init__(filepath: str)`: Initialize checker
- `load_dataset()`: Load dataset from HDF5
- `check_label_distribution(min_failure_rate) -> QualityCheckResult`: Check label distribution
- `check_trajectory_continuity(max_jump) -> QualityCheckResult`: Check trajectory continuity
- `check_state_normalization(max_quaternion_norm) -> QualityCheckResult`: Check quaternion normalization
- `check_sample_balance(min_samples_per_failure_type) -> QualityCheckResult`: Check sample balance
- `check_data_range(max_velocity, max_angular_velocity) -> QualityCheckResult`: Check data range
- `run_all_checks() -> Dict[str, QualityCheckResult]`: Run all quality checks
- `generate_quality_report() -> str`: Generate quality report

## Statistical Anomaly Detection

### `control_layer.anomaly_detector`

#### Module Constants

The following constants are defined at module level for configurable thresholds:

- `Z_SCORE_THRESHOLD_CRITICAL` (float): Z-score threshold for CRITICAL severity (default: 5.0)
- `Z_SCORE_THRESHOLD_WARNING` (float): Z-score threshold for WARNING severity (default: 4.0)
- `ISOLATION_FOREST_CRITICAL_SCORE` (float): Anomaly score threshold for CRITICAL severity (default: 0.5)
- `ISOLATION_FOREST_WARNING_SCORE` (float): Anomaly score threshold for WARNING severity (default: 0.3)
- `NORM_EPSILON` (float): Small epsilon value for numerical stability (default: 1e-8)

#### `AlertSeverity`

Enum of alert severities:
- `INFO`: Informational
- `WARNING`: Warning
- `CRITICAL`: Critical

#### `AlertAction`

Enum of response actions:
- `LOG`: Log alert
- `REDUCE_GAIN`: Reduce control gain
- `SAFE_SET_HOLD`: Trigger safe-set hold

#### `AnomalyAlert`

Represents an anomaly detection alert.

**Attributes:**
- `timestamp` (float): Alert timestamp
- `severity` (AlertSeverity): Alert severity
- `detector_name` (str): Detector name
- `anomaly_score` (float): Anomaly score
- `threshold` (float): Detection threshold
- `affected_packet_id` (int): Affected packet ID
- `action` (AlertAction): Response action
- `message` (str): Alert message

#### `FailureEvent`

Represents a single failure event.

**Attributes:**
- `failure_type` (FailureType): Type of failure
- `severity` (float): Severity level 0.0-1.0
- `timestamp` (float): Time of failure in seconds
- `affected_packet_id` (Optional[int]): Packet ID affected by failure
- `metadata` (Dict, field(default_factory=dict)): Additional metadata

#### `ZScoreDetector`

Z-score based anomaly detector.

**Methods:**
- `__init__(threshold, window_size, state_dim)`: Initialize detector
- `update_statistics(state) -> Tuple[np.ndarray, np.ndarray]`: Update rolling statistics
- `detect(state, packet_id, timestamp) -> Optional[AnomalyAlert]`: Detect anomalies

**Parameters:**
- `threshold` (float): Z-score threshold for anomaly detection (default: 3.0)
- `window_size` (int): Window size for moving statistics (default: 100)
- `state_dim` (int): State vector dimension (default: 7)

**Note**: Uses `samples_added` counter to track buffer initialization, ensuring correct initialization after filling buffer once.

#### `IsolationForestDetector`

Isolation Forest based anomaly detector.

**Methods:**
- `__init__(contamination, n_estimators, state_dim)`: Initialize detector
- `add_training_sample(state)`: Add training sample
- `train()`: Train the model
- `detect(state, packet_id, timestamp) -> Optional[AnomalyAlert]`: Detect anomalies

**Parameters:**
- `contamination`: Expected anomaly proportion (default: 0.1)
- `n_estimators`: Number of trees (default: 100)
- `state_dim`: State dimension (default: 7)

**Note:** Requires scikit-learn

#### `StatisticalAnomalyDetector`

Statistical anomaly detection system combining multiple detectors.

**Methods:**
- `__init__(use_z_score, use_isolation_forest, state_dim)`: Initialize detector
- `detect(state, packet_id, timestamp) -> List[AnomalyAlert]`: Detect anomalies
- `train_isolation_forest()`: Train isolation forest detector

**Parameters:**
- `use_z_score`: Enable z-score detector (default: True)
- `use_isolation_forest`: Enable isolation forest detector (default: True)
- `state_dim`: State dimension (default: 7)

#### `ResponseHandler`

Handle anomaly detection responses.

**Methods:**
- `__init__()`: Initialize handler
- `handle_alert(alert: AnomalyAlert) -> bool`: Handle an alert
- `get_alert_summary() -> dict`: Get alert summary

## Training

### `control_layer.train_vmd_enhanced`

#### `train_enhanced_predictor`

Train enhanced VMD-IRCNN predictor on synthetic data.

**Parameters:**
- `n_samples` (int): Number of training samples (default: 10000)
- `n_epochs` (int): Number of training epochs (default: 50)
- `batch_size` (int): Batch size (default: 32)
- `learning_rate` (float): Learning rate (default: 0.001)
- `device` (str): Device (default: "cpu")
- `save_path` (str): Path to save model

**Returns:**
- Training metrics dictionary

#### `validate_against_rom`

Validate trained predictor against ROM predictor baseline.

**Parameters:**
- `trained_predictor` (DeepResidualPredictor): Trained predictor
- `n_test_samples` (int): Number of test samples (default: 100)
- `device` (str): Device (default: "cpu")

**Returns:**
- Validation metrics dictionary
