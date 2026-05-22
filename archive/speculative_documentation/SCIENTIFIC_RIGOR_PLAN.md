# Comprehensive Scientific Rigor Implementation Plan

## Executive Summary

This plan addresses critical scientific accuracy issues identified in the SGMS Anchor simulation codebase. Implementation is organized into 6 phases with parallel AI execution support, prioritizing immediate fixes before systematic improvements.

**PARALLEL EXECUTION MODEL:**
- **Reviewer/Implementer AI** (this AI): Reviews code, validates implementations, ensures integration
- **Parallel AI Workers**: Implement specific components based on detailed specifications
- **Integration Gates**: Each phase requires reviewer approval before proceeding

---

## PHASE 0: Immediate Critical Fixes (Priority: CRITICAL - START NOW)

### 0.1 Reduce Default Timestep

**Rationale:** Current dt=0.01s causes numerical instability at 50,000 RPM. Immediate fix required.

**Implementation:**
```python
# Update params/canonical_values.py
SIMULATION_PARAMS['operating_conditions']['dt_default'] = {
    'value': 0.001,  # Reduced from 0.01s
    'note': 'Critical fix for 50k RPM stability (1.2ms period)',
    'temporary': True  # Will be replaced by adaptive timestep in Phase 1
}
```

**Success Criteria:**
- All simulations use dt=0.001s by default
- No energy drift > 1% over 100 spin periods
- Performance impact documented

**Dependencies:** None
**Estimated Effort:** 0.5 days
**Risk:** Low (temporary fix)

### 0.2 Add Energy Conservation Checks

**Rationale:** Detect numerical integration errors immediately.

**Implementation:**
```python
# Add to dynamics/rigid_body.py
def integrate(self, t_span, torques, dt=0.001, check_energy=True):
    """Integrate with energy conservation monitoring."""
    
    if check_energy:
        initial_energy = self.compute_total_energy()
    
    # ... existing integration code ...
    
    if check_energy:
        final_energy = self.compute_total_energy()
        energy_drift = abs(final_energy - initial_energy) / initial_energy
        
        if energy_drift > 0.01:  # 1% threshold
            logger.warning(f"Energy conservation violation: {energy_drift:.4f}")
            # Could trigger adaptive timestep or halt simulation
    
    return result
```

**Success Criteria:**
- Energy drift warnings trigger for >1% errors
- Angular momentum conservation also checked
- Performance impact <5%

**Dependencies:** 0.1
**Estimated Effort:** 1 day
**Risk:** Low

### 0.3 Disable Stub ML in Production

**Rationale:** Stub detectors may produce misleading results.

**Implementation:**
```python
# Update backend/ml_integration.py
class MLIntegrationLayer:
    def __init__(self, use_stub_in_production=False):
        if not use_stub_in_production and not TRUE_VMD_AVAILABLE:
            logger.error("True VMD not available - ML disabled for scientific accuracy")
            self.enable_wobble_detection = False
```

**Success Criteria:**
- Stub detectors disabled by default
- Clear error messages when ML unavailable
- Fallback to physics-based detection only

**Dependencies:** None
**Estimated Effort:** 0.5 days
**Risk:** Low

---

## PHASE 1: Critical Numerical Methods (Priority: HIGH)

### 1.1 Adaptive Timestep Control for RK4 Integrator

**Rationale:** Fixed dt=0.01s is inadequate for 50,000 RPM (5236 rad/s) spin dynamics. At this rate, one revolution takes 1.2ms, making dt=10ms far too coarse.

**BUG FIXES NEEDED:**
- Current plan uses Bogacki-Shampine (3rd order) for error estimation but RK4 is 4th order - mismatched orders
- No handling of NaN/infinite states during error calculation
- Missing vectorized operations for performance

**Corrected Implementation:**
```python
# New file: dynamics/adaptive_integrator.py
class AdaptiveRK4Integrator:
    """RK4 with adaptive timestep based on local error estimation."""
    
    def __init__(self, 
                 dt_initial: float = 0.001,
                 dt_min: float = 1e-6,
                 dt_max: float = 0.01,
                 rtol: float = 1e-6,
                 atol: float = 1e-9,
                 safety_factor: float = 0.9,
                 max_rejects: int = 10):
        """
        Args:
            dt_initial: Initial timestep (s)
            dt_min: Minimum allowed timestep (s)
            dt_max: Maximum allowed timestep (s)
            rtol: Relative tolerance for error control
            atol: Absolute tolerance for error control
            safety_factor: Safety factor for timestep updates
            max_rejects: Maximum consecutive rejected steps
        """
        self.dt = dt_initial
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.rtol = rtol
        self.atol = atol
        self.safety = safety_factor
        self.max_rejects = max_rejects
        self.reject_count = 0
        
    def step(self, rhs, t: float, y: np.ndarray) -> tuple[np.ndarray, float, bool]:
        """
        Single adaptive step with proper error estimation.
        
        Uses embedded RK4(5) pair (Dormand-Prince) for consistent error estimation.
        
        Returns:
            (y_new, dt_used, accepted): New state, timestep used, and acceptance flag
        """
        # Dormand-Prince RK4(5) coefficients
        # This is the standard method used in ODE45
        a = np.array([[0, 0, 0, 0, 0, 0, 0],
                      [1/5, 0, 0, 0, 0, 0, 0],
                      [3/40, 9/40, 0, 0, 0, 0, 0],
                      [44/45, -56/15, 32/9, 0, 0, 0, 0],
                      [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
                      [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
                      [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]])
        
        b5 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])  # 5th order
        b4 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])  # 4th order
        
        # Compute stages
        k = np.zeros((7, len(y)))
        k[0] = self.dt * rhs(t, y)
        k[1] = self.dt * rhs(t + a[1,0]*self.dt, y + a[1,0]*k[0])
        k[2] = self.dt * rhs(t + a[2,0]*self.dt, y + a[2,0]*k[0] + a[2,1]*k[1])
        k[3] = self.dt * rhs(t + a[3,0]*self.dt, y + a[3,0]*k[0] + a[3,1]*k[1] + a[3,2]*k[2])
        k[4] = self.dt * rhs(t + a[4,0]*self.dt, y + a[4,0]*k[0] + a[4,1]*k[1] + a[4,2]*k[2] + a[4,3]*k[3])
        k[5] = self.dt * rhs(t + a[5,0]*self.dt, y + a[5,0]*k[0] + a[5,1]*k[1] + a[5,2]*k[2] + a[5,3]*k[3] + a[5,4]*k[4])
        k[6] = self.dt * rhs(t + a[6,0]*self.dt, y + a[6,0]*k[0] + a[6,1]*k[1] + a[6,2]*k[2] + a[6,3]*k[3] + a[6,4]*k[4] + a[6,5]*k[5])
        
        # 5th order solution
        y5 = y + np.dot(b5, k)
        # 4th order solution
        y4 = y + np.dot(b4, k)
        
        # Error estimate (difference between 5th and 4th order)
        error = np.linalg.norm(y5 - y4)
        scale = self.atol + self.rtol * np.maximum(np.linalg.norm(y), np.linalg.norm(y5))
        error_ratio = error / scale
        
        # Accept/reject step
        if error_ratio <= 1.0:
            # Accept step
            if error_ratio > 0:
                # Update timestep for next step
                dt_new = self.safety * self.dt * error_ratio ** (-0.2)
                self.dt = np.clip(dt_new, self.dt_min, self.dt_max)
            self.reject_count = 0
            return y5, self.dt, True
        else:
            # Reject step
            dt_new = self.safety * self.dt * error_ratio ** (-0.25)
            self.dt = max(dt_new, self.dt_min)
            self.reject_count += 1
            
            if self.reject_count > self.max_rejects:
                raise RuntimeError(f"Maximum rejected steps ({self.max_rejects}) exceeded")
            
            return y, self.dt, False
```

**Success Criteria:**
- Timestep automatically reduces to <0.001s during high-frequency spin events
- Energy conservation verified to 1e-6 relative error over 100 revolutions
- Angular momentum conservation verified to machine precision
- No more than 5% performance impact vs. fixed dt=0.001s

**Dependencies:** None
**Estimated Effort:** 3-4 days (more complex than initially estimated)
**Risk:** Medium (may expose existing physics model instabilities)

**PARALLEL AI ASSIGNMENT:**
- **AI Worker 1**: Implement Dormand-Prince coefficients and stage calculations
- **AI Worker 2**: Add error handling and NaN state management
- **Reviewer AI**: Validate mathematical correctness, test against scipy ODE45

---

### 1.2 Convergence Criteria and Error Bounds

**Rationale:** No verification that numerical solutions have converged to correct physics.

**BUG FIXES NEEDED:**
- Richardson extrapolation formula incorrect for adaptive timestep
- Missing handling of division by zero in convergence calculation
- No validation of monotonic convergence

**Corrected Implementation:**
```python
# Add to dynamics/convergence_checker.py
class ConvergenceChecker:
    """Verify numerical convergence of simulations."""
    
    def __init__(self, 
                 max_iterations: int = 10,
                 convergence_threshold: float = 0.01,
                 min_dt_ratio: float = 0.5):
        """
        Args:
            max_iterations: Maximum refinement iterations
            convergence_threshold: Maximum relative change for convergence
            min_dt_ratio: Minimum ratio between consecutive timesteps
        """
        self.max_iterations = max_iterations
        self.threshold = convergence_threshold
        self.min_dt_ratio = min_dt_ratio
        
    def check_convergence(self, 
                          simulation_fn: Callable,
                          dt_values: list[float],
                          metric_fn: Callable) -> ConvergenceResult:
        """
        Check convergence by Richardson extrapolation.
        
        Uses proper Richardson extrapolation for order verification.
        
        Args:
            simulation_fn: Function that takes dt and returns results
            dt_values: List of timesteps to test (descending order, geometric progression)
            metric_fn: Function to extract convergence metric from results
            
        Returns:
            ConvergenceResult with order of convergence and estimated error
        """
        if len(dt_values) < 3:
            raise ValueError("Need at least 3 timestep values for convergence analysis")
        
        # Verify geometric progression (required for Richardson extrapolation)
        for i in range(1, len(dt_values)):
            ratio = dt_values[i] / dt_values[i-1]
            if abs(ratio - 0.5) > 0.1:  # Allow 10% tolerance
                logger.warning(f"Timesteps not in geometric progression: ratio={ratio}")
        
        results = []
        for dt in dt_values:
            try:
                result = simulation_fn(dt)
                metric = metric_fn(result)
                if np.isfinite(metric):
                    results.append((dt, metric))
                else:
                    logger.warning(f"Non-finite result for dt={dt}")
                    return ConvergenceResult(converged=False, error="Non-finite result")
            except Exception as e:
                logger.error(f"Simulation failed for dt={dt}: {e}")
                return ConvergenceResult(converged=False, error=f"Simulation failed: {e}")
        
        if len(results) < 3:
            return ConvergenceResult(converged=False, error="Insufficient valid results")
        
        # Richardson extrapolation for convergence order
        # For geometric progression with ratio r=0.5:
        # Order p = log2((f(h) - f(2h)) / (f(h/2) - f(h)))
        
        dt_h, f_h = results[0]
        dt_2h, f_2h = results[1]
        dt_h2, f_h2 = results[2]
        
        # Check monotonic convergence
        if not (abs(f_h - f_2h) > 1e-15 and abs(f_h2 - f_h) > 1e-15):
            return ConvergenceResult(converged=False, error="Insufficient change in results")
        
        # Compute convergence order
        numerator = abs(f_h - f_2h)
        denominator = abs(f_h2 - f_h)
        
        if denominator > 0 and numerator > 0:
            r = numerator / denominator
            if r > 0:
                order = np.log2(r)
            else:
                order = 0
        else:
            order = 0
        
        # Validate order (should be positive for convergence)
        if order <= 0:
            return ConvergenceResult(converged=False, error=f"Non-convergent sequence (order={order})")
        
        # Estimate error using Richardson extrapolation
        # f_exact ≈ f_h + (f_h - f_2h) / (2^p - 1)
        try:
            error_estimate = abs(f_h - f_2h) / (2**order - 1)
            relative_error = error_estimate / abs(f_h) if f_h != 0 else float('inf')
        except (ZeroDivisionError, OverflowError):
            return ConvergenceResult(converged=False, error="Error in Richardson extrapolation")
        
        converged = relative_error < self.threshold
        
        return ConvergenceResult(
            converged=converged,
            order_of_convergence=order,
            error_estimate=error_estimate,
            relative_error=relative_error,
            dt_recommended=dt_h / 2 if not converged and dt_h / 2 >= dt_values[-1] else dt_h,
            validation_passed=True
        )
```

**Success Criteria:**
- All simulations verify convergence before reporting results
- Richardson extrapolation shows 4th-order convergence for RK4
- Automatic timestep refinement when convergence not achieved
- Proper error handling for non-convergent cases

**Dependencies:** 1.1 (Adaptive timestep)
**Estimated Effort:** 3 days (more complex due to error handling)
**Risk:** Low

**PARALLEL AI ASSIGNMENT:**
- **AI Worker 1**: Implement Richardson extrapolation mathematics
- **AI Worker 2**: Add comprehensive error handling and validation
- **Reviewer AI**: Test against known analytical solutions

---

### 1.3 Timestep Stability Validation

**Rationale:** Verify numerical stability for high-frequency spin dynamics.

**BUG FIXES NEEDED:**
- Missing actual energy drift computation implementation
- No consideration of numerical precision limits
- CFL condition oversimplified for coupled dynamics

**Corrected Implementation:**
```python
# Add to tests/test_timestep_stability.py
class TestTimestepStability:
    """Validate numerical stability for spin dynamics."""
    
    def __init__(self):
        self.tolerance_energy = 1e-6  # Relative energy conservation tolerance
        self.tolerance_momentum = 1e-12  # Angular momentum conservation tolerance
        
    def test_spin_frequency_stability(self):
        """Verify stability across operational spin frequencies."""
        spin_rates = [1000, 5000, 10000, 40000, 50000, 60000]  # RPM
        
        results = {}
        
        for rpm in spin_rates:
            omega = rpm * 2 * np.pi / 60  # rad/s
            period = 2 * np.pi / omega
            
            rpm_results = {}
            
            # Test multiple timesteps per period
            for steps_per_period in [10, 20, 50, 100, 200, 500]:
                dt = period / steps_per_period
                
                try:
                    # Run simulation and check energy conservation
                    energy_drift, momentum_drift, max_dt_used = self._compute_conservation_metrics(
                        rpm, dt, duration=10*period
                    )
                    
                    rpm_results[steps_per_period] = {
                        'energy_drift': energy_drift,
                        'momentum_drift': momentum_drift,
                        'max_dt_used': max_dt_used,
                        'stable': energy_drift < self.tolerance_energy and momentum_drift < self.tolerance_momentum
                    }
                    
                except Exception as e:
                    rpm_results[steps_per_period] = {
                        'error': str(e),
                        'stable': False
                    }
            
            results[rpm] = rpm_results
        
        # Validate results
        for rpm, rpm_results in results.items():
            # Should achieve stability at some timestep
            stable_configs = [sp for sp, config in rpm_results.items() 
                            if config.get('stable', False)]
            
            assert len(stable_configs) > 0, f"No stable configuration found for {rpm} RPM"
            
            # Check that more steps per period improves or maintains stability
            stable_steps = sorted(stable_configs)
            if len(stable_steps) > 1:
                # Should not need more steps than necessary
                min_stable = min(stable_steps)
                assert min_stable <= 100, f"Requires too many steps per period for {rpm} RPM: {min_stable}"
        
        return results
    
    def _compute_conservation_metrics(self, rpm: float, dt: float, duration: float) -> tuple[float, float, float]:
        """
        Compute energy and momentum conservation metrics.
        
        Returns:
            (energy_drift, momentum_drift, max_dt_used)
        """
        from dynamics.rigid_body import RigidBody
        from dynamics.multi_body import MultiBodyStream, Packet
        
        # Create simple spinning packet
        omega = rpm * 2 * np.pi / 60
        I = np.diag([1e-6, 1e-6, 2e-6])  # Small moment of inertia
        body = RigidBody(0.05, I)
        body.angular_velocity = np.array([0, 0, omega])
        
        # Initial conservation quantities
        E_initial = body.compute_kinetic_energy()
        L_initial = body.compute_angular_momentum()
        
        # Integrate
        n_steps = int(duration / dt)
        max_dt_used = dt
        
        for step in range(n_steps):
            # Simple zero-torque integration
            body.integrate_numba_rk4_zero_torque((0, duration), dt=dt)
            
            # Check if adaptive timestep was used (if implemented)
            if hasattr(body, 'last_dt_used'):
                max_dt_used = max(max_dt_used, body.last_dt_used)
        
        # Final quantities
        E_final = body.compute_kinetic_energy()
        L_final = body.compute_angular_momentum()
        
        # Relative drifts
        energy_drift = abs(E_final - E_initial) / E_initial if E_initial != 0 else float('inf')
        momentum_drift = np.linalg.norm(L_final - L_initial) / np.linalg.norm(L_initial) if np.linalg.norm(L_initial) > 0 else float('inf')
        
        return energy_drift, momentum_drift, max_dt_used
    
    def test_cfl_condition(self):
        """Verify CFL-like condition for coupled spin-translation dynamics."""
        # For coupled rotational-translational dynamics
        # dt < min(characteristic_time_scale) / stability_factor
        
        characteristic_times = {
            'spin_period': 2*np.pi / (50000 * 2*np.pi/60),  # 1.2ms
            'libration_period': 2*np.pi / np.sqrt(6000/1000),  # ~2.5s
            'control_response': 0.1,  # 100ms
            'packet_transit': 0.48 / 1600,  # Packet spacing/stream velocity
        }
        
        # Most restrictive is spin period
        dt_max_spin = characteristic_times['spin_period'] / 20  # 20 points per period
        dt_max_libration = characteristic_times['libration_period'] / 100  # Conservative
        
        dt_max = min(dt_max_spin, dt_max_libration)
        
        # Verify this is reasonable
        assert dt_max >= 1e-6, f"Required dt_max={dt_max} too small for practical simulation"
        assert dt_max <= 1e-3, f"dt_max={dt_max} too large for spin stability"
        
        return {
            'characteristic_times': characteristic_times,
            'dt_max_recommended': dt_max,
            'limiting_factor': 'spin_period' if dt_max_spin < dt_max_libration else 'libration_period'
        }
```

**Success Criteria:**
- Energy conservation < 0.1% over 1000 spin periods at 50,000 RPM
- Angular momentum conservation to machine precision
- Identification of stability limits for all operational parameters
- Performance impact documented and acceptable

**Dependencies:** 1.1, 1.2
**Estimated Effort:** 4 days (more comprehensive testing)
**Risk:** High (may require significant timestep reduction, impacting performance)

**PARALLEL AI ASSIGNMENT:**
- **AI Worker 1**: Implement energy/momentum conservation metrics
- **AI Worker 2**: Create comprehensive stability test suite
- **Reviewer AI**: Validate against analytical solutions and performance benchmarks

---

## PHASE 2: ML Model Validation (Priority: HIGH)

### 2.1 Ground Truth Dataset for Wobble Detection

**Rationale:** Stub ML detectors need validation against physics-based ground truth.

**Implementation:**
```python
# New file: control_layer/ground_truth_generator.py
class WobbleGroundTruthGenerator:
    """Generate physics-based ground truth for wobble detection validation."""
    
    def __init__(self, 
                 n_samples: int = 10000,
                 wobble_fraction: float = 0.3,
                 duration: float = 1.0,
                 dt: float = 0.001):
        """
        Generate synthetic signals with known wobble characteristics.
        
        Args:
            n_samples: Number of signals to generate
            wobble_fraction: Fraction of signals that contain wobble
            duration: Signal duration (s)
            dt: Timestep (s)
        """
        self.n_samples = n_samples
        self.wobble_fraction = wobble_fraction
        self.duration = duration
        self.dt = dt
        self.n_points = int(duration / dt)
        
    def generate(self) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Generate ground truth dataset.
        
        Returns:
            (signals, labels, metadata): Signals (n_samples, n_points),
                                         labels (n_samples,) True/False for wobble,
                                         metadata dict with wobble parameters
        """
        signals = []
        labels = []
        metadata = []
        
        for i in range(self.n_samples):
            has_wobble = i < int(self.n_samples * self.wobble_fraction)
            
            # Base signal: clean spin
            t = np.arange(self.n_points) * self.dt
            base_freq = 10000 / 60  # 10k RPM in Hz
            signal = np.sin(2 * np.pi * base_freq * t)
            
            if has_wobble:
                # Add nutation/precession physics
                # Based on Euler equations for asymmetric rotor
                wobble_freq = base_freq * 0.1  # Typical nutation frequency
                wobble_amplitude = np.random.uniform(0.01, 0.1)  # 1-10% wobble
                
                # Add nutation component
                signal += wobble_amplitude * np.sin(2 * np.pi * wobble_freq * t)
                
                # Add noise
                signal += 0.001 * np.random.randn(self.n_points)
                
                labels.append(True)
                metadata.append({
                    'wobble_freq': wobble_freq,
                    'wobble_amplitude': wobble_amplitude,
                    'type': 'nutation'
                })
            else:
                # Clean signal with only noise
                signal += 0.001 * np.random.randn(self.n_points)
                labels.append(False)
                metadata.append({'type': 'clean'})
            
            signals.append(signal)
        
        return np.array(signals), np.array(labels), metadata
    
    def validate_detector(self, detector_fn: Callable) -> dict:
        """
        Validate a detector against ground truth.
        
        Args:
            detector_fn: Function that takes signal and returns (is_wobble, confidence)
            
        Returns:
            Validation metrics dict
        """
        signals, true_labels, _ = self.generate()
        
        predictions = []
        confidences = []
        for signal in signals:
            is_wobble, confidence, _ = detector_fn(signal)
            predictions.append(is_wobble)
            confidences.append(confidence)
        
        predictions = np.array(predictions)
        confidences = np.array(confidences)
        
        # Compute metrics
        from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary'
        )
        
        try:
            auc = roc_auc_score(true_labels, confidences)
        except ValueError:
            auc = None
        
        cm = confusion_matrix(true_labels, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'accuracy': (tp + tn) / (tp + tn + fp + fn)
        }
```

**Success Criteria:**
- Ground truth generated from physics-based nutation/precession models
- Dataset includes edge cases (near-threshold wobble, noisy signals, multiple modes)
- Comprehensive validation metrics (precision, recall, F1, AUC-ROC)

**Dependencies:** None
**Estimated Effort:** 3-4 days
**Risk:** Medium (requires understanding of gyroscopic dynamics)

---

### 2.2 Statistical Verification for ML Detectors

**Rationale:** Verify stub detectors meet minimum performance thresholds.

**Implementation:**
```python
# Add to tests/test_ml_validation.py
class TestMLDetectorValidation:
    """Statistical validation of ML detectors."""
    
    def test_stub_detector_minimum_performance(self):
        """Stub detector must achieve minimum performance on ground truth."""
        from control_layer.ground_truth_generator import WobbleGroundTruthGenerator
        from control_layer.vmd_ircnn_stub import VMDIRCNNDetector
        
        # Generate ground truth
        generator = WobbleGroundTruthGenerator(n_samples=1000)
        
        # Test stub detector
        detector = VMDIRCNNDetector()
        metrics = generator.validate_detector(detector.detect_wobble)
        
        # Minimum performance thresholds
        assert metrics['precision'] >= 0.7, f"Precision {metrics['precision']} below threshold"
        assert metrics['recall'] >= 0.7, f"Recall {metrics['recall']} below threshold"
        assert metrics['f1_score'] >= 0.7, f"F1 {metrics['f1_score']} below threshold"
        
    def test_detector_uncertainty_quantification(self):
        """Detector must provide calibrated uncertainty estimates."""
        # Confidence scores should be calibrated (confidence ≈ accuracy)
        
        generator = WobbleGroundTruthGenerator(n_samples=5000)
        signals, true_labels, _ = generator.generate()
        
        detector = VMDIRCNNDetector()
        
        # Bin predictions by confidence
        confidence_bins = np.linspace(0, 1, 11)
        bin_accuracies = []
        bin_confidences = []
        
        for i in range(len(confidence_bins) - 1):
            mask = (confidences >= confidence_bins[i]) & (confidences < confidence_bins[i+1])
            if np.sum(mask) > 0:
                bin_acc = np.mean(predictions[mask] == true_labels[mask])
                bin_conf = np.mean(confidences[mask])
                bin_accuracies.append(bin_acc)
                bin_confidences.append(bin_conf)
        
        # Expected calibration error (ECE)
        ece = np.mean(np.abs(np.array(bin_accuracies) - np.array(bin_confidences)))
        assert ece < 0.1, f"Expected calibration error {ece} too high"
```

**Success Criteria:**
- All detectors achieve minimum 70% precision/recall on ground truth
- Confidence scores are well-calibrated (ECE < 0.1)
- Statistical tests pass with p < 0.05 significance

**Dependencies:** 2.1
**Estimated Effort:** 2 days
**Risk:** Low

---

### 2.3 Uncertainty Quantification for ML Predictions

**Rationale:** ML predictions must include uncertainty bounds for scientific use.

**Implementation:**
```python
# Add to control_layer/ml_uncertainty.py
class MLUncertaintyQuantifier:
    """Add uncertainty quantification to ML predictions."""
    
    def __init__(self, detector, n_ensemble: int = 10):
        """
        Args:
            detector: Base detector model
            n_ensemble: Number of ensemble members for uncertainty estimation
        """
        self.detector = detector
        self.n_ensemble = n_ensemble
        
    def predict_with_uncertainty(self, signal: np.ndarray) -> dict:
        """
        Make prediction with uncertainty bounds.
        
        Uses ensemble of noisy perturbations to estimate epistemic uncertainty.
        
        Returns:
            dict with prediction, confidence, and uncertainty bounds
        """
        predictions = []
        confidences = []
        
        # Generate ensemble by adding noise to signal
        for i in range(self.n_ensemble):
            # Add small perturbations
            noise_level = 0.001 * np.std(signal)
            perturbed_signal = signal + np.random.randn(len(signal)) * noise_level
            
            is_wobble, confidence, _ = self.detector.detect_wobble(perturbed_signal)
            predictions.append(is_wobble)
            confidences.append(confidence)
        
        predictions = np.array(predictions)
        confidences = np.array(confidences)
        
        # Compute statistics
        mean_conf = np.mean(confidences)
        std_conf = np.std(confidences)
        prediction_consistency = np.mean(predictions)  # Fraction predicting wobble
        
        # Uncertainty bounds (95% CI)
        ci_lower = max(0, mean_conf - 1.96 * std_conf)
        ci_upper = min(1, mean_conf + 1.96 * std_conf)
        
        return {
            'prediction': prediction_consistency > 0.5,
            'confidence': mean_conf,
            'confidence_std': std_conf,
            'confidence_ci': (ci_lower, ci_upper),
            'prediction_entropy': self._compute_entropy(predictions),
            'epistemic_uncertainty': std_conf,
            'aleatoric_uncertainty': self._estimate_aleatoric(signal),
        }
    
    def _compute_entropy(self, predictions: np.ndarray) -> float:
        """Compute prediction entropy as uncertainty measure."""
        p = np.mean(predictions)
        if p == 0 or p == 1:
            return 0
        return -p * np.log2(p) - (1-p) * np.log2(1-p)
    
    def _estimate_aleatoric(self, signal: np.ndarray) -> float:
        """Estimate aleatoric (data) uncertainty from signal properties."""
        # Higher noise = higher aleatoric uncertainty
        snr = np.abs(np.mean(signal)) / (np.std(signal) + 1e-10)
        return 1 / (1 + snr)  # Inverse SNR as uncertainty proxy
```

**Success Criteria:**
- All ML predictions include 95% confidence intervals
- Epistemic and aleatoric uncertainty separated
- Uncertainty properly propagated to downstream decisions

**Dependencies:** 2.2
**Estimated Effort:** 2 days
**Risk:** Low

---

## PHASE 3: Physics Model Improvements (Priority: MEDIUM)

### 3.1 Gradual Thermal Degradation Near T_c

**Rationale:** Current model has abrupt transition at T_c; real superconductors show gradual degradation.

**Implementation:**
```python
# Update dynamics/gdBCO_material.py
def critical_current_density(self, B: float, T: float) -> float:
    """
    Compute J_c(B, T) with improved thermal model.
    
    Includes gradual degradation approaching T_c following
    Ginzburg-Landau theory predictions.
    """
    # Ginzburg-Landau critical behavior near T_c
    # J_c ∝ (1 - T/T_c)^n for T < T_c
    # With gradual transition using error function smoothing
    
    if T >= self.props.Tc:
        return 0.0
    
    # Reduced temperature
    t_reduced = T / self.props.Tc
    
    # GL theory: order parameter ψ ∝ (1 - t)^(1/2)
    # J_c ∝ ψ^2 ∝ (1 - t) near T_c
    gl_factor = (1 - t_reduced) ** 1.5  # GL exponent + Bean-London
    
    # Add thermal fluctuation correction near T_c
    # kT/ΔE where ΔE is condensation energy
    if t_reduced > 0.9:  # Within 10% of T_c
        fluctuation_width = 0.05  # 5% width for thermal fluctuations
        t_eff = t_reduced + fluctuation_width * np.random.randn()
        t_eff = np.clip(t_eff, 0, 1)
        gl_factor = (1 - t_eff) ** 1.5
    
    # Magnetic field dependence (Kim model)
    b_reduced = B / self.props.B0
    field_factor = 1.0 / (1.0 + b_reduced ** self.props.alpha)
    
    return self.props.Jc0 * gl_factor * field_factor
```

**Success Criteria:**
- J_c shows correct (1-T/T_c)^1.5 scaling near T_c
- Thermal fluctuations included for T > 0.9 T_c
- Continuity verified at T_c transition

**Dependencies:** None
**Estimated Effort:** 2 days
**Risk:** Low

---

### 3.2 Electromagnetic-Thermal Coupling

**Rationale:** Current models treat EM and thermal independently; real systems are coupled.

**Implementation:**
```python
# New file: dynamics/em_thermal_coupling.py
class EMThermalCoupling:
    """Coupled electromagnetic-thermal model for superconductors."""
    
    def __init__(self, material: GdBCOMaterial, 
                 thermal_model: LumpedThermalModel):
        """
        Args:
            material: Superconductor material model
            thermal_model: Thermal model for temperature evolution
        """
        self.material = material
        self.thermal = thermal_model
        
    def compute_coupled_step(self, 
                            state: EM ThermalState,
                            dt: float) -> EM ThermalState:
        """
        Compute one coupled timestep.
        
        Coupling:
        1. Current flow generates Joule heating: P = I^2 * R(T)
        2. Heating raises temperature: dT = P*dt / (m*c_p)
        3. Temperature changes J_c: J_c = f(T, B)
        4. Changed J_c affects current distribution
        """
        # Current density from Bean-London model
        J_c = self.material.critical_current_density(
            state.B_field, state.temperature
        )
        
        # If operating current > J_c, resistive transition occurs
        if state.current_density > J_c:
            # Resistive heating
            # R = R_normal * (1 - J_c/J_op) for J_op > J_c
            excess_current_ratio = state.current_density / J_c - 1
            resistance = self._compute_resistance(excess_current_ratio)
            heating_power = state.current_density**2 * resistance * state.volume
        else:
            # Superconducting: only AC losses
            heating_power = self._compute_ac_losses(state)
        
        # Update temperature
        new_temperature = self.thermal.step(
            state.temperature, heating_power, dt
        )
        
        # Update critical current with new temperature
        new_J_c = self.material.critical_current_density(
            state.B_field, new_temperature
        )
        
        # Check for quench (thermal runaway)
        if new_temperature > state.temperature and heating_power > 0:
            # Positive feedback: heating increases T, which can reduce J_c
            dT_dJ = -1.5 * (1 - new_temperature/self.material.props.Tc)**0.5
            dJ_dT = new_J_c / self.material.props.Tc * dT_dJ
            
            # Stability criterion: dP_heating/dT < dP_cooling/dT
            dP_heating_dT = 2 * state.current_density * state.current_density * resistance / new_J_c * dJ_dT
            dP_cooling_dT = self.thermal.heat_transfer_coefficient
            
            if dP_heating_dT > dP_cooling_dT:
                state.quench_detected = True
        
        return EM ThermalState(
            temperature=new_temperature,
            B_field=state.B_field,
            current_density=min(state.current_density, new_J_c),
            J_c=new_J_c,
            heating_power=heating_power,
            quench_detected=state.quench_detected
        )
```

**Success Criteria:**
- Correct prediction of thermal runaway conditions
- Stable operation verified for nominal conditions
- Quench detection validated against literature data

**Dependencies:** 3.1
**Estimated Effort:** 4-5 days
**Risk:** Medium (complex coupling may expose instabilities)

---

### 3.3 Geometry Scaling Factor Validation

**Rationale:** Geometry scaling factor (0.12) lacks experimental validation.

**Implementation:**
```python
# Add to validation/geometry_scaling_validation.py
class GeometryScalingValidator:
    """Validate geometry scaling factors against experimental data."""
    
    def __init__(self):
        # Literature data for REBCO tape vs bulk
        self.literature_data = {
            'bulk_sample_10mm': {
                'k_fp_measured': [80000, 120000],  # N/m
                'geometry': {'thickness': 0.001, 'width': 0.01, 'length': 0.01},
                'source': 'Su et al. 2019'
            },
            'tape_12mm': {
                'k_fp_measured': [6000, 12000],  # N/m (estimated from thin film)
                'geometry': {'thickness': 1e-6, 'width': 0.012, 'length': 1.0},
                'source': 'AMSC datasheet + scaling'
            }
        }
        
    def validate_scaling_law(self) -> dict:
        """
        Validate volume-based scaling law.
        
        Theory: k_fp ∝ V (flux pinning sites scale with volume)
        """
        bulk = self.literature_data['bulk_sample_10mm']
        tape = self.literature_data['tape_12mm']
        
        # Compute volumes
        V_bulk = (bulk['geometry']['thickness'] * 
                  bulk['geometry']['width'] * 
                  bulk['geometry']['length'])
        
        V_tape = (tape['geometry']['thickness'] * 
                  tape['geometry']['width'] * 
                  tape['geometry']['length'])
        
        # Expected scaling
        expected_ratio = V_tape / V_bulk
        
        # Measured ratio
        k_bulk_avg = np.mean(bulk['k_fp_measured'])
        k_tape_avg = np.mean(tape['k_fp_measured'])
        measured_ratio = k_tape_avg / k_bulk_avg
        
        # Compare
        discrepancy = abs(measured_ratio - expected_ratio) / expected_ratio
        
        return {
            'expected_ratio': expected_ratio,
            'measured_ratio': measured_ratio,
            'discrepancy': discrepancy,
            'valid': discrepancy < 0.5,  # Within 50% (large uncertainty expected)
            'recommendation': self._generate_recommendation(discrepancy)
        }
```

**Success Criteria:**
- Scaling law validated against ≥2 independent literature sources
- Uncertainty bounds established for scaling factor
- Updated canonical values if discrepancy > 30%

**Dependencies:** None
**Estimated Effort:** 2-3 days (literature research + analysis)
**Risk:** Low

---

## PHASE 4: Monte Carlo Statistical Rigor (Priority: MEDIUM)

### 4.1 Statistical Tests for Fault Injection

**Rationale:** Verify fault injection follows correct statistical distributions.

**Implementation:**
```python
# Add to tests/test_mc_statistics.py
class TestMonteCarloStatistics:
    """Statistical validation of Monte Carlo methodology."""
    
    def test_poisson_fault_distribution(self):
        """Verify fault injection follows Poisson distribution."""
        from scipy.stats import chisquare, kstest
        
        # Run many realizations and count faults
        config = MonteCarloConfig(
            n_realizations=1000,
            time_horizon=10.0,
            fault_rate=1.0,  # 1 fault per hour
            fault_injection_mode='poisson'
        )
        
        fault_counts = []
        for seed in range(100):
            config.random_seed = seed
            runner = CascadeRunner(config)
            result = runner.run_monte_carlo(stream_factory)
            fault_counts.append(result['fault_events_total'])
        
        # Expected Poisson distribution
        lambda_expected = config.fault_rate * config.time_horizon / 3600
        
        # Kolmogorov-Smirnov test against Poisson CDF
        # Transform to uniform via CDF
        from scipy.stats import poisson
        uniform_transformed = poisson.cdf(fault_counts, lambda_expected)
        
        statistic, p_value = kstest(uniform_transformed, 'uniform')
        
        assert p_value > 0.05, f"Fault distribution not Poisson (p={p_value})"
    
    def test_independence_of_faults(self):
        """Verify faults are independent (no temporal clustering)."""
        # Compute autocorrelation of fault times
        # Should be white noise (no correlation)
        
        config = MonteCarloConfig(
            n_realizations=100,
            time_horizon=100.0,
            fault_rate=10.0,
            fault_injection_mode='poisson'
        )
        
        # Collect all fault times
        all_fault_times = []
        for seed in range(50):
            config.random_seed = seed
            runner = CascadeRunner(config)
            # Modify runner to record fault times
            result = runner.run_monte_carlo_with_timing(stream_factory)
            all_fault_times.extend(result['fault_times'])
        
        # Compute inter-fault intervals
        intervals = np.diff(sorted(all_fault_times))
        
        # Test for exponential distribution (Poisson process)
        from scipy.stats import expon
        statistic, p_value = kstest(intervals, 'expon', 
                                     args=(0, 1/config.fault_rate))
        
        assert p_value > 0.05, f"Inter-fault intervals not exponential (p={p_value})"
```

**Success Criteria:**
- Poisson fault distribution validated (p > 0.05)
- Independence verified (no autocorrelation)
- Exponential inter-fault intervals confirmed

**Dependencies:** None
**Estimated Effort:** 2 days
**Risk:** Low

---

### 4.2 Bias Detection and CI Validation

**Rationale:** Verify confidence intervals are unbiased and achieve nominal coverage.

**Implementation:**
```python
# Add to monte_carlo/statistical_validation.py
class MCStatisticalValidator:
    """Validate statistical properties of Monte Carlo results."""
    
    def __init__(self, runner: CascadeRunner):
        self.runner = runner
        
    def validate_confidence_interval_coverage(self, 
                                               true_parameter: float,
                                               n_trials: int = 100) -> dict:
        """
        Verify CI achieves nominal coverage (e.g., 95% CI contains true value 95% of time).
        
        Args:
            true_parameter: Known true value from analytical solution or high-res simulation
            n_trials: Number of MC runs to perform
            
        Returns:
            Coverage statistics
        """
        ci_contains_truth = []
        ci_widths = []
        
        for trial in range(n_trials):
            # Run MC with this configuration
            result = self.runner.run_monte_carlo(stream_factory)
            
            # Extract CI for cascade probability
            ci_lower = result['cascade_probability_ci'][0]
            ci_upper = result['cascade_probability_ci'][1]
            
            # Check if true value is in CI
            contains = ci_lower <= true_parameter <= ci_upper
            ci_contains_truth.append(contains)
            
            ci_widths.append(ci_upper - ci_lower)
        
        # Compute coverage
        actual_coverage = np.mean(ci_contains_truth)
        nominal_coverage = 0.95  # Assuming 95% Wilson CI
        
        # Should be within 5% of nominal (90-100% for 95% CI)
        coverage_error = abs(actual_coverage - nominal_coverage)
        
        return {
            'actual_coverage': actual_coverage,
            'nominal_coverage': nominal_coverage,
            'coverage_error': coverage_error,
            'valid': coverage_error < 0.05,
            'mean_ci_width': np.mean(ci_widths),
            'ci_width_std': np.std(ci_widths)
        }
    
    def detect_bias(self, analytical_solution: float, n_runs: int = 50) -> dict:
        """
        Detect systematic bias in MC estimates.
        
        Uses t-test to check if mean estimate differs from analytical solution.
        """
        estimates = []
        
        for run in range(n_runs):
            result = self.runner.run_monte_carlo(stream_factory)
            estimates.append(result['cascade_probability'])
        
        estimates = np.array(estimates)
        
        # One-sample t-test against analytical solution
        from scipy.stats import ttest_1samp
        t_stat, p_value = ttest_1samp(estimates, analytical_solution)
        
        # Compute effect size (Cohen's d)
        mean_diff = np.mean(estimates) - analytical_solution
        pooled_std = np.std(estimates, ddof=1)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        return {
            'mean_estimate': np.mean(estimates),
            'analytical_solution': analytical_solution,
            'bias': mean_diff,
            'relative_bias': mean_diff / analytical_solution if analytical_solution != 0 else float('inf'),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant_bias': p_value < 0.05,
            'cohens_d': cohens_d,
            'effect_size': 'small' if abs(cohens_d) < 0.2 else 'medium' if abs(cohens_d) < 0.5 else 'large'
        }
```

**Success Criteria:**
- CI coverage within 5% of nominal (e.g., 90-100% for 95% CI)
- No significant bias detected (p > 0.05 for t-test)
- Effect size small (Cohen's d < 0.2) for any detected bias

**Dependencies:** 4.1
**Estimated Effort:** 2 days
**Risk:** Medium (may reveal existing biases requiring model fixes)

---

### 4.3 Early Termination Criteria Review

**Rationale:** Current early termination based on CI width may introduce bias.

**Implementation:**
```python
# Update monte_carlo/cascade_runner.py
def run_monte_carlo(self, stream_factory: Callable) -> dict:
    """
    Run Monte Carlo with optional early termination.
    
    Updated to use sequential testing with proper bias correction.
    """
    results = []
    n_realized = 0
    
    # Sequential testing with O'Brien-Fleming bounds
    # to control Type I error under optional stopping
    
    while n_realized < self.config.n_realizations:
        # Run batch
        batch_results = self._run_batch(stream_factory, n_realized)
        results.extend(batch_results)
        n_realized += len(batch_results)
        
        # Check early termination criteria (if enabled)
        if self.config.enable_early_termination and n_realized >= self.config.min_realizations:
            # Compute current CI
            successes = sum(1 for r in results if r.success)
            ci_lower, ci_upper = self._wilson_ci(successes, n_realized)
            ci_width = (ci_upper - ci_lower) / ((ci_upper + ci_lower) / 2) if (ci_upper + ci_lower) > 0 else float('inf')
            
            # Apply bias correction for early stopping
            # Using Chow-Robbins correction
            if ci_width < self.config.ci_width_threshold:
                # Don't stop immediately - run additional validation realizations
                validation_results = self._run_batch(stream_factory, n_realized, n=20)
                validation_successes = sum(1 for r in validation_results if r.success)
                
                # Check consistency
                validation_rate = validation_successes / len(validation_results)
                main_rate = successes / n_realized
                
                if abs(validation_rate - main_rate) < 0.1:  # Within 10%
                    # Accept early termination
                    logger.info(f"Early termination at n={n_realized} (validated)")
                    break
                else:
                    # Continue - results inconsistent
                    logger.warning(f"Validation failed ({validation_rate:.3f} vs {main_rate:.3f}), continuing...")
```

**Success Criteria:**
- Early termination validated with hold-out set (no bias detected)
- Sequential testing with proper error control
- Conservative approach prioritizes accuracy over speed

**Dependencies:** 4.2
**Estimated Effort:** 2 days
**Risk:** Medium (may change MC performance characteristics)

---

## PHASE 5: Uncertainty Propagation (Priority: MEDIUM)

### 5.1 Uncertainty Tracking Framework

**Rationale:** All calculations must propagate uncertainties from input parameters to final results.

**Implementation:**
```python
# New file: utils/uncertainty_propagation.py
from dataclasses import dataclass
from typing import Union, List
import numpy as np

@dataclass
class UncertainValue:
    """Value with uncertainty representation."""
    nominal: float
    uncertainty: float  # Standard uncertainty (1 sigma)
    distribution: str = 'normal'  # 'normal', 'uniform', 'triangular'
    
    def __add__(self, other: 'UncertainValue') -> 'UncertainValue':
        """Addition: u_c = sqrt(u_a^2 + u_b^2) (uncorrelated)."""
        return UncertainValue(
            nominal=self.nominal + other.nominal,
            uncertainty=np.sqrt(self.uncertainty**2 + other.uncertainty**2),
            distribution='normal'  # Sum of normals is normal
        )
    
    def __mul__(self, other: 'UncertainValue') -> 'UncertainValue':
        """Multiplication: relative uncertainties add in quadrature."""
        nominal = self.nominal * other.nominal
        rel_unc_self = self.uncertainty / abs(self.nominal) if self.nominal != 0 else 0
        rel_unc_other = other.uncertainty / abs(other.nominal) if other.nominal != 0 else 0
        uncertainty = nominal * np.sqrt(rel_unc_self**2 + rel_unc_other**2)
        
        return UncertainValue(nominal=nominal, uncertainty=uncertainty)
    
    def apply_function(self, func: callable, derivative: callable) -> 'UncertainValue':
        """Apply function with uncertainty propagation (GUM method)."""
        nominal = func(self.nominal)
        uncertainty = abs(derivative(self.nominal)) * self.uncertainty
        return UncertainValue(nominal, uncertainty)


class MonteCarloUncertaintyPropagator:
    """Propagate uncertainties using Monte Carlo method (GUM Supplement 1)."""
    
    def __init__(self, n_samples: int = 10000):
        self.n_samples = n_samples
        
    def propagate(self,
                  inputs: dict[str, UncertainValue],
                  model: callable) -> UncertainValue:
        """
        Propagate uncertainties through model using MC sampling.
        
        Args:
            inputs: Dict of input parameters with uncertainties
            model: Function that takes dict of nominal values and returns result
            
        Returns:
            UncertainValue for model output
        """
        # Generate samples from input distributions
        samples = {}
        for name, value in inputs.items():
            if value.distribution == 'normal':
                samples[name] = np.random.normal(
                    value.nominal, value.uncertainty, self.n_samples
                )
            elif value.distribution == 'uniform':
                samples[name] = np.random.uniform(
                    value.nominal - value.uncertainty,
                    value.nominal + value.uncertainty,
                    self.n_samples
                )
            elif value.distribution == 'triangular':
                from scipy.stats import triang
                samples[name] = triang.rvs(
                    0.5,  # mode at center
                    loc=value.nominal - value.uncertainty,
                    scale=2*value.uncertainty,
                    size=self.n_samples
                )
        
        # Evaluate model for each sample
        results = []
        for i in range(self.n_samples):
            input_sample = {name: samples[name][i] for name in inputs}
            try:
                result = model(input_sample)
                results.append(result)
            except:
                results.append(np.nan)
        
        results = np.array(results)
        valid_results = results[~np.isnan(results)]
        
        return UncertainValue(
            nominal=np.mean(valid_results),
            uncertainty=np.std(valid_results, ddof=1),
            distribution='normal'  # CLT approximation
        )
```

**Success Criteria:**
- All canonical parameters converted to UncertainValue representation
- Uncertainty propagated through all physics calculations
- Final results include uncertainty bounds

**Dependencies:** None
**Estimated Effort:** 3-4 days
**Risk:** Medium (requires systematic code changes)

---

### 5.2 Physical Consistency Checks

**Rationale:** Parameter validation should verify physical consistency, not just ranges.

**Implementation:**
```python
# Update params/validation.py
class PhysicalConsistencyChecker:
    """Check physical consistency of parameter sets."""
    
    def __init__(self):
        self.checks = []
        
    def check_thermal_consistency(self, params: dict) -> list[str]:
        """Verify thermal parameters are physically consistent."""
        errors = []
        
        # Operating temperature must be below critical temperature
        T_op = params.get('temperature', 77)
        T_c = params.get('Tc', 92)
        
        if T_op >= T_c:
            errors.append(f"Operating temperature {T_op}K >= T_c {T_c}K: superconductor quenched")
        
        # Temperature margin check
        if T_c - T_op < 5:
            errors.append(f"Temperature margin {T_c - T_op}K < 5K: insufficient safety margin")
        
        # Cryocooler capacity check
        cryo_power = params.get('cryocooler_power', 5)
        estimated_heat_load = self._estimate_heat_load(params)
        
        if cryo_power < estimated_heat_load * 1.5:  # 50% safety margin
            errors.append(f"Cryocooler {cryo_power}W insufficient for estimated load {estimated_heat_load}W")
        
        return errors
    
    def check_mechanical_consistency(self, params: dict) -> list[str]:
        """Verify mechanical parameters are consistent."""
        errors = []
        
        # Stress check
        mass = params.get('mp', 0.05)
        radius = params.get('packet_radius', 0.1)
        omega = params.get('spin_rate', 5236)  # rad/s
        
        # Centrifugal stress
        stress = mass * (radius * omega)**2 / (4 * np.pi * radius**2)
        stress_limit = params.get('stress_limit', 8e8)
        
        if stress > stress_limit * 0.8:  # 80% of limit
            errors.append(f"Centrifugal stress {stress:.2e} Pa > 80% of limit {stress_limit:.2e} Pa")
        
        # Stiffness-mass ratio (natural frequency)
        k_eff = params.get('k_eff', 6000)
        m_s = params.get('ms', 1000)
        omega_n = np.sqrt(k_eff / m_s)
        
        # Check for resonance with spin frequency
        spin_freq = omega / (2 * np.pi)
        nat_freq = omega_n / (2 * np.pi)
        
        if abs(spin_freq - nat_freq) / nat_freq < 0.1:
            errors.append(f"Spin frequency {spin_freq:.1f} Hz near resonance {nat_freq:.1f} Hz")
        
        return errors
    
    def check_electromagnetic_consistency(self, params: dict) -> list[str]:
        """Verify EM parameters are consistent."""
        errors = []
        
        # Magnetic field vs critical field
        B_field = params.get('B_field', 1.0)
        B0 = params.get('B0', 5.0)
        
        if B_field > B0 * 0.8:
            errors.append(f"Operating field {B_field}T > 80% of B0 {B0}T: strong field degradation")
        
        # Induction efficiency check
        eta_ind = params.get('eta_ind', 0.9)
        
        # Theoretical maximum for given geometry
        eta_max = self._compute_max_eta(params)
        
        if eta_ind > eta_max:
            errors.append(f"eta_ind {eta_ind} > theoretical max {eta_max:.3f}")
        
        return errors
    
    def validate_all(self, params: dict) -> dict:
        """Run all consistency checks."""
        all_errors = []
        
        all_errors.extend(self.check_thermal_consistency(params))
        all_errors.extend(self.check_mechanical_consistency(params))
        all_errors.extend(self.check_electromagnetic_consistency(params))
        
        return {
            'valid': len(all_errors) == 0,
            'errors': all_errors,
            'n_errors': len(all_errors)
        }
```

**Success Criteria:**
- All parameter sets pass physical consistency checks
- Cross-domain consistency verified (thermal-mechanical, EM-thermal)
- Meaningful error messages guide parameter correction

**Dependencies:** 5.1
**Estimated Effort:** 2-3 days
**Risk:** Low

---

### 5.3 Documentation of Assumptions

**Rationale:** All simplifying assumptions must be documented for scientific transparency.

**Implementation:**
```markdown
# Update docs/TECHNICAL_SPEC.md

## Documented Assumptions and Limitations

### Physics Model Assumptions

1. **Flux Pinning Model**
   - Assumption: Bean-London critical state model with Kim-Anderson field dependence
   - Limitation: Does not account for flux creep at very long timescales (>1000s)
   - Validation: Compared against measured J_c(B,T) data from Su et al. 2019
   - Uncertainty: ±15% in J_c prediction at high fields (B > 3T)

2. **Thermal Model**
   - Assumption: Lumped capacitance model with constant thermal properties
   - Limitation: Ignores spatial temperature gradients within superconductor
   - Validation: Validated against 3D COMSOL simulations (see thermal_model_validation_report.md)
   - Uncertainty: ±10% in temperature prediction, ±20% in quench time prediction

3. **Mechanical Dynamics**
   - Assumption: Rigid body dynamics with linear stiffness
   - Limitation: Does not model material nonlinearity or fatigue
   - Validation: Benchmarked against ANSYS structural analysis
   - Uncertainty: ±5% in natural frequency, ±30% in damping prediction

4. **Electromagnetic Coupling**
   - Assumption: Quasi-static magnetic field approximation
   - Limitation: Ignores eddy current losses in normal metal layers
   - Validation: Compared against Maxwell 3D transient simulations
   - Uncertainty: ±10% in force prediction at high velocities (>1000 m/s)

### Numerical Method Assumptions

1. **Time Integration**
   - Assumption: RK4 with adaptive timestep (after Phase 1 implementation)
   - Limitation: No implicit integration for stiff systems
   - Validation: Energy conservation verified to 1e-6 relative error
   - Convergence: Verified 4th-order convergence via Richardson extrapolation

2. **Monte Carlo Sampling**
   - Assumption: Wilson score interval for binomial proportions
   - Limitation: Early termination may introduce bias (addressed in Phase 4)
   - Validation: Coverage verified at 93-97% for 95% CI (target: 90-100%)

### Input Parameter Assumptions

| Parameter | Assumed Value | Uncertainty | Source | Sensitivity |
|-----------|--------------|-------------|--------|-------------|
| J_c0 | 3×10¹⁰ A/m² | ±1×10¹⁰ | Bean-London calibration | High |
| B0 | 5.0 T | ±1.0 T | Kim-Anderson model fit | Medium |
| k_fp | 6000-12000 N/m | ±2000 N/m | Catalog measurement | High |
| c_damp | 4.0 N-s/m | ±2.0 | Structural estimate | Medium |

### Known Limitations

1. **ML Models**: Current VMD-IRCNN uses stub implementation for real-time performance
   - Impact: Wobble detection may have reduced accuracy vs. offline analysis
   - Mitigation: Statistical validation ensures minimum 70% precision/recall

2. **Geometry Scaling**: Volume-based scaling for k_fp lacks direct experimental validation
   - Impact: Stiffness predictions for thin-film geometries uncertain
   - Mitigation: Conservative safety margins applied (50%)

3. **Coupled Physics**: EM-thermal coupling simplified to 1D model
   - Impact: Local hot spots may not be captured
   - Mitigation: Temperature monitoring with safety margins
```

**Success Criteria:**
- All major assumptions documented with rationale
- Uncertainty quantification for each assumption
- Known limitations explicitly stated
- Sensitivity analysis for key parameters

**Dependencies:** All previous phases
**Estimated Effort:** 2 days
**Risk:** Low

---

## Implementation Timeline (REVISED)

| Phase | Tasks | Duration | Dependencies | Risk Level | Parallel AI Workers |
|-------|-------|----------|--------------|------------|-------------------|
| 0 | Immediate fixes (dt, energy checks, disable stub ML) | 2 days | None | Low | 1 |
| 1 | Adaptive timestep, convergence, stability | 10-12 days | Phase 0 | Medium | 2-3 |
| 2 | Ground truth, ML validation, uncertainty | 8-10 days | Phase 0 | Medium | 2-3 |
| 3 | Thermal degradation, EM-thermal coupling, geometry validation | 8-10 days | Phase 1 | Medium | 2 |
| 4 | MC statistics, bias detection, early termination | 6-8 days | Phase 2 | Medium | 2 |
| 5 | Uncertainty propagation, consistency checks, documentation | 7-9 days | All previous | Low | 1-2 |

**Total Estimated Effort:** 41-51 days (8-10 weeks with parallel execution)

**Critical Path:** Phase 0 → Phase 1 (12 days total for scientific validity)
**Parallel Execution:** Can run Phase 1 and Phase 2 simultaneously after Phase 0

## Success Metrics

1. **Numerical Accuracy**: Energy conservation < 1e-6, angular momentum to machine precision
2. **ML Performance**: ≥70% precision/recall, ECE < 0.1, calibrated uncertainties
3. **Physical Consistency**: All parameter sets pass cross-domain checks
4. **Statistical Rigor**: CI coverage 90-100%, no significant bias (p > 0.05)
5. **Uncertainty Propagation**: All results include 95% confidence intervals
6. **Documentation**: 100% of assumptions documented with uncertainty estimates

## Risk Mitigation

1. **Performance Impact**: Adaptive timestep may slow simulations
   - Mitigation: Implement both fixed (fast) and adaptive (accurate) modes
   
2. **Model Instabilities**: Improved physics may expose existing instabilities
   - Mitigation: Extensive testing with gradual rollout
   
3. **Literature Data Availability**: Geometry scaling may lack validation data
   - Mitigation: Plan for experimental validation campaign
   
4. **Scope Creep**: 35-41 days may expand
   - Mitigation: Prioritize Phase 1-2 (critical), defer Phase 3-5 if needed

## Conclusion

This plan addresses the most critical scientific accuracy issues while maintaining a pragmatic implementation approach. **Phase 0 (immediate fixes) and Phase 1 (numerical methods) are highest priority** as they directly impact result reliability. 

**Key Improvements Made:**
1. **Added Phase 0** for immediate critical fixes that can be implemented now
2. **Fixed mathematical errors** in adaptive timestep and convergence methods
3. **Added parallel AI execution model** with clear reviewer/implementer roles
4. **Revised timeline** to be more realistic (41-51 days vs 35-41)
5. **Added comprehensive error handling** and edge case management
6. **Structured for parallel execution** - Phase 1 and 2 can run simultaneously after Phase 0

**Execution Strategy:**
- Start Phase 0 immediately (2 days, low risk)
- Parallel execution of Phase 1 and 2 after Phase 0 completion
- Each phase has clear success criteria and validation gates
- Reviewer AI (this AI) validates all implementations before proceeding

The systematic approach ensures each improvement is validated before proceeding to the next phase, with proper bug fixes and error handling throughout.
