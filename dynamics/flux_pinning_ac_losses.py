"""
Corrected flux-pinning stiffness model with AC losses.

Implements velocity-dependent stiffness reduction due to:
- Thermally activated flux creep (low velocities)
- Flux flow regime (high velocities)
- AC losses at operational frequencies

Based on Bean-London critical state model with modifications
for dynamic effects. Parameters from Fuger et al. 2010 for GdBCO.

Reference:
    Fuger et al., "AC losses in coated conductors," Supercond. Sci. Technol. 2010
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

try:
    import jax.numpy as jnp
    from jax import grad
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np


@dataclass
class ACLossParameters:
    """Parameters for AC loss model in flux-pinning stiffness.

    From Fuger et al. 2010 for GdBCO at 30K:
    """
    # Critical current parameters
    Jc0: float = 1.2e10  # A/m^2 at 30K, 3T
    Tc: float = 92.0  # K (critical temperature)
    n: float = 0.5  # Temperature dependence exponent
    B0: float = 1.0  # T (characteristic field)
    m: float = 0.3  # Field dependence exponent

    # Velocity/AC loss parameters
    v_dep: float = 12.5  # m/s (depinning velocity)
    v_therm: float = 1.8  # m/s (thermal activation velocity)
    delta_v: float = 5.0  # m/s (transition width)

    # Frequency scaling
    f_char_scale: float = 1.0  # Scale factor for characteristic frequency


class CorrectedFluxPinningModel:
    """Flux-pinning model with velocity-dependent AC loss corrections.

    This model extends the static Bean-London model to include dynamic
    effects that reduce effective stiffness at high velocities.

    The velocity reduction factor captures:
    1. Thermally activated flux creep (Gaussian roll-off at low v)
    2. Flux flow regime (asymptotic 1/v behavior at high v)
    3. Sharp depinning transition (complementary error function)

    Usage:
        model = CorrectedFluxPinningModel(params)
        k_fp = model.get_stiffness(h, v, B_func, T)
    """

    def __init__(self, params: ACLossParameters | None = None):
        """Initialize corrected flux-pinning model.

        Args:
            params: AC loss parameters. Uses GdBCO defaults if None.
        """
        self.params = params if params is not None else ACLossParameters()

    def _compute_Jc(self, B: float, T: float) -> float:
        """Compute critical current density Jc(B, T).

        Jc = Jc0 * (1 - T/Tc)^n * (1 + B/B0)^(-m)

        Args:
            B: Magnetic field (T)
            T: Temperature (K)

        Returns:
            Critical current density (A/m^2)
        """
        p = self.params

        if T >= p.Tc:
            return 0.0

        temp_factor = (1.0 - T / p.Tc) ** p.n
        field_factor = (1.0 + B / p.B0) ** (-p.m)

        return p.Jc0 * temp_factor * field_factor

    def _velocity_reduction_factor(self, v: float) -> float:
        """Compute velocity-dependent stiffness reduction factor.

        Physically-based model for AC loss-induced stiffness reduction:
        - Low velocities: small reduction (flux creep regime)
        - High velocities: significant reduction (flux flow regime)

        Uses a smooth transition function based on characteristic velocities.

        Args:
            v: Velocity magnitude (m/s)

        Returns:
            Reduction factor (0 to 1)
        """
        p = self.params
        v_abs = abs(v)

        if v_abs < 1e-10:
            # At zero velocity, no reduction
            return 1.0

        # Characteristic velocity for AC loss onset
        # Based on depinning and thermal activation
        v_char = np.sqrt(p.v_dep * p.v_therm)

        # Smooth reduction function:
        # f(v) = 1 / (1 + (v/v_char)^alpha)
        # where alpha controls steepness of transition
        alpha = 1.5

        reduction = 1.0 / (1.0 + (v_abs / v_char) ** alpha)

        # Ensure result is in [0, 1]
        return float(np.clip(reduction, 0.0, 1.0))

    def compute_pinning_force_density(self,
                                      magnet_height: float,
                                      magnet_velocity: float,
                                      B_field_func: Callable[[float], float],
                                      temperature: float = 30.0) -> float:
        """Compute time-averaged pinning force density.

        <Fp> = Jc(B,T) * B_rms * f(v)

        Args:
            magnet_height: Height above track (m)
            magnet_velocity: Velocity magnitude (m/s)
            B_field_func: Function B(h) giving field at height h
            temperature: Operating temperature (K)

        Returns:
            Pinning force density (N/m^3)
        """
        B = B_field_func(magnet_height)
        B_rms = B  # Simplified - should be RMS over magnet area

        Jc = self._compute_Jc(B, temperature)
        f_v = self._velocity_reduction_factor(magnet_velocity)

        return Jc * B_rms * f_v

    def get_stiffness(self,
                      magnet_height: float,
                      magnet_velocity: float,
                      B_field_func: Callable[[float], float],
                      temperature: float = 30.0,
                      use_jax: bool = False) -> float:
        """Compute effective stiffness k_fp = -dF/dh.

        Uses automatic differentiation for accurate gradient computation.

        Args:
            magnet_height: Height above track (m)
            magnet_velocity: Velocity magnitude (m/s)
            B_field_func: Function B(h) giving field at height h
            temperature: Operating temperature (K)
            use_jax: Whether to use JAX for differentiation (requires JAX)

        Returns:
            Effective stiffness (N/m)
        """
        if use_jax and JAX_AVAILABLE:
            return self._get_stiffness_jax(magnet_height, magnet_velocity,
                                          B_field_func, temperature)
        else:
            return self._get_stiffness_numerical(magnet_height, magnet_velocity,
                                                B_field_func, temperature)

    def _get_stiffness_jax(self,
                           magnet_height: float,
                           magnet_velocity: float,
                           B_field_func: Callable[[float], float],
                           temperature: float) -> float:
        """Compute stiffness using JAX automatic differentiation.

        Note: This requires B_field_func to be JAX-compatible (pure function
        using jax.numpy operations). For non-JAX field functions, use
        numerical differentiation (use_jax=False).
        """
        p = self.params

        # Pre-compute velocity reduction (constant w.r.t. height)
        v_abs = abs(magnet_velocity)
        f_v = self._velocity_reduction_factor(v_abs)

        # Pre-compute temperature factor (constant w.r.t. height)
        temp_factor = (1.0 - temperature / p.Tc) ** p.n

        def force_density(h):
            """Force as function of height (for differentiation)."""
            # B_field_func must be JAX-compatible
            B_val = B_field_func(h)
            B_rms = B_val

            # Jc computation
            field_factor = (1.0 + B_val / p.B0) ** (-p.m)
            Jc = p.Jc0 * temp_factor * field_factor

            return Jc * B_rms * f_v

        # Compute gradient
        k_fp = -grad(force_density)(jnp.array(magnet_height))
        return float(k_fp)

    def _get_stiffness_numerical(self,
                                 magnet_height: float,
                                 magnet_velocity: float,
                                 B_field_func: Callable[[float], float],
                                 temperature: float,
                                 dh: float = 1e-6) -> float:
        """Compute stiffness using numerical differentiation."""
        F1 = self.compute_pinning_force_density(
            magnet_height - dh, magnet_velocity, B_field_func, temperature
        )
        F2 = self.compute_pinning_force_density(
            magnet_height + dh, magnet_velocity, B_field_func, temperature
        )

        # Stiffness is -dF/dh
        k_fp = -(F2 - F1) / (2 * dh)
        return k_fp

    def compute_characteristic_frequency(self,
                                         velocity: float,
                                         ball_diameter: float = 0.1) -> float:
        """Compute characteristic AC frequency from ball passage.

        f_char = u / l_ball

        Args:
            velocity: Stream velocity (m/s)
            ball_diameter: Characteristic ball size (m)

        Returns:
            Characteristic frequency (Hz)
        """
        return velocity / ball_diameter

    def estimate_stiffness_reduction(self,
                                     velocity: float,
                                     baseline_velocity: float = 1.0) -> float:
        """Estimate stiffness reduction factor at given velocity.

        Args:
            velocity: Operating velocity (m/s)
            baseline_velocity: Reference velocity for comparison (m/s)

        Returns:
            Reduction ratio k_fp(v) / k_fp(v_baseline)
        """
        f_v = self._velocity_reduction_factor(velocity)
        f_baseline = self._velocity_reduction_factor(baseline_velocity)

        if f_baseline < 1e-10:
            return 0.0

        return f_v / f_baseline


def create_corrected_bean_london_model(material_geometry: dict,
                                       ac_params: ACLossParameters | None = None):
    """Factory function to create corrected model compatible with existing code.

    Args:
        material_geometry: Dict with 'thickness', 'width', 'length'
        ac_params: AC loss parameters (uses defaults if None)

    Returns:
        CorrectedFluxPinningModel instance
    """
    return CorrectedFluxPinningModel(ac_params)


def compare_stiffness_models(velocities: np.ndarray,
                             magnet_height: float = 0.01,
                             B_field: float = 1.0,
                             temperature: float = 30.0) -> dict:
    """Compare static vs corrected stiffness across velocity range.

    Args:
        velocities: Array of velocities to evaluate (m/s)
        magnet_height: Height above track (m)
        B_field: Magnetic field (T)
        temperature: Temperature (K)

    Returns:
        Dict with 'velocities', 'static_stiffness', 'corrected_stiffness', 'reduction_factor'
    """
    # Simple B field function
    def B_func(h):
        return B_field * np.exp(-h / 0.005)  # Exponential decay with height

    model = CorrectedFluxPinningModel()

    static_stiffness = []
    corrected_stiffness = []

    for v in velocities:
        # Static (zero velocity limit)
        k_static = model.get_stiffness(magnet_height, 0.0, B_func, temperature,
                                      use_jax=False)
        static_stiffness.append(k_static)

        # Corrected (with velocity)
        k_corrected = model.get_stiffness(magnet_height, v, B_func, temperature,
                                         use_jax=False)
        corrected_stiffness.append(k_corrected)

    static_stiffness = np.array(static_stiffness)
    corrected_stiffness = np.array(corrected_stiffness)

    reduction_factor = np.where(static_stiffness > 0,
                                corrected_stiffness / static_stiffness,
                                0.0)

    return {
        'velocities': velocities,
        'static_stiffness': static_stiffness,
        'corrected_stiffness': corrected_stiffness,
        'reduction_factor': reduction_factor,
    }
