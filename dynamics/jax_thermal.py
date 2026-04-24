"""
JAX-based thermal models for packet stream.

Accelerated thermal prediction using JAX for JIT compilation.
"""

import logging

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None

logger = logging.getLogger(__name__)


if JAX_AVAILABLE:
    class JAXThermalModel:
        """
        JAX-accelerated thermal model for packet stream.

        Uses JIT compilation for fast thermal prediction across
        multiple packets.
        """

        def __init__(
            self,
            dt: float = 0.01,
            thermal_mass: float = 1000.0,  # J/K
            heat_capacity: float = 500.0,  # J/(kg·K)
            convection_coeff: float = 10.0,  # W/(m²·K)
            surface_area: float = 0.1,  # m²
        ):
            """
            Initialize JAX thermal model.

            Args:
                dt: Time step (s)
                thermal_mass: Thermal mass (J/K)
                heat_capacity: Specific heat capacity (J/(kg·K))
                convection_coeff: Convection coefficient (W/(m²·K))
                surface_area: Surface area (m²)
            """
            if not JAX_AVAILABLE:
                raise ImportError(
                    "JAX is required for thermal models. "
                    "Install with: poetry install --extras jax"
                )

            self.dt = dt
            self.thermal_mass = thermal_mass
            self.heat_capacity = heat_capacity
            self.convection_coeff = convection_coeff
            self.surface_area = surface_area

            # Compile thermal update function
            self._thermal_update_jit = jax.jit(self._thermal_update)
            # Pre-compile vmap for batch prediction to avoid re-compilation in loop
            self._thermal_update_vmap = jax.jit(jax.vmap(self._thermal_update))

            logger.info("JAX thermal model initialized with JIT compilation")

        def _thermal_update(
            self,
            T: jnp.ndarray,  # noqa: N803
            Q_in: jnp.ndarray,  # noqa: N803
            T_amb: float,  # noqa: N803
        ) -> jnp.ndarray:
            """
            Thermal update step (JAX-compatible).

            Args:
                T: Current temperatures [N]
                Q_in: Heat input rates [N]
                T_amb: Ambient temperature

            Returns:
                Updated temperatures [N]
            """
            # Convection heat loss
            Q_conv = self.convection_coeff * self.surface_area * (T - T_amb)  # noqa: N806

            # Temperature update
            dT = (Q_in - Q_conv) / self.thermal_mass * self.dt  # noqa: N806
            T_new = T + dT  # noqa: N806

            return T_new  # noqa: N806

        def predict_temperatures(
            self,
            T_initial: np.ndarray,  # noqa: N803
            Q_in: np.ndarray,  # noqa: N803
            T_amb: float = 293.15,  # noqa: N803
            n_steps: int = 100,
        ) -> tuple[np.ndarray, dict]:
            """
            Predict temperatures over time horizon.

            Args:
                T_initial: Initial temperatures [N]
                Q_in: Heat input rates over time [n_steps, N]
                T_amb: Ambient temperature (K)
                n_steps: Number of prediction steps

            Returns:
                (temperatures, metadata) where temperatures is [n_steps+1, N]
            """
            # Convert to JAX arrays
            T = jnp.array(T_initial)  # noqa: N806
            Q_jax = jnp.array(Q_in)  # noqa: N806

            # Time evolution
            temperatures = [T]
            for i in range(n_steps):
                T = self._thermal_update_jit(T, Q_jax[i], T_amb)  # noqa: N806
                temperatures.append(T)  # noqa: N806

            # Convert back to numpy
            temperatures = np.array([np.array(t) for t in temperatures])

            metadata = {
                'n_packets': len(T_initial),
                'n_steps': n_steps,
                'dt': self.dt,
                'max_temp': np.max(temperatures),
                'min_temp': np.min(temperatures),
            }

            return temperatures, metadata

        def batch_predict(
            self,
            T_initial: np.ndarray,  # noqa: N803
            Q_in: np.ndarray,  # noqa: N803
            T_amb: float = 293.15,  # noqa: N803
        ) -> np.ndarray:
            """
            Batch prediction for multiple scenarios.

            Args:
                T_initial: Initial temperatures [batch, N]
                Q_in: Heat input rates [batch, n_steps, N]
                T_amb: Ambient temperature

            Returns:
                Predicted temperatures [batch, n_steps+1, N]
            """
            # Validate input shapes
            if T_initial.ndim != 2:
                raise ValueError(f"T_initial must be 2D [batch, N], got shape {T_initial.shape}")
            if Q_in.ndim != 3:
                raise ValueError(f"Q_in must be 3D [batch, n_steps, N], got shape {Q_in.shape}")
            if T_initial.shape[0] != Q_in.shape[0]:
                raise ValueError(f"Batch size mismatch: T_initial {T_initial.shape[0]} vs Q_in {Q_in.shape[0]}")
            if T_initial.shape[1] != Q_in.shape[2]:
                raise ValueError(f"Packet count mismatch: T_initial {T_initial.shape[1]} vs Q_in {Q_in.shape[2]}")

            # Vectorized prediction using JAX
            T = jnp.array(T_initial)  # noqa: N806
            Q_jax = jnp.array(Q_in)  # noqa: N806

            def scan_fn(carry, Q_step):  # noqa: N803
                T_current = carry  # noqa: N806
                T_new = self._thermal_update_vmap(T_current, Q_step, T_amb)  # noqa: N806
                return T_new, T_new  # noqa: N806

            _, temperatures = jax.lax.scan(scan_fn, T, Q_jax.T)

            # Add initial temperature
            temperatures = jnp.concatenate([T[None], temperatures], axis=0)

            return np.array(temperatures)

        def get_model_info(self) -> dict:
            """Get model metadata."""
            return {
                'dt': self.dt,
                'thermal_mass': self.thermal_mass,
                'heat_capacity': self.heat_capacity,
                'convection_coeff': self.convection_coeff,
                'surface_area': self.surface_area,
                'jit_compiled': True,
            }
