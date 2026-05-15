"""
Structure-preserving integrators for long-term stability.

Replaces explicit Euler with:
- Symplectic integrators for conservative Hamiltonian systems
- Symplectic-Euler for dissipative systems
- Adaptive RK for non-conservative orbits
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable, Union
import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class IntegrationResult:
    """Result of an integration step."""
    t_new: float
    y_new: np.ndarray
    error_estimate: float
    step_accepted: bool
    n_evals: int
    suggested_dt: Optional[float] = None


class Integrator(ABC):
    """Abstract base for time integrators."""
    
    @abstractmethod
    def step(
        self,
        f: Callable[[float, np.ndarray], np.ndarray],
        t: float,
        y: np.ndarray,
        dt: float
    ) -> IntegrationResult:
        """
        Take one integration step.
        
        Args:
            f: ODE function dy/dt = f(t, y)
            t: Current time
            y: Current state
            dt: Step size
        
        Returns:
            IntegrationResult
        """
        pass
    
    @abstractmethod
    def is_symplectic(self) -> bool:
        """Returns True if integrator preserves Hamiltonian structure."""
        pass


# =============================================================================
# SYMPLECTIC INTEGRATORS (for conservative mechanics)
# =============================================================================

class VelocityVerlet(Integrator):
    """
    Velocity Verlet integrator for separable Hamiltonian systems.
    
    For H = T(p) + V(q), the system is:
        dq/dt = ∂H/∂p = p/m
        dp/dt = -∂H/∂q = F(q)
    
    Algorithm:
        v_{n+1/2} = v_n + (dt/2) * F(q_n) / m
        q_{n+1} = q_n + dt * v_{n+1/2}
        v_{n+1} = v_{n+1/2} + (dt/2) * F(q_{n+1}) / m
    
    Properties:
    - Second-order accurate
    - Symplectic (preserves phase space volume)
    - Time-reversible
    - Good energy conservation over long times
    """
    
    def __init__(self, mass: Union[float, np.ndarray] = 1.0):
        self.mass = np.asarray(mass)
    
    def step(
        self,
        f: Callable[[float, np.ndarray], np.ndarray],
        t: float,
        y: np.ndarray,
        dt: float
    ) -> IntegrationResult:
        """
        Take one Velocity Verlet step.
        
        For this integrator, f should return forces (not full derivative).
        State y = [position, velocity] concatenated.
        """
        n = len(y) // 2
        q = y[:n].copy()
        v = y[n:].copy()
        
        # Get force at current position
        # f returns full state derivative, extract force part
        dydt = f(t, y)
        a = dydt[n:]  # acceleration (force/mass)
        
        # Half-step velocity
        v_half = v + 0.5 * dt * a
        
        # Full-step position
        q_new = q + dt * v_half
        
        # Construct new state for force evaluation
        y_new_partial = np.concatenate([q_new, v_half])
        
        # Get force at new position
        dydt_new = f(t + dt, y_new_partial)
        a_new = dydt_new[n:]
        
        # Half-step velocity to complete
        v_new = v_half + 0.5 * dt * a_new
        
        y_new = np.concatenate([q_new, v_new])
        
        # Error estimate (simplified - compare to one Euler step)
        y_euler = y + dt * dydt
        error = np.linalg.norm(y_new - y_euler)
        
        return IntegrationResult(
            t_new=t + dt,
            y_new=y_new,
            error_estimate=error,
            step_accepted=True,
            n_evals=2,
            suggested_dt=None
        )
    
    def is_symplectic(self) -> bool:
        return True


class StormerVerlet(Integrator):
    """
    Störmer-Verlet (leapfrog) for second-order ODEs.
    
    For q'' = F(q), equivalent to Velocity Verlet but often more stable
    for position-dependent forces only.
    
    Algorithm:
        q_{n+1} = 2*q_n - q_{n-1} + dt² * F(q_n)
    
    Or in one-step form:
        p_{n+1/2} = p_n + (dt/2) * F(q_n)
        q_{n+1} = q_n + dt * p_{n+1/2}
        p_{n+1} = p_{n+1/2} + (dt/2) * F(q_{n+1})
    """
    
    def __init__(self):
        self._prev_q = None
        self._prev_t = None
    
    def step(
        self,
        f: Callable[[float, np.ndarray], np.ndarray],
        t: float,
        y: np.ndarray,
        dt: float
    ) -> IntegrationResult:
        """
        Take one Stormer-Verlet step.
        
        Same interface as Velocity Verlet - state y = [q, v].
        """
        # Implementation is identical to Velocity Verlet for separable Hamiltonians
        # The difference is in how forces are computed (F(q) vs F(q,v,t))
        n = len(y) // 2
        q = y[:n].copy()
        v = y[n:].copy()
        
        dydt = f(t, y)
        a = dydt[n:]
        
        v_half = v + 0.5 * dt * a
        q_new = q + dt * v_half
        
        y_new_partial = np.concatenate([q_new, v_half])
        dydt_new = f(t + dt, y_new_partial)
        a_new = dydt_new[n:]
        
        v_new = v_half + 0.5 * dt * a_new
        y_new = np.concatenate([q_new, v_new])
        
        error = np.linalg.norm(dt**2 * a)  # O(dt²) local truncation error
        
        return IntegrationResult(
            t_new=t + dt,
            y_new=y_new,
            error_estimate=error,
            step_accepted=True,
            n_evals=2,
            suggested_dt=None
        )
    
    def is_symplectic(self) -> bool:
        return True


class SymplecticEuler(Integrator):
    """
    Symplectic Euler (semi-implicit Euler) for dissipative systems.
    
    Algorithm:
        p_{n+1} = p_n + dt * F(q_n)
        q_{n+1} = q_n + dt * p_{n+1}
    
    Note: This is first-order, but symplectic for separable Hamiltonians.
    For dissipative systems (damping), use this instead of Velocity Verlet.
    """
    
    def step(
        self,
        f: Callable[[float, np.ndarray], np.ndarray],
        t: float,
        y: np.ndarray,
        dt: float
    ) -> IntegrationResult:
        """Take one symplectic Euler step."""
        n = len(y) // 2
        q = y[:n].copy()
        v = y[n:].copy()
        
        # Get acceleration (may depend on velocity for damping)
        dydt = f(t, y)
        a = dydt[n:]
        
        # Update velocity first
        v_new = v + dt * a
        
        # Update position with new velocity
        q_new = q + dt * v_new
        
        y_new = np.concatenate([q_new, v_new])
        
        # First-order error
        error = np.linalg.norm(dt * dydt)
        
        return IntegrationResult(
            t_new=t + dt,
            y_new=y_new,
            error_estimate=error,
            step_accepted=True,
            n_evals=1,
            suggested_dt=None
        )
    
    def is_symplectic(self) -> bool:
        return True  # For separable Hamiltonians


# =============================================================================
# RUNGE-KUTTA INTEGRATORS (for general ODEs)
# =============================================================================

class RK4(Integrator):
    """
    Classical 4th-order Runge-Kutta.
    
    NOT symplectic - use for non-conservative systems or when
    high accuracy on short timescales is needed.
    """
    
    def step(
        self,
        f: Callable[[float, np.ndarray], np.ndarray],
        t: float,
        y: np.ndarray,
        dt: float
    ) -> IntegrationResult:
        """Take one RK4 step."""
        k1 = f(t, y)
        k2 = f(t + dt/2, y + dt*k1/2)
        k3 = f(t + dt/2, y + dt*k2/2)
        k4 = f(t + dt, y + dt*k3)
        
        y_new = y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Error estimate: compare to embedded 3rd-order method
        y_embedded = y + (dt/6) * (k1 + 4*k2 + k3)  # Simplified
        error = np.linalg.norm(y_new - y_embedded)
        
        return IntegrationResult(
            t_new=t + dt,
            y_new=y_new,
            error_estimate=error,
            step_accepted=True,
            n_evals=4,
            suggested_dt=None
        )
    
    def is_symplectic(self) -> bool:
        return False


class AdaptiveRK45(Integrator):
    """
    Adaptive Dormand-Prince (RK45) via scipy.integrate.solve_ivp wrapper.
    
    For problems where you need adaptive stepping but don't require
    symplectic properties (orbital perturbations, thermal, etc.)
    """
    
    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-9,
        max_step: Optional[float] = None
    ):
        self.rtol = rtol
        self.atol = atol
        self.max_step = max_step
    
    def step(
        self,
        f: Callable[[float, np.ndarray], np.ndarray],
        t: float,
        y: np.ndarray,
        dt: float
    ) -> IntegrationResult:
        """
        Take one adaptive RK45 step.
        
        Note: This uses solve_ivp internally, so it's not efficient for
        single steps. Use for testing or when adaptive stepping is essential.
        """
        # Use solve_ivp for one step
        t_span = [t, t + dt]
        
        sol = solve_ivp(
            f, t_span, y,
            method='RK45',
            rtol=self.rtol,
            atol=self.atol,
            max_step=min(dt, self.max_step) if self.max_step else dt,
            dense_output=False
        )
        
        if not sol.success:
            return IntegrationResult(
                t_new=t,
                y_new=y,
                error_estimate=np.inf,
                step_accepted=False,
                n_evals=sol.nfev,
                suggested_dt=dt/2
            )
        
        y_new = sol.y[:, -1]
        
        # Estimate error from solver internals
        error = self.rtol * np.linalg.norm(y_new) + self.atol
        
        return IntegrationResult(
            t_new=sol.t[-1],
            y_new=y_new,
            error_estimate=error,
            step_accepted=True,
            n_evals=sol.nfev,
            suggested_dt=dt if sol.t[-1] - t > 0.9*dt else dt*1.5
        )
    
    def is_symplectic(self) -> bool:
        return False


# =============================================================================
# INTEGRATOR SELECTION HELPER
# =============================================================================

def select_integrator(
    system_type: str,
    timescale: float,
    accuracy_required: str = "medium"
) -> Integrator:
    """
    Select appropriate integrator for a physics domain.
    
    Args:
        system_type: One of:
            - "conservative_mechanics": Position-dependent forces only
            - "dissipative_mechanics": Velocity-dependent forces (damping)
            - "general_ode": Non-mechanical system (thermal, etc.)
            - "orbital": Orbital mechanics (use specialized orbit integrator)
        timescale: Characteristic timescale of system (seconds)
        accuracy_required: "low", "medium", "high"
    
    Returns:
        Appropriate Integrator instance
    """
    if system_type == "conservative_mechanics":
        # For stream mechanics, interball forces (conservative)
        return VelocityVerlet()
    
    elif system_type == "dissipative_mechanics":
        # For attitude with damping, flux-pinning with hysteresis
        return SymplecticEuler()
    
    elif system_type == "general_ode":
        # For thermal, cryocooler, etc.
        if accuracy_required == "high":
            return AdaptiveRK45(rtol=1e-8, atol=1e-10)
        else:
            return RK4()
    
    elif system_type == "orbital":
        # Use specialized orbital propagator
        # For now, adaptive RK
        return AdaptiveRK45(rtol=1e-10, atol=1e-12)
    
    else:
        raise ValueError(f"Unknown system_type: {system_type}")


# =============================================================================
# ADAPTIVE TIMESTEP CALCULATOR (CRITICAL for 50k RPM stability)
# =============================================================================

def calculate_adaptive_timestep(
    spin_rate_rad_s: float,
    dt_max_ratio: float = 0.1,
    dt_min: float = 1e-6,
    dt_max: float = 0.001
) -> float:
    """
    Calculate adaptive timestep based on spin rate for gyroscopic stability.

    CRITICAL: At 50k RPM (5236 rad/s), the spin period is ~1.2ms.
    Numerical stability requires dt << T_spin.

    Formula: dt = min(dt_max, max(dt_min, ratio * T_spin))
    where T_spin = 2*pi / spin_rate

    Args:
        spin_rate_rad_s: Angular velocity in rad/s
        dt_max_ratio: Max timestep as fraction of spin period (default 0.1)
        dt_min: Minimum timestep to prevent excessive computation
        dt_max: Maximum timestep (hard limit for stability)

    Returns:
        Recommended timestep in seconds

    Example:
        >>> calculate_adaptive_timestep(5236)  # 50k RPM
        0.0001  # 0.1 ms = T_spin / 12
    """
    if spin_rate_rad_s <= 0:
        return dt_max

    T_spin = 2 * np.pi / spin_rate_rad_s
    dt_recommended = dt_max_ratio * T_spin

    # Clamp to bounds
    dt = max(dt_min, min(dt_max, dt_recommended))

    return float(dt)


def validate_timestep_for_spin(
    dt: float,
    spin_rate_rad_s: float,
    max_ratio: float = 0.1
) -> dict:
    """
    Validate if timestep is appropriate for given spin rate.

    Args:
        dt: Timestep to validate
        spin_rate_rad_s: Angular velocity in rad/s
        max_ratio: Maximum allowed ratio of dt to spin period

    Returns:
        Dictionary with validation results
    """
    if spin_rate_rad_s <= 0:
        return {"valid": True, "ratio": 0, "warning": None}

    T_spin = 2 * np.pi / spin_rate_rad_s
    ratio = dt / T_spin

    result = {
        "valid": ratio <= max_ratio,
        "ratio": ratio,
        "T_spin": T_spin,
        "dt": dt,
        "max_ratio": max_ratio,
    }

    if ratio > 1.0:
        result["warning"] = f"CRITICAL: dt={dt}s spans {ratio:.1f} rotations! Numerical instability guaranteed."
    elif ratio > max_ratio:
        result["warning"] = f"WARNING: dt={dt}s exceeds recommended max ({max_ratio} * T_spin). Energy drift likely."
    else:
        result["warning"] = None

    return result


# =============================================================================
# ENERGY/MOMENTUM MONITORING
# =============================================================================

@dataclass
class ConservationMonitor:
    """
    Monitor energy and momentum conservation during integration.
    
    For symplectic integrators, energy should oscillate but not drift.
    For non-symplectic, energy will drift - this quantifies the drift.
    """
    
    energy_func: Optional[Callable[[np.ndarray], float]] = None
    momentum_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
    
    energy_history: list[tuple[float, float]] = None
    momentum_history: list[tuple[float, np.ndarray]] = None
    
    def __post_init__(self):
        if self.energy_history is None:
            self.energy_history = []
        if self.momentum_history is None:
            self.momentum_history = []
    
    def record(self, t: float, y: np.ndarray):
        """Record energy and momentum at time t."""
        if self.energy_func:
            E = self.energy_func(y)
            self.energy_history.append((t, E))
        
        if self.momentum_func:
            p = self.momentum_func(y)
            self.momentum_history.append((t, p))
    
    def check_conservation(self) -> dict:
        """
        Check if energy/momentum are conserved.
        
        Returns dictionary with statistics.
        """
        results = {}
        
        if len(self.energy_history) > 1:
            energies = [e for _, e in self.energy_history]
            E0 = energies[0]
            dE = np.array(energies) - E0
            
            results['energy_drift_relative'] = np.abs(dE[-1]) / abs(E0) if E0 != 0 else 0
            results['energy_drift_absolute'] = np.abs(dE[-1])
            results['energy_oscillation'] = np.std(dE)
            results['energy_conserved'] = results['energy_drift_relative'] < 0.01  # 1% threshold
        
        if len(self.momentum_history) > 1:
            momenta = np.array([p for _, p in self.momentum_history])
            p0 = momenta[0]
            dp = momenta - p0
            
            results['momentum_drift'] = np.linalg.norm(dp[-1])
            results['momentum_conserved'] = results['momentum_drift'] < 0.01 * np.linalg.norm(p0)
        
        return results
    
    def plot_history(self):
        """Plot conservation history (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            if self.energy_history:
                times, energies = zip(*self.energy_history)
                E0 = energies[0]
                dE = np.array(energies) - E0
                
                axes[0].plot(times, dE)
                axes[0].set_xlabel('Time (s)')
                axes[0].set_ylabel('ΔE (J)')
                axes[0].set_title('Energy Conservation')
                axes[0].grid(True)
            
            if self.momentum_history:
                times, momenta = zip(*self.momentum_history)
                p0 = momenta[0]
                dp = [np.linalg.norm(p - p0) for p in momenta]
                
                axes[1].plot(times, dp)
                axes[1].set_xlabel('Time (s)')
                axes[1].set_ylabel('|Δp| (kg·m/s)')
                axes[1].set_title('Momentum Conservation')
                axes[1].grid(True)
            
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("matplotlib not available for plotting")


# =============================================================================
# TESTING
# =============================================================================

def test_integrators():
    """Test integrators on simple harmonic oscillator."""
    
    # SHO: x'' = -omega² * x
    omega = 1.0
    
    def f_sho(t, y):
        """State: [x, v], return [v, a]."""
        x, v = y
        a = -omega**2 * x
        return np.array([v, a])
    
    def energy(y):
        x, v = y
        return 0.5 * v**2 + 0.5 * omega**2 * x**2
    
    # Initial conditions: x=1, v=0
    y0 = np.array([1.0, 0.0])
    t0 = 0.0
    dt = 0.01
    n_steps = 1000
    
    print("Testing integrators on harmonic oscillator")
    print("=" * 60)
    
    integrators = {
        'VelocityVerlet': VelocityVerlet(),
        'StormerVerlet': StormerVerlet(),
        'SymplecticEuler': SymplecticEuler(),
        'RK4': RK4(),
    }
    
    for name, integrator in integrators.items():
        y = y0.copy()
        t = t0
        
        monitor = ConservationMonitor(energy_func=energy)
        monitor.record(t, y)
        
        for _ in range(n_steps):
            result = integrator.step(f_sho, t, y, dt)
            y = result.y_new
            t = result.t_new
            monitor.record(t, y)
        
        checks = monitor.check_conservation()
        
        print(f"\n{name}:")
        print(f"  Symplectic: {integrator.is_symplectic()}")
        print(f"  Energy drift (relative): {checks.get('energy_drift_relative', 'N/A'):.6f}")
        print(f"  Energy oscillation (std): {checks.get('energy_oscillation', 'N/A'):.6f}")
        print(f"  Final x: {y[0]:.6f} (expected ~1.0)")


if __name__ == "__main__":
    test_integrators()
