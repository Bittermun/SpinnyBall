"""
True Variational Mode Decomposition (VMD) implementation.

Implements the VMD algorithm from Dragomiretskiy & Zosso (2014)
for adaptive signal decomposition into variational modes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import scipy.fft

logger = logging.getLogger(__name__)


@dataclass
class VMDParameters:
    """VMD hyperparameters."""
    num_modes: int = 4
    alpha: float = 2000.0  # Bandwidth constraint
    tau: float = 0.0  # Time-step of dual ascent
    K: int = 4  # Number of DC components
    DC: bool = False  # Include DC component
    init: int = 1  # Initialization (1: uniform, 2: random, 3: noise)
    tol: float = 1e-7  # Convergence tolerance
    max_iter: int = 100  # Maximum iterations


class VMDDecomposer:
    """
    True VMD implementation using variational optimization.

    Decomposes signals into K variational modes using the
    ADMM (Alternating Direction Method of Multipliers) algorithm.
    """

    def __init__(self, params: VMDParameters | None = None):
        """
        Initialize VMD decomposer.

        Args:
            params: VMD hyperparameters. If None, uses defaults.
        """
        self.params = params or VMDParameters()
        
        # Precompute frequency grid (will be set on first decompose call)
        self.n_samples = 0
        self.freqs = None
        self.t = None
        
        logger.info(f"VMD decomposer initialized with {self.params.num_modes} modes")

    def decompose(self, signal: np.ndarray) -> np.ndarray:
        """
        Perform variational mode decomposition.

        Algorithm:
        1. Initialize center frequencies {u_k} and modes {u_k}
        2. Update modes: u_k = argmin L (ADMM optimization)
        3. Update center frequencies: ω_k = argmin L
        4. Update Lagrange multipliers: λ = λ + τ (f - Σ u_k)
        5. Check convergence: |u_k^n+1 - u_k^n| / |u_k^n| < tol
        6. Repeat until convergence or max iterations

        Args:
            signal: Input signal [N]

        Returns:
            modes: [num_modes × N] decomposed modes
        """
        n = len(signal)
        
        # Initialize frequency grid
        if self.n_samples != n:
            self.n_samples = n
            self.freqs = scipy.fft.fftfreq(n)
            self.t = np.arange(n) / n
        
        # Initialize modes and center frequencies
        modes = np.zeros((self.params.num_modes, n))
        omega = self._initialize_omega()
        
        # Initialize Lagrange multipliers
        lambda_hat = np.zeros(n, dtype=complex)
        
        # Precompute signal FFT
        signal_hat = scipy.fft.fft(signal)
        
        # ADMM optimization loop
        modes_prev = modes.copy()
        for iteration in range(self.params.max_iter):
            # Update each mode
            for k in range(self.params.num_modes):
                modes[k] = self._update_mode(signal_hat, modes, omega, k, lambda_hat)
            
            # Update center frequencies
            for k in range(self.params.num_modes):
                omega[k] = self._update_frequency(modes[k])
            
            # Update Lagrange multipliers
            lambda_hat = lambda_hat + self.params.tau * (signal_hat - np.sum(scipy.fft.fft(modes, axis=1), axis=0))
            
            # Check convergence
            if self._check_convergence(modes, modes_prev):
                logger.info(f"VMD converged at iteration {iteration}")
                break
            
            modes_prev = modes.copy()
        else:
            logger.warning(f"VMD did not converge after {self.params.max_iter} iterations")
        
        # Validate energy conservation
        energy_original = np.sum(signal ** 2)
        energy_reconstructed = np.sum(np.sum(modes, axis=0) ** 2)
        energy_error = abs(energy_original - energy_reconstructed) / energy_original
        
        if energy_error > 0.05:
            logger.warning(f"VMD energy conservation error: {energy_error:.4f} (> 5%)")
        else:
            logger.info(f"VMD energy conservation error: {energy_error:.4f}")
        
        return modes

    def _initialize_omega(self) -> np.ndarray:
        """Initialize center frequencies."""
        if self.params.init == 1:
            # Uniform initialization
            omega = np.linspace(0, 0.5, self.params.num_modes)
        elif self.params.init == 2:
            # Random initialization
            omega = np.random.rand(self.params.num_modes) * 0.5
        elif self.params.init == 3:
            # Noise initialization
            omega = np.sort(np.random.rand(self.params.num_modes) * 0.5)
        else:
            raise ValueError(f"Invalid init method: {self.params.init}")
        
        return omega

    def _update_mode(
        self,
        signal_hat: np.ndarray,
        modes: np.ndarray,
        omega: np.ndarray,
        k: int,
        lambda_hat: np.ndarray,
    ) -> np.ndarray:
        """
        Update mode k using Wiener filter.

        Args:
            signal_hat: FFT of signal
            modes: Current modes [num_modes × N]
            omega: Center frequencies
            k: Mode index to update
            lambda_hat: Lagrange multipliers in frequency domain

        Returns:
            Updated mode k [N]
        """
        # Sum of all other modes
        sum_modes_hat = np.sum(scipy.fft.fft(modes, axis=1), axis=0)
        sum_modes_hat -= scipy.fft.fft(modes[k])
        
        # Wiener filter with bandwidth constraint
        # u_k_hat = (signal_hat - Σ_{j≠k} u_j_hat + lambda_hat/2) / (1 + 2α(ω - ω_k)²)
        numerator = signal_hat - sum_modes_hat + lambda_hat / 2
        
        # Bandwidth constraint
        freq_shift = (self.freqs - omega[k]) ** 2
        denominator = 1 + 2 * self.params.alpha * freq_shift
        
        u_k_hat = numerator / denominator
        
        # Handle DC component if enabled
        if self.params.DC and k == 0:
            u_k_hat[0] = 0  # Remove DC from first mode
        
        # Inverse FFT
        u_k = np.real(scipy.fft.ifft(u_k_hat))
        
        return u_k

    def _update_frequency(self, mode: np.ndarray) -> float:
        """
        Update center frequency for mode.

        Args:
            mode: Single mode [N]

        Returns:
            Updated center frequency ω_k
        """
        # Compute gradient: ω_k = argmin ||∂_t (u_k * exp(-jω_k t))||²
        mode_hat = scipy.fft.fft(mode)
        
        # Compute derivative in frequency domain
        derivative = 1j * 2 * np.pi * self.freqs * mode_hat
        
        # Find frequency that minimizes derivative magnitude
        power_spectrum = np.abs(derivative) ** 2
        omega_k = self.freqs[np.argmin(power_spectrum)]
        
        # Ensure positive frequency
        omega_k = abs(omega_k)
        
        return omega_k

    def _check_convergence(self, modes: np.ndarray, modes_prev: np.ndarray) -> bool:
        """
        Check convergence criterion.

        Args:
            modes: Current modes [num_modes × N]
            modes_prev: Previous modes [num_modes × N]

        Returns:
            True if converged, False otherwise
        """
        # Σ_k ||u_k^n+1 - u_k^n||² / Σ_k ||u_k^n||² < tol
        numerator = np.sum((modes - modes_prev) ** 2)
        denominator = np.sum(modes_prev ** 2) + 1e-10
        
        relative_error = numerator / denominator
        
        return relative_error < self.params.tol

    def get_model_info(self) -> dict:
        """Get model metadata."""
        return {
            'num_modes': self.params.num_modes,
            'alpha': self.params.alpha,
            'tau': self.params.tau,
            'K': self.params.K,
            'DC': self.params.DC,
            'init': self.params.init,
            'tol': self.params.tol,
            'max_iter': self.params.max_iter,
        }
