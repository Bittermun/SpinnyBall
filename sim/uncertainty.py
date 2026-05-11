"""
Uncertainty quantification for physics outputs.

Lightweight first-order error propagation. Not Monte Carlo.
"""

from dataclasses import dataclass
from typing import Optional, Union, Self
import numpy as np


@dataclass(frozen=True)
class ValidityRegime:
    """Parameter ranges where uncertainty estimates apply."""
    param_name: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    def check(self, value: float) -> bool:
        """Check if value is within valid regime."""
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        return True
    
    def __str__(self) -> str:
        parts = []
        if self.min_value is not None:
            parts.append(f">= {self.min_value}")
        if self.max_value is not None:
            parts.append(f"<= {self.max_value}")
        return f"{self.param_name}: {' and '.join(parts)}"


@dataclass
class UncertainQuantity:
    """
    Scalar quantity with first-order uncertainty.
    
    Stores:
    - Central value
    - Standard deviation (statistical/random error)
    - Systematic error (bias)
    - Validity regime (where these uncertainty estimates apply)
    
    Example:
        >>> B = UncertainQuantity(
        ...     value=1.5,  # Tesla
        ...     std_dev=0.05,  # 3.3% random error
        ...     systematic_error=0.1,  # 7% systematic from model simplification
        ...     validity=[ValidityRegime("r", 0.1, 10.0)]  # valid 0.1-10m
        ... )
    """
    value: float
    std_dev: float = 0.0
    systematic_error: float = 0.0
    validity: list[ValidityRegime] = None
    source: str = ""  # Description of how value was computed
    
    def __post_init__(self):
        if self.validity is None:
            self.validity = []
    
    @property
    def total_error(self) -> float:
        """Total uncertainty (RSS of random and systematic)."""
        return np.sqrt(self.std_dev**2 + self.systematic_error**2)
    
    @property
    def relative_error(self) -> float:
        """Relative uncertainty."""
        if abs(self.value) < 1e-15:
            return float('inf')
        return self.total_error / abs(self.value)
    
    @property
    def lower_bound(self, n_sigma: float = 2.0) -> float:
        """Lower bound at n-sigma confidence (approximate)."""
        return self.value - n_sigma * self.total_error
    
    @property
    def upper_bound(self, n_sigma: float = 2.0) -> float:
        """Upper bound at n-sigma confidence (approximate)."""
        return self.value + n_sigma * self.total_error
    
    def check_validity(self, **params) -> tuple[bool, list[str]]:
        """
        Check if quantity is valid given current parameters.
        
        Args:
            **params: Parameter values to check (e.g., r=5.0, T=77.0)
            
        Returns:
            (is_valid, list of violation messages)
        """
        violations = []
        for regime in self.validity:
            if regime.param_name in params:
                if not regime.check(params[regime.param_name]):
                    violations.append(
                        f"{regime.param_name}={params[regime.param_name]} "
                        f"outside validity: {regime}"
                    )
        return len(violations) == 0, violations
    
    # Arithmetic with uncertainty propagation
    def __add__(self, other: Union[float, Self]) -> Self:
        """Addition: variances add for independent quantities."""
        if isinstance(other, (int, float)):
            return UncertainQuantity(
                value=self.value + other,
                std_dev=self.std_dev,
                systematic_error=self.systematic_error,
                validity=self.validity.copy(),
                source=f"{self.source} + {other}"
            )
        
        # Combine validity regimes (intersection)
        new_validity = self._intersect_validity(other)
        
        return UncertainQuantity(
            value=self.value + other.value,
            std_dev=np.sqrt(self.std_dev**2 + other.std_dev**2),
            systematic_error=self.systematic_error + other.systematic_error,
            validity=new_validity,
            source=f"({self.source}) + ({other.source})"
        )
    
    def __radd__(self, other: float) -> Self:
        return self.__add__(other)
    
    def __sub__(self, other: Union[float, Self]) -> Self:
        """Subtraction: same as addition for independent quantities."""
        if isinstance(other, (int, float)):
            return UncertainQuantity(
                value=self.value - other,
                std_dev=self.std_dev,
                systematic_error=self.systematic_error,
                validity=self.validity.copy(),
                source=f"{self.source} - {other}"
            )
        
        new_validity = self._intersect_validity(other)
        
        return UncertainQuantity(
            value=self.value - other.value,
            std_dev=np.sqrt(self.std_dev**2 + other.std_dev**2),
            systematic_error=self.systematic_error + other.systematic_error,
            validity=new_validity,
            source=f"({self.source}) - ({other.source})"
        )
    
    def __rsub__(self, other: float) -> Self:
        return UncertainQuantity(
            value=other - self.value,
            std_dev=self.std_dev,
            systematic_error=self.systematic_error,
            validity=self.validity.copy(),
            source=f"{other} - ({self.source})"
        )
    
    def __mul__(self, other: Union[float, Self]) -> Self:
        """
        Multiplication: relative errors add in quadrature.
        
        For A * B: (δC/C)² = (δA/A)² + (δB/B)²
        """
        if isinstance(other, (int, float)):
            return UncertainQuantity(
                value=self.value * other,
                std_dev=self.std_dev * abs(other),
                systematic_error=self.systematic_error * abs(other),
                validity=self.validity.copy(),
                source=f"{self.source} * {other}"
            )
        
        new_value = self.value * other.value
        
        # Relative error propagation
        rel1 = self.relative_error
        rel2 = other.relative_error
        new_rel_error = np.sqrt(rel1**2 + rel2**2)
        
        new_validity = self._intersect_validity(other)
        
        return UncertainQuantity(
            value=new_value,
            std_dev=new_value * new_rel_error * 0.5,  # Split between random/systematic
            systematic_error=new_value * new_rel_error * 0.5,
            validity=new_validity,
            source=f"({self.source}) * ({other.source})"
        )
    
    def __rmul__(self, other: float) -> Self:
        return self.__mul__(other)
    
    def __truediv__(self, other: Union[float, Self]) -> Self:
        """Division: same error propagation as multiplication."""
        if isinstance(other, (int, float)):
            return UncertainQuantity(
                value=self.value / other,
                std_dev=self.std_dev / abs(other),
                systematic_error=self.systematic_error / abs(other),
                validity=self.validity.copy(),
                source=f"{self.source} / {other}"
            )
        
        new_value = self.value / other.value
        
        rel1 = self.relative_error
        rel2 = other.relative_error
        new_rel_error = np.sqrt(rel1**2 + rel2**2)
        
        new_validity = self._intersect_validity(other)
        
        return UncertainQuantity(
            value=new_value,
            std_dev=new_value * new_rel_error * 0.5,
            systematic_error=new_value * new_rel_error * 0.5,
            validity=new_validity,
            source=f"({self.source}) / ({other.source})"
        )
    
    def __rtruediv__(self, other: float) -> Self:
        return UncertainQuantity(
            value=other / self.value,
            std_dev=other * self.std_dev / self.value**2,
            systematic_error=other * self.systematic_error / self.value**2,
            validity=self.validity.copy(),
            source=f"{other} / ({self.source})"
        )
    
    def __pow__(self, exponent: float) -> Self:
        """Power: δ(C^n) = n * C^(n-1) * δC"""
        new_value = self.value ** exponent
        derivative = exponent * self.value ** (exponent - 1)
        
        return UncertainQuantity(
            value=new_value,
            std_dev=abs(derivative) * self.std_dev,
            systematic_error=abs(derivative) * self.systematic_error,
            validity=self.validity.copy(),
            source=f"({self.source})^{exponent}"
        )
    
    def __neg__(self) -> Self:
        return UncertainQuantity(
            value=-self.value,
            std_dev=self.std_dev,
            systematic_error=self.systematic_error,
            validity=self.validity.copy(),
            source=f"-({self.source})"
        )
    
    def __abs__(self) -> Self:
        # Note: uncertainty increases near zero (derivative discontinuity)
        return UncertainQuantity(
            value=abs(self.value),
            std_dev=self.std_dev,
            systematic_error=self.systematic_error,
            validity=self.validity.copy(),
            source=f"|{self.source}|"
        )
    
    def sqrt(self) -> Self:
        """Square root with error propagation."""
        # δ(√x) = δx / (2√x)
        new_value = np.sqrt(self.value)
        if new_value > 1e-15:
            derivative = 1.0 / (2.0 * new_value)
            return UncertainQuantity(
                value=new_value,
                std_dev=self.std_dev * derivative,
                systematic_error=self.systematic_error * derivative,
                validity=self.validity.copy(),
                source=f"sqrt({self.source})"
            )
        else:
            return UncertainQuantity(
                value=new_value,
                std_dev=float('inf'),
                systematic_error=float('inf'),
                validity=self.validity.copy(),
                source=f"sqrt({self.source})"
            )
    
    def _intersect_validity(self, other: Self) -> list[ValidityRegime]:
        """Compute intersection of validity regimes."""
        # For now, just concatenate; smarter intersection is domain-specific
        return self.validity + other.validity
    
    def __repr__(self) -> str:
        return (f"UncertainQuantity({self.value:.6g} "
                f"± {self.total_error:.2g} "
                f"[{100*self.relative_error:.1f}%])")
    
    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            'value': self.value,
            'std_dev': self.std_dev,
            'systematic_error': self.systematic_error,
            'validity': [str(v) for v in self.validity],
            'source': self.source,
        }


@dataclass
class UncertainArray:
    """Array quantity with element-wise uncertainty."""
    values: np.ndarray
    std_devs: np.ndarray
    systematic_errors: np.ndarray
    validity: list[ValidityRegime] = None
    
    def __post_init__(self):
        if self.validity is None:
            self.validity = []
        self.values = np.asarray(self.values)
        self.std_devs = np.asarray(self.std_devs)
        self.systematic_errors = np.asarray(self.systematic_errors)
    
    @property
    def total_errors(self) -> np.ndarray:
        return np.sqrt(self.std_devs**2 + self.systematic_errors**2)
    
    def __getitem__(self, idx) -> UncertainQuantity:
        """Get element as UncertainQuantity."""
        return UncertainQuantity(
            value=self.values[idx],
            std_dev=self.std_devs[idx],
            systematic_error=self.systematic_errors[idx],
            validity=self.validity.copy()
        )
    
    def magnitude(self) -> UncertainQuantity:
        """Compute magnitude (norm) with error propagation."""
        mag = np.linalg.norm(self.values)
        # Error propagation: δ|v| = (v·δv)/|v|
        rel_error = np.dot(self.values, self.total_errors) / mag**2
        return UncertainQuantity(
            value=mag,
            std_dev=mag * rel_error * 0.5,
            systematic_error=mag * rel_error * 0.5,
            validity=self.validity.copy()
        )


# Convenience constructors

def certain(value: float, source: str = "") -> UncertainQuantity:
    """Create quantity with zero uncertainty (e.g., physical constant)."""
    return UncertainQuantity(value=value, std_dev=0.0, systematic_error=0.0, source=source)


def from_relative(value: float, relative_error: float, source: str = "") -> UncertainQuantity:
    """Create quantity from relative error (split evenly random/systematic)."""
    abs_error = abs(value * relative_error)
    return UncertainQuantity(
        value=value,
        std_dev=abs_error * 0.5,
        systematic_error=abs_error * 0.5,
        source=source
    )


def from_bounds(value: float, lower: float, upper: float, source: str = "") -> UncertainQuantity:
    """
    Create quantity from confidence bounds (assumes 2-sigma, normal distribution).
    
    Uses: total_error = (upper - lower) / 4
    """
    total_error = (upper - lower) / 4.0
    return UncertainQuantity(
        value=value,
        std_dev=total_error * 0.5,
        systematic_error=total_error * 0.5,
        source=source
    )
