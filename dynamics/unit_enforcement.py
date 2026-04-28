"""
Unit enforcement for physics calculations using Pint.

This module provides type-safe unit handling to prevent silent unit mismatch
errors across the SpinnyBall physics engine.
"""

import pint
from typing import TypeVar, Union

# Create unit registry
ureg = pint.UnitRegistry()

# Type aliases for common physical quantities
Quantity = pint.Quantity
Meter = ureg.meter
Second = ureg.second
Kilogram = ureg.kilogram
MeterPerSecond = ureg.meter / ureg.second
MeterPerSecondSquared = ureg.meter / ureg.second**2
Newton = ureg.newton
Joule = ureg.joule
Watt = ureg.watt
Tesla = ureg.tesla
Pascal = ureg.pascal
Kelvin = ureg.kelvin

# Convenience function for creating quantities with units
def Q_(value, units):
    """Create a Pint quantity from value and unit string."""
    return ureg.Quantity(value, units)


class UnitEnforcedCalculator:
    """
    Wrapper class that enforces unit consistency in physics calculations.
    
    Example usage:
        calc = UnitEnforcedCalculator()
        force = calc.calculate_force(mass=10*ureg.kg, acceleration=5*ureg.m/ureg.s**2)
        # Returns: 50 newton
    """
    
    def __init__(self):
        self.ureg = ureg
    
    def ensure_units(self, value, expected_units):
        """
        Ensure a value has the expected units, converting if necessary.
        
        Args:
            value: A scalar or Pint Quantity
            expected_units: Expected unit string or Pint Unit
            
        Returns:
            Value converted to expected units
            
        Raises:
            pint.DimensionalityError: If units are incompatible
        """
        if not isinstance(value, pint.Quantity):
            # Assume SI base units if no units provided
            value = value * expected_units
        else:
            value = value.to(expected_units)
        return value
    
    def strip_units(self, value, target_units=None):
        """
        Strip units from a quantity, optionally converting first.
        
        Args:
            value: Pint Quantity
            target_units: Optional unit string to convert to before stripping
            
        Returns:
            Scalar value without units
        """
        if target_units is not None:
            value = value.to(target_units)
        return value.magnitude


# Global instance for convenience
unit_calc = UnitEnforcedCalculator()


# Decorator for enforcing units on function arguments
def enforce_units(**expected_unit_kwargs):
    """
    Decorator to enforce units on function keyword arguments.
    
    Example:
        @enforce_units(mass='kg', velocity='m/s')
        def calculate_momentum(mass, velocity):
            return mass * velocity
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Convert kwargs to proper units
            for param_name, expected_unit in expected_unit_kwargs.items():
                if param_name in kwargs:
                    value = kwargs[param_name]
                    if not isinstance(value, pint.Quantity):
                        kwargs[param_name] = Q_(value, expected_unit)
                    else:
                        kwargs[param_name] = value.to(expected_unit)
            return func(*args, **kwargs)
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator
