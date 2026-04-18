# Post-Mortem Analysis: Weeks 1–9 Implementation

## Overview
This document analyzes issues found during Weeks 1–9 implementation. Weeks 10–12 are now complete (dashboard extension, FastAPI backend, digital twin visualization).

## Issues Found and Root Causes

### 1. Manual Quaternion Indexing (Error-Prone)
**Issue**: Used manual array indexing `np.array([q[3], q[0], q[1], q[2]])` for quaternion conversion
**Root Cause**: Didn't consider numpy's built-in optimized functions
**Fix by User**: Changed to `np.roll(q, 1)` - zero-copy view-based conversion
**Lesson**: Always check for numpy built-ins before manual array manipulation

### 2. Incomplete Input Validation
**Issue**: Only checked `shape != (3,)` for vectors, didn't check ndim
**Root Cause**: Assumed shape check was sufficient, didn't consider 2D arrays
**Fix by User**: Added `if omega.ndim != 1` validation
**Lesson**: Validate both shape and ndim for array inputs to catch all invalid cases

### 3. Missing Torque Function Validation
**Issue**: No validation of torque function return value (shape/ndim)
**Root Cause**: Trusted external function contracts without verification
**Fix by User**: Added ndim and shape validation for torque returns
**Lesson**: Never trust external function contracts - validate all inputs/outputs

### 4. Performance Issue: Repeated Matrix Inversion
**Issue**: Computed `np.linalg.inv(I)` on every euler_equations call
**Root Cause**: Didn't consider that inertia is constant for a rigid body
**Fix by User**: Added lazy caching with `_I_inv` property and optional precomputed parameter
**Lesson**: Profile performance for constant values - cache expensive computations

### 5. Generic Error Messages
**Issue**: Error messages like "Quaternion norm too small for normalization"
**Root Cause**: Didn't include context for debugging
**Fix by User**: Added detailed messages with norm value, quaternion, and actionable suggestions
**Lesson**: Error messages should include: actual value, expected value, context, and actionable suggestions

### 6. Weak Type Hints
**Issue**: Used generic `callable` type hint instead of specific signature
**Root Cause**: Didn't use proper typing for callable signatures
**Fix by User**: Changed to `Callable[[float, np.ndarray], np.ndarray]`
**Lesson**: Use specific type hints for callables to document expected signatures

### 7. StubMPCController solve Method Bug
**Issue**: Referenced undefined `kwargs` in solve method
**Root Cause**: Copied pattern from main MPCController without adapting
**Fix by Me**: Changed signature to include `horizon: int = 10` parameter
**Lesson**: When copying patterns, adapt to the specific context - don't leave undefined references

### 8. Test Syntax Error
**Issue**: Used incorrect assert syntax `assert p.type == ... for p in ... if ...`
**Root Cause**: Confused list comprehension with assert syntax
**Fix by Me**: Changed to proper list comprehension then assert membership
**Lesson**: Test code should be as carefully written as production code - verify syntax

### 9. PassFailGate Warning Logic Bug
**Issue**: Warning logic only triggered when value failed main threshold but passed warning threshold (impossible for ">=")
**Root Cause**: Didn't think through the logic for different comparison operators
**Fix by Me**: Rewrote logic to handle ">=" and "<=" operators correctly
**Lesson**: Test edge cases for all comparison operators - logic must work for each type

## Patterns for Excellence

### Input Validation Pattern
```python
def validate_vector(omega: np.ndarray, name: str = "omega") -> np.ndarray:
    """Validate 3D vector input."""
    omega = np.asarray(omega, dtype=float)
    if omega.ndim != 1:
        raise ValueError(f"{name} must be 1D array, got ndim={omega.ndim}")
    if omega.shape != (3,):
        raise ValueError(f"{name} must be 3-element vector, got shape {omega.shape}")
    return omega
```

### Performance Caching Pattern
```python
@property
def I_inv(self) -> np.ndarray:
    """Lazy-computed inertia tensor inverse."""
    if self._I_inv is None:
        self._I_inv = np.linalg.inv(self.I)
    return self._I_inv
```

### Error Message Pattern
```python
raise ValueError(
    f"Quaternion norm too small for normalization: norm={norm:.2e}, "
    f"quaternion={q}. This may indicate numerical integration instability. "
    f"Consider smaller timestep or higher-order integrator."
)
```

### Type Hint Pattern
```python
from typing import Callable

def integrate(
    self,
    dt: float,
    torques: Callable[[float, np.ndarray], np.ndarray],
) -> dict:
    ...
```

### Test Pattern
```python
# Instead of: assert condition for x in items if condition
# Use:
items_filtered = [x for x in items if condition]
assert expected in items_filtered
```

## Guidelines for Weeks 10–12 (Completed)

Weeks 10–12 (dashboard extension, FastAPI backend, digital twin visualization) are now complete. These guidelines remain applicable for future development.

### Before Writing Code
1. Check for numpy/scipy built-ins before manual array manipulation
2. Validate all inputs (shape, ndim, dtype) with specific error messages
3. Profile performance for repeated operations - consider caching
4. Use specific type hints for all function signatures
5. Write error messages with: actual value, expected value, context, action

### During Code Review
1. Check for undefined variable references
2. Verify logic works for all comparison operators (>=, <=, >, <, ==, !=)
3. Ensure test code compiles and runs
4. Validate that external function returns are checked
5. Confirm constant values are cached if expensive to compute

### Testing Strategy
1. Test all validation paths (valid inputs, invalid shapes, invalid ndim)
2. Test edge cases for comparison operators
3. Test with both numpy arrays and Python lists where applicable
4. Verify error messages are actionable
5. Test performance for repeated operations

### Code Quality Checklist
- [ ] All inputs validated (shape, ndim, dtype)
- [ ] All external function outputs validated
- [ ] Error messages include actual/expected values + context + action
- [ ] Type hints are specific (not generic)
- [ ] No undefined variable references
- [ ] Performance-critical code uses caching
- [ ] Uses numpy built-ins instead of manual manipulation
- [ ] Test code compiles and runs
- [ ] Logic tested for all comparison operators

## Dependencies for Weeks 10–12 (Completed)

The Weeks 10–12 dashboard extension required:
- Frontend: Three.js for 3D visualization
- Backend: FastAPI for web interface
- Integration: Pyodide/PyScript for in-browser Python (optional)
- Validation: MuJoCo for 6-DoF oracle comparison (pending)

These dependencies are now installed in pyproject.toml with the `backend` extra.
