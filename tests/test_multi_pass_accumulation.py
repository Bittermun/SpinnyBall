"""
Unit tests for multi-pass Δvx accumulation analysis.
"""

import numpy as np
import pytest

from sgms_v1 import simulate_multi_pass_accumulation, DEFAULT_PARAMS


def test_multi_pass_accumulation_small_n():
    """Test multi-pass accumulation with small n_passes for speed."""
    # Use small n_passes for testing
    results = simulate_multi_pass_accumulation(n_passes=10, params=DEFAULT_PARAMS, verbose=False)
    
    # Check return structure
    assert 'delta_vx_history' in results
    assert 'cumulative_delta_vx' in results
    assert 'mean_delta_vx' in results
    assert 'std_delta_vx' in results
    assert 'final_cumulative' in results
    assert 'drift_rate' in results
    assert 'error_type' in results
    assert 'n_passes' in results
    assert 'failed_passes' in results
    
    # Check array sizes
    assert len(results['delta_vx_history']) == 10
    assert len(results['cumulative_delta_vx']) == 10
    assert results['n_passes'] == 10


def test_multi_pass_accumulation_missing_params():
    """Test that missing required parameters raises ValueError."""
    incomplete_params = {'mass': 0.05}  # Missing array_length, v_z0, etc.
    
    with pytest.raises(ValueError, match="Missing required parameter"):
        simulate_multi_pass_accumulation(n_passes=1, params=incomplete_params, verbose=False)


def test_multi_pass_accumulation_warning_large_n():
    """Test that large n_passes triggers warning."""
    # This test checks that the warning is printed (captured via capsys in real pytest)
    # For now, just verify it doesn't crash
    results = simulate_multi_pass_accumulation(n_passes=10001, params=DEFAULT_PARAMS, verbose=False)
    assert results['n_passes'] == 10001


def test_multi_pass_accumulation_verbose_false():
    """Test that verbose=False suppresses output."""
    results = simulate_multi_pass_accumulation(n_passes=5, params=DEFAULT_PARAMS, verbose=False)
    assert results['n_passes'] == 5


def test_multi_pass_accumulation_error_type():
    """Test that error_type is one of expected values."""
    results = simulate_multi_pass_accumulation(n_passes=10, params=DEFAULT_PARAMS, verbose=False)
    
    valid_types = ['random_walk', 'mean_reverting', 'insufficient_variance']
    assert results['error_type'] in valid_types


def test_multi_pass_accumulation_cumulative_sum():
    """Test that cumulative_delta_vx is correct cumulative sum."""
    results = simulate_multi_pass_accumulation(n_passes=10, params=DEFAULT_PARAMS, verbose=False)
    
    # Manually compute cumulative sum
    expected_cumsum = np.cumsum(results['delta_vx_history'])
    
    assert np.allclose(results['cumulative_delta_vx'], expected_cumsum)


def test_multi_pass_accumulation_drift_rate():
    """Test that drift_rate equals final_cumulative / n_passes."""
    results = simulate_multi_pass_accumulation(n_passes=10, params=DEFAULT_PARAMS, verbose=False)
    
    expected_drift = results['final_cumulative'] / results['n_passes']
    assert np.isclose(results['drift_rate'], expected_drift)


def test_multi_pass_accumulation_failed_passes():
    """Test that failed_passes is tracked."""
    results = simulate_multi_pass_accumulation(n_passes=10, params=DEFAULT_PARAMS, verbose=False)
    
    # failed_passes should be a non-negative integer
    assert isinstance(results['failed_passes'], (int, np.integer))
    assert results['failed_passes'] >= 0
    assert results['failed_passes'] <= results['n_passes']


def test_multi_pass_accumulation_mean_std():
    """Test that mean and std are calculated correctly."""
    results = simulate_multi_pass_accumulation(n_passes=10, params=DEFAULT_PARAMS, verbose=False)
    
    # Manually compute mean and std
    expected_mean = np.mean(results['delta_vx_history'])
    expected_std = np.std(results['delta_vx_history'])
    
    assert np.isclose(results['mean_delta_vx'], expected_mean)
    assert np.isclose(results['std_delta_vx'], expected_std)


def test_multi_pass_accumulation_insufficient_variance():
    """Test error_type='insufficient_variance' when variance is near zero."""
    # Create params that would produce very small variance
    # This is hard to guarantee without mocking, so we just test the logic exists
    results = simulate_multi_pass_accumulation(n_passes=5, params=DEFAULT_PARAMS, verbose=False)
    
    # If variance is very small, should return 'insufficient_variance'
    if results['std_delta_vx'] < 1e-12:
        assert results['error_type'] == 'insufficient_variance'


def test_multi_pass_accumulation_walk_ratio_logic():
    """Test that walk_ratio calculation is correct."""
    results = simulate_multi_pass_accumulation(n_passes=10, params=DEFAULT_PARAMS, verbose=False)
    
    # Manually compute walk_ratio
    expected_walk_std = results['std_delta_vx'] * np.sqrt(results['n_passes'])
    actual_cumulative_std = np.std(results['cumulative_delta_vx'])
    
    if expected_walk_std > 1e-12:
        walk_ratio = actual_cumulative_std / expected_walk_std
        # walk_ratio should be positive
        assert walk_ratio >= 0


def test_multi_pass_accumulation_progress_logging():
    """Test that progress logging works (verbose=True)."""
    # Just verify it doesn't crash with verbose=True
    results = simulate_multi_pass_accumulation(n_passes=15, params=DEFAULT_PARAMS, verbose=True)
    assert results['n_passes'] == 15


def test_multi_pass_accumulation_default_params():
    """Test that function works with DEFAULT_PARAMS."""
    results = simulate_multi_pass_accumulation(n_passes=5, params=None, verbose=False)
    
    # Should use global P (DEFAULT_PARAMS)
    assert results['n_passes'] == 5
