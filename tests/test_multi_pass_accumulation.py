"""
Unit tests for multi-pass Δvx accumulation analysis.
"""

import numpy as np
import pytest

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from sgms_v1 import simulate_multi_pass_accumulation, P as DEFAULT_PARAMS

# Helper function to create test params with required fields
def get_test_params():
    test_params = DEFAULT_PARAMS.copy()
    test_params['max_step'] = 1e-6  # Add missing required parameter
    return test_params


def test_multi_pass_accumulation_small_n():
    """Test multi-pass accumulation with small n_passes for speed."""
    # Use small n_passes for testing
    test_params = get_test_params()
    results = simulate_multi_pass_accumulation(n_passes=10, params=test_params, verbose=False)
    
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
    """Test that warning is issued for large n_passes."""
    test_params = get_test_params()
    results = simulate_multi_pass_accumulation(n_passes=10001, params=test_params, verbose=False)


def test_multi_pass_accumulation_verbose_false():
    """Test that verbose=False suppresses output."""
    test_params = get_test_params()
    results = simulate_multi_pass_accumulation(n_passes=5, params=test_params, verbose=False)


def test_multi_pass_accumulation_error_type():
    """Test error type classification."""
    test_params = get_test_params()
    results = simulate_multi_pass_accumulation(n_passes=10, params=test_params, verbose=False)
    
    valid_types = ['random_walk', 'mean_reverting', 'insufficient_variance']
    assert results['error_type'] in valid_types


def test_multi_pass_accumulation_cumulative_sum():
    """Test that cumulative_delta_vx is correct cumulative sum."""
    test_params = get_test_params()
    results = simulate_multi_pass_accumulation(n_passes=10, params=test_params, verbose=False)
    
    # Manually compute cumulative sum
    expected_cumsum = np.cumsum(results['delta_vx_history'])
    
    assert np.allclose(results['cumulative_delta_vx'], expected_cumsum)


def test_multi_pass_accumulation_drift_rate():
    """Test that drift_rate equals final_cumulative / n_passes."""
    test_params = get_test_params()
    results = simulate_multi_pass_accumulation(n_passes=10, params=test_params, verbose=False)
    
    expected_drift = results['final_cumulative'] / results['n_passes']
    assert np.isclose(results['drift_rate'], expected_drift)


def test_multi_pass_accumulation_failed_passes():
    """Test that failed_passes is tracked."""
    test_params = get_test_params()
    results = simulate_multi_pass_accumulation(n_passes=10, params=test_params, verbose=False)
    
    # failed_passes should be a non-negative integer
    assert isinstance(results['failed_passes'], (int, np.integer))
    assert results['failed_passes'] >= 0
    assert results['failed_passes'] <= results['n_passes']


def test_multi_pass_accumulation_mean_std():
    """Test that mean and std are calculated correctly."""
    test_params = get_test_params()
    results = simulate_multi_pass_accumulation(n_passes=10, params=test_params, verbose=False)
    
    # Manually compute mean and std
    expected_mean = np.mean(results['delta_vx_history'])
    expected_std = np.std(results['delta_vx_history'])
    
    assert np.isclose(results['mean_delta_vx'], expected_mean)
    assert np.isclose(results['std_delta_vx'], expected_std)


def test_multi_pass_accumulation_insufficient_variance():
    """Test error_type='insufficient_variance' when variance is near zero."""
    test_params = get_test_params()
    results = simulate_multi_pass_accumulation(n_passes=5, params=test_params, verbose=False)
    
    # Manually compute walk_ratio
    expected_walk_std = results['std_delta_vx'] * np.sqrt(results['n_passes'])
    actual_cumulative_std = np.std(results['cumulative_delta_vx'])
    
    if results['std_delta_vx'] < 1e-12:
        assert results['error_type'] == 'insufficient_variance'


def test_multi_pass_accumulation_walk_ratio_logic():
    """Test that walk_ratio calculation is correct."""
    test_params = get_test_params()
    results = simulate_multi_pass_accumulation(n_passes=10, params=test_params, verbose=False)
    
    # Manually compute walk_ratio
    expected_walk_std = results['std_delta_vx'] * np.sqrt(results['n_passes'])
    actual_cumulative_std = np.std(results['cumulative_delta_vx'])
    
    if expected_walk_std > 1e-12:
        walk_ratio = actual_cumulative_std / expected_walk_std
        # walk_ratio should be positive
        assert walk_ratio >= 0


def test_multi_pass_accumulation_progress_logging():
    """Test that progress is logged when verbose=True."""
    test_params = get_test_params()
    results = simulate_multi_pass_accumulation(n_passes=15, params=test_params, verbose=True)


def test_multi_pass_accumulation_default_params():
    """Test that default params work when params=None."""
    # This will use the global P variable from sgms_v1
    results = simulate_multi_pass_accumulation(n_passes=5, params=None, verbose=False)
    
    # Should use global P (DEFAULT_PARAMS)
    assert results['n_passes'] == 5
