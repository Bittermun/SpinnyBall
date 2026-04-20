"""
Unit tests for IRCNN predictor implementation.
"""

import pytest

# Skip tests if PyTorch is not available
pytest.importorskip("torch", reason="PyTorch not available")

import torch
import torch.nn as nn

from control_layer.ircnn_predictor import IRCNNBlock, IRCNNPredictor, IRCNNParameters


class TestIRCNNParameters:
    """Test IRCNN parameters dataclass."""

    def test_default_parameters(self):
        """Test default parameter values."""
        params = IRCNNParameters()
        assert params.input_dim == 7
        assert params.hidden_dim == 64
        assert params.num_blocks == 4

    def test_custom_parameters(self):
        """Test custom parameter values."""
        params = IRCNNParameters(
            input_dim=10,
            hidden_dim=128,
            num_blocks=8,
        )
        assert params.input_dim == 10
        assert params.hidden_dim == 128
        assert params.num_blocks == 8


class TestIRCNNBlock:
    """Test IRCNN invertible residual block."""

    def test_initialization(self):
        """Test IRCNN block initialization."""
        block = IRCNNBlock(hidden_dim=64)
        assert block.hidden_dim == 64
        assert block.f_net is not None
        assert block.g_net is not None

    def test_forward_shape(self):
        """Test forward pass preserves shape."""
        block = IRCNNBlock(hidden_dim=64)
        x = torch.randn(10, 128)  # batch=10, dim=128 (hidden_dim * 2)
        x_out = block(x)
        assert x_out.shape == x.shape

    def test_invertibility(self):
        """Test IRCNN block is exactly invertible (error < 1e-6)."""
        block = IRCNNBlock(hidden_dim=64)
        x = torch.randn(10, 128)
        
        x_forward = block(x)
        x_inverse = block.inverse(x_forward)
        
        error = torch.max(torch.abs(x - x_inverse))
        assert error < 1e-6, f"Invertibility error {error:.2e} exceeds 1e-6"

    def test_log_det(self):
        """Test log determinant computation."""
        block = IRCNNBlock(hidden_dim=64)
        log_det = block.log_det()
        # For this architecture, log_det should be 0
        assert log_det == 0.0

    def test_different_hidden_dim(self):
        """Test IRCNN block with different hidden dimension."""
        block = IRCNNBlock(hidden_dim=32)
        x = torch.randn(10, 64)  # hidden_dim * 2
        x_out = block(x)
        assert x_out.shape == (10, 64)


class TestIRCNNPredictor:
    """Test IRCNN predictor implementation."""

    def test_initialization(self):
        """Test IRCNN predictor initialization."""
        predictor = IRCNNPredictor(input_dim=7, hidden_dim=64, num_blocks=4)
        assert predictor.input_dim == 7
        assert predictor.hidden_dim == 64
        assert predictor.num_blocks == 4
        assert len(predictor.blocks) == 4

    def test_initialization_default_params(self):
        """Test IRCNN predictor with default parameters."""
        predictor = IRCNNPredictor()
        assert predictor.input_dim == 7
        assert predictor.hidden_dim == 64
        assert predictor.num_blocks == 4

    def test_forward_shape(self):
        """Test forward pass preserves input shape."""
        predictor = IRCNNPredictor(input_dim=7, hidden_dim=64, num_blocks=4)
        x = torch.randn(10, 7)
        x_out = predictor(x)
        assert x_out.shape == x.shape

    def test_forward_different_batch_size(self):
        """Test forward pass with different batch sizes."""
        predictor = IRCNNPredictor(input_dim=7, hidden_dim=64, num_blocks=4)
        
        for batch_size in [1, 5, 10, 20]:
            x = torch.randn(batch_size, 7)
            x_out = predictor(x)
            assert x_out.shape == (batch_size, 7)

    def test_compute_log_likelihood(self):
        """Test log likelihood computation."""
        predictor = IRCNNPredictor(input_dim=7, hidden_dim=64, num_blocks=4)
        x = torch.randn(10, 7)
        x_pred = predictor(x)
        
        log_likelihood = predictor.compute_log_likelihood(x, x_pred)
        assert log_likelihood is not None
        assert isinstance(log_likelihood.item(), float)

    def test_get_model_info(self):
        """Test get_model_info returns correct metadata."""
        predictor = IRCNNPredictor(input_dim=7, hidden_dim=64, num_blocks=4)
        
        info = predictor.get_model_info()
        assert info['input_dim'] == 7
        assert info['hidden_dim'] == 64
        assert info['num_blocks'] == 4
        assert info['total_parameters'] > 0
        assert info['trainable_parameters'] > 0

    def test_different_input_dim(self):
        """Test IRCNN predictor with different input dimension."""
        predictor = IRCNNPredictor(input_dim=10, hidden_dim=64, num_blocks=4)
        x = torch.randn(5, 10)
        x_out = predictor(x)
        assert x_out.shape == (5, 10)

    def test_different_hidden_dim(self):
        """Test IRCNN predictor with different hidden dimension."""
        predictor = IRCNNPredictor(input_dim=7, hidden_dim=32, num_blocks=4)
        x = torch.randn(5, 7)
        x_out = predictor(x)
        assert x_out.shape == (5, 7)

    def test_different_num_blocks(self):
        """Test IRCNN predictor with different number of blocks."""
        predictor = IRCNNPredictor(input_dim=7, hidden_dim=64, num_blocks=8)
        x = torch.randn(5, 7)
        x_out = predictor(x)
        assert x_out.shape == (5, 7)
        assert len(predictor.blocks) == 8

    def test_single_block(self):
        """Test IRCNN predictor with single block."""
        predictor = IRCNNPredictor(input_dim=7, hidden_dim=64, num_blocks=1)
        x = torch.randn(5, 7)
        x_out = predictor(x)
        assert x_out.shape == (5, 7)
        assert len(predictor.blocks) == 1

    def test_model_in_eval_mode(self):
        """Test model in eval mode."""
        predictor = IRCNNPredictor(input_dim=7, hidden_dim=64, num_blocks=4)
        predictor.eval()
        
        x = torch.randn(10, 7)
        with torch.no_grad():
            x_out = predictor(x)
        assert x_out.shape == (10, 7)

    def test_model_in_train_mode(self):
        """Test model in train mode."""
        predictor = IRCNNPredictor(input_dim=7, hidden_dim=64, num_blocks=4)
        predictor.train()
        
        x = torch.randn(10, 7)
        x_out = predictor(x)
        assert x_out.shape == (10, 7)


class TestIRCNNPerformance:
    """Performance benchmarks for IRCNN predictor."""

    def test_inference_latency(self):
        """Test IRCNN inference meets latency target (< 5 ms)."""
        import time
        
        predictor = IRCNNPredictor(input_dim=7, hidden_dim=64, num_blocks=4)
        predictor.eval()
        
        x = torch.randn(10, 7)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                predictor(x)
        
        # Benchmark
        n_iterations = 100
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_iterations):
                predictor(x)
        elapsed = time.perf_counter() - start
        latency_ms = (elapsed / n_iterations) * 1000
        
        assert latency_ms < 5.0, f"Latency {latency_ms:.2f} ms exceeds target 5 ms"

    def test_parameter_count(self):
        """Test model parameter count is reasonable (< 50 MB)."""
        predictor = IRCNNPredictor(input_dim=7, hidden_dim=64, num_blocks=4)
        
        info = predictor.get_model_info()
        # Assuming float32 (4 bytes per parameter)
        param_size_mb = info['total_parameters'] * 4 / (1024 ** 2)
        
        assert param_size_mb < 50.0, f"Model size {param_size_mb:.2f} MB exceeds target 50 MB"
