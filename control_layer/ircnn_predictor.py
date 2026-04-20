"""
Invertible Residual CNN (IRCNN) predictor implementation.

Implements iResNet architecture for nonlinear prediction with exact
likelihood computation via invertible residual blocks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

logger = logging.getLogger(__name__)


@dataclass
class IRCNNParameters:
    """IRCNN hyperparameters."""
    input_dim: int = 7
    hidden_dim: int = 64
    num_blocks: int = 4


class IRCNNBlock(nn.Module):
    """
    Invertible residual block for exact likelihood computation.

    Architecture:
    - Split input into two streams: x1, x2
    - x1' = x1 + f(x2)
    - x2' = x2 + g(x1')
    - Inverse: x2 = x2' - g(x1'), x1 = x1' - f(x2)
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim

        # f network: transforms x2 to update x1
        self.f_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # g network: transforms x1' to update x2
        self.g_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (invertible).

        Args:
            x: Input tensor [batch, hidden_dim * 2]

        Returns:
            Output tensor [batch, hidden_dim * 2]
        """
        x1, x2 = torch.chunk(x, 2, dim=-1)
        x1_new = x1 + self.f_net(x2)
        x2_new = x2 + self.g_net(x1_new)
        return torch.cat([x1_new, x2_new], dim=-1)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse pass (exact).

        Args:
            x: Input tensor [batch, hidden_dim * 2]

        Returns:
            Reconstructed input tensor [batch, hidden_dim * 2]
        """
        x1_new, x2_new = torch.chunk(x, 2, dim=-1)
        x2 = x2_new - self.g_net(x1_new)
        x1 = x1_new - self.f_net(x2)
        return torch.cat([x1, x2], dim=-1)

    def log_det(self) -> torch.Tensor:
        """
        Compute log determinant of Jacobian.

        For this architecture, the Jacobian determinant is 1
        (triangular structure with ones on diagonal).

        Returns:
            Log determinant (scalar 0 for this architecture)
        """
        return torch.tensor(0.0)


class IRCNNPredictor(nn.Module):
    """
    Invertible Residual CNN for nonlinear prediction.

    Uses invertible residual blocks for exact likelihood computation
    and reversible inference.
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 64,
        num_blocks: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        # Input projection: input_dim -> hidden_dim * 2 (for split into two streams)
        self.input_proj = nn.Linear(input_dim, hidden_dim * 2)

        # Invertible residual blocks (operate on hidden_dim * 2)
        self.blocks = nn.ModuleList([
            IRCNNBlock(hidden_dim) for _ in range(num_blocks)
        ])

        # Output projection: hidden_dim * 2 -> input_dim
        self.output_proj = nn.Linear(hidden_dim * 2, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward prediction.

        Args:
            x: Input tensor [batch, input_dim]

        Returns:
            Predicted tensor [batch, input_dim]
        """
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)

    def compute_log_likelihood(self, x: torch.Tensor, x_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute exact log likelihood using invertibility.

        Args:
            x: Input tensor [batch, input_dim]
            x_pred: Predicted tensor [batch, input_dim]

        Returns:
            Log likelihood (scalar)
        """
        # Forward pass to get prediction
        h = self.input_proj(x)
        log_det = torch.tensor(0.0, device=x.device)

        for block in self.blocks:
            h = block(h)
            # Log determinant of Jacobian (tractable due to invertibility)
            log_det += block.log_det()

        x_pred = self.output_proj(h)

        # Gaussian likelihood
        log_likelihood = -0.5 * torch.sum((x_pred - x) ** 2) + log_det
        return log_likelihood

    def get_model_info(self) -> dict:
        """Get model metadata."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_blocks': self.num_blocks,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
        }
