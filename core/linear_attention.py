import torch
import torch.nn as nn
from typing import Optional, Tuple
from core.kernels import get_kernel, numerically_stable_divide
from core.accumulator import LinearAccumulator, PrefixSumAccumulator


class LinearAttention(nn.Module):
    """
    Linear Attention layer with O(N) complexity.

    Key features:
    - Maintains fixed-size state accumulator
    - Supports both training (parallel prefix sums) and inference (sequential updates)
    - Uses kernel functions to ensure feature positivity
    - Numerically stable normalization
    """

    def __init__(self, dim: int, num_heads: int = 8, kernel_type: str = 'elu',
                 dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel = get_kernel(kernel_type)
        self.dropout = nn.Dropout(dropout)

        # Projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # Initialize projections
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x: torch.Tensor, is_inference: bool = False,
                accumulator: Optional[LinearAccumulator] = None) -> Tuple[torch.Tensor, Optional[LinearAccumulator]]:
        """
        Forward pass with support for both training and inference.

        Args:
            x: Input features (batch, seq_len, dim)
            is_inference: If True, use sequential updates with accumulator
            accumulator: For inference, pass the accumulator state

        Returns:
            output: Attention output (batch, seq_len, dim)
            accumulator: Updated accumulator (only in inference mode)
        """
        batch_size, seq_len, dim = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply kernel to ensure positivity
        q = self.kernel.apply(q)
        k = self.kernel.apply(k)

        if is_inference:
            # Sequential inference with accumulator
            output = self._forward_inference(q, k, v, accumulator, batch_size, seq_len)
        else:
            # Parallel training with prefix sums
            output = self._forward_training(q, k, v)

        # Reshape back and apply output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        output = self.out_proj(output)
        output = self.dropout(output)

        return output, None

    def _forward_training(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Parallel training path using prefix sums."""
        batch_size, num_heads, seq_len, head_dim = q.shape

        # Reshape for prefix sum computation
        q_flat = q.transpose(1, 2).contiguous().view(-1, seq_len, head_dim)
        k_flat = k.transpose(1, 2).contiguous().view(-1, seq_len, head_dim)
        v_flat = v.transpose(1, 2).contiguous().view(-1, seq_len, head_dim)

        # Compute prefix sums
        kv_sums, k_sums = PrefixSumAccumulator.compute_prefix_sums(k_flat, v_flat)

        # Compute outputs for all positions
        outputs = PrefixSumAccumulator.compute_outputs(q_flat, kv_sums, k_sums)

        # Reshape back to multi-head format
        outputs = outputs.view(batch_size, num_heads, seq_len, head_dim)

        return outputs

    def _forward_inference(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          accumulator: Optional[LinearAccumulator],
                          batch_size: int, seq_len: int) -> torch.Tensor:
        """Sequential inference path with accumulator state."""
        _, num_heads, seq_len, head_dim = q.shape

        outputs = torch.zeros_like(q)

        if accumulator is None:
            accumulator = [LinearAccumulator(head_dim, device=q.device) for _ in range(num_heads)]

        for t in range(seq_len):
            for h in range(num_heads):
                k_t = k[:, h, t, :]
                v_t = v[:, h, t, :]
                q_t = q[:, h, t, :]

                context, normalizer = accumulator[h].update(k_t, v_t)
                output_t = numerically_stable_divide(
                    torch.matmul(context, q_t.unsqueeze(-1)).squeeze(-1),
                    normalizer.squeeze(-1)
                )
                outputs[:, h, t, :] = output_t

        return outputs


class MultiHeadLinearAttention(nn.Module):
    """Wrapper for multi-head linear attention with convenience interface."""

    def __init__(self, dim: int, num_heads: int = 8, kernel_type: str = 'elu',
                 dropout: float = 0.0):
        super().__init__()
        self.attention = LinearAttention(dim, num_heads, kernel_type, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.attention(x, is_inference=False)
        return output
