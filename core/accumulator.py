import torch
import torch.nn as nn
from typing import Tuple, Optional


class LinearAccumulator(nn.Module):
    """
    Fixed-size accumulator for linear attention state.

    Maintains:
    - kv_state: Accumulated key-value products (D x D matrix)
    - k_sum: Accumulated key norms for normalization (D,)
    - length: Running count of processed tokens
    """

    def __init__(self, dim: int, device=None):
        super().__init__()
        self.dim = dim
        self.device = device

        self.register_buffer('kv_state', torch.zeros(dim, dim, device=device))
        self.register_buffer('k_sum', torch.zeros(dim, device=device))
        self.register_buffer('length', torch.tensor(0, device=device))

    def reset(self):
        """Reset accumulator state."""
        self.kv_state.zero_()
        self.k_sum.zero_()
        self.length.zero_()

    def update(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update accumulator with new key-value pair.

        Args:
            k: Key features (batch, seq_len, dim) or (batch, dim)
            v: Value features (batch, seq_len, dim) or (batch, dim)

        Returns:
            context: Normalized context vector (batch, seq_len, dim) or (batch, dim)
            normalizer: Normalization factor (batch, seq_len, 1) or (batch, 1)
        """
        # Handle both 2D and 3D inputs
        input_shape = k.shape
        if len(k.shape) == 3:
            batch_size, seq_len, _ = k.shape
            k_flat = k.view(-1, self.dim)
            v_flat = v.view(-1, self.dim)
        else:
            batch_size = k.shape[0]
            seq_len = 1
            k_flat = k
            v_flat = v

        # Update KV state: kv_state += k^T @ v
        kv_product = torch.bmm(k_flat.unsqueeze(1).transpose(1, 2), v_flat.unsqueeze(1))
        self.kv_state = self.kv_state + kv_product.squeeze(0)

        # Update K sum: k_sum += k^T (for normalization)
        self.k_sum = self.k_sum + k_flat.sum(dim=0)

        # Update length counter
        self.length = self.length + seq_len

        # Compute context: v' = KV_state @ k / (k_sum @ k + eps)
        context = torch.matmul(self.kv_state, k_flat.t()).t()
        normalizer = torch.matmul(k_flat, self.k_sum.unsqueeze(1))

        # Reshape back to original shape
        if len(input_shape) == 3:
            context = context.view(*input_shape)
            normalizer = normalizer.view(batch_size, seq_len, 1)

        return context, normalizer

    def get_state(self) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Get current accumulator state."""
        return self.kv_state.clone(), self.k_sum.clone(), self.length.item()

    def set_state(self, kv_state: torch.Tensor, k_sum: torch.Tensor, length: int):
        """Set accumulator state from previous checkpoint."""
        self.kv_state.copy_(kv_state)
        self.k_sum.copy_(k_sum)
        self.length.fill_(length)


class PrefixSumAccumulator:
    """
    Efficient computation of parallel prefix sums for training.

    Enables simultaneous computation of attention outputs for all positions
    during training, avoiding sequential computation.
    """

    @staticmethod
    def compute_prefix_sums(k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute prefix KV accumulations for all positions efficiently.

        Args:
            k: Key features (batch, seq_len, dim)
            v: Value features (batch, seq_len, dim)

        Returns:
            kv_sums: Prefix accumulated KV products (batch, seq_len, dim, dim)
            k_sums: Prefix accumulated K norms (batch, seq_len, dim)
        """
        batch_size, seq_len, dim = k.shape

        # Initialize accumulators
        kv_sums = torch.zeros(batch_size, seq_len, dim, dim, device=k.device, dtype=k.dtype)
        k_sums = torch.zeros(batch_size, seq_len, dim, device=k.device, dtype=k.dtype)

        # Compute cumulative products
        kv_acc = torch.zeros(batch_size, dim, dim, device=k.device, dtype=k.dtype)
        k_acc = torch.zeros(batch_size, dim, device=k.device, dtype=k.dtype)

        for i in range(seq_len):
            k_i = k[:, i, :]  # (batch, dim)
            v_i = v[:, i, :]  # (batch, dim)

            # kv_acc += k_i^T @ v_i
            kv_product = torch.bmm(k_i.unsqueeze(2), v_i.unsqueeze(1))
            kv_acc = kv_acc + kv_product

            # k_acc += k_i
            k_acc = k_acc + k_i

            kv_sums[:, i, :, :] = kv_acc
            k_sums[:, i, :] = k_acc

        return kv_sums, k_sums

    @staticmethod
    def compute_outputs(q: torch.Tensor, kv_sums: torch.Tensor, k_sums: torch.Tensor,
                       eps=1e-6) -> torch.Tensor:
        """
        Compute attention outputs using prefix sums.

        Args:
            q: Query features (batch, seq_len, dim)
            kv_sums: Prefix KV products (batch, seq_len, dim, dim)
            k_sums: Prefix K sums (batch, seq_len, dim)
            eps: Numerical stability epsilon

        Returns:
            outputs: Attention outputs (batch, seq_len, dim)
        """
        batch_size, seq_len, dim = q.shape

        outputs = torch.zeros_like(q)

        for i in range(seq_len):
            q_i = q[:, i, :]  # (batch, dim)
            kv_acc = kv_sums[:, i, :, :]  # (batch, dim, dim)
            k_acc = k_sums[:, i, :]  # (batch, dim)

            # context = kv_acc @ q_i / (k_acc @ q_i + eps)
            numerator = torch.bmm(kv_acc, q_i.unsqueeze(2)).squeeze(2)
            denominator = torch.sum(k_acc * q_i, dim=1, keepdim=True)

            outputs[:, i, :] = numerator / (denominator + eps)

        return outputs
