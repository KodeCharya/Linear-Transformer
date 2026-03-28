import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SlidingWindowAttention(nn.Module):
    """
    Standard scaled dot-product attention with causal sliding window.

    For local context (last N tokens), use quadratic attention.
    Beyond the window, use linear attention.
    """

    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply sliding window attention.

        Args:
            x: Input features (batch, seq_len, dim)

        Returns:
            output: Attention output (batch, seq_len, dim)
        """
        batch_size, seq_len, dim = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply sliding window mask
        mask = self._create_sliding_window_mask(seq_len, self.window_size, device=scores.device)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0) == 1, float('-inf'))

        # Softmax
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        output = torch.matmul(attn, v)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        output = self.out_proj(output)

        return output

    @staticmethod
    def _create_sliding_window_mask(seq_len: int, window_size: int, device) -> torch.Tensor:
        """Create sliding window mask for causal attention."""
        mask = torch.zeros(seq_len, seq_len, device=device)

        for i in range(seq_len):
            start = max(0, i - window_size + 1)
            mask[i, start:i+1] = 1

        return mask


class HybridAttention(nn.Module):
    """
    Combines sliding window attention (local) with linear attention (global).

    Strategy:
    - Use quadratic attention for the most recent window_size tokens
    - Use linear attention for everything before that
    """

    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 64,
                 kernel_type: str = 'elu', dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size

        self.window_attention = SlidingWindowAttention(dim, num_heads, window_size, dropout)

        # Import here to avoid circular dependency
        from core.linear_attention import LinearAttention
        self.linear_attention = LinearAttention(dim, num_heads, kernel_type, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply hybrid attention combining local and global context.

        Args:
            x: Input features (batch, seq_len, dim)

        Returns:
            output: Blended attention output (batch, seq_len, dim)
        """
        batch_size, seq_len, dim = x.shape

        # Get linear attention output (global context)
        linear_out, _ = self.linear_attention(x, is_inference=False)

        # Get sliding window attention output (local context)
        window_out = self.window_attention(x)

        # Blend outputs with learned weight
        # Early tokens use more linear attention, recent tokens use more window
        blend_weights = torch.linspace(0, 1, seq_len, device=x.device)
        blend_weights = blend_weights.view(1, seq_len, 1)

        output = (1 - blend_weights) * linear_out + blend_weights * window_out

        return output


class ContextFusionLayer(nn.Module):
    """
    Advanced fusion mechanism combining multiple attention types.

    Learns optimal blending of:
    1. Full sequence linear attention
    2. Local sliding window attention
    3. Adaptive context based on content
    """

    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 64):
        super().__init__()
        self.dim = dim

        # Multi-view attention modules
        from core.linear_attention import LinearAttention
        self.linear_attn = LinearAttention(dim, num_heads, 'elu')
        self.window_attn = SlidingWindowAttention(dim, num_heads, window_size)

        # Learnable fusion gates
        self.linear_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.window_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

        # Normalization
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply learned context fusion."""
        # Get attention outputs
        linear_out, _ = self.linear_attn(x, is_inference=False)
        window_out = self.window_attn(x)

        # Compute gating values
        linear_gate = self.linear_gate(x)
        window_gate = self.window_gate(x)

        # Normalize gates
        total_gate = linear_gate + window_gate
        linear_gate = linear_gate / (total_gate + 1e-6)
        window_gate = window_gate / (total_gate + 1e-6)

        # Fuse outputs
        output = linear_gate * linear_out + window_gate * window_out

        return self.norm(output)
