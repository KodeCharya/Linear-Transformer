import torch
import torch.nn as nn
from typing import Optional


class RotaryPositionalEmbedding(nn.Module):
    """RoPE (Rotary Position Embedding) for efficient position encoding."""

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """Apply rotary positional embedding."""
        if seq_len is None:
            seq_len = x.shape[1]

        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)

        # Expand to match input dimensions
        emb = torch.cat([freqs, freqs], dim=-1)

        # Reshape for broadcasting
        batch_size = x.shape[0]
        emb = emb.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)

        return emb


class FeedForward(nn.Module):
    """Feed-forward network with gating."""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 4

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Single transformer block with linear attention and feedforward.

    Architecture:
    1. LayerNorm -> Linear Attention -> Residual
    2. LayerNorm -> FeedForward -> Residual
    """

    def __init__(self, dim: int, num_heads: int = 8, kernel_type: str = 'elu',
                 dropout: float = 0.1, use_hybrid: bool = False, window_size: int = 64):
        super().__init__()
        self.dim = dim
        self.use_hybrid = use_hybrid

        # Attention layer
        if use_hybrid:
            from core.hybrid_attention import HybridAttention
            self.attention = HybridAttention(dim, num_heads, window_size, kernel_type, dropout)
        else:
            from core.linear_attention import LinearAttention
            self.attention = LinearAttention(dim, num_heads, kernel_type, dropout)

        # Feed-forward network
        self.ff = FeedForward(dim, dim * 4, dropout)

        # Layer normalization and residual connections
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformer block."""
        # Attention with residual
        if self.use_hybrid:
            attn_out = self.attention(self.norm1(x))
        else:
            attn_out, _ = self.attention(self.norm1(x))
        x = x + attn_out

        # Feed-forward with residual
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out

        return x


class LinearTransformer(nn.Module):
    """
    Complete Linear Transformer model with configurable depth and parameters.

    Features:
    - Linear-time attention mechanism
    - Configurable sliding window hybrid mode
    - RoPE positional embeddings
    - Multi-layer stacking
    """

    def __init__(self, vocab_size: int, dim: int = 512, num_layers: int = 6,
                 num_heads: int = 8, kernel_type: str = 'elu', dropout: float = 0.1,
                 use_hybrid: bool = False, window_size: int = 64, max_seq_len: int = 4096):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # Token embeddings
        self.token_embed = nn.Embedding(vocab_size, dim)

        # Positional embeddings (RoPE)
        self.pos_embed = RotaryPositionalEmbedding(dim, max_seq_len)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, kernel_type, dropout, use_hybrid, window_size)
            for _ in range(num_layers)
        ])

        # Output layer
        self.norm = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize embeddings
        nn.init.normal_(self.token_embed.weight, mean=0, std=dim ** -0.5)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transformer.

        Args:
            input_ids: Token indices (batch, seq_len)

        Returns:
            logits: Predicted token logits (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape

        # Token embedding
        x = self.token_embed(input_ids)
        x = x * (self.dim ** 0.5)

        # Add positional information via attention mechanism
        # (RoPE is applied within attention, not here)
        x = self.dropout(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)

        # Output projection
        x = self.norm(x)
        logits = self.output_proj(x)

        return logits

    def generate(self, input_ids: torch.Tensor, max_length: int = 100,
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Starting tokens (batch, seq_len)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter

        Returns:
            generated_ids: Generated token sequence (batch, seq_len + max_length)
        """
        device = input_ids.device

        for _ in range(max_length):
            # Get model predictions
            logits = self.forward(input_ids[:, -self.max_seq_len:])
            next_token_logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Apply top-p filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumsum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumsum_probs > top_p
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[:, indices_to_remove] = float('-inf')

            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)

        return input_ids
