import torch
import pytest
from core.linear_attention import LinearAttention, MultiHeadLinearAttention


def test_linear_attention_initialization():
    """Test linear attention layer initialization."""
    attn = LinearAttention(dim=64, num_heads=4)
    assert attn.dim == 64
    assert attn.num_heads == 4
    assert attn.head_dim == 16


def test_linear_attention_training_forward():
    """Test linear attention forward pass in training mode."""
    batch_size, seq_len, dim = 2, 32, 64
    attn = LinearAttention(dim=dim, num_heads=4)

    x = torch.randn(batch_size, seq_len, dim)
    output, _ = attn(x, is_inference=False)

    assert output.shape == (batch_size, seq_len, dim)
    assert not torch.isnan(output).any()


def test_multihead_linear_attention():
    """Test multi-head linear attention wrapper."""
    batch_size, seq_len, dim = 2, 32, 64
    attn = MultiHeadLinearAttention(dim=dim, num_heads=4)

    x = torch.randn(batch_size, seq_len, dim)
    output = attn(x)

    assert output.shape == (batch_size, seq_len, dim)
    assert not torch.isnan(output).any()


def test_linear_attention_gradient_flow():
    """Test that gradients flow through attention."""
    batch_size, seq_len, dim = 2, 16, 32
    attn = LinearAttention(dim=dim, num_heads=2)

    x = torch.randn(batch_size, seq_len, dim, requires_grad=True)
    output, _ = attn(x, is_inference=False)

    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert (x.grad != 0).any()
