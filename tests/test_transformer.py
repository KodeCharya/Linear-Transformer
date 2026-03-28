import torch
import pytest
from core.transformer import LinearTransformer


def test_transformer_initialization():
    """Test transformer model initialization."""
    model = LinearTransformer(
        vocab_size=256,
        dim=64,
        num_layers=2,
        num_heads=4
    )

    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 0


def test_transformer_forward():
    """Test transformer forward pass."""
    batch_size, seq_len = 2, 32
    model = LinearTransformer(
        vocab_size=256,
        dim=64,
        num_layers=2,
        num_heads=4
    )

    input_ids = torch.randint(0, 256, (batch_size, seq_len))
    logits = model(input_ids)

    assert logits.shape == (batch_size, seq_len, 256)
    assert not torch.isnan(logits).any()


def test_transformer_generation():
    """Test transformer text generation."""
    model = LinearTransformer(
        vocab_size=256,
        dim=32,
        num_layers=1,
        num_heads=2,
        max_seq_len=64
    )

    input_ids = torch.randint(0, 256, (1, 10))
    generated = model.generate(input_ids, max_length=20, temperature=1.0)

    assert generated.shape[0] == 1
    assert generated.shape[1] == 30


def test_transformer_with_hybrid_attention():
    """Test transformer with hybrid attention."""
    model = LinearTransformer(
        vocab_size=256,
        dim=64,
        num_layers=2,
        num_heads=4,
        use_hybrid=True,
        window_size=16
    )

    input_ids = torch.randint(0, 256, (2, 32))
    logits = model(input_ids)

    assert logits.shape == (2, 32, 256)
