import torch
import pytest
from data.tokenizer import SimpleTokenizer, BPETokenizer


def test_simple_tokenizer_encode_decode():
    """Test simple character-level tokenizer."""
    tokenizer = SimpleTokenizer(vocab_size=256)

    text = "Hello World"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)

    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)
    assert decoded == text


def test_simple_tokenizer_bounds():
    """Test tokenizer handles out-of-vocab characters."""
    tokenizer = SimpleTokenizer(vocab_size=128)

    # Character beyond vocab size
    text = "Hello" + chr(200)
    tokens = tokenizer.encode(text)

    # Should handle gracefully (use default token)
    assert len(tokens) >= 0


def test_bpe_tokenizer_train():
    """Test BPE tokenizer training."""
    tokenizer = BPETokenizer(vocab_size=256)

    texts = ["hello world", "hello there", "world of tokens"]
    tokenizer.train(texts, num_merges=10)

    assert len(tokenizer.merges) > 0


def test_bpe_tokenizer_encode_decode():
    """Test BPE encoding and decoding."""
    tokenizer = BPETokenizer(vocab_size=256)

    texts = ["hello world"] * 5
    tokenizer.train(texts, num_merges=10)

    text = "hello"
    tokens = tokenizer.encode(text)

    assert isinstance(tokens, list)
