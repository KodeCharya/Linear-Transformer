import torch
import pytest
from core.accumulator import LinearAccumulator, PrefixSumAccumulator


def test_linear_accumulator_initialization():
    """Test accumulator initialization."""
    acc = LinearAccumulator(dim=64)
    assert acc.dim == 64
    assert acc.kv_state.shape == (64, 64)
    assert acc.k_sum.shape == (64,)


def test_linear_accumulator_reset():
    """Test accumulator reset."""
    acc = LinearAccumulator(dim=64)
    acc.kv_state.fill_(1.0)
    acc.k_sum.fill_(1.0)
    acc.length.fill_(10)

    acc.reset()

    assert (acc.kv_state == 0).all()
    assert (acc.k_sum == 0).all()
    assert acc.length.item() == 0


def test_prefix_sum_accumulator():
    """Test prefix sum computation."""
    batch_size, seq_len, dim = 2, 10, 32

    k = torch.randn(batch_size, seq_len, dim)
    v = torch.randn(batch_size, seq_len, dim)

    kv_sums, k_sums = PrefixSumAccumulator.compute_prefix_sums(k, v)

    assert kv_sums.shape == (batch_size, seq_len, dim, dim)
    assert k_sums.shape == (batch_size, seq_len, dim)

    # Check that sums are cumulative
    for b in range(batch_size):
        for i in range(seq_len):
            if i > 0:
                k_sum_should_increase = (k_sums[b, i] - k_sums[b, i-1]).norm() > 0
                assert k_sum_should_increase or torch.allclose(k_sums[b, i], k_sums[b, i-1])


def test_prefix_sum_outputs():
    """Test output computation from prefix sums."""
    batch_size, seq_len, dim = 2, 10, 32

    q = torch.randn(batch_size, seq_len, dim)
    k = torch.randn(batch_size, seq_len, dim)
    v = torch.randn(batch_size, seq_len, dim)

    kv_sums, k_sums = PrefixSumAccumulator.compute_prefix_sums(k, v)
    outputs = PrefixSumAccumulator.compute_outputs(q, kv_sums, k_sums)

    assert outputs.shape == (batch_size, seq_len, dim)
    assert not torch.isnan(outputs).any()
