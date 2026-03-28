import torch
import pytest
from core.kernels import ReluKernel, EluKernel, get_kernel


def test_relu_kernel():
    """Test ReLU kernel ensures positivity."""
    x = torch.randn(4, 64)
    output = ReluKernel.apply(x)
    assert (output >= 0).all()


def test_elu_kernel():
    """Test ELU kernel ensures positivity."""
    x = torch.randn(4, 64)
    output = EluKernel.apply(x)
    assert (output >= 0).all()


def test_kernel_factory():
    """Test kernel factory function."""
    kernel = get_kernel('relu')
    assert kernel == ReluKernel

    kernel = get_kernel('elu')
    assert kernel == EluKernel

    with pytest.raises(ValueError):
        get_kernel('invalid')
