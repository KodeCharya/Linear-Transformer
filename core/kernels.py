import torch
import torch.nn.functional as F


class KernelFunction:
    """Base class for feature map kernels."""

    @staticmethod
    def apply(x):
        raise NotImplementedError


class ReluKernel(KernelFunction):
    """ReLU-based kernel: ensures positive features."""

    @staticmethod
    def apply(x):
        return F.relu(x)


class EluKernel(KernelFunction):
    """ELU-based kernel: smoother positive projection."""

    @staticmethod
    def apply(x):
        return F.elu(x) + 1


class IdentityKernel(KernelFunction):
    """Identity kernel for testing."""

    @staticmethod
    def apply(x):
        return x


def get_kernel(kernel_type: str) -> KernelFunction:
    """Get kernel function by type."""
    kernels = {
        'relu': ReluKernel,
        'elu': EluKernel,
        'identity': IdentityKernel,
    }
    if kernel_type not in kernels:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    return kernels[kernel_type]


def numerically_stable_divide(numerator, denominator, eps=1e-6):
    """Divide with numerical stability."""
    return numerator / (denominator + eps)
