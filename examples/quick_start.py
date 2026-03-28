"""
Quick Start Example for Linear Transformer

Demonstrates:
1. Model creation
2. Forward pass
3. Text generation
"""

import torch
from core.transformer import LinearTransformer
from core.kernels import get_kernel
from data.tokenizer import SimpleTokenizer


def example_basic_forward():
    """Basic forward pass example."""
    print("=== Basic Forward Pass ===\n")

    # Create model
    model = LinearTransformer(
        vocab_size=256,
        dim=64,
        num_layers=2,
        num_heads=4,
        kernel_type='elu'
    )

    # Create random input
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, 256, (batch_size, seq_len))

    # Forward pass
    logits = model(input_ids)

    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Output range: [{logits.min():.2f}, {logits.max():.2f}]")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")


def example_kernel_functions():
    """Demonstrate kernel functions."""
    print("=== Kernel Functions ===\n")

    x = torch.randn(10, 32)

    for kernel_name in ['relu', 'elu', 'identity']:
        kernel = get_kernel(kernel_name)
        output = kernel.apply(x)

        print(f"{kernel_name.upper()} Kernel:")
        print(f"  Input range: [{x.min():.2f}, {x.max():.2f}]")
        print(f"  Output range: [{output.min():.2f}, {output.max():.2f}]")
        print(f"  Min value: {output.min():.4f}")
        print()


def example_generation():
    """Generate text with Linear Transformer."""
    print("=== Text Generation ===\n")

    # Create small model for demonstration
    model = LinearTransformer(
        vocab_size=256,
        dim=32,
        num_layers=1,
        num_heads=2,
        max_seq_len=128
    )

    # Tokenizer
    tokenizer = SimpleTokenizer(vocab_size=256)

    # Prompt
    prompt = "Hello"
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

    print(f"Prompt: {prompt}")
    print(f"Prompt tokens: {input_ids.shape}")

    # Generate
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_length=50,
            temperature=1.0,
            top_p=0.9
        )

    generated_text = tokenizer.decode(generated[0].tolist())
    print(f"\nGenerated ({len(generated_text)} chars):")
    print(f"{generated_text}\n")


def example_memory_efficiency():
    """Show memory efficiency of linear attention."""
    print("=== Memory Efficiency ===\n")

    print("Sequence length: 10,000 tokens")
    print("Model dimension: 512")
    print()

    # Standard attention memory
    seq_len = 10_000
    dim = 512
    bytes_per_param = 4

    standard_attn_memory = seq_len * seq_len * bytes_per_param / (1024 * 1024)
    linear_attn_memory = dim * dim * bytes_per_param / (1024 * 1024)

    print(f"Standard Attention Memory: {standard_attn_memory:.1f} MB")
    print(f"Linear Attention Memory: {linear_attn_memory:.2f} MB")
    print(f"Reduction: {standard_attn_memory / linear_attn_memory:.0f}x\n")


def example_hybrid_attention():
    """Demonstrate hybrid attention."""
    print("=== Hybrid Attention ===\n")

    # Model with hybrid attention
    model = LinearTransformer(
        vocab_size=256,
        dim=64,
        num_layers=2,
        num_heads=4,
        use_hybrid=True,
        window_size=64
    )

    input_ids = torch.randint(0, 256, (1, 256))
    logits = model(input_ids)

    print(f"Hybrid Attention Configuration:")
    print(f"  Local window size: 64 tokens")
    print(f"  Global context: Linear attention for older tokens")
    print(f"  Sequence length: {input_ids.shape[1]} tokens")
    print(f"  Output shape: {logits.shape}\n")


if __name__ == '__main__':
    example_basic_forward()
    example_kernel_functions()
    example_memory_efficiency()
    example_hybrid_attention()
    example_generation()

    print("=" * 50)
    print("All examples completed successfully!")
