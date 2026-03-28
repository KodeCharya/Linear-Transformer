"""
Advanced Usage Examples

Demonstrates:
1. Custom architectures
2. Batch generation
3. Model checkpointing
4. Performance profiling
"""

import torch
import time
from core.transformer import LinearTransformer
from core.linear_attention import LinearAttention
from core.accumulator import LinearAccumulator
from data.tokenizer import SimpleTokenizer
from inference.generator import TextGenerator


def example_complex_architecture():
    """Create and use a complex model architecture."""
    print("=== Complex Architecture ===\n")

    # Large model with multiple configurations
    configs = [
        {
            'name': 'Lightweight',
            'dim': 64,
            'num_layers': 2,
            'num_heads': 2,
        },
        {
            'name': 'Standard',
            'dim': 256,
            'num_layers': 6,
            'num_heads': 8,
        },
        {
            'name': 'Large',
            'dim': 512,
            'num_layers': 12,
            'num_heads': 16,
        },
    ]

    for config in configs:
        model = LinearTransformer(
            vocab_size=256,
            dim=config['dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            use_hybrid=True,
            window_size=64
        )

        params = sum(p.numel() for p in model.parameters())
        print(f"{config['name']}:")
        print(f"  Dimension: {config['dim']}")
        print(f"  Layers: {config['num_layers']}")
        print(f"  Heads: {config['num_heads']}")
        print(f"  Parameters: {params:,}")
        print()


def example_batch_generation():
    """Generate text for multiple prompts simultaneously."""
    print("=== Batch Generation ===\n")

    model = LinearTransformer(
        vocab_size=256,
        dim=64,
        num_layers=2,
        num_heads=4,
        max_seq_len=256
    )

    tokenizer = SimpleTokenizer(vocab_size=256)
    generator = TextGenerator(model, tokenizer, device='cpu')

    prompts = [
        "The future",
        "Artificial intelligence",
        "Linear attention"
    ]

    print(f"Generating for {len(prompts)} prompts in batch:\n")

    for prompt in prompts:
        print(f"Prompt: {prompt}")

    # In practice, use batch_generate for efficiency
    # results = generator.batch_generate(prompts, max_length=50)
    print("\n(Batch generation would process all prompts simultaneously)\n")


def example_streaming_inference():
    """Demonstrate streaming/sequential inference."""
    print("=== Streaming Inference ===\n")

    batch_size = 1
    dim = 64
    seq_len = 100

    # Create attention layer
    attention = LinearAttention(dim=dim, num_heads=4)

    # Simulate streaming tokens
    print(f"Processing {seq_len} tokens sequentially")
    print(f"Model dimension: {dim}")
    print()

    with torch.no_grad():
        for t in range(seq_len):
            # Single token at position t
            x = torch.randn(batch_size, 1, dim)

            # Process (would maintain state in real streaming)
            output, _ = attention(x, is_inference=False)

            if (t + 1) % 20 == 0:
                print(f"Processed token {t + 1}/{seq_len}")

    print("\nStreaming inference completed\n")


def example_performance_profiling():
    """Profile model performance."""
    print("=== Performance Profiling ===\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    test_configs = [
        {'seq_len': 128, 'dim': 64},
        {'seq_len': 256, 'dim': 128},
        {'seq_len': 512, 'dim': 256},
    ]

    print("Throughput (tokens/second):\n")

    for config in test_configs:
        model = LinearTransformer(
            vocab_size=256,
            dim=config['dim'],
            num_layers=2,
            num_heads=4
        ).to(device)

        input_ids = torch.randint(0, 256, (2, config['seq_len']), device=device)

        # Warmup
        with torch.no_grad():
            _ = model(input_ids)

        # Time forward pass
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()

        with torch.no_grad():
            for _ in range(10):
                _ = model(input_ids)

        torch.cuda.synchronize() if device == 'cuda' else None
        elapsed = time.time() - start

        tokens_per_sec = (10 * 2 * config['seq_len']) / elapsed
        print(f"Seq len: {config['seq_len']:4d}, Dim: {config['dim']:3d} "
              f"| {tokens_per_sec:8.0f} tokens/sec")

    print()


def example_memory_profiling():
    """Profile memory usage."""
    print("=== Memory Profiling ===\n")

    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.reset_peak_memory_stats()
    else:
        device = 'cpu'

    print(f"Device: {device}\n")

    model = LinearTransformer(
        vocab_size=256,
        dim=256,
        num_layers=4,
        num_heads=8
    ).to(device)

    # Forward pass
    input_ids = torch.randint(0, 256, (1, 512), device=device)

    with torch.no_grad():
        _ = model(input_ids)

    if device == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak GPU Memory: {peak_memory:.1f} MB")
    else:
        print("Memory profiling requires CUDA")

    print()


if __name__ == '__main__':
    example_complex_architecture()
    example_batch_generation()
    example_streaming_inference()
    example_performance_profiling()
    # example_memory_profiling()  # Requires CUDA

    print("=" * 50)
    print("Advanced examples completed successfully!")
