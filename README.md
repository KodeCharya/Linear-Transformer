# Linear Transformer Implementation

A complete implementation of Linear Transformers with O(N) complexity, enabling efficient sequence processing with constant memory requirements.

## Architecture Overview

### Core Components

1. **Kernel Functions** (`core/kernels.py`)
   - Feature mapping to ensure positivity
   - Supports ReLU, ELU, and identity kernels
   - Prevents information cancellation in accumulator

2. **Linear Accumulator** (`core/accumulator.py`)
   - Fixed-size state for streaming computation
   - Stores key-value products (D x D matrix)
   - Maintains normalization factors
   - Supports both sequential and parallel computation

3. **Linear Attention** (`core/linear_attention.py`)
   - O(N) complexity compared to O(N²) standard attention
   - Multi-head support with independent accumulators
   - Kernel-based feature mapping for stability
   - Training path: parallel prefix sums for efficiency
   - Inference path: sequential updates for streaming

4. **Hybrid Attention** (`core/hybrid_attention.py`)
   - Sliding window attention for local context
   - Linear attention for global context
   - Automatic blending of short-term and long-term dependencies
   - Context fusion with learnable gates

5. **Transformer Model** (`core/transformer.py`)
   - Multi-layer transformer blocks
   - Rotary positional embeddings (RoPE)
   - Feedforward networks with residual connections
   - Text generation with sampling strategies

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
# Basic training with default parameters
python main.py --mode train --epochs 3 --batch_size 16

# Advanced training with hybrid attention
python main.py --mode train \
    --epochs 5 \
    --batch_size 32 \
    --model_dim 128 \
    --num_layers 4 \
    --num_heads 8 \
    --use_hybrid \
    --window_size 64 \
    --learning_rate 1e-3
```

### Generation

```bash
# Generate from trained model
python main.py --mode generate \
    --prompt "The future of AI" \
    --max_length 200 \
    --temperature 0.8 \
    --top_p 0.9 \
    --checkpoint checkpoints/linear_transformer/best_model.pt
```

## Key Features

### Efficiency Characteristics

- **Quadratic to Linear**: O(N²) → O(N) complexity
- **Constant VRAM**: Memory usage independent of sequence length
- **Streaming Inference**: Process documents of any length without reloading
- **Parallel Training**: Efficient batch training with prefix sums

### Stability Mechanisms

1. **Kernel Function**: Ensures all features positive to prevent cancellation
2. **Normalization**: Parallel denominator tracking prevents output explosion
3. **Numerical Safety**: Epsilon-guarded division for stability

### Hybrid Approach

- **Local Window**: Recent tokens use standard attention (128 token window)
- **Global Context**: Older tokens use linear attention
- **Adaptive Blending**: Learned mixing of attention types
- **Perfect Recall**: Never forgets any historical context

## Configuration

### Model Parameters

```python
model = LinearTransformer(
    vocab_size=256,           # Token vocabulary size
    dim=512,                  # Model dimension
    num_layers=6,             # Transformer blocks
    num_heads=8,              # Attention heads
    kernel_type='elu',        # Feature mapping: 'relu', 'elu', 'identity'
    use_hybrid=False,         # Enable sliding window hybrid attention
    window_size=64,           # Local attention window size
    max_seq_len=4096          # Maximum sequence length
)
```

### Training Parameters

```python
trainer.train(
    train_loader,
    val_loader,
    num_epochs=10,
    learning_rate=1e-3,
    weight_decay=1e-5,
    warmup_epochs=1,
    gradient_clip=1.0,
    log_interval=100
)
```

## Database Integration

The implementation integrates with Supabase for tracking experiments:

### Tables

- `model_configurations`: Store model architecture configs
- `training_runs`: Track training experiments
- `training_metrics`: Per-epoch metrics (loss, perplexity, time)
- `model_checkpoints`: Checkpoint metadata and paths

### Usage

```python
from db.supabase_client import SupabaseClient

db = SupabaseClient()

# Save model configuration
config_id = db.save_model_config(config, "my_model_v1")

# Create training run
run_id = db.create_training_run("my_model", config_id, training_config)

# Log metrics during training
db.save_training_metrics(run_id, epoch, train_loss, val_loss, perplexity, epoch_time, lr)

# Get training statistics
stats = db.get_run_statistics(run_id)
```

## Performance Comparison

### Complexity Analysis

| Operation | Standard Attention | Linear Attention |
|-----------|-------------------|-----------------|
| Time Complexity | O(N²) | O(N) |
| Space Complexity | O(N²) | O(1) |
| Seq Length: 1K | 1.0M ops | 1.0K ops |
| Seq Length: 10K | 100M ops | 10K ops |
| Seq Length: 100K | 10B ops | 100K ops |

### Memory Usage

At sequence length 10,000:
- Standard Attention: 10,000² × 4 bytes ≈ 400 MB
- Linear Attention: 512 × 512 × 4 bytes ≈ 1 MB

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_transformer.py -v

# Run with coverage
pytest tests/ --cov=core --cov-report=html
```

## Project Structure

```
project/
├── core/                 # Core transformer components
│   ├── kernels.py       # Feature mapping kernels
│   ├── accumulator.py   # Linear accumulator state
│   ├── linear_attention.py  # Linear attention layer
│   ├── hybrid_attention.py  # Hybrid attention mechanism
│   └── transformer.py   # Complete transformer model
├── data/                # Data loading and preprocessing
│   ├── tokenizer.py     # Character and BPE tokenizers
│   └── dataset.py       # Text datasets
├── training/            # Training infrastructure
│   └── trainer.py       # Training loop and metrics
├── inference/           # Generation and inference
│   └── generator.py     # Text generation with sampling
├── db/                  # Database integration
│   └── supabase_client.py
├── tests/               # Unit tests
└── main.py             # Training and generation script
```

## Advanced Usage

### Custom Kernel Functions

```python
from core.kernels import KernelFunction

class CustomKernel(KernelFunction):
    @staticmethod
    def apply(x):
        # Your custom feature mapping
        return torch.sigmoid(x) * 2 - 1  # Example

kernel = CustomKernel()
output = kernel.apply(features)
```

### Streaming Inference

```python
from core.accumulator import LinearAccumulator

# For streaming token-by-token generation
accumulator = LinearAccumulator(dim=512, device='cuda')

for token_id in tokens:
    # Process one token at a time
    # Accumulator state is updated internally
    output, _ = attention(token_embedding, is_inference=True, accumulator=accumulator)
```

### Multi-GPU Training

```python
model = LinearTransformer(...)
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

trainer = LinearTransformerTrainer(model, device='cuda:0')
trainer.train(train_loader, val_loader, ...)
```

## Research Applications

1. **Long Document Processing**: Handle documents longer than standard memory allows
2. **Real-time Streaming**: Process continuous streams with constant memory
3. **Mobile Deployment**: Inference on edge devices with limited memory
4. **Knowledge Distillation**: Use linear attention as student model
5. **Retrieval Augmented Generation**: Efficient context incorporation

## Future Enhancements

- Flash Attention optimizations for linear kernels
- Multi-query attention for improved training speed
- Sparse linear attention for selective information flow
- Learnable kernel functions via neural networks
- Mixed precision training (FP16/BF16)

## References

- Katharopoulos et al. (2020): "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
- Su et al. (2021): "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- Dao et al. (2022): "FlashAttention: Fast and Memory-Efficient Exact Attention"

## License

MIT License
