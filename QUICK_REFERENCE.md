# Linear Transformer - Quick Reference

## Installation & Setup

```bash
pip install -r requirements.txt
```

## Basic Commands

### Training
```bash
# Quick train (3 epochs, small model)
python main.py --mode train

# Full training
python main.py --mode train \
    --epochs 10 \
    --batch_size 32 \
    --model_dim 128 \
    --num_layers 4 \
    --learning_rate 1e-3
```

### Generation
```bash
python main.py --mode generate \
    --prompt "Your prompt here" \
    --max_length 100 \
    --temperature 0.8
```

### Examples
```bash
python examples/quick_start.py
python examples/advanced_usage.py
```

## Model Configuration

```python
from core.transformer import LinearTransformer

model = LinearTransformer(
    vocab_size=256,        # Token vocabulary
    dim=128,               # Model dimension
    num_layers=4,          # Number of layers
    num_heads=8,           # Attention heads
    kernel_type='elu',     # 'relu', 'elu', or 'identity'
    use_hybrid=False,      # Enable hybrid attention
    window_size=64,        # Window size for hybrid
    max_seq_len=4096       # Max sequence length
)
```

## Data Preparation

```python
from data.tokenizer import SimpleTokenizer
from data.dataset import create_data_loaders

# Tokenizer
tokenizer = SimpleTokenizer(vocab_size=256)

# Create data loaders
texts = ["sample text 1", "sample text 2"]
train_loader, val_loader = create_data_loaders(
    texts, tokenizer, seq_len=256, batch_size=32
)
```

## Training

```python
from core.transformer import LinearTransformer
from training.trainer import LinearTransformerTrainer

model = LinearTransformer(vocab_size=256)
trainer = LinearTransformerTrainer(model, device='cuda')

history = trainer.train(
    train_loader, val_loader,
    num_epochs=10,
    learning_rate=1e-3,
    checkpoint_dir='checkpoints/'
)

trainer.save_history('training_history.json')
```

## Generation

```python
from core.transformer import LinearTransformer
from inference.generator import TextGenerator
from data.tokenizer import SimpleTokenizer

model = LinearTransformer(vocab_size=256)
tokenizer = SimpleTokenizer()
generator = TextGenerator(model, tokenizer)

# Generate text
text = generator.generate(
    "Starting prompt",
    max_length=100,
    temperature=0.8,
    top_p=0.9
)
```

## Database Integration

```python
from db.supabase_client import SupabaseClient

db = SupabaseClient()

# Save configuration
config_id = db.save_model_config(config, "model_v1")

# Create training run
run_id = db.create_training_run("my_model", config_id, training_config)

# Log metrics
db.save_training_metrics(run_id, epoch, train_loss, val_loss, perp, time, lr)

# Get statistics
stats = db.get_run_statistics(run_id)
```

## Module Reference

### core/kernels.py
```python
from core.kernels import get_kernel

kernel = get_kernel('elu')      # 'relu', 'elu', 'identity'
features = kernel.apply(x)      # Ensures positivity
```

### core/accumulator.py
```python
from core.accumulator import LinearAccumulator, PrefixSumAccumulator

# Streaming updates
acc = LinearAccumulator(dim=512)
context, norm = acc.update(keys, values)

# Parallel training
kv_sums, k_sums = PrefixSumAccumulator.compute_prefix_sums(keys, values)
outputs = PrefixSumAccumulator.compute_outputs(queries, kv_sums, k_sums)
```

### core/linear_attention.py
```python
from core.linear_attention import LinearAttention, MultiHeadLinearAttention

# Create attention layer
attn = LinearAttention(dim=512, num_heads=8)
output, _ = attn(x, is_inference=False)

# Or multi-head wrapper
attn = MultiHeadLinearAttention(dim=512, num_heads=8)
output = attn(x)
```

### core/hybrid_attention.py
```python
from core.hybrid_attention import HybridAttention, SlidingWindowAttention, ContextFusionLayer

# Hybrid attention
hybrid = HybridAttention(dim=512, window_size=64)
output = hybrid(x)

# Sliding window only
window = SlidingWindowAttention(dim=512, window_size=64)
output = window(x)

# Learned fusion
fusion = ContextFusionLayer(dim=512)
output = fusion(x)
```

### core/transformer.py
```python
from core.transformer import LinearTransformer, TransformerBlock

# Full model
model = LinearTransformer(vocab_size=256, dim=128, num_layers=6)
logits = model(input_ids)
generated = model.generate(input_ids, max_length=100)

# Individual block
block = TransformerBlock(dim=128, num_heads=8)
output = block(x)
```

### data/tokenizer.py
```python
from data.tokenizer import SimpleTokenizer, BPETokenizer

# Character tokenizer
tokenizer = SimpleTokenizer(vocab_size=256)
tokens = tokenizer.encode("text")
text = tokenizer.decode(tokens)

# BPE tokenizer
tokenizer = BPETokenizer(vocab_size=10000)
tokenizer.train(texts, num_merges=1000)
```

### data/dataset.py
```python
from data.dataset import TextDataset, FileDataset, create_data_loaders

# From texts
train_loader, val_loader = create_data_loaders(texts, tokenizer)

# From files
dataset = FileDataset('data/', tokenizer, seq_len=256, max_files=100)
```

### training/trainer.py
```python
from training.trainer import LinearTransformerTrainer

trainer = LinearTransformerTrainer(model, device='cuda')
history = trainer.train(train_loader, val_loader, num_epochs=10)
trainer.save_checkpoint('model.pt')
trainer.load_checkpoint('model.pt')
```

### inference/generator.py
```python
from inference.generator import TextGenerator

generator = TextGenerator(model, tokenizer)
text = generator.generate("prompt", max_length=100)
batch_texts = generator.batch_generate(["p1", "p2"], max_length=100)
```

### db/supabase_client.py
```python
from db.supabase_client import SupabaseClient

db = SupabaseClient()
db.save_model_config(config, name)
db.create_training_run(model_name, config_id, training_config)
db.save_training_metrics(run_id, epoch, train_loss, val_loss, perp, time, lr)
db.get_run_statistics(run_id)
```

## Key Parameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| model_dim | 128 | 64-1024 | Model dimension |
| num_layers | 4 | 1-24 | Number of transformer layers |
| num_heads | 8 | 2-16 | Attention heads (must divide dim) |
| kernel_type | 'elu' | relu/elu/identity | Feature mapping |
| batch_size | 32 | 1-256 | Batch size for training |
| seq_len | 256 | 64-4096 | Sequence length |
| learning_rate | 1e-3 | 1e-4 to 1e-2 | Learning rate |
| window_size | 64 | 32-256 | Sliding window size |
| max_seq_len | 4096 | 256+ | Max generation length |

## Performance Tips

1. **Use GPU**: 10-100x faster than CPU
2. **Larger Batches**: More efficient resource usage
3. **Shorter Sequences**: Faster training
4. **Hybrid Mode**: Better accuracy on long sequences
5. **Warm Cache**: First run slower than subsequent

## File Organization

| File | Purpose |
|------|---------|
| main.py | Entry point for training/generation |
| core/*.py | Core transformer components |
| data/*.py | Data loading and preprocessing |
| training/*.py | Training infrastructure |
| inference/*.py | Generation system |
| db/*.py | Database integration |
| tests/*.py | Unit tests |
| examples/*.py | Usage examples |

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Out of Memory | Reduce batch_size or seq_len |
| Slow Training | Use GPU, increase batch_size |
| NaN Loss | Reduce learning_rate, check data |
| Poor Generation | Train longer, use larger model |
| Slow Inference | Use GPU, reduce seq_len |

## Next Steps

1. Start with `examples/quick_start.py`
2. Try `python main.py --mode train --epochs 3`
3. Generate with `python main.py --mode generate --prompt "test"`
4. Read TRAINING_GUIDE.md for advanced usage
5. Check ARCHITECTURE.md for technical details

## Support

- README.md: Overview and quick start
- ARCHITECTURE.md: Mathematical foundations
- TRAINING_GUIDE.md: Detailed training instructions
- IMPLEMENTATION_SUMMARY.md: Implementation details
- SYSTEM_OVERVIEW.txt: Complete system overview

---

**Production Ready**: Yes
**GPU Support**: Yes
**Database Integration**: Yes
**Tests**: Yes
**Documentation**: Comprehensive
