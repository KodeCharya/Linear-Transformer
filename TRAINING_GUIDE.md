# Training Guide for Linear Transformer

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Make sure `.env` is configured with Supabase credentials (already set).

### 3. Prepare Data

Create training data files or use built-in sample data:

```python
from data.tokenizer import SimpleTokenizer
from data.dataset import create_data_loaders

texts = [
    "Your training text 1",
    "Your training text 2",
    ...
]

tokenizer = SimpleTokenizer(vocab_size=256)
train_loader, val_loader = create_data_loaders(texts, tokenizer, seq_len=256)
```

## Training

### Basic Training

```bash
python main.py --mode train --epochs 5 --batch_size 32
```

### Advanced Training

```bash
python main.py --mode train \
    --epochs 20 \
    --batch_size 64 \
    --model_dim 256 \
    --num_layers 6 \
    --num_heads 8 \
    --kernel_type elu \
    --use_hybrid \
    --window_size 128 \
    --learning_rate 1e-3
```

### With Monitoring

Training metrics are automatically logged to:
1. Console output
2. JSON file: `checkpoints/linear_transformer/training_history.json`
3. Supabase database (if available)

### Hyperparameter Guide

| Parameter | Range | Notes |
|-----------|-------|-------|
| `model_dim` | 64-1024 | Larger = more capacity but slower |
| `num_layers` | 1-24 | 6-12 typical for language models |
| `num_heads` | 2-16 | Usually dim/num_heads = 64 |
| `kernel_type` | relu/elu | ELU more stable |
| `learning_rate` | 1e-4 to 1e-2 | Start with 1e-3 |
| `batch_size` | 8-256 | Larger = faster but more memory |
| `seq_len` | 128-2048 | Longer = more context but slower |
| `window_size` | 32-256 | For hybrid attention |

## Checkpoint Management

### Saving Checkpoints

Checkpoints are automatically saved to `checkpoints/linear_transformer/`:
- `best_model.pt`: Best validation loss checkpoint
- `training_history.json`: Loss curves

### Loading Checkpoints

```bash
python main.py --mode generate \
    --checkpoint checkpoints/linear_transformer/best_model.pt \
    --prompt "Hello world"
```

Or programmatically:

```python
from training.trainer import LinearTransformerTrainer

trainer = LinearTransformerTrainer(model, device)
trainer.load_checkpoint("checkpoints/linear_transformer/best_model.pt")
```

## Troubleshooting

### Problem: Out of Memory

**Solution**: Reduce batch size or sequence length
```bash
python main.py --mode train --batch_size 8 --seq_len 128
```

### Problem: Training Loss Not Decreasing

**Solution**:
- Increase model size
- Reduce learning rate
- Check data quality
- Train longer

```bash
python main.py --mode train \
    --epochs 20 \
    --model_dim 256 \
    --learning_rate 1e-4
```

### Problem: NaN in Loss

**Solution**: Enable gradient clipping (default: 1.0)
- Check data for invalid values
- Use smaller learning rate

## Performance Tips

1. **Use GPU**: 10-100x faster than CPU
2. **Batch Training**: Larger batches use resources more efficiently
3. **Sequence Length**: Shorter sequences train faster
4. **Mixed Precision**: Use `torch.cuda.amp` for 2x speedup
5. **Gradient Accumulation**: Train on larger effective batches

### Hardware Requirements

| Model Size | Batch | Seq Len | GPU VRAM | Time/Epoch |
|------------|-------|---------|----------|-----------|
| Small (64d, 2L) | 64 | 256 | 2GB | 10s |
| Medium (256d, 6L) | 32 | 512 | 8GB | 1m |
| Large (512d, 12L) | 16 | 1024 | 24GB | 5m |

## Validation Strategy

1. **Train/Val Split**: 80/20 recommended
2. **Validation Metrics**:
   - Loss: Lower is better
   - Perplexity: exp(loss), lower is better
   - Human evaluation: Sample generated text

3. **Early Stopping**:
   ```python
   if val_loss < best_val_loss:
       best_val_loss = val_loss
       save_checkpoint()
   ```

## Data Preparation Tips

### Tokenization

```python
from data.tokenizer import SimpleTokenizer, BPETokenizer

# Simple character-level
tokenizer = SimpleTokenizer(vocab_size=256)

# Or byte-pair encoding
tokenizer = BPETokenizer(vocab_size=10000)
tokenizer.train(texts, num_merges=10000)
```

### Data Loading

```python
from data.dataset import create_data_loaders

# From text list
train_loader, val_loader = create_data_loaders(
    texts,
    tokenizer,
    seq_len=256,
    batch_size=32
)

# From files
from data.dataset import FileDataset
dataset = FileDataset(
    'data/raw/',
    tokenizer,
    seq_len=256,
    max_files=100
)
```

## Experiment Tracking with Supabase

### Log Metrics

Metrics are automatically tracked in the database:

```python
db = SupabaseClient()

# View your experiments
runs = db.list_training_runs(limit=10)
for run in runs:
    print(f"Run {run['id']}: {run['status']}")
    stats = db.get_run_statistics(run['id'])
    print(f"  Best val loss: {stats['best_val_loss']:.4f}")
```

### Compare Models

```python
# Get all model configs
configs = db.client.table('model_configurations').select('*').execute().data

# Compare training runs
for config in configs:
    runs = db.client.table('training_runs') \
        .select('*') \
        .eq('model_config_id', config['id']) \
        .execute().data

    for run in runs:
        stats = db.get_run_statistics(run['id'])
        print(f"{config['name']}: {stats['final_val_perplexity']:.2f} ppl")
```

## Advanced Techniques

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop
with autocast():
    output = model(input_ids)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Gradient Accumulation

```python
accumulation_steps = 4

for batch_idx, (input_ids, targets) in enumerate(train_loader):
    output = model(input_ids)
    loss = criterion(output, targets) / accumulation_steps
    loss.backward()

    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Multi-GPU Training

```python
model = LinearTransformer(...)
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

trainer = LinearTransformerTrainer(model, device='cuda:0')
```

## Next Steps

1. Start with small model for quick iteration
2. Validate on your specific task
3. Scale up model size as needed
4. Fine-tune hyperparameters based on validation
5. Deploy best checkpoint for inference
