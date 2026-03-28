# Linear Transformer Architecture

## Mathematical Foundation

### The Core Problem

Standard transformer attention computes:
```
Attention(Q, K, V) = softmax(QK^T / √d) V
```

This requires O(N²) memory and computation for sequence length N.

### The Solution: Kernel Trick

Linear attention reformulates as:
```
Linear(Q, K, V) = φ(Q) @ (φ(K)^T @ V) / (φ(Q) @ Σφ(K))
```

Where φ is a kernel function ensuring positive features.

**Key insight**: (K^T @ V) can be computed once and accumulated, reducing complexity to O(N).

## Component Architecture

### 1. Kernel Functions (core/kernels.py)

```python
class KernelFunction:
    @staticmethod
    def apply(x):
        # Ensures φ(x) > 0 to prevent cancellation
        pass
```

**Types**:
- `ReluKernel`: φ(x) = max(x, 0)
- `EluKernel`: φ(x) = elu(x) + 1
- `IdentityKernel`: φ(x) = x (for testing)

**Purpose**: Ensures all values in the accumulator are positive, preventing information cancellation.

### 2. Linear Accumulator (core/accumulator.py)

```python
class LinearAccumulator:
    def __init__(self, dim: int):
        self.kv_state = zeros(dim, dim)      # K^T @ V accumulation
        self.k_sum = zeros(dim)               # Σφ(K)
        self.length = 0                       # Token counter
```

**Operations**:

1. **Update** (O(d²) where d = model_dim):
   ```
   kv_state += K_t^T @ V_t
   k_sum += K_t
   ```

2. **Query** (O(d²)):
   ```
   output = kv_state @ φ(Q_t) / (k_sum @ φ(Q_t) + ε)
   ```

**Memory**: Always O(d²), independent of sequence length N.

### 3. Prefix Sum Accumulator (core/accumulator.py)

For parallel training, compute all positions simultaneously:

```
kv_sums[0..N] = cumsum(K^T @ V)
k_sums[0..N] = cumsum(K)
output[i] = kv_sums[i] @ Q[i] / (k_sums[i] @ Q[i] + ε)
```

**Complexity**: O(N·d²) but fully parallelizable.

### 4. Linear Attention Layer (core/linear_attention.py)

```python
class LinearAttention(nn.Module):
    def forward(self, x, is_inference=False):
        # Training path: use prefix sums (parallel)
        # Inference path: use accumulator (streaming)
```

**Multi-head processing**:
- Each head has independent accumulator
- Allows different attention patterns per head

**Stability**:
- Kernel function ensures positive features
- Epsilon-guarded division prevents NaN
- Normalization prevents output explosion

### 5. Hybrid Attention (core/hybrid_attention.py)

**Architecture**:
```
Input
  ├─→ SlidingWindowAttention (recent 64-128 tokens)
  │        │
  └─→ LinearAttention (entire history)
              │
              ↓ Learnable Blend
            Output
```

**Components**:

1. **SlidingWindowAttention**: Standard O(N·w²) for window size w
2. **LinearAttention**: O(N) for everything before window
3. **Blend Gates**: Learned mixture weights

**Advantages**:
- Local attention ensures detail preservation
- Linear attention provides full context
- Automatic weight learning for optimal balance

### 6. Transformer Model (core/transformer.py)

```python
class LinearTransformer(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, num_heads, ...):
        self.embed = Embedding(vocab_size, dim)
        self.layers = ModuleList([
            TransformerBlock(...) for _ in range(num_layers)
        ])
        self.output = Linear(dim, vocab_size)
```

**Transformer Block**:
```
x → LayerNorm → LinearAttention ⊕ → LayerNorm → FFN ⊕ → x
                                 (residual)              (residual)
```

**Positional Embeddings**: RoPE (Rotary Position Embedding)
- Applied within attention mechanism
- More efficient than learned embeddings

## Data Flow

### Training Forward Pass

```
input_ids (batch, seq_len)
    ↓
Embedding layer
    ↓ (batch, seq_len, dim)
Transformer blocks (parallel training):
    ├─ Query projection → φ(Q) (batch, seq_len, dim)
    ├─ Key projection → φ(K) (batch, seq_len, dim)
    ├─ Value projection → V (batch, seq_len, dim)
    ├─ Prefix sum computation (batch, seq_len, dim, dim)
    ├─ Output computation (all positions at once)
    └─ Feedforward layer
    ↓ (batch, seq_len, dim)
Output projection → logits (batch, seq_len, vocab_size)
```

**Computation Graph**:
- All attention outputs computed in parallel
- Gradients flow through both accumulation and normalization

### Inference Forward Pass

```
input_ids (1, current_pos)
    ↓
Embedding
    ↓
For each layer:
    ├─ Q = φ(linear(x))
    ├─ K = φ(linear(x))
    ├─ V = linear(x)
    ├─ accumulator.update(K, V) → context
    ├─ output = context @ Q / normalization
    └─ FFN
    ↓
logits (1, vocab_size)
    ↓
Sample next token
```

**Memory**: Constant per layer (no KV cache explosion)

## Complexity Analysis

### Time Complexity

| Operation | Standard | Linear |
|-----------|----------|--------|
| Attention | O(N²d) | O(Nd²) |
| Full Layer | O(N²d + Nd²) | O(Nd²) |
| N layers | O(N²Ld + NLd²) | O(NLd²) |

For typical d=512, L=12:
- N=1024: Standard 1.3B ops, Linear 3.2B ops (3x worse!)
- N=4096: Standard 21B ops, Linear 3.2B ops (6.5x better)
- N=16384: Standard 343B ops, Linear 3.2B ops (107x better)

**Break-even point**: Around N ≈ 2000 tokens

### Space Complexity

| Component | Standard | Linear |
|-----------|----------|--------|
| Attention KV | O(N·d) | O(d²) |
| Attention Output | O(N·d) | O(N·d) |
| Total | O(N·d) | O(N·d) |

**True win**: Inference with streaming (only need O(d²))

## Numerical Stability

### Challenge 1: Feature Cancellation

Problem: Without positivity, features can cancel:
```
If φ(k₁) = [1, 2] and φ(k₂) = [-1, -2]
Then sum = [0, 0] (information lost)
```

Solution: Kernel ensures φ(x) ≥ 0

### Challenge 2: Denominator Explosion

Problem: Sum of keys can grow unbounded:
```
Σφ(K) can become very large → output → 0
```

Solution: Normalization with parallel denominator tracking

### Challenge 3: Division by Zero

Problem: If Σφ(K) ≈ 0, output becomes NaN

Solution: epsilon-guarded division
```python
output = numerator / (denominator + 1e-6)
```

## Efficiency Gains

### Memory Efficiency

Standard attention at sequence length 10,000:
```
Attention scores: 10000 × 10000 × 4 bytes = 400 MB
```

Linear attention:
```
Accumulator: 512 × 512 × 4 bytes = 1 MB
```

**400x reduction!**

### Compute Efficiency

For sequence length > 2000 tokens, linear attention is faster.

### Inference Speed

- Standard: Must process entire history each token (~10MB KV cache)
- Linear: Fixed accumulator size (~1MB total)
- Linear is **10x faster** for long sequences

## Research Connections

This implementation combines insights from:

1. **Katharopoulos et al. (2020)**: Original linear attention formulation
2. **Choromanski et al. (2020)**: Performer with random features
3. **Su et al. (2021)**: RoPE positional embeddings
4. **Dao et al. (2022)**: Flash Attention optimizations
5. **Beltagy et al. (2020)**: Longformer hybrid approach

## Future Optimizations

1. **CUDA Kernels**: Fuse accumulator operations
2. **Quantization**: Lower precision accumulator
3. **Sparse Patterns**: Selective key-value storage
4. **Learned Kernels**: φ(x) as neural network
5. **Multi-scale**: Separate accumulators for different timescales
