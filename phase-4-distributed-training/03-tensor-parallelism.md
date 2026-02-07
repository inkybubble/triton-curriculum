# Phase 4, Lesson 3: Tensor Parallelism and Beyond

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain why tensor parallelism is needed when FSDP is not enough
- Describe column-parallel and row-parallel linear layers
- Understand how Megatron-LM parallelizes transformer MLP and attention blocks
- Explain pipeline parallelism and the "bubble" problem
- Understand how data, tensor, and pipeline parallelism combine in 3D parallelism

---

## Prerequisites

You should have completed Lessons 1 and 2 and understand:
- Data parallelism (DDP): same model on every GPU, split the batch
- FSDP (ZeRO): shard parameters/gradients/optimizer states across GPUs
- AllReduce, AllGather, ReduceScatter communication primitives
- Why larger models need more aggressive distribution strategies

---

## When FSDP Is Not Enough

FSDP (ZeRO Stage 3) is powerful. It shards everything -- parameters, gradients, and optimizer states -- across GPUs, so each GPU only stores 1/N of the total. But there is a crucial detail that limits it:

**FSDP reconstructs the full layer parameters on each GPU for every forward and backward pass.**

Here is why that matters. Consider a single linear layer in a large model:

```
Layer: nn.Linear(16384, 49152)   (common in 70B+ models)

Parameters: 16384 * 49152 = ~800M parameters
In fp16: 800M * 2 bytes = 1.6 GB

With FSDP (4 GPUs), each GPU stores 1/4 = 400 MB at rest.
But during forward/backward, AllGather reconstructs the FULL 1.6 GB on EACH GPU.
```

For very large models, even this temporary 1.6 GB materialization per layer can strain memory, especially when combined with activations. And for truly massive layers, the AllGather itself becomes a bottleneck.

**Tensor parallelism takes a fundamentally different approach:** instead of each GPU reconstructing the full layer and computing the full output, each GPU computes only a **portion** of the layer's output. The matrix multiplication itself is distributed.

```
FSDP approach:
  AllGather full weight → compute full output → discard weight

Tensor Parallelism approach:
  Each GPU has a SLICE of the weight → compute a SLICE of the output → communicate to combine
```

---

## Column-Parallel Linear Layers

The simplest form of tensor parallelism splits a linear layer's weight matrix by **columns**.

### Regular linear layer

A standard linear layer computes:

```
y = x @ A      where x is (batch, d_in) and A is (d_in, d_out)
```

For example, with `d_in = 4` and `d_out = 8`:

```
                        A (4 x 8)
                 ┌─────────────────────┐
                 │ a00 a01 a02 a03 a04 │ a05 a06 a07 │
  x (batch x 4) │ a10 a11 a12 a13 a14 │ a15 a16 a17 │  =  y (batch x 8)
                 │ a20 a21 a22 a23 a24 │ a25 a26 a27 │
                 │ a30 a31 a32 a33 a34 │ a35 a36 a37 │
                 └─────────────────────┘
```

### Splitting by columns (2 GPUs)

We split A into two halves along the column dimension. Each GPU stores half the columns:

```
GPU 0 has A₁ (4 x 4):              GPU 1 has A₂ (4 x 4):
┌───────────────────┐               ┌───────────────────┐
│ a00  a01  a02  a03│               │ a04  a05  a06  a07│
│ a10  a11  a12  a13│               │ a14  a15  a16  a17│
│ a20  a21  a22  a23│               │ a24  a25  a26  a27│
│ a30  a31  a32  a33│               │ a34  a35  a36  a37│
└───────────────────┘               └───────────────────┘
```

Each GPU receives the **full** input `x` and computes its portion:

```
GPU 0: y₁ = x @ A₁     →  shape (batch, 4)   (first 4 outputs)
GPU 1: y₂ = x @ A₂     →  shape (batch, 4)   (last 4 outputs)
```

To get the full output, we **concatenate** the results -- this requires an **AllGather**:

```
y = [y₁ | y₂]    →  shape (batch, 8)
```

Visually:

```
                   ┌────────────┐
                   │   A₁       │
  x ──────────────►│  (4 x 4)  │──────► y₁ (batch x 4) ──┐
  (replicated on   └────────────┘                          │
   both GPUs)                                              ├── AllGather ──► y (batch x 8)
                   ┌────────────┐                          │
  x ──────────────►│   A₂       │──────► y₂ (batch x 4) ──┘
                   │  (4 x 4)  │
                   └────────────┘
```

**Key properties:**
- Each GPU stores only half the weight matrix: memory per GPU is halved.
- Each GPU does only half the computation: compute per GPU is halved.
- Communication: one AllGather to collect the full output.
- The input `x` must be available on all GPUs (replicated).

---

## Row-Parallel Linear Layers

Row parallelism splits the weight matrix by **rows** instead of columns. This requires the input to be split as well.

### Splitting by rows (2 GPUs)

We split A into two halves along the row dimension:

```
         A₁ (2 x 8)                   GPU 0
┌───────────────────────────┐
│ a00 a01 a02 a03 a04 a05 a06 a07 │
│ a10 a11 a12 a13 a14 a15 a16 a17 │
└───────────────────────────────────┘

         A₂ (2 x 8)                   GPU 1
┌───────────────────────────────────┐
│ a20 a21 a22 a23 a24 a25 a26 a27 │
│ a30 a31 a32 a33 a34 a35 a36 a37 │
└───────────────────────────────────┘
```

The input `x` must also be split along its last dimension. If `x` is `(batch, 4)`, each GPU gets half:

```
GPU 0 gets x₁ = x[:, 0:2]    shape (batch, 2)
GPU 1 gets x₂ = x[:, 2:4]    shape (batch, 2)
```

Each GPU computes a **partial sum**:

```
GPU 0: y₁ = x₁ @ A₁     →  shape (batch, 8)   (partial result)
GPU 1: y₂ = x₂ @ A₂     →  shape (batch, 8)   (partial result)
```

The full output is the **sum** of these partial results -- this requires an **AllReduce**:

```
y = y₁ + y₂
```

This works because matrix multiplication distributes over addition. If we split `x` and `A` correctly:

```
y = x @ A = [x₁ | x₂] @ [A₁] = x₁ @ A₁ + x₂ @ A₂
                         [A₂]
```

Visually:

```
  x₁ (batch x 2) ──────► ┌────────────┐
  (first half of x)       │   A₁       │──────► y₁ (batch x 8) ──┐
                          │  (2 x 8)   │                          │
                          └────────────┘                          ├── AllReduce (sum) ──► y (batch x 8)
  x₂ (batch x 2) ──────► ┌────────────┐                          │
  (second half of x)      │   A₂       │──────► y₂ (batch x 8) ──┘
                          │  (2 x 8)   │
                          └────────────┘
```

**Key properties:**
- Each GPU stores only half the weight matrix.
- Each GPU does only half the computation.
- Communication: one AllReduce to sum the partial results.
- The input must be **split** across GPUs (not replicated).

---

## The Megatron-LM Insight: Pairing Column and Row Parallel

Here is the clever part. In a transformer, the MLP block consists of **two** linear layers back to back:

```
MLP(x) = GELU(x @ W₁) @ W₂

Where:
  W₁ is (d_model, 4 * d_model)   -- "up projection"
  W₂ is (4 * d_model, d_model)   -- "down projection"
```

If we make W₁ **column-parallel** and W₂ **row-parallel**, something magical happens: the output of the column-parallel layer is naturally split along the last dimension, which is exactly what the row-parallel layer needs as input.

### Step-by-step for 2-GPU tensor parallelism

```
Input: x (batch, d_model) -- replicated on both GPUs


STEP 1: Column-parallel W₁
─────────────────────────────
  Split W₁ by columns: W₁ = [W₁ₐ | W₁ᵦ]

  GPU 0: h₁ = x @ W₁ₐ     →  (batch, 2*d_model)
  GPU 1: h₂ = x @ W₁ᵦ     →  (batch, 2*d_model)

  Communication needed? NO! x is replicated and each GPU just uses its half of W₁.


STEP 2: GELU activation
─────────────────────────────
  GPU 0: h₁ = GELU(h₁)
  GPU 1: h₂ = GELU(h₂)

  Communication needed? NO! GELU is element-wise.
  (This only works because we split by columns. If we had split by rows,
   GELU(a + b) ≠ GELU(a) + GELU(b), and we would need to communicate.)


STEP 3: Row-parallel W₂
─────────────────────────────
  Split W₂ by rows: W₂ = [W₂ₐ]
                          [W₂ᵦ]

  GPU 0: y₁ = h₁ @ W₂ₐ    →  (batch, d_model)   (partial sum)
  GPU 1: y₂ = h₂ @ W₂ᵦ    →  (batch, d_model)   (partial sum)


STEP 4: AllReduce
─────────────────────────────
  y = y₁ + y₂              →  (batch, d_model)

  Communication needed? YES -- one AllReduce.
```

**Total communication for the entire MLP block: ONE AllReduce.**

Compare this to the naive approach of doing AllGather after every layer -- Megatron-LM's pairing cuts communication in half.

### Diagram of the full MLP with tensor parallelism

```
                          GPU 0                         GPU 1
                    ┌──────────────┐              ┌──────────────┐
                    │              │              │              │
  x (replicated) ──►  x @ W₁ₐ    │   x ─────────►  x @ W₁ᵦ    │
                    │     │        │              │     │        │
                    │     ▼        │              │     ▼        │
                    │  GELU(h₁)   │              │  GELU(h₂)   │
                    │     │        │              │     │        │
                    │     ▼        │              │     ▼        │
                    │  h₁ @ W₂ₐ   │              │  h₂ @ W₂ᵦ   │
                    │     │        │              │     │        │
                    └─────┼────────┘              └─────┼────────┘
                          │                             │
                          └──────────┬──────────────────┘
                                     │
                                 AllReduce
                                   (sum)
                                     │
                                     ▼
                              y (batch, d_model)
                              (replicated on both GPUs)
```

---

## Tensor-Parallel Attention

The same column/row trick applies to multi-head attention. The key insight: attention heads are independent, so we can assign different heads to different GPUs.

### How multi-head attention works (quick recap)

```
MultiHeadAttention(x):
    Q = x @ W_Q     # (batch, seq, d_model) @ (d_model, d_model) → (batch, seq, d_model)
    K = x @ W_K
    V = x @ W_V

    # Split into heads: (batch, seq, d_model) → (batch, n_heads, seq, d_head)
    # Compute attention per head
    # Concatenate heads back

    output = Concat(head_0, head_1, ..., head_n) @ W_O
```

### Tensor-parallel attention (2 GPUs, 8 heads)

Split Q, K, V projections by **columns** (each GPU handles 4 heads):

```
GPU 0: Q₁ = x @ W_Q₁   (handles heads 0-3)
        K₁ = x @ W_K₁
        V₁ = x @ W_V₁
        attn₁ = Attention(Q₁, K₁, V₁)    →  (batch, seq, d_model/2)

GPU 1: Q₂ = x @ W_Q₂   (handles heads 4-7)
        K₂ = x @ W_K₂
        V₂ = x @ W_V₂
        attn₂ = Attention(Q₂, K₂, V₂)    →  (batch, seq, d_model/2)
```

The output projection W_O is split by **rows**:

```
GPU 0: o₁ = attn₁ @ W_O₁   →  (batch, seq, d_model)  (partial sum)
GPU 1: o₂ = attn₂ @ W_O₂   →  (batch, seq, d_model)  (partial sum)

output = AllReduce(o₁ + o₂)  →  (batch, seq, d_model)
```

Again, just **one AllReduce** for the entire attention block.

### Full transformer layer with tensor parallelism

```
┌──────────────────────────────────────────────────────────────┐
│                   Transformer Layer (2-way TP)                │
│                                                              │
│  Input x (replicated on both GPUs)                           │
│      │                                                       │
│      ▼                                                       │
│  ┌─────────┐                                                 │
│  │LayerNorm│  (replicated -- no communication)               │
│  └────┬────┘                                                 │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────────────────────────────┐                        │
│  │  Column-Parallel Q, K, V         │                        │
│  │  GPU 0: heads 0-3                │  No communication      │
│  │  GPU 1: heads 4-7                │                        │
│  └──────────┬───────────────────────┘                        │
│             │                                                │
│             ▼                                                │
│  ┌──────────────────────────────────┐                        │
│  │  Independent Attention per GPU    │  No communication      │
│  └──────────┬───────────────────────┘                        │
│             │                                                │
│             ▼                                                │
│  ┌──────────────────────────────────┐                        │
│  │  Row-Parallel Output Projection   │                        │
│  │  + AllReduce                      │  ◄── 1st AllReduce     │
│  └──────────┬───────────────────────┘                        │
│             │                                                │
│             ▼                                                │
│  ┌─────────┐                                                 │
│  │ Residual │  + LayerNorm (replicated)                      │
│  └────┬────┘                                                 │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────────────────────────────┐                        │
│  │  Column-Parallel W₁ (MLP up)     │  No communication      │
│  │  + GELU                           │                        │
│  └──────────┬───────────────────────┘                        │
│             │                                                │
│             ▼                                                │
│  ┌──────────────────────────────────┐                        │
│  │  Row-Parallel W₂ (MLP down)      │                        │
│  │  + AllReduce                      │  ◄── 2nd AllReduce     │
│  └──────────┬───────────────────────┘                        │
│             │                                                │
│             ▼                                                │
│  Residual connection                                         │
│       │                                                      │
│       ▼                                                      │
│  Output (replicated on both GPUs)                            │
│                                                              │
│  TOTAL COMMUNICATION: 2 AllReduces per transformer layer     │
└──────────────────────────────────────────────────────────────┘
```

This is remarkably efficient. An entire transformer layer, with attention and MLP, requires only 2 AllReduce operations regardless of model size.

---

## Implementation

Here is a concrete implementation of column-parallel and row-parallel linear layers:

```python
import torch
import torch.nn as nn
import torch.distributed as dist


class ColumnParallelLinear(nn.Module):
    """
    Linear layer split by columns across GPUs.

    Each GPU stores columns [rank * local_out : (rank+1) * local_out] of the
    weight matrix and computes the corresponding slice of the output.

    Input: x (batch, in_features) -- replicated on all GPUs
    Output: y (batch, out_features // world_size) -- each GPU has a slice
    """

    def __init__(self, in_features, out_features, world_size, rank, bias=True):
        super().__init__()
        assert out_features % world_size == 0, (
            f"out_features ({out_features}) must be divisible by "
            f"world_size ({world_size})"
        )
        self.in_features = in_features
        self.local_out_features = out_features // world_size
        self.rank = rank
        self.world_size = world_size

        # Each GPU stores only its slice of the weight
        self.weight = nn.Parameter(
            torch.empty(in_features, self.local_out_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.local_out_features))
        else:
            self.bias = None

        # Initialize weights (important: use the same seed logic as the
        # original layer for reproducibility, or initialize the full weight
        # and slice it)
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        """
        Args:
            x: (batch, in_features) -- replicated on all GPUs

        Returns:
            local_output: (batch, local_out_features) -- this GPU's slice
        """
        # Standard matrix multiply with this GPU's weight slice
        local_output = x @ self.weight
        if self.bias is not None:
            local_output = local_output + self.bias
        return local_output


class RowParallelLinear(nn.Module):
    """
    Linear layer split by rows across GPUs.

    Each GPU stores rows [rank * local_in : (rank+1) * local_in] of the
    weight matrix and expects the corresponding slice of the input.

    Input: x (batch, in_features // world_size) -- each GPU has a slice
    Output: y (batch, out_features) -- AllReduced across GPUs
    """

    def __init__(self, in_features, out_features, world_size, rank, bias=True):
        super().__init__()
        assert in_features % world_size == 0, (
            f"in_features ({in_features}) must be divisible by "
            f"world_size ({world_size})"
        )
        self.local_in_features = in_features // world_size
        self.out_features = out_features
        self.rank = rank
        self.world_size = world_size

        # Each GPU stores only its slice of the weight
        self.weight = nn.Parameter(
            torch.empty(self.local_in_features, out_features)
        )
        if bias:
            # Only one GPU needs the bias (it's added after AllReduce)
            # By convention, rank 0 holds the bias
            if rank == 0:
                self.bias = nn.Parameter(torch.empty(out_features))
            else:
                self.bias = None
        else:
            self.bias = None

        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        """
        Args:
            x: (batch, local_in_features) -- this GPU's slice of the input

        Returns:
            output: (batch, out_features) -- full output, same on all GPUs
        """
        # Compute partial result with this GPU's weight slice
        local_output = x @ self.weight

        # Sum partial results across all GPUs
        dist.all_reduce(local_output, op=dist.ReduceOp.SUM)

        # Add bias (only rank 0 has it, but after AllReduce everyone
        # has the same local_output, so this is fine)
        if self.bias is not None:
            local_output = local_output + self.bias

        return local_output


class TensorParallelMLP(nn.Module):
    """
    Megatron-LM style tensor-parallel MLP block.

    Combines column-parallel (up projection) with row-parallel (down projection)
    to minimize communication: only ONE AllReduce per forward pass.
    """

    def __init__(self, d_model, d_ff, world_size, rank):
        super().__init__()
        # Up projection: column-parallel (no communication needed)
        self.w1 = ColumnParallelLinear(d_model, d_ff, world_size, rank)
        # Down projection: row-parallel (AllReduce at the end)
        self.w2 = RowParallelLinear(d_ff, d_model, world_size, rank)
        self.activation = nn.GELU()

    def forward(self, x):
        # x: (batch, d_model) -- replicated on all GPUs

        # Column-parallel: each GPU computes a slice of the hidden state
        h = self.w1(x)           # (batch, d_ff // world_size) -- no comm
        h = self.activation(h)    # Element-wise, no comm needed

        # Row-parallel: each GPU computes a partial sum, then AllReduce
        output = self.w2(h)       # (batch, d_model) -- AllReduce happens here

        return output  # Replicated on all GPUs
```

### Usage example

```python
def train_with_tensor_parallelism(rank, world_size):
    setup(rank, world_size)

    d_model = 4096
    d_ff = 16384

    # Each GPU creates its shard of the MLP
    mlp = TensorParallelMLP(d_model, d_ff, world_size, rank).to(rank)

    print(f"GPU {rank}: MLP parameters = "
          f"{sum(p.numel() for p in mlp.parameters()) / 1e6:.1f}M")
    # With 4-way TP, each GPU has ~1/4 of the parameters

    x = torch.randn(8, 128, d_model, device=rank)

    output = mlp(x)
    # output is (8, 128, d_model) and identical on all GPUs

    loss = output.sum()
    loss.backward()
    # Gradients are computed in a distributed fashion too

    cleanup()
```

---

## Tensor Parallelism vs FSDP: When to Use What

| Aspect | FSDP (ZeRO-3) | Tensor Parallelism |
|--------|---------------|-------------------|
| What is distributed | Storage of params/grads/optim | The computation itself |
| During forward | AllGather full layer, compute full output | Each GPU computes partial output |
| Communication | AllGather + ReduceScatter per layer | AllReduce per transformer block |
| Memory (params) | 1/N per GPU at rest, full during compute | 1/N per GPU always |
| Memory (activations) | Full activations per GPU | 1/N of intermediate activations |
| Bandwidth requirement | High (AllGather full layers) | Moderate (AllReduce on smaller tensors) |
| Latency sensitivity | Less (can overlap) | More (AllReduce is on critical path) |
| Best for | Models that fit layer-by-layer | Very large layers, within a node |

**The practical rule:** Tensor parallelism works best **within a single machine** where GPUs are connected by NVLink (very fast). FSDP works well **across machines** where the network is slower. This is because TP requires frequent, latency-sensitive AllReduces (on the critical path of computation), while FSDP's AllGathers can overlap with computation.

---

## Pipeline Parallelism

Pipeline parallelism is the third major strategy. Instead of splitting individual layers (tensor parallelism) or replicating the model (data parallelism), it splits the model by **groups of layers**.

### The basic idea

Divide the model's layers into stages, and assign each stage to a different GPU:

```
Model: 16 transformer layers
4 GPUs

GPU 0: Layers 0-3    (Stage 0)
GPU 1: Layers 4-7    (Stage 1)
GPU 2: Layers 8-11   (Stage 2)
GPU 3: Layers 12-15  (Stage 3)

Data flows: GPU 0 → GPU 1 → GPU 2 → GPU 3
```

Each GPU only stores its assigned layers, so memory per GPU is reduced by a factor of N.

### The bubble problem

The naive implementation has a critical efficiency problem:

```
Time ──────────────────────────────────────────────────────►

GPU 0: [Forward 0]                              [Backward 0]
GPU 1:            [Forward 1]            [Backward 1]
GPU 2:                       [Forward 2][Backward 2]
GPU 3:                                  [Fwd 3][Bwd 3]

        ████████ = active computation
                 = idle (the "bubble")
```

While GPU 0 is computing the forward pass for Stage 0, GPUs 1-3 are sitting idle. While GPU 3 is computing, GPUs 0-2 are idle. This idle time is called the **pipeline bubble**, and it wastes GPU resources.

With 4 stages, the bubble fraction is approximately `(N-1)/N` -- with 4 GPUs, 75% of GPU-time is wasted. This is terrible.

### Micro-batching: filling the bubble

The solution is to split the batch into smaller **micro-batches** and pipeline them:

```
Batch is split into 4 micro-batches (m0, m1, m2, m3)

Time ──────────────────────────────────────────────────────────────────────────►

GPU 0: [F(m0)] [F(m1)] [F(m2)] [F(m3)]           [B(m3)] [B(m2)] [B(m1)] [B(m0)]
GPU 1:         [F(m0)] [F(m1)] [F(m2)] [F(m3)]  [B(m3)] [B(m2)] [B(m1)] [B(m0)]
GPU 2:                 [F(m0)] [F(m1)] [F(m2)] [F(m3)][B(m3)][B(m2)] [B(m1)] [B(m0)]
GPU 3:                         [F(m0)] [F(m1)] [F(m2)] [F(m3)B(m3)B(m2)] [B(m1)] [B(m0)]

F = Forward, B = Backward
```

With M micro-batches and N stages, the bubble fraction drops to `(N-1) / (M + N - 1)`. With 4 stages and 16 micro-batches, the bubble is `3/19 ≈ 16%` -- much better than 75%.

### Interleaved scheduling (1F1B)

The schedule above processes all forward passes before any backward passes. A more efficient schedule, called **1F1B** (one forward, one backward), interleaves them:

```
1F1B Schedule (4 stages, 4 micro-batches):

GPU 0: [F0] [F1] [F2] [F3] [B0] [B1] [B2] [B3]
GPU 1:      [F0] [F1] [F2] [B0] [F3] [B1] [B2] [B3]
GPU 2:           [F0] [F1] [B0] [F2] [B1] [F3] [B2] [B3]
GPU 3:                [F0] [B0] [F1] [B1] [F2] [B2] [F3] [B3]
```

In 1F1B, each GPU alternates between forward and backward passes once the pipeline is full. This reduces the peak memory usage because fewer micro-batch activations need to be stored simultaneously.

### Pipeline parallelism: pros and cons

**Pros:**
- Each GPU only stores a fraction of the model layers.
- Communication is minimal -- only activations (not parameters) are passed between stages.
- The data transferred between stages is small (just the hidden states).

**Cons:**
- The pipeline bubble wastes some compute.
- Implementing it correctly (especially the schedule) is complex.
- Uneven layer sizes can cause load imbalance.

---

## 3D Parallelism: Combining Everything

For the largest models (100B+ parameters), no single strategy is sufficient. The solution is to combine all three:

```
3D Parallelism = Data Parallel x Tensor Parallel x Pipeline Parallel
```

### Example: Training a 175B model on 64 GPUs

```
Configuration:
  - 8-way tensor parallelism (within each node of 8 GPUs)
  - 4-way pipeline parallelism (model split into 4 stages)
  - 2-way data parallelism (2 replicas of the full pipeline)

  8 x 4 x 2 = 64 GPUs


  ┌──────────── Data Parallel Replica 0 ────────────┐
  │                                                   │
  │  Stage 0      Stage 1      Stage 2      Stage 3   │
  │  ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐   │
  │  │8 GPUs│───►│8 GPUs│───►│8 GPUs│───►│8 GPUs│   │
  │  │ (TP) │    │ (TP) │    │ (TP) │    │ (TP) │   │
  │  └──────┘    └──────┘    └──────┘    └──────┘   │
  │                                                   │
  └───────────────────────────────────────────────────┘
         │              │              │           │
    AllReduce      AllReduce      AllReduce    AllReduce
    (gradients)    (gradients)    (gradients)  (gradients)
         │              │              │           │
  ┌───────────────────────────────────────────────────┐
  │                                                   │
  │  Stage 0      Stage 1      Stage 2      Stage 3   │
  │  ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐   │
  │  │8 GPUs│───►│8 GPUs│───►│8 GPUs│───►│8 GPUs│   │
  │  │ (TP) │    │ (TP) │    │ (TP) │    │ (TP) │   │
  │  └──────┘    └──────┘    └──────┘    └──────┘   │
  │                                                   │
  └──────────── Data Parallel Replica 1 ────────────┘
```

**How the strategies are layered:**

1. **Tensor parallelism (innermost, 8-way):** Within each group of 8 GPUs (one node), individual matrix multiplications are split. This uses fast NVLink within the node.

2. **Pipeline parallelism (middle, 4-way):** The model is split into 4 stages. Each stage lives on a group of 8 GPUs (that are tensor-parallel). Activations flow between stages over the inter-node network.

3. **Data parallelism (outermost, 2-way):** The entire pipeline is replicated twice. Each replica processes different data. Gradients are synchronized via AllReduce between replicas.

### Why this layering?

The layering is dictated by **communication bandwidth**:

```
Communication speed hierarchy:

Fastest:  NVLink within a node     (~600 GB/s per GPU on DGX H100)
          Used for: Tensor Parallelism (frequent, small AllReduces)

Medium:   Inter-node InfiniBand    (~50-100 GB/s)
          Used for: Pipeline Parallelism (send activations between stages)

Slowest:  Cross-replica gradient sync
          Used for: Data Parallelism (AllReduce gradients, can overlap with compute)
```

Tensor parallelism requires the most frequent communication (AllReduce every transformer layer), so it goes on the fastest links. Data parallelism can overlap communication with computation, so it tolerates slower links.

---

## Connection to Your Triton Knowledge

This is where your Triton kernel expertise becomes directly relevant to distributed training:

### 1. Fused kernels reduce communication pressure

In tensor parallelism, AllReduce is on the critical path. If your Triton kernels for attention or MLP are faster (through fusion, better memory access patterns, etc.), the AllReduce has more time to complete without being a bottleneck.

### 2. Custom tensor-parallel kernels

You could write Triton kernels that are aware of tensor parallelism. For example, a fused column-parallel linear + GELU kernel that avoids materializing the intermediate result:

```python
@triton.jit
def column_parallel_linear_gelu_kernel(
    x_ptr, w_ptr, output_ptr,
    M, N, K,  # N is local_out_features (already sharded)
    ...
):
    # Standard tiled matmul, but N is 1/world_size of the full output
    # Apply GELU fused into the same kernel
    # No communication needed -- this is purely local compute
    ...
```

### 3. Communication-aware kernel design

When writing kernels for models that use tensor parallelism, you need to know which tensors are sharded and which are replicated. A kernel that assumes the full tensor is available will give wrong results if it receives only a shard.

### 4. Overlap communication with custom kernels

Advanced: you can launch NCCL AllReduce operations asynchronously and use CUDA streams to overlap them with your Triton kernel execution. This is how production training frameworks achieve near-linear scaling.

---

## Exercises

### Exercise 1: Communication pattern

Draw the complete communication pattern for a 2-GPU tensor-parallel MLP block with `d_model=4096` and `d_ff=16384`. Label each operation (matmul, GELU, AllReduce) and the tensor shapes at each step.

### Exercise 2: Memory calculation

For a linear layer with shape `(4096, 16384)` in fp16:
- How much memory does the full layer require?
- With 4-way tensor parallelism, how much does each GPU need?
- What about the activations for a batch of 8 sequences of length 2048?

**Solution sketch:**
- Full layer: `4096 * 16384 * 2 bytes = 128 MB`
- Per GPU with 4-way TP: `128 MB / 4 = 32 MB`
- Activations for the output: `8 * 2048 * 16384 * 2 bytes = 512 MB` (full) or `128 MB` per GPU with TP

### Exercise 3: Why TP within nodes?

Explain why tensor parallelism is typically used within a single node (8 GPUs) while data parallelism spans across nodes. Consider:
- NVLink bandwidth (~600 GB/s) vs InfiniBand (~100 GB/s)
- AllReduce frequency (every layer vs once per step)
- Overlap potential

### Exercise 4: 3D parallelism design

You need to train a 70B parameter model on 32 GPUs (4 nodes, 8 GPUs each). Design a 3D parallelism configuration. How would you split TP, PP, and DP?

**Hints:**
- TP should stay within nodes (max 8-way)
- PP should use as few stages as possible to minimize bubble
- DP fills in the rest

One possible answer: 8-way TP (within each node) x 2-way PP (model in 2 halves) x 2-way DP (2 replicas). `8 x 2 x 2 = 32 GPUs`.

---

## Phase 4 Summary

Over these three lessons, you have learned the complete toolkit for distributed training:

```
Strategy        | What it distributes     | Communication    | When to use
────────────────┼─────────────────────────┼──────────────────┼────────────────────────
DDP             | Data (batches)          | AllReduce grads  | Model fits on 1 GPU
                | (full model per GPU)    | 1x per step      | Simplest, fastest
────────────────┼─────────────────────────┼──────────────────┼────────────────────────
FSDP            | Data + model storage    | AllGather +      | Model too big for DDP
(ZeRO)          | (sharded at rest,       | ReduceScatter    | Up to ~30B parameters
                |  gathered for compute)  | per layer        |
────────────────┼─────────────────────────┼──────────────────┼────────────────────────
Tensor          | Computation itself      | AllReduce per    | Very large layers
Parallelism     | (each GPU computes      | transformer      | Within a single node
                |  part of each layer)    | block            |
────────────────┼─────────────────────────┼──────────────────┼────────────────────────
Pipeline        | Model layers            | Send activations | Very deep models
Parallelism     | (each GPU has a         | between stages   | Combined with TP + DP
                |  subset of layers)      |                  |
────────────────┼─────────────────────────┼──────────────────┼────────────────────────
3D Parallelism  | All of the above        | All of the above | 100B+ parameter models
                |                         |                  | Large GPU clusters
```

The progression from DDP to 3D parallelism follows a clear logic:

1. **DDP** -- the default. Use it when the model fits on one GPU.
2. **FSDP** -- when DDP runs out of memory. Shards storage but keeps computation per-GPU.
3. **Tensor Parallelism** -- when even FSDP's temporary materialization of full layers is too much, or when you need to reduce activation memory.
4. **Pipeline Parallelism** -- to distribute across more GPUs than TP can efficiently handle.
5. **3D Parallelism** -- combine all three for maximum scale.

---

## Milestone

After completing this lesson (and Phase 4), you should be able to:

- Explain column-parallel and row-parallel linear layers, including the math
- Describe how Megatron-LM parallelizes transformer MLP and attention with minimal communication
- Explain pipeline parallelism, the bubble problem, and micro-batching
- Design a 3D parallelism configuration for a given model size and GPU count
- Decide which combination of DP, FSDP, TP, and PP to use for a given training scenario

---

## Resources

- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) -- the original Megatron paper introducing tensor-parallel transformers
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473) -- 3D parallelism paper with detailed analysis
- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965) -- foundational pipeline parallelism paper
- [PipeDream: Generalized Pipeline Parallelism for DNN Training](https://arxiv.org/abs/1806.03377) -- 1F1B scheduling and more
- [HuggingFace Parallelism Guide](https://huggingface.co/docs/transformers/parallelism) -- practical guide with HuggingFace integration
- [Lilian Weng: How to Train Really Large Models on Many GPUs](https://lilianweng.github.io/posts/2021-09-25-train-large/) -- excellent blog post covering all strategies
- [NVIDIA Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM) -- production implementation
