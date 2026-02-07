# 02 — Flash Attention: The Most Important Kernel

> **Prerequisites:** You have completed Phase 2 (softmax, matmul, layernorm kernels)
> and the autotuning lesson. You understand tiling, online reductions, and the
> memory hierarchy.

---

## Learning Objectives

By the end of this lesson you will be able to:

1. Explain the memory problem with standard attention and why it limits sequence length.
2. Describe the Flash Attention algorithm step by step, including online softmax.
3. Implement a working Flash Attention forward pass in Triton.
4. Explain why Flash Attention is O(N) memory instead of O(N^2).
5. Answer interview questions about Flash Attention with confidence.

---

## Why This Kernel Matters

Flash Attention is the single most important kernel in modern LLM infrastructure. It is
used in every major model — GPT-4, Claude, Llama, Gemini. Interviewers at Anthropic,
OpenAI, and NVIDIA ask about it frequently because it tests three things at once:

1. **Algorithmic thinking:** The core insight is not about GPU tricks — it is about
   restructuring the softmax computation to avoid materializing a large matrix.
2. **Systems knowledge:** The implementation requires understanding memory hierarchy,
   tiling, and the tradeoff between compute and memory access.
3. **Mathematical precision:** The online softmax trick requires careful handling of
   numerical stability across tiles.

If you can explain Flash Attention clearly and implement it, you demonstrate all three.

---

## Standard Attention — The Memory Problem

Let's start with the standard attention computation from the transformer architecture.
Given queries Q, keys K, and values V, each of shape (seq_len, head_dim):

```python
# Standard attention — what PyTorch does naively
def standard_attention(Q, K, V):
    # Q, K, V: (seq_len, head_dim)  — ignoring batch and heads for clarity
    d_k = Q.shape[-1]

    # Step 1: Compute attention scores
    S = Q @ K.T                    # (seq_len, seq_len) — this is the problem
    S = S / math.sqrt(d_k)        # Scale

    # Step 2: Softmax
    P = torch.softmax(S, dim=-1)  # (seq_len, seq_len) — still huge

    # Step 3: Weighted sum of values
    O = P @ V                     # (seq_len, head_dim) — back to reasonable size

    return O
```

The problem is step 1. The matrix `S = Q @ K.T` has shape (seq_len, seq_len).

For a concrete example with a single attention head:

```
seq_len = 2,048   ->  S is 2048 x 2048   =   4M elements  =  16 MB (FP32)
seq_len = 8,192   ->  S is 8192 x 8192   =  67M elements  = 268 MB
seq_len = 32,768  ->  S is 32K x 32K     =   1B elements  =   4 GB
seq_len = 131,072 ->  S is 128K x 128K   =  16B elements  =  64 GB
```

Now multiply by the number of heads (e.g., 32) and batch size (e.g., 8):

```
seq_len = 8,192 with 32 heads, batch 8:
  S memory = 268 MB * 32 * 8 = 68 GB  — does not fit on any single GPU!
```

Even when it fits in memory, the cost is severe. The attention matrix must be:
1. Written to HBM after the Q @ K.T matmul.
2. Read back from HBM for the softmax.
3. Written to HBM after the softmax.
4. Read back from HBM for the P @ V matmul.

That is four full passes over an N^2-sized matrix through slow HBM. The arithmetic is
O(N^2 * d), but the memory traffic is also O(N^2), and on modern GPUs memory bandwidth
is the bottleneck, not compute.

### The Key Insight

Look at the final output: `O = softmax(Q @ K.T / sqrt(d_k)) @ V`. The output has shape
(seq_len, head_dim) — it is **not** N^2-sized. The intermediate N^2 attention matrix
is just a means to compute a weighted average of V. We never actually need the full
matrix to exist simultaneously. If we could compute the output **tile by tile**, we
could avoid ever storing the full attention matrix.

This is exactly what Flash Attention does.

---

## Flash Attention Algorithm — Step by Step

The Flash Attention algorithm processes the attention computation in tiles, iterating
over blocks of K and V while maintaining running statistics for the softmax. The output
is built up incrementally, and at no point does an N x N matrix exist in memory.

### The Tiling Strategy

We split the matrices into blocks along the sequence dimension:

```
Q is (seq_len, head_dim). Split into row blocks of size BLOCK_M:
  Q_0 = Q[0:BLOCK_M, :]
  Q_1 = Q[BLOCK_M:2*BLOCK_M, :]
  ...

K is (seq_len, head_dim). Split into row blocks of size BLOCK_N:
  K_0 = K[0:BLOCK_N, :]
  K_1 = K[BLOCK_N:2*BLOCK_N, :]
  ...

V is (seq_len, head_dim). Split the same way as K:
  V_0 = V[0:BLOCK_N, :]
  V_1 = V[BLOCK_N:2*BLOCK_N, :]
  ...
```

Each Triton program is responsible for one Q block and must iterate over ALL K/V blocks
to compute its portion of the output:

```
Program 0 handles Q_0:    Q_0 x K_0, Q_0 x K_1, Q_0 x K_2, ...  -> O_0
Program 1 handles Q_1:    Q_1 x K_0, Q_1 x K_1, Q_1 x K_2, ...  -> O_1
Program 2 handles Q_2:    Q_2 x K_0, Q_2 x K_1, Q_2 x K_2, ...  -> O_2
...
```

At each step, a program computes a small (BLOCK_M, BLOCK_N) attention tile and uses it
to update its running output. The attention tile is computed, used, and discarded — it
is never written to HBM.

```
For Program 0 (handles Q_0):

Step 0:  S_00 = Q_0 @ K_0.T   (BLOCK_M x BLOCK_N — fits in SRAM!)
         Update output with softmax(S_00) @ V_0

Step 1:  S_01 = Q_0 @ K_1.T   (BLOCK_M x BLOCK_N — fits in SRAM!)
         Update output with softmax(S_01) @ V_1

Step 2:  S_02 = Q_0 @ K_2.T   (BLOCK_M x BLOCK_N — fits in SRAM!)
         Update output with softmax(S_02) @ V_2

...continue until all K/V blocks processed...

Final:   Write O_0 to HBM.
```

### But Wait — The Softmax Problem

There is a catch. Softmax over a full row is defined as:

```
softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```

This requires `max(x)` over the **entire row** — all N elements. But we are processing
the row in chunks of BLOCK_N at a time. When we process tile S_00, we do not yet know
the global max — the max might be in tile S_03, which we have not seen yet.

The naive approach would be: make two passes. First pass: iterate over all K blocks to
find the row-wise max. Second pass: iterate again to compute the actual softmax and
output. But this doubles the memory traffic.

Flash Attention uses a single-pass approach called **online softmax**.

---

## Online Softmax — The Key Trick

### Connection to Phase 2

In the softmax kernel from Phase 2, the entire row fit in a single block. You computed
`max(row)` and `sum(exp(row - max))` in one shot. Online softmax handles the case where
the row is too long to fit in one block, so we process it chunk by chunk.

### The Algorithm

We maintain three running statistics for each query row:

- `m_i` — the running maximum seen so far (initialized to negative infinity)
- `l_i` — the running sum of exponentials (initialized to 0)
- `O_i` — the running un-normalized output (initialized to 0)

For each new K/V tile j, we update these statistics:

```
# --- Processing tile j ---

# 1. Compute attention scores for this tile
S_j = Q_block @ K_j.T * scale          # shape: (BLOCK_M, BLOCK_N)

# 2. Find the max of this tile (per query row)
m_j = rowwise_max(S_j)                 # shape: (BLOCK_M,)

# 3. Update the running max
m_new = max(m_i, m_j)                  # shape: (BLOCK_M,)

# 4. Compute the correction factor for previously accumulated values
#    If the new max is bigger, all our old exp() values were computed
#    relative to a smaller max and need to be scaled down.
alpha = exp(m_i - m_new)               # shape: (BLOCK_M,)

# 5. Compute attention weights for this tile (relative to new max)
P_j = exp(S_j - m_new)                # shape: (BLOCK_M, BLOCK_N)

# 6. Update the running sum of exponentials
#    Old sum needs the correction factor; new tile's sum is added fresh.
l_new = alpha * l_i + rowwise_sum(P_j) # shape: (BLOCK_M,)

# 7. Update the running output
#    Old output was weighted by l_i with max m_i. Rescale it, then add
#    the new tile's contribution.
O_i = O_i * alpha + P_j @ V_j         # shape: (BLOCK_M, HEAD_DIM)

# 8. Update running statistics
m_i = m_new
l_i = l_new
```

After processing ALL tiles:

```
# Final normalization: divide the accumulated output by the total sum
O_final = O_i / l_i
```

### Why This Works — The Math

Let's trace through a small example with 2 tiles to build intuition.

Suppose we have a single query row and 6 keys, split into two tiles of 3:

```
Full scores: s = [2, 5, 1, 3, 7, 4]
             tile 0: [2, 5, 1]    tile 1: [3, 7, 4]
```

**Standard softmax (for reference):**

```
max = 7
exp(s - 7) = [exp(-5), exp(-2), exp(-6), exp(-4), exp(0), exp(-3)]
           = [0.0067,  0.1353,  0.0025,  0.0183,  1.0,    0.0498]
sum = 1.2126
softmax = each / 1.2126
```

**Online softmax (what Flash Attention does):**

Processing tile 0: `[2, 5, 1]`

```
m_0 = max(2, 5, 1) = 5
P_0 = exp([2, 5, 1] - 5) = [exp(-3), exp(0), exp(-4)] = [0.0498, 1.0, 0.0183]
l_0 = 0.0498 + 1.0 + 0.0183 = 1.0681
O_0 = P_0 @ V_0   (un-normalized; weighted by these P values)
```

Processing tile 1: `[3, 7, 4]`

```
m_1 = max(3, 7, 4) = 7
m_new = max(m_0, m_1) = max(5, 7) = 7

# Correction: our old values used max=5, but the true max is 7
alpha = exp(5 - 7) = exp(-2) = 0.1353

# New tile's attention weights
P_1 = exp([3, 7, 4] - 7) = [exp(-4), exp(0), exp(-3)] = [0.0183, 1.0, 0.0498]

# Update running sum
l_new = alpha * l_0 + sum(P_1)
      = 0.1353 * 1.0681 + (0.0183 + 1.0 + 0.0498)
      = 0.1445 + 1.0681
      = 1.2126                     <-- matches the standard softmax sum!

# Update running output
O = O_0 * alpha + P_1 @ V_1       <-- rescales old output, adds new contribution
```

After all tiles:

```
O_final = O / l_new                <-- divide by total sum to get proper softmax
```

The key insight is the **alpha correction factor**. When we discover a new maximum, we
rescale all previously accumulated values by `exp(old_max - new_max)`. This is exact,
not approximate. The final result is bit-for-bit identical to standard attention (up to
floating point rounding).

### Why `alpha = exp(m_old - m_new)` Is Always <= 1

Since `m_new >= m_old` by definition (we take the max of old and new), the exponent
`m_old - m_new` is always <= 0, so `alpha` is always <= 1. This means we are always
scaling old values **down** when a new max appears. This is numerically stable — we
never multiply by a large number.

---

## The Triton Implementation

Here is a complete, working Flash Attention forward pass. Read it carefully — every
line will be explained afterward.

```python
import math
import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention_forward(
    # Pointers to input/output tensors
    Q_ptr, K_ptr, V_ptr, O_ptr,
    # Strides for Q: (batch, head, seq, dim)
    stride_qb, stride_qh, stride_qm, stride_qk,
    # Strides for K: (batch, head, seq, dim)
    stride_kb, stride_kh, stride_kn, stride_kk,
    # Strides for V: (batch, head, seq, dim)
    stride_vb, stride_vh, stride_vn, stride_vk,
    # Strides for O: (batch, head, seq, dim)
    stride_ob, stride_oh, stride_om, stride_ok,
    # Dimensions
    num_heads,
    N_CTX,                         # sequence length
    # Compile-time constants
    HEAD_DIM: tl.constexpr,        # dimension of each head (e.g., 64 or 128)
    BLOCK_M: tl.constexpr,         # block size for Q (rows of output per program)
    BLOCK_N: tl.constexpr,         # block size for K/V (columns to process per step)
):
    # Scale factor: 1 / sqrt(d_k)
    scale = 1.0 / tl.sqrt(HEAD_DIM * 1.0)

    # ----- Identify which Q block and which (batch, head) this program handles -----
    pid_m = tl.program_id(0)      # which block of Q rows
    pid_bh = tl.program_id(1)     # which (batch, head) pair

    # Decode batch and head indices from the combined pid
    batch_idx = pid_bh // num_heads
    head_idx = pid_bh % num_heads

    # ----- Compute base pointers for this (batch, head) -----
    q_base = Q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    k_base = K_ptr + batch_idx * stride_kb + head_idx * stride_kh
    v_base = V_ptr + batch_idx * stride_vb + head_idx * stride_vh
    o_base = O_ptr + batch_idx * stride_ob + head_idx * stride_oh

    # ----- Offsets for this program's Q block -----
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)    # row indices: [start, start+1, ..., start+BLOCK_M-1]
    offs_d = tl.arange(0, HEAD_DIM)                       # head dimension indices: [0, 1, ..., HEAD_DIM-1]

    # ----- Load the Q block: shape (BLOCK_M, HEAD_DIM) -----
    # This Q block stays in SRAM for the entire inner loop.
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # ----- Initialize accumulators -----
    # These track the running output and softmax statistics.
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)   # running output (un-normalized)
    m_i = tl.full((BLOCK_M,), value=float('-inf'), dtype=tl.float32)  # running max per row
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)            # running sum of exp per row

    # ----- Inner loop: iterate over all K/V blocks -----
    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Load K block: shape (BLOCK_N, HEAD_DIM)
        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

        # Load V block: shape (BLOCK_N, HEAD_DIM)
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

        # --- Step 1: Compute attention scores ---
        # qk = Q_block @ K_block.T, shape (BLOCK_M, BLOCK_N)
        qk = tl.dot(q, tl.trans(k)) * scale

        # Mask out-of-bounds positions (when N_CTX is not a multiple of BLOCK_N)
        qk = tl.where(offs_n[None, :] < N_CTX, qk, float('-inf'))

        # --- Step 2: Online softmax update ---
        # Find the max of this tile's scores (per query row)
        m_ij = tl.max(qk, axis=1)                 # (BLOCK_M,)

        # New running max: element-wise max of old running max and this tile's max
        m_new = tl.maximum(m_i, m_ij)             # (BLOCK_M,)

        # Correction factor: rescale old accumulations to account for new max
        alpha = tl.exp(m_i - m_new)                # (BLOCK_M,)  — always <= 1.0

        # Attention weights for this tile (un-normalized)
        p = tl.exp(qk - m_new[:, None])            # (BLOCK_M, BLOCK_N)

        # Update running sum of exponentials
        l_new = alpha * l_i + tl.sum(p, axis=1)    # (BLOCK_M,)

        # --- Step 3: Update output accumulator ---
        # Rescale the old accumulator (it was computed with the old max)
        # and add the new tile's contribution: P_tile @ V_block
        acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), v)

        # --- Step 4: Update running statistics ---
        m_i = m_new
        l_i = l_new

    # ----- Final normalization -----
    # The accumulator holds sum(exp(s_j - m_final) * v_j) for all j.
    # Divide by l_i (the total sum of exponentials) to get the proper softmax-weighted average.
    acc = acc / l_i[:, None]

    # ----- Store the output -----
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(q.dtype), mask=offs_m[:, None] < N_CTX)
```

---

## Line-by-Line Explanation

### The Grid: How Work Is Divided

```python
pid_m = tl.program_id(0)      # which block of Q rows
pid_bh = tl.program_id(1)     # which (batch, head) pair
```

The kernel uses a 2D grid:
- **Axis 0:** One program per block of Q rows. If seq_len=2048 and BLOCK_M=64, that is
  2048/64 = 32 programs along this axis.
- **Axis 1:** One program per (batch, head) pair. If batch_size=4 and num_heads=8,
  that is 32 programs along this axis.

Total programs: 32 * 32 = 1024, all running in parallel.

Each program computes BLOCK_M rows of the output for one attention head in one batch
element. This is the outer parallelism — no communication between programs is needed.

### Loading Q Once, K/V Many Times

```python
q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
```

The Q block is loaded **once** and stays in registers for the entire inner loop. Each
iteration of the inner loop loads a new K and V block, but Q is reused. This is the
same data reuse pattern from the matmul kernel in Phase 2 — load one operand once,
stream the other operand through.

### The Online Softmax Update (The Critical Lines)

These four lines are the heart of Flash Attention:

```python
m_new = tl.maximum(m_i, m_ij)
alpha = tl.exp(m_i - m_new)
p = tl.exp(qk - m_new[:, None])
l_new = alpha * l_i + tl.sum(p, axis=1)
```

**`m_new = tl.maximum(m_i, m_ij)`** — Update the running max. After processing this
tile, the max is either the old running max or the max from this tile, whichever is
larger.

**`alpha = tl.exp(m_i - m_new)`** — The correction factor. If `m_new == m_i` (the old
max is still the biggest), then `alpha = exp(0) = 1` and nothing changes. If `m_new >
m_i` (this tile had a bigger value), then `alpha < 1` and we scale down everything we
accumulated before. This is the key to making online softmax work: we retroactively
adjust past computations when new information arrives.

**`p = tl.exp(qk - m_new[:, None])`** — The attention weights for this tile, computed
relative to the new running max. This is numerically stable because we subtract the
max before exponentiating (same trick as in the Phase 2 softmax kernel, but applied per
tile rather than per row).

**`l_new = alpha * l_i + tl.sum(p, axis=1)`** — Update the running sum. The old sum
`l_i` was computed relative to the old max, so we multiply by `alpha` to adjust it to
the new max. Then we add the sum from the current tile.

### The Accumulator Update

```python
acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), v)
```

This is where the output is built incrementally:

- `acc * alpha[:, None]` — Rescale the old accumulator. The old `acc` held the
  un-normalized sum `sum(exp(s_j - m_old) * v_j)` over previous tiles. Multiplying by
  `alpha = exp(m_old - m_new)` converts it to `sum(exp(s_j - m_new) * v_j)`.

- `tl.dot(p.to(q.dtype), v)` — The new tile's contribution: attention weights times
  values. Shape: (BLOCK_M, BLOCK_N) @ (BLOCK_N, HEAD_DIM) = (BLOCK_M, HEAD_DIM).

- The sum is the new accumulator, which now holds `sum(exp(s_j - m_new) * v_j)` over
  ALL tiles processed so far.

Note: `p.to(q.dtype)` converts the float32 attention weights to the input dtype
(typically float16) before the matrix multiply, which allows Tensor Core acceleration.

### Final Normalization

```python
acc = acc / l_i[:, None]
```

After the loop, `acc` holds `sum(exp(s_j - m_final) * v_j)` for all j, and `l_i`
holds `sum(exp(s_j - m_final))` for all j. Dividing gives:

```
O = sum(exp(s_j - m_final) * v_j) / sum(exp(s_j - m_final))
  = sum(softmax(s_j) * v_j)
  = softmax(S) @ V
```

This is exactly the standard attention output.

### Why We Do NOT Store the Attention Matrix

This is the entire point of Flash Attention. The attention scores `qk` have shape
(BLOCK_M, BLOCK_N) and live in SRAM (registers/shared memory) only. They are computed,
used to update `acc`, and then overwritten on the next loop iteration. At no point does
an (N_CTX, N_CTX) matrix exist in HBM. The only things written to HBM are the final
output O and (optionally) the softmax statistics m and l for the backward pass.

---

## The Launcher Function

```python
def flash_attention(Q, K, V):
    """
    Q, K, V: (batch, heads, seq_len, head_dim) tensors on GPU.
    Returns: O of the same shape.
    """
    assert Q.is_cuda, "Inputs must be on GPU"
    batch, heads, seq_len, head_dim = Q.shape

    # Allocate output
    O = torch.empty_like(Q)

    # Block sizes — these could be autotuned
    BLOCK_M = 64
    BLOCK_N = 64

    # Grid: one program per (Q block, batch-head pair)
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * heads)

    flash_attention_forward[grid](
        Q, K, V, O,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        heads,
        seq_len,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return O
```

### Understanding the Grid

```python
grid = (triton.cdiv(seq_len, BLOCK_M), batch * heads)
```

- **Axis 0:** `ceil(seq_len / BLOCK_M)` programs. Each handles BLOCK_M rows of Q.
- **Axis 1:** `batch * heads` programs. Each handles one (batch, head) pair.

For batch=4, heads=32, seq_len=2048, BLOCK_M=64:
- Axis 0: 2048/64 = 32 programs
- Axis 1: 4 * 32 = 128 programs
- Total: 32 * 128 = 4096 programs running in parallel

Each program does `ceil(seq_len / BLOCK_N)` iterations of the inner loop (32 iterations
for seq_len=2048, BLOCK_N=64).

---

## Verification

Always verify that your kernel produces correct results before benchmarking.

```python
def test_flash_attention():
    torch.manual_seed(42)
    batch, heads, seq_len, head_dim = 2, 4, 1024, 64
    Q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    K = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    V = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)

    # Our implementation
    triton_output = flash_attention(Q, K, V)

    # Reference: PyTorch's scaled_dot_product_attention
    reference_output = torch.nn.functional.scaled_dot_product_attention(Q, K, V)

    # Check correctness (allow some tolerance for FP16)
    max_diff = (triton_output - reference_output).abs().max().item()
    print(f"Max absolute difference: {max_diff:.6f}")
    assert max_diff < 1e-2, f"Results differ too much: max_diff={max_diff}"
    print("Correctness verified!")

    # Also test with different sequence lengths
    for seq_len in [128, 256, 512, 2048]:
        Q = torch.randn(1, 1, seq_len, head_dim, device='cuda', dtype=torch.float16)
        K = torch.randn(1, 1, seq_len, head_dim, device='cuda', dtype=torch.float16)
        V = torch.randn(1, 1, seq_len, head_dim, device='cuda', dtype=torch.float16)
        triton_out = flash_attention(Q, K, V)
        ref_out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        diff = (triton_out - ref_out).abs().max().item()
        print(f"  seq_len={seq_len}: max_diff={diff:.6f}")


test_flash_attention()
```

A note on tolerance: FP16 has limited precision (about 3 decimal digits). The online
softmax involves many exp() and division operations that accumulate rounding errors.
A max difference of 1e-2 or less is normal and acceptable for FP16. If you want
tighter accuracy, cast inputs to FP32 for testing.

---

## Memory Complexity Analysis

### Standard Attention: O(N^2)

```
Q @ K.T       ->  Materializes (N, N) matrix in HBM       : O(N^2) memory
softmax(S)    ->  Reads and writes (N, N) matrix from HBM  : O(N^2) memory
P @ V         ->  Reads (N, N) matrix from HBM             : O(N^2) memory

Total HBM memory: O(N^2)   (for the attention matrix)
Total HBM reads:  O(N^2)   (multiple passes over the attention matrix)
```

### Flash Attention: O(N)

```
Q is loaded once per program            : O(BLOCK_M * d) per program
K, V are streamed in tiles of BLOCK_N   : O(BLOCK_N * d) per tile, discarded after use
Attention scores qk                     : O(BLOCK_M * BLOCK_N) in SRAM, never in HBM
Output accumulator acc                  : O(BLOCK_M * d) in registers
Statistics m_i, l_i                     : O(BLOCK_M) in registers

Total HBM memory: O(N * d)   (just the input and output tensors — no attention matrix!)
Total HBM reads:  O(N^2 * d / BLOCK_M)   (more compute-focused, less memory-focused)
```

The critical difference: Flash Attention never writes the N x N attention matrix to HBM.
The only HBM tensors are Q, K, V, and O, all of which are O(N * d).

### Empirical Verification

You can verify this by measuring peak memory usage:

```python
import torch

def measure_memory_standard(seq_len, head_dim=64):
    """Measure peak memory of standard attention."""
    torch.cuda.reset_peak_memory_stats()
    Q = torch.randn(1, 1, seq_len, head_dim, device='cuda', dtype=torch.float16)
    K = torch.randn(1, 1, seq_len, head_dim, device='cuda', dtype=torch.float16)
    V = torch.randn(1, 1, seq_len, head_dim, device='cuda', dtype=torch.float16)

    # Standard attention — materializes the N x N matrix
    S = Q @ K.transpose(-2, -1) / math.sqrt(head_dim)
    P = torch.softmax(S, dim=-1)
    O = P @ V

    peak = torch.cuda.max_memory_allocated() / 1e6  # MB
    del Q, K, V, S, P, O
    torch.cuda.empty_cache()
    return peak

def measure_memory_flash(seq_len, head_dim=64):
    """Measure peak memory of Flash Attention."""
    torch.cuda.reset_peak_memory_stats()
    Q = torch.randn(1, 1, seq_len, head_dim, device='cuda', dtype=torch.float16)
    K = torch.randn(1, 1, seq_len, head_dim, device='cuda', dtype=torch.float16)
    V = torch.randn(1, 1, seq_len, head_dim, device='cuda', dtype=torch.float16)

    O = flash_attention(Q, K, V)

    peak = torch.cuda.max_memory_allocated() / 1e6  # MB
    del Q, K, V, O
    torch.cuda.empty_cache()
    return peak

# Compare
for seq_len in [512, 1024, 2048, 4096, 8192]:
    std_mem = measure_memory_standard(seq_len)
    flash_mem = measure_memory_flash(seq_len)
    ratio = std_mem / flash_mem
    print(f"seq_len={seq_len:5d}  standard={std_mem:8.1f} MB  flash={flash_mem:8.1f} MB  ratio={ratio:.1f}x")
```

You should see the standard attention memory growing quadratically while Flash Attention
memory grows linearly.

---

## Benchmarking Against PyTorch

```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N_CTX'],
        x_vals=[2**i for i in range(8, 14)],   # 256 to 8192
        line_arg='provider',
        line_vals=['flash_triton', 'sdpa'],
        line_names=['Flash Attention (Triton)', 'PyTorch SDPA'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='ms',
        plot_name='flash-attention-benchmark',
        args={'batch': 4, 'heads': 8, 'head_dim': 64},
    )
)
def bench_flash(N_CTX, batch, heads, head_dim, provider):
    Q = torch.randn(batch, heads, N_CTX, head_dim, device='cuda', dtype=torch.float16)
    K = torch.randn(batch, heads, N_CTX, head_dim, device='cuda', dtype=torch.float16)
    V = torch.randn(batch, heads, N_CTX, head_dim, device='cuda', dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'flash_triton':
        fn = lambda: flash_attention(Q, K, V)
    elif provider == 'sdpa':
        fn = lambda: torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    return ms, max_ms, min_ms


bench_flash.run(print_data=True, show_plots=True)
```

What to expect:
- At short sequence lengths (256-512), PyTorch's SDPA may be faster because it uses
  highly optimized cuDNN/cuBLAS kernels.
- At longer sequence lengths (2048+), our Triton Flash Attention should become
  competitive because the memory savings matter more.
- PyTorch's SDPA on newer versions actually uses Flash Attention under the hood (via
  the C++ flash_attn library), so you may see similar performance. The goal is to
  understand how it works, not necessarily to beat the built-in implementation.

---

## Adding Causal Masking

In autoregressive models (GPT-style), each token can only attend to tokens at the same
position or earlier. This means the attention matrix has a lower-triangular structure:
position i can attend to positions 0, 1, ..., i but not i+1, i+2, ....

In standard attention, you add a mask before softmax:

```python
# Causal mask: -inf for positions where query < key
causal_mask = torch.triu(torch.full((N, N), float('-inf')), diagonal=1)
S = S + causal_mask
```

In Flash Attention, the causal mask is applied tile by tile:

```python
# Inside the inner loop, after computing qk:
qk = tl.dot(q, tl.trans(k)) * scale

# Apply causal mask: position m can only attend to position n if m >= n
causal_mask = offs_m[:, None] >= offs_n[None, :]
qk = tl.where(causal_mask, qk, float('-inf'))
```

This is a single line change. But there is an optimization: for tiles where ALL
positions are masked (i.e., the entire Q block comes before the entire K block), we
can skip the tile entirely:

```python
# Optimization: only iterate over K/V blocks that are not fully masked
for start_n in range(0, min(N_CTX, (pid_m + 1) * BLOCK_M), BLOCK_N):
    # ... process tile ...
```

Instead of iterating over ALL K/V blocks, each program only iterates up to its own row
position. Programs handling early Q positions (small pid_m) do less work, and programs
handling late Q positions do more. On average, this cuts the total work roughly in half
— which is why causal attention is about 2x faster than full attention in Flash
Attention.

---

## How Flash Attention Relates to What You Already Know

The following table maps Flash Attention concepts to kernels you wrote in Phase 2:

| Flash Attention concept | Where you saw it before |
|---|---|
| Tiling over sequence length | Tiling over K dimension in matmul |
| Online softmax (running max, running sum) | Softmax kernel (but over the full row in one shot) |
| `tl.dot(q, tl.trans(k))` | The tile matmul `tl.dot(a, b)` in matmul |
| Accumulator rescaling (`acc * alpha`) | New — unique to online softmax |
| Final normalization (`acc / l_i`) | The `/ sum_exp` in your softmax kernel |
| Masking for out-of-bounds | The `mask = offsets < n_elements` pattern from vector add |

Flash Attention is a synthesis of everything from Phase 2. If you understood each
component individually, combining them here should feel natural.

---

## The Flash Attention 2 Improvements

The original Flash Attention paper (Dao, 2022) established the algorithm. Flash
Attention 2 (Dao, 2023) kept the same algorithm but restructured the implementation
for better GPU utilization. The key changes:

1. **Swapped the loop order.** Flash Attention 1 has the outer loop over K/V blocks
   and the inner loop over Q blocks. Flash Attention 2 reverses this: outer loop over
   Q blocks, inner loop over K/V blocks. This reduces the amount of shared memory
   reads/writes.

2. **Better parallelism.** Flash Attention 1 parallelizes over batch and heads only.
   Flash Attention 2 also parallelizes over the sequence dimension (the Q blocks),
   which is what our Triton implementation already does.

3. **Reduced non-matmul FLOPs.** The online softmax rescaling involves element-wise
   operations that do not use Tensor Cores. Flash Attention 2 minimizes these by
   deferring the `l_i` normalization to the very end.

Our Triton implementation above actually follows the Flash Attention 2 structure
(outer loop over Q blocks per program, inner loop over K/V blocks). The key insight
from the implementation in this lesson is the same as Flash Attention 2: each program
owns a Q block and iterates over K/V, rather than the other way around.

---

## Exercises

### Exercise 1: Add Causal Masking

Modify the kernel to support causal (autoregressive) attention. You need to:

1. Add the causal mask inside the inner loop:
   ```python
   causal_mask = offs_m[:, None] >= offs_n[None, :]
   qk = tl.where(causal_mask, qk, float('-inf'))
   ```

2. Optimize the loop bound to skip fully-masked tiles:
   ```python
   for start_n in range(0, min(N_CTX, (pid_m + 1) * BLOCK_M), BLOCK_N):
   ```

3. Verify against `torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)`.

### Exercise 2: Find the Crossover Point

Benchmark standard attention (materializing the full N x N matrix) against Flash
Attention at sequence lengths from 128 to 8192. At what sequence length does Flash
Attention become faster? Why does it lose at short sequence lengths? (Hint: think about
the overhead of the online softmax bookkeeping vs. the benefit of reduced memory
traffic.)

### Exercise 3: Memory Scaling

Using the `measure_memory_standard` and `measure_memory_flash` functions from above,
plot peak memory vs. sequence length for both approaches. Verify that:
- Standard attention memory scales as O(N^2)
- Flash Attention memory scales as O(N)

Fit a curve to your measurements (linear regression on log-log scale) and confirm the
exponent.

### Exercise 4: Read the Flash Attention 2 Paper

Read [Flash Attention 2](https://arxiv.org/abs/2307.08691) (particularly Sections 3
and 4). Identify:
- What is the difference in parallelism strategy between FA1 and FA2?
- Why does FA2 achieve better GPU utilization on A100s?
- What is the "online softmax trick" they reference, and how does it relate to what
  you implemented?

Write a one-paragraph summary of what FA2 changes and why it is faster.

---

## Interview Preparation

If you are asked about Flash Attention in an interview, here are the key points to hit:

**The problem:** Standard attention materializes an O(N^2) attention matrix, which
dominates memory and memory bandwidth for long sequences.

**The solution:** Process the attention computation in tiles. Each tile computes a small
(BLOCK_M, BLOCK_N) piece of the attention matrix in SRAM, uses it immediately to update
the output, and discards it. The attention matrix never exists in HBM.

**The trick:** Online softmax allows computing the correct softmax output incrementally,
tile by tile, without knowing the global max in advance. When a new max appears, all
previously accumulated values are rescaled by `exp(old_max - new_max)`.

**The result:** O(N) memory instead of O(N^2). The same total FLOPs, but much less
memory traffic. This makes it IO-aware — it is designed around the GPU memory hierarchy.

**Be ready to:** Draw the tiling diagram, write the online softmax update equations
from memory, and explain why the correction factor `exp(m_old - m_new)` is always
less than or equal to 1.

---

## Milestone Checklist

You are ready to move on to the next lesson when you can:

- [ ] Explain why standard attention is O(N^2) memory and what specific matrix causes it.
- [ ] Draw the Flash Attention tiling strategy from memory.
- [ ] Write the online softmax update equations without looking at notes.
- [ ] Explain why `alpha = exp(m_old - m_new)` correctly rescales previous values.
- [ ] Implement Flash Attention in Triton (or at least outline the kernel structure
      from memory).
- [ ] Explain the difference between Flash Attention 1 and 2 at a high level.

---

## Resources

- [Flash Attention paper (Dao et al., 2022)](https://arxiv.org/abs/2205.14135) — the original paper. Read Sections 2 and 3 carefully.
- [Flash Attention 2 paper (Dao, 2023)](https://arxiv.org/abs/2307.08691) — the optimized version. Focus on Section 3.
- [Triton Flash Attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) — the official Triton tutorial, which is a more optimized version of what we built here.
- [Tri Dao's talk on Flash Attention](https://www.youtube.com/watch?v=gMOAud7hZg4) — a conference talk walking through the algorithm and implementation.
- [Online softmax explanation by Aleksa Gordic](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad) — a clear blog post explaining the online softmax trick.
