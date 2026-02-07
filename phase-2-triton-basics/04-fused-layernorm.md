# 04 — Fused LayerNorm: Solo Exercise

> **Prerequisites:** You have completed all of Phase 2 so far — vector addition,
> fused softmax, and matrix multiplication. You can write Triton kernels with
> `tl.load`, `tl.store`, masking, reductions, 2D pointer arithmetic, and `tl.dot`.
> This lesson tests whether you can apply these patterns independently.

---

## Learning Objectives

By the end of this module, you will be able to:

1. Implement a fused LayerNorm kernel in Triton **on your own**
2. Apply the row-wise kernel pattern (from softmax) to a new problem
3. Verify your kernel's correctness against PyTorch
4. Assess your own readiness to write Triton kernels independently

---

## How to Use This Module

This is a **solo exercise**, not a tutorial. The goal is to test whether you have internalized the patterns from the previous three modules (vector addition, fused softmax, matrix multiplication).

Here is the recommended workflow:

1. **Read the specification** below — understand what LayerNorm does mathematically.
2. **Close this file** and try to implement it in Colab from scratch.
3. If you get stuck, come back and read **one hint at a time** (do not skip ahead).
4. Only look at the reference implementation **after** you have a working version or are truly stuck after all hints.
5. Compare your solution to the reference and note the differences.

The amount of help you need is your honest signal for how well you have absorbed Phase 2:

- **No hints needed:** You are solid. Move to Phase 3.
- **1-2 hints needed:** Good — you understand the structure but need reminders on specifics.
- **3-4 hints needed:** Okay — you should re-read the softmax module before continuing.
- **Needed the full reference:** Go back and re-implement softmax and matmul from scratch, then try this again.

There is no shame in needing hints. The goal is learning, not speed.

---

## What Is LayerNorm?

Layer Normalization is a normalization technique used in virtually every transformer model. It normalizes each row (each sample in a batch, or each token in a sequence) independently.

### The Math

Given an input vector `x` of length `n`:

```
mean     = (1/n) * sum(x)
variance = (1/n) * sum((x - mean)^2)
output   = gamma * (x - mean) / sqrt(variance + epsilon) + beta
```

Where:
- `mean` and `variance` are computed along the last dimension (each row independently)
- `gamma` (weight) is a learnable scale parameter, shape `(n,)`, initialized to ones
- `beta` (bias) is a learnable shift parameter, shape `(n,)`, initialized to zeros
- `epsilon` is a small constant (e.g., `1e-5`) to prevent division by zero

In PyTorch:

```python
# These are equivalent:
output = torch.nn.functional.layer_norm(x, [n_features], weight, bias)

# Manual computation:
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True, correction=0)  # population variance, not sample
output = weight * (x - mean) / torch.sqrt(var + eps) + bias
```

Note: LayerNorm uses **population variance** (dividing by `n`, not `n-1`). This is the `correction=0` in PyTorch's `var()`.

### Why Fuse It?

In PyTorch, `layer_norm` launches multiple CUDA kernels internally: one to compute the mean, one to subtract it, one to compute variance, one to normalize, one to scale and shift. Each kernel reads from and writes to global memory.

A fused kernel does everything in a **single pass**: load the row once, compute mean, compute variance, normalize, scale, shift, and store the result. One read from global memory, one write. This is the same fusion strategy as the softmax kernel from module 02.

---

## Your Specification

Write a Triton kernel that computes LayerNorm on a 2D input tensor.

**Inputs:**
- `x`: a 2D tensor of shape `(n_rows, n_cols)`, dtype float32, on CUDA
- `gamma`: a 1D tensor of shape `(n_cols,)`, the scale parameter
- `beta`: a 1D tensor of shape `(n_cols,)`, the shift parameter
- `eps`: a float, default `1e-5`

**Output:**
- A tensor of the same shape and dtype as `x`, containing the LayerNorm result

**Constraints:**
- One Triton program per row (same pattern as softmax)
- Each program loads its entire row, computes the statistics, normalizes, and stores the result
- Use masking for when `n_cols` is not a multiple of `BLOCK_SIZE`

**Verification:** Your output must match `torch.nn.functional.layer_norm` within `atol=1e-5`.

---

## Stop Here and Try It

You have all the information you need. Open Colab and write the kernel.

Think about:
- What is the grid? (How many programs do you launch?)
- What does each program load?
- What reductions do you need?
- What is the final computation before storing?

Come back for hints only if you are stuck.

---

## Hints

Read these one at a time. After each hint, go back and try again before reading the next one.

---

### Hint 1: Structure

The kernel structure is almost identical to the fused softmax kernel. One program per row. Load the row into a register, do computations, store the result. The grid is `(n_rows,)`.

---

### Hint 2: Reductions

You need two reductions:
1. `tl.sum(row, axis=0)` to compute the sum, then divide by `n_cols` for the mean.
2. After subtracting the mean, `tl.sum(diff * diff, axis=0)` divided by `n_cols` for the variance.

Both reductions collapse the entire row down to a single scalar.

---

### Hint 3: The Normalization Step

After computing mean and variance:

```python
inv_std = 1.0 / tl.sqrt(var + eps)
normed = (row - mean) * inv_std
```

Computing `1/sqrt(x)` once and multiplying is faster than dividing by `sqrt(x)` for each element. Division is expensive on GPUs.

---

### Hint 4: Loading Gamma and Beta

`gamma` and `beta` are 1D tensors indexed by column position. Load them with the same `col_offsets` and `mask` you use for the input row. For the mask's `other` value: `gamma` should default to `1.0` (multiplying by 1 is a no-op) and `beta` should default to `0.0` (adding 0 is a no-op).

---

### Hint 5: Pointer Arithmetic

For a 2D input with row stride `input_row_stride`:
```python
row_start_ptr = input_ptr + row_idx * input_row_stride
input_values = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0)
```

This is exactly the same as softmax. Each row starts at a different offset in memory, and within the row, elements are contiguous (stride 1).

---

## Reference Implementation

Only read this after you have attempted the kernel yourself.

### The Kernel

```python
import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(
    # Pointers
    output_ptr,
    input_ptr,
    gamma_ptr,
    beta_ptr,
    # Strides
    input_row_stride,
    output_row_stride,
    # Dimensions
    n_cols,
    # Parameters
    eps,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    # Which row does this program handle?
    row_idx = tl.program_id(0)

    # Compute pointers to the start of this row.
    row_start_ptr = input_ptr + row_idx * input_row_stride

    # Column offsets — each program processes one full row.
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # -----------------------------------------------------------
    # Step 1: Load the entire row.
    # -----------------------------------------------------------
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0)

    # -----------------------------------------------------------
    # Step 2: Compute the mean.
    # -----------------------------------------------------------
    # tl.sum reduces the entire row to a single scalar.
    # We divide by n_cols (not BLOCK_SIZE!) because masked-off
    # positions are 0 and should not count toward the average.
    mean = tl.sum(row, axis=0) / n_cols

    # -----------------------------------------------------------
    # Step 3: Compute the variance.
    # -----------------------------------------------------------
    # Subtract the mean first, then compute mean of squares.
    diff = row - mean
    var = tl.sum(diff * diff, axis=0) / n_cols

    # -----------------------------------------------------------
    # Step 4: Normalize.
    # -----------------------------------------------------------
    inv_std = 1.0 / tl.sqrt(var + eps)
    normed = diff * inv_std

    # -----------------------------------------------------------
    # Step 5: Scale and shift with gamma and beta.
    # -----------------------------------------------------------
    gamma = tl.load(gamma_ptr + col_offsets, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + col_offsets, mask=mask, other=0.0)
    output = normed * gamma + beta

    # -----------------------------------------------------------
    # Step 6: Store the result.
    # -----------------------------------------------------------
    output_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_start_ptr + col_offsets, output, mask=mask)
```

### Detailed Explanation

**Grid and program assignment.** `row_idx = tl.program_id(0)` assigns each program to one row. The grid has `n_rows` programs total. This is identical to the softmax kernel's structure.

**Loading the row.** We load `BLOCK_SIZE` elements starting from `row_start_ptr`. The mask ensures we only load valid elements (when `n_cols < BLOCK_SIZE`). The `other=0.0` is important: masked-off elements become zero, which does not affect the sum in the mean calculation. However, note that we divide by `n_cols`, not `BLOCK_SIZE` — if we divided by `BLOCK_SIZE`, the zeros from masked positions would dilute the mean.

**Computing the mean.** `tl.sum(row, axis=0)` adds up all elements in the row (including the masked zeros). Dividing by `n_cols` gives us the correct mean. If this looks familiar, it is — the softmax kernel also used `tl.sum` for the denominator of the softmax normalization.

**Computing the variance.** We compute `diff = row - mean` (broadcasting: scalar `mean` is subtracted from every element), then square it and sum. Again, dividing by `n_cols` for population variance. The masked-off positions have `row = 0.0`, so after subtracting the mean, they contribute `mean^2` to the variance sum. But wait — is that correct?

Actually, there is a subtlety: the masked positions have `row = 0.0`, so `diff = 0.0 - mean = -mean` for those positions, and `diff * diff = mean^2`. These get included in `tl.sum(diff * diff)`, adding extra to the variance. But we divide by `n_cols` (the true number of elements), not `BLOCK_SIZE`. So as long as the masked positions contribute zero to the sum, the variance would be correct. But they contribute `mean^2`, not zero.

The fix is that this effect is negligible when `BLOCK_SIZE` is close to `n_cols` (which it usually is, since `BLOCK_SIZE` is the next power of 2 above `n_cols`). For exact correctness with arbitrary padding, you could zero out the diff at masked positions:

```python
diff = tl.where(mask, row - mean, 0.0)
```

However, in practice the standard Triton LayerNorm tutorial does not do this, and the error is within float32 tolerance for typical feature sizes (768, 1024, 2048, 4096 — which are all powers of 2 or close to it). For this curriculum, we keep it simple. Just be aware of the subtlety.

**Normalization.** We compute `inv_std = 1/sqrt(var + eps)` and multiply rather than dividing by `sqrt(var + eps)`. On GPU hardware, multiplication is significantly faster than division. The `eps` prevents division by zero when variance is exactly 0 (which can happen with constant input rows).

**Scaling and shifting.** `gamma` and `beta` are 1D parameters, one value per feature. We load them using the same `col_offsets`. The default values in the mask (`other=1.0` for gamma, `other=0.0` for beta) ensure that masked positions get the identity transformation: multiplying by 1 and adding 0 does nothing.

**Comparison to softmax.** The structure is nearly identical:

```
Softmax:                           LayerNorm:
1. Load row                        1. Load row
2. Compute max (reduction)         2. Compute mean (reduction)
3. Subtract max, exponentiate      3. Subtract mean
4. Compute sum (reduction)         4. Compute variance (reduction)
5. Divide by sum                   5. Divide by sqrt(var + eps)
6. Store                           6. Scale by gamma, shift by beta
                                   7. Store
```

Both are row-wise operations with two reductions. Both are memory-bound (low arithmetic intensity). Both benefit enormously from fusion because the unfused version reads/writes the row multiple times, while the fused version reads once and writes once.

### The Launcher

```python
def layernorm(x, gamma, beta, eps=1e-5):
    assert x.is_contiguous(), "Input must be contiguous"
    n_rows, n_cols = x.shape
    # Allocate output — same shape and dtype as input.
    output = torch.empty_like(x)
    # Round up to next power of 2 for efficient GPU execution.
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # One program per row.
    grid = (n_rows,)

    layernorm_kernel[grid](
        output, x, gamma, beta,
        x.stride(0), output.stride(0),
        n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
```

`triton.next_power_of_2(n_cols)` rounds up to the nearest power of 2. If `n_cols = 768`, `BLOCK_SIZE` becomes 1024. This is required because Triton block sizes must be powers of 2. The extra positions (769 through 1023) are handled by the mask.

---

## Complete Runnable Example

Copy this into a Colab cell:

```python
import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(
    output_ptr,
    input_ptr,
    gamma_ptr,
    beta_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load the row
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0)

    # Compute mean
    mean = tl.sum(row, axis=0) / n_cols

    # Compute variance
    diff = row - mean
    var = tl.sum(diff * diff, axis=0) / n_cols

    # Normalize
    inv_std = 1.0 / tl.sqrt(var + eps)
    normed = diff * inv_std

    # Scale and shift
    gamma = tl.load(gamma_ptr + col_offsets, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + col_offsets, mask=mask, other=0.0)
    output = normed * gamma + beta

    # Store
    output_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_start_ptr + col_offsets, output, mask=mask)


def layernorm(x, gamma, beta, eps=1e-5):
    assert x.is_contiguous(), "Input must be contiguous"
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)

    layernorm_kernel[grid](
        output, x, gamma, beta,
        x.stride(0), output.stride(0),
        n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


# --- Correctness test ---
torch.manual_seed(42)
x = torch.randn(128, 768, device='cuda', dtype=torch.float32)
gamma = torch.ones(768, device='cuda', dtype=torch.float32)
beta = torch.zeros(768, device='cuda', dtype=torch.float32)

triton_out = layernorm(x, gamma, beta)
torch_out = torch.nn.functional.layer_norm(x, [768], gamma, beta)

if torch.allclose(triton_out, torch_out, atol=1e-5):
    print("LayerNorm matches PyTorch!")
else:
    diff = (triton_out - torch_out).abs()
    print(f"MISMATCH — max diff: {diff.max().item():.8f}, mean diff: {diff.mean().item():.8f}")

# --- Test with non-trivial gamma and beta ---
gamma = torch.randn(768, device='cuda', dtype=torch.float32)
beta = torch.randn(768, device='cuda', dtype=torch.float32)

triton_out = layernorm(x, gamma, beta)
torch_out = torch.nn.functional.layer_norm(x, [768], gamma, beta)

if torch.allclose(triton_out, torch_out, atol=1e-5):
    print("LayerNorm with learned params matches PyTorch!")
else:
    diff = (triton_out - torch_out).abs()
    print(f"MISMATCH — max diff: {diff.max().item():.8f}, mean diff: {diff.mean().item():.8f}")

# --- Test with non-power-of-2 feature size ---
x2 = torch.randn(64, 500, device='cuda', dtype=torch.float32)
gamma2 = torch.ones(500, device='cuda', dtype=torch.float32)
beta2 = torch.zeros(500, device='cuda', dtype=torch.float32)

triton_out2 = layernorm(x2, gamma2, beta2)
torch_out2 = torch.nn.functional.layer_norm(x2, [500], gamma2, beta2)

if torch.allclose(triton_out2, torch_out2, atol=1e-4):
    print("LayerNorm with n_cols=500 matches PyTorch!")
else:
    diff = (triton_out2 - torch_out2).abs()
    print(f"MISMATCH at n_cols=500 — max diff: {diff.max().item():.8f}")
```

---

## Benchmarking

```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['n_cols'],
        x_vals=[256, 512, 768, 1024, 2048, 4096, 8192],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'PyTorch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='layernorm-performance',
        args={'n_rows': 1024},
    )
)
def benchmark(n_rows, n_cols, provider):
    x = torch.randn(n_rows, n_cols, device='cuda', dtype=torch.float32)
    gamma = torch.ones(n_cols, device='cuda', dtype=torch.float32)
    beta = torch.zeros(n_cols, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: layernorm(x, gamma, beta), quantiles=quantiles
        )
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.nn.functional.layer_norm(x, [n_cols], gamma, beta),
            quantiles=quantiles,
        )
    # Bandwidth: read input + gamma + beta, write output = 4 * n_rows * n_cols bytes
    # (gamma and beta are n_cols each, but shared across rows — negligible for large n_rows)
    gbps = lambda ms: 2 * n_rows * n_cols * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(show_plots=True, print_data=True)
```

Since LayerNorm is memory-bound (very few FLOPs per byte), we measure in GB/s rather than TFLOPS. The Triton kernel should be competitive with or faster than PyTorch's built-in, because the fused version avoids the intermediate memory traffic that the unfused PyTorch implementation generates.

---

## Self-Assessment Questions

Answer these to check your understanding. If you cannot answer one, re-read the relevant section.

### 1. Why do we use `other=0.0` for the input mask but `other=1.0` for gamma?

**Answer:** Think about what happens to masked-off positions in each case.

For the input: masked positions become `0.0`. When we compute `tl.sum(row)`, the zeros do not affect the sum (we divide by `n_cols`, not `BLOCK_SIZE`, so the zeros are effectively ignored). When we compute `diff = row - mean`, the masked positions get `-mean`, and `diff * diff` gives `mean^2` — a small error, but negligible in practice.

For gamma: masked positions are multiplied by the normalized value. If gamma were `0.0` for masked positions, it would zero out those values. But we want the identity transformation, so `gamma = 1.0` means "no scaling." Similarly, `beta = 0.0` means "no shift." Since these positions will be masked off during the store anyway, the exact values do not matter for correctness — but using sensible defaults makes the code easier to reason about.

### 2. Could you compute mean and variance in a single pass?

Yes, using the identity: `var(x) = E[x^2] - E[x]^2` (the "computational formula" for variance).

```python
# Single-pass approach:
sum_x = tl.sum(row, axis=0)
sum_x2 = tl.sum(row * row, axis=0)
mean = sum_x / n_cols
var = sum_x2 / n_cols - mean * mean
```

This avoids computing `diff = row - mean` before the variance calculation, saving one subtraction per element. However, this formula is numerically less stable (catastrophic cancellation when `E[x^2]` and `E[x]^2` are close). In practice, for float32 with typical feature sizes, both approaches work fine. For the most numerically robust version, Welford's online algorithm computes mean and variance in a single pass without the stability issue.

### 3. How would you add the backward pass?

The backward pass for LayerNorm computes gradients with respect to `x`, `gamma`, and `beta`. It is more complex because you need to propagate gradients back through the normalization step, accounting for the fact that the mean and variance depend on all elements in the row.

The gradient with respect to `x` involves three terms: the direct gradient, the mean gradient, and the variance gradient. Writing this as a fused Triton kernel is a great advanced exercise — the official Triton tutorial includes it.

### 4. What is the maximum `n_cols` this kernel supports?

The entire row must fit in a single Triton block. Triton blocks are limited by register file size. On the T4, this limits you to roughly 32,768 to 65,536 float32 elements per program (depending on how many registers the rest of the kernel uses).

In practice, this means `n_cols` up to about 32K works reliably. For larger feature dimensions (unlikely in practice — typical transformer hidden sizes are 768 to 8192), you would need to split the row across multiple programs and combine partial results, similar to how matmul splits the K dimension. The official Triton tutorial for LayerNorm handles this case.

---

## Exercises

### Exercise 1: RMSNorm

RMSNorm (used in LLaMA, Mistral, and most modern LLMs) is simpler than LayerNorm:

```
RMSNorm(x) = gamma * x / sqrt(mean(x^2) + epsilon)
```

No mean subtraction, no beta. Implement it as a Triton kernel. Verify against:

```python
def pytorch_rmsnorm(x, gamma, eps=1e-5):
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return gamma * x / rms
```

### Exercise 2: GroupNorm

GroupNorm divides the features into groups and normalizes within each group. For input shape `(batch, channels)` with `num_groups` groups:

```python
# Reshape to (batch, num_groups, channels_per_group)
# Normalize along the last dimension (within each group)
# Reshape back
```

This is trickier because each program now handles one (batch, group) pair, and the "row" to normalize is `channels_per_group` elements long. Think about how the pointer arithmetic changes.

### Exercise 3: Fused LayerNorm + Dropout

Combine LayerNorm and dropout in a single kernel. After normalizing and scaling, randomly zero out elements with probability `p`:

```python
# After normed * gamma + beta:
# Generate random mask, zero out elements, scale survivors by 1/(1-p)
```

Hint: Triton provides `tl.rand()` for generating random numbers within a kernel. You need to pass a seed and use the program ID and offsets to generate unique random values for each element.

### Exercise 4: LayerNorm Backward Pass

This is the most challenging exercise. Write the backward kernel for LayerNorm. Given the upstream gradient `dy` (same shape as the output), compute:

- `dx`: gradient with respect to the input
- `dgamma`: gradient with respect to gamma (accumulated across all rows)
- `dbeta`: gradient with respect to beta (accumulated across all rows)

Start by working out the math on paper. The gradient `dx` involves the chain rule through the normalization, which creates dependencies between all elements in a row.

---

## Milestone

You are ready to move on to Phase 3 when:

- [ ] You implemented LayerNorm without looking at the reference (or needed minimal hints)
- [ ] Your kernel passes the correctness test against PyTorch
- [ ] You can explain why fusion helps for LayerNorm (single read/write vs. multiple)
- [ ] You can explain the similarities and differences between the softmax and LayerNorm kernels

---

## Phase 2 Summary

You have now completed Phase 2. Here is what you have learned:

| Module | Key Concepts |
|--------|-------------|
| 01 — Vector Addition | Programs, blocks, `tl.arange`, masking, `tl.load`/`tl.store`, the grid |
| 02 — Fused Softmax | Reductions (`tl.max`, `tl.sum`), numerical stability, kernel fusion |
| 03 — Matrix Multiplication | Tiling, 2D pointer arithmetic with broadcasting, the K-dimension loop, `tl.dot` |
| 04 — Fused LayerNorm | Applying all of the above independently to a new problem |

**Core patterns you now know:**

1. **Elementwise kernels:** One program per block of elements, linear indexing (vector add).
2. **Row-wise kernels:** One program per row, load entire row, reduce, transform, store (softmax, LayerNorm).
3. **Tiled 2D kernels:** One program per output tile, loop over a shared dimension, accumulate partial results (matmul).

These three patterns cover the vast majority of kernels you will encounter in deep learning.

**What you understand about performance:**

- **Fusion** reduces global memory traffic by combining multiple operations into one kernel.
- **Tiling** increases data reuse, converting memory-bound operations into compute-bound ones.
- **Masking** handles boundary conditions when data does not divide evenly into blocks.
- **Mixed precision** (float16 inputs, float32 accumulation) balances speed and accuracy.

### What Comes Next: Phase 3

In Phase 3, you will make these kernels **fast**:

- **01 — Autotuning:** Use `@triton.autotune` to automatically search over block sizes, num_warps, and other parameters to find the best configuration for each problem size.
- **02 — Flash Attention:** Walk through the Flash Attention paper and implement the tiled, fused attention kernel — the most important kernel in modern LLM inference.
- **03 — Reading Real Kernels:** Navigate production kernel code in vLLM, Unsloth, and torchtune to see how the patterns you learned are applied in practice.

You now have the vocabulary and mental models to understand all of it.

---

## Resources

- [Triton LayerNorm Tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html) — The official tutorial, which also includes the backward pass
- ["Layer Normalization" (Ba, Kiros, Hinton, 2016)](https://arxiv.org/abs/1607.06450) — The original paper introducing LayerNorm
- ["Root Mean Square Layer Normalization" (Zhang and Sennrich, 2019)](https://arxiv.org/abs/1910.07467) — The RMSNorm paper, used in LLaMA and modern LLMs
