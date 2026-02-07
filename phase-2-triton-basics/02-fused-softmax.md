# 02 — Fused Softmax: Your First Reduction Kernel

> **Prerequisites:** You have completed lesson 01 (vector addition) and can write a
> basic Triton kernel with `tl.load`, `tl.store`, masking, and a launcher. You
> understand `program_id`, `BLOCK_SIZE`, and the grid.

---

## Learning Objectives

By the end of this lesson you will be able to:

1. Implement softmax from scratch in a Triton kernel.
2. Write **reduction** operations (max, sum) inside a kernel — a fundamentally
   different pattern from the element-wise operations in lesson 01.
3. Apply the numerical stability trick (subtract the max) and explain why it is
   necessary.
4. Explain what "fused" means and why a fused kernel is faster than separate
   PyTorch operations, connecting the answer to the memory hierarchy from Phase 1.

---

## Why Softmax?

Softmax is one of the most important operations in modern deep learning:

- It is the final step in every **attention head** in every transformer model.
  GPT, LLaMA, BERT, Stable Diffusion — they all compute softmax thousands of times
  per forward pass.
- It involves **reductions** (computing the max and sum of a row), which is a new
  programming pattern beyond the element-wise operations you saw in lesson 01.
- It is a textbook example of why **kernel fusion** matters for performance.

By the end of this lesson, you will have a kernel that outperforms PyTorch's
built-in `torch.softmax` for moderate-sized inputs, and you will understand
exactly why.

---

## Softmax Math — Quick Refresher

Given a vector **x** of length *n*, softmax maps it to a probability distribution:

```
softmax(x_i) = exp(x_i) / sum_j(exp(x_j))
```

Each output is between 0 and 1, and all outputs sum to 1.

### The Numerical Stability Problem

There is a serious practical problem with the naive formula above. Consider
`x = [1000, 1001, 1002]`:

```
exp(1000) = 1.97 * 10^434     <-- this overflows float32 (max ~3.4 * 10^38)
```

The result is `inf`, and `inf / inf = NaN`. Your model produces garbage.

### The Fix: Subtract the Maximum

A beautiful mathematical property of softmax is that subtracting any constant *c*
from every element does not change the result:

```
softmax(x_i) = exp(x_i - c) / sum_j(exp(x_j - c))     for any constant c
```

Proof: `exp(x_i - c) = exp(x_i) * exp(-c)`. The `exp(-c)` factor appears in both
the numerator and denominator and cancels out.

By choosing `c = max(x)`, the largest exponent becomes `exp(0) = 1`, and all other
exponents are `exp(negative) < 1`. No overflow is possible.

The numerically stable softmax formula is:

```
m = max(x)
softmax(x_i) = exp(x_i - m) / sum_j(exp(x_j - m))
```

This is what every production softmax implementation uses, and it is what we will
implement.

---

## Why Fusion Matters — The Core Insight

### How PyTorch Computes Softmax (Unfused)

When you write `torch.softmax(x, dim=-1)`, PyTorch calls a single optimized CUDA
kernel under the hood. But let's imagine we had to build softmax from basic PyTorch
operations to understand the fusion problem:

```python
# Each line launches a separate CUDA kernel
m = x.max(dim=-1, keepdim=True).values   # Kernel 1: read x,       write m
safe = x - m                              # Kernel 2: read x and m, write safe
e = torch.exp(safe)                       # Kernel 3: read safe,    write e
s = e.sum(dim=-1, keepdim=True)           # Kernel 4: read e,       write s
out = e / s                               # Kernel 5: read e and s, write out
```

Five separate kernel launches. Here is the problem — trace the data movement for
a single row of *n* elements:

```
                     HBM (Slow)                      SM Registers (Fast)
                    +-----------+                    +------------------+
Kernel 1: Read x    | n floats  | -----> compute max  |                  |
          Write m   | 1 float   | <-----              |                  |
                    +-----------+                    +------------------+

Kernel 2: Read x,m  | n+1 floats| -----> subtract     |                  |
          Write safe| n floats  | <-----              |                  |
                    +-----------+                    +------------------+

Kernel 3: Read safe | n floats  | -----> exp          |                  |
          Write e   | n floats  | <-----              |                  |
                    +-----------+                    +------------------+

Kernel 4: Read e    | n floats  | -----> sum           |                  |
          Write s   | 1 float   | <-----              |                  |
                    +-----------+                    +------------------+

Kernel 5: Read e,s  | n+1 floats| -----> divide        |                  |
          Write out | n floats  | <-----              |                  |
                    +-----------+                    +------------------+

Total HBM reads:  n + (n+1) + n + n + (n+1) = ~5n
Total HBM writes: 1 + n + n + 1 + n         = ~3n
Grand total: ~8n memory operations through slow HBM
```

### How a Fused Kernel Works

A fused kernel does everything in a single pass:

```
                     HBM (Slow)                      SM Registers (Fast)
                    +-----------+                    +------------------+
Fused:   Read x     | n floats  | -----> max         |  row data lives  |
                    |           |        subtract    |  here the whole  |
                    |           |        exp          |  time — never    |
                    |           |        sum          |  written back    |
                    |           |        divide       |  until the end   |
         Write out  | n floats  | <-----              |                  |
                    +-----------+                    +------------------+

Total HBM reads:  n
Total HBM writes: n
Grand total: 2n memory operations through slow HBM
```

The same arithmetic happens in both cases — the same additions, the same `exp`
calls, the same divisions. But the fused kernel moves **4x less data** through the
slow HBM bottleneck. Since softmax is memory-bound (very few FLOPs per byte), this
translates directly into a roughly 4x speedup.

This is the central lesson:

```
+-----------------------------------------------------------------------+
|                                                                       |
|  Kernel fusion is NOT about doing less math.                          |
|  It is about doing less memory traffic.                               |
|                                                                       |
|  The data stays in fast registers/SRAM instead of being written       |
|  to slow HBM and read back between every operation.                   |
|                                                                       |
+-----------------------------------------------------------------------+
```

---

## The Kernel Design

Our strategy: **one program per row.** Each program loads an entire row of the input
matrix into registers, computes the full softmax within those registers, and writes
the result back. No inter-program communication is needed because each row is
independent.

```
Input matrix (n_rows x n_cols):

     col 0   col 1   col 2  ...  col n_cols-1
    +-------+-------+-------+---+-------------+
row 0 |       |       |       |   |             |  <-- Program 0 handles this row
    +-------+-------+-------+---+-------------+
row 1 |       |       |       |   |             |  <-- Program 1 handles this row
    +-------+-------+-------+---+-------------+
row 2 |       |       |       |   |             |  <-- Program 2 handles this row
    +-------+-------+-------+---+-------------+
  :   :       :       :       :   :             :
    +-------+-------+-------+---+-------------+
row m |       |       |       |   |             |  <-- Program m handles this row
    +-------+-------+-------+---+-------------+

Grid size = n_rows (one program per row)
BLOCK_SIZE >= n_cols (entire row must fit in one block)
```

This is a different parallelization strategy than vector addition. In lesson 01, we
split a single vector across many programs. Here, we give each program a complete
row to process independently.

---

## The Kernel — Line by Line

```python
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one complete row
    row_idx = tl.program_id(0)

    # Compute the pointer to the start of this program's row
    row_start_ptr = input_ptr + row_idx * input_row_stride

    # Generate column offsets: [0, 1, 2, ..., BLOCK_SIZE-1]
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # Mask: only load columns that actually exist
    mask = col_offsets < n_cols

    # Load the entire row (masked positions get -inf)
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))

    # Step 1: Find the maximum value in the row (for numerical stability)
    row_max = tl.max(row, axis=0)

    # Step 2: Subtract the max (prevents overflow in exp)
    safe_row = row - row_max

    # Step 3: Exponentiate
    numerator = tl.exp(safe_row)

    # Step 4: Sum the exponentials
    denominator = tl.sum(numerator, axis=0)

    # Step 5: Normalize
    softmax_output = numerator / denominator

    # Store the result
    output_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_start_ptr + col_offsets, softmax_output, mask=mask)
```

Now let's go through every detail.

---

### The Parameters

```python
def softmax_kernel(
    output_ptr,          # Pointer to the output tensor
    input_ptr,           # Pointer to the input tensor
    input_row_stride,    # Number of elements between consecutive rows of input
    output_row_stride,   # Number of elements between consecutive rows of output
    n_cols,              # Number of columns in each row
    BLOCK_SIZE: tl.constexpr,  # Must be >= n_cols, must be a power of 2
):
```

**`input_row_stride` and `output_row_stride` — Understanding Strides:**

In Phase 1 you learned that GPU memory is a flat, one-dimensional sequence of bytes.
A 2D matrix is stored by laying out each row consecutively:

```
Logical view:             Memory layout (row-major):

  [[a, b, c, d],          [a, b, c, d, e, f, g, h, i, j, k, l]
   [e, f, g, h],           ^           ^           ^
   [i, j, k, l]]           row 0       row 1       row 2
                            stride=4 -->|
```

The **stride** tells you how many elements to skip in the flat memory to get from
the start of one row to the start of the next row. For a contiguous tensor with
`n_cols = 4`, the stride is 4. For row *i*, the data starts at
`input_ptr + i * stride`.

Why pass the stride as a parameter instead of just using `n_cols`? Because PyTorch
tensors can have non-contiguous memory layouts (for example, after a `.transpose()`
or `.slice()`). The stride accounts for this. Using `tensor.stride(0)` gives the
correct stride regardless of memory layout.

**`BLOCK_SIZE: tl.constexpr`:**
Same as in lesson 01 — a compile-time constant. Here it must be at least as large
as `n_cols` because each program must load the entire row at once. We round up to
the next power of 2 (more on why below).

---

### Identifying the Row

```python
row_idx = tl.program_id(0)
row_start_ptr = input_ptr + row_idx * input_row_stride
```

Each program gets a unique `row_idx` from 0 to `n_rows - 1`. Multiplying by the
stride and adding to the base pointer gives us a pointer to the first element of
this program's row.

```
input_ptr ---> [row 0 data...][row 1 data...][row 2 data...]...
                ^               ^               ^
                row_idx=0       row_idx=1       row_idx=2
                + 0*stride      + 1*stride      + 2*stride
```

---

### Column Offsets and Masking

```python
col_offsets = tl.arange(0, BLOCK_SIZE)
mask = col_offsets < n_cols
```

`tl.arange(0, BLOCK_SIZE)` generates `[0, 1, 2, ..., BLOCK_SIZE-1]`. These are the
column indices within the row.

Since `BLOCK_SIZE` is rounded up to the next power of 2, it may be larger than
`n_cols`. For example, if `n_cols = 100`, then `BLOCK_SIZE = 128`, and columns
100 through 127 do not exist. The mask is `True` for valid columns and `False` for
the padding.

---

### Loading the Row with `-inf` Padding

```python
row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
```

This is the most subtle line in the kernel. The `other=-float('inf')` argument
tells `tl.load`: "For masked-out positions (where `mask` is `False`), use the
value negative infinity instead of loading from memory."

Why `-inf` and not `0`? Because of how the subsequent operations work:

- **For `tl.max`:** We need `max(row)` to be the max of the *real* values. Since
  `-inf` is less than any real number, padded positions never become the max.
- **For `tl.exp`:** `exp(-inf) = 0`, so padded positions contribute zero to the
  sum. They are effectively invisible.

If we used `other=0` instead, a padded position would have value 0. Then
`exp(0) = 1`, which would incorrectly add 1 to the denominator for each padded
position, producing wrong results.

This is a common and important pattern:

```
+-----------------------------------------------------------------------+
|                                                                       |
|  When masking, always choose the "other" value based on how           |
|  downstream operations will interact with it:                         |
|                                                                       |
|    - Before max:  use -inf  (won't become the max)                    |
|    - Before sum:  use  0    (won't affect the sum)                    |
|    - Before min:  use +inf  (won't become the min)                    |
|                                                                       |
|  For softmax, -inf handles both max and sum correctly because         |
|  exp(-inf) = 0.                                                       |
|                                                                       |
+-----------------------------------------------------------------------+
```

---

### The Reduction Operations

Now we reach the new programming pattern that distinguishes this kernel from
vector addition.

```python
row_max = tl.max(row, axis=0)
```

In lesson 01, every operation was element-wise: load a vector, add a vector, store
a vector. Here, `tl.max` takes a vector of `BLOCK_SIZE` elements and reduces it to
a **single scalar** — the maximum value. This is a **reduction** operation.

Under the hood, Triton generates an efficient parallel reduction tree. Conceptually:

```
  row = [3, 7, 1, 9, 2, 8, 4, 6]     (BLOCK_SIZE = 8)

  Step 1: Compare pairs
          [max(3,7), max(1,9), max(2,8), max(4,6)]  =  [7, 9, 8, 6]

  Step 2: Compare pairs again
          [max(7,9), max(8,6)]  =  [9, 8]

  Step 3: Final comparison
          max(9, 8) = 9

  Result: row_max = 9
```

This completes in `log2(BLOCK_SIZE)` steps, not `BLOCK_SIZE` steps. For
`BLOCK_SIZE = 1024`, that is 10 steps instead of 1024 — a massive speedup from
parallelism. This is also why `BLOCK_SIZE` must be a power of 2: the reduction tree
divides evenly at every level.

The `axis=0` argument means "reduce along the first (and only) axis of this
vector." This is a reduction *within a single program*, not across programs. Each
program independently finds the max of its own row.

---

### Subtract, Exponentiate, Sum, Divide

```python
safe_row = row - row_max           # Subtract max (element-wise: vector - scalar)
numerator = tl.exp(safe_row)       # Exponentiate (element-wise)
denominator = tl.sum(numerator, axis=0)  # Sum reduction (vector -> scalar)
softmax_output = numerator / denominator  # Normalize (element-wise: vector / scalar)
```

After finding `row_max`, these four lines implement the softmax formula directly:

1. **`safe_row = row - row_max`:** Broadcasts the scalar `row_max` across all
   elements and subtracts. After this, the largest element is 0 and all others are
   negative. This prevents overflow in the next step.

2. **`numerator = tl.exp(safe_row)`:** Computes `exp()` element-wise. Since all
   values are <= 0, all results are in the safe range [0, 1].

3. **`denominator = tl.sum(numerator, axis=0)`:** Another reduction, this time
   summing all the exponentials into a single scalar. Same parallel reduction tree
   as `tl.max`, but with addition instead of max.

4. **`softmax_output = numerator / denominator`:** Broadcasts the scalar denominator
   and divides every element. The result is a proper probability distribution.

All of this happens in registers. The data was loaded from HBM once at the
beginning, and it will be written to HBM once at the end. No intermediate results
ever touch global memory. This is what makes the fused kernel fast.

---

### Storing the Result

```python
output_start_ptr = output_ptr + row_idx * output_row_stride
tl.store(output_start_ptr + col_offsets, softmax_output, mask=mask)
```

Same pattern as loading, but in reverse. We compute the pointer to the start of this
program's output row and store the softmax results. The mask ensures we only write
to valid columns.

---

## The Launcher

```python
def softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2, "Input must be 2D"
    n_rows, n_cols = x.shape

    # Allocate output
    output = torch.empty_like(x)

    # BLOCK_SIZE must be a power of 2 and >= n_cols
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # One program per row
    grid = (n_rows,)

    softmax_kernel[grid](
        output, x,
        x.stride(0), output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
```

### `triton.next_power_of_2(n_cols)`

This rounds `n_cols` up to the nearest power of 2. For example:
- `n_cols = 100` becomes `BLOCK_SIZE = 128`
- `n_cols = 1000` becomes `BLOCK_SIZE = 1024`
- `n_cols = 1024` stays `BLOCK_SIZE = 1024`

Powers of 2 are important for two reasons:

1. **Efficient reductions.** The parallel reduction tree (for `tl.max` and `tl.sum`)
   divides the data in half at each step. With a power-of-2 size, every step divides
   evenly with no leftover elements.

2. **Memory alignment.** GPU memory is organized in aligned segments. Power-of-2
   block sizes align naturally with these segments, which means loads and stores can
   be done with fewer memory transactions.

### `x.stride(0)`

PyTorch's `.stride()` method returns the number of elements (not bytes) you must
skip in the flat memory layout to advance by one step along each dimension.
For a contiguous 2D tensor of shape `(n_rows, n_cols)`:

```python
x.stride(0)  # = n_cols  (skip n_cols elements to reach the next row)
x.stride(1)  # = 1       (skip 1 element to reach the next column)
```

We pass `x.stride(0)` as `input_row_stride` so the kernel knows how to navigate
between rows in memory.

### The Grid

```python
grid = (n_rows,)
```

We launch one program per row. The grid is simply the number of rows. This is a
1-tuple (one-dimensional grid). Program 0 gets row 0, program 1 gets row 1, and so
on. All programs execute simultaneously (up to the GPU's capacity to schedule them).

---

## Complete Runnable Code

Paste this into a Colab notebook with a T4 runtime.

```python
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))

    row_max = tl.max(row, axis=0)
    safe_row = row - row_max
    numerator = tl.exp(safe_row)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    output_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_start_ptr + col_offsets, softmax_output, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2, "Input must be 2D"
    n_rows, n_cols = x.shape

    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    grid = (n_rows,)
    softmax_kernel[grid](
        output, x,
        x.stride(0), output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


# ---- Verify correctness ----
x = torch.randn(128, 512, device='cuda', dtype=torch.float32)

triton_output = softmax(x)
torch_output = torch.softmax(x, dim=-1)

assert torch.allclose(triton_output, torch_output, atol=1e-6), "Results don't match!"
print("Correctness verified")

# Verify that outputs are proper probability distributions
assert torch.allclose(triton_output.sum(dim=-1), torch.ones(128, device='cuda'), atol=1e-5)
print("All rows sum to 1 (valid probability distributions)")
```

---

## Benchmarking: Fused Triton vs PyTorch

```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['n_cols'],
        x_vals=[128 * i for i in range(2, 100)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'PyTorch'],
        styles=[('blue', '-'), ('red', '-')],
        ylabel='GB/s',
        plot_name='softmax-performance',
        args={'n_rows': 4096},
    )
)
def benchmark(n_rows, n_cols, provider):
    x = torch.randn(n_rows, n_cols, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.softmax(x, dim=-1), quantiles=quantiles
        )
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: softmax(x), quantiles=quantiles
        )
    # 2 memory ops per element: 1 read + 1 write (this is the fused count)
    gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(print_data=True, show_plots=True)
```

### Reading the Results

**What to expect on a T4:**

- For small `n_cols` (under ~256), the Triton kernel may be comparable to or slower
  than PyTorch. PyTorch's built-in `torch.softmax` is a single fused CUDA kernel
  that is heavily optimized, and for small rows the kernel launch overhead dominates.
- For moderate `n_cols` (256 to ~4096), the Triton kernel often matches or beats
  PyTorch because our kernel is doing the same fusion that PyTorch's native kernel
  does, and Triton's code generation is competitive.
- For large `n_cols` (beyond ~8192), our kernel will run into trouble because the
  entire row must fit in `BLOCK_SIZE`, and very large block sizes consume too many
  registers per program (leaving too few programs to keep the GPU busy). This is a
  limitation we will address later.

**Why GB/s again?**

Even though softmax involves more arithmetic than vector addition (exp, division),
it is still predominantly **memory-bound** for typical row sizes. The number of
FLOPs per byte loaded is still small, placing softmax in the memory-bound region of
the roofline chart. The bottleneck is HBM bandwidth, not compute throughput.

---

## Understanding the Limitation: `BLOCK_SIZE >= n_cols`

Our kernel requires the entire row to fit within one program's `BLOCK_SIZE`. This
means:

- `n_cols = 512` needs `BLOCK_SIZE = 512` (uses 2KB of registers per program for float32)
- `n_cols = 4096` needs `BLOCK_SIZE = 4096` (uses 16KB of registers per program)
- `n_cols = 16384` needs `BLOCK_SIZE = 16384` (uses 64KB of registers — may exceed
  the SM's register file)

What happens when the row is too long to fit in registers? The kernel either fails
to compile or suffers severe performance degradation due to register spilling
(overflow into slow local memory).

The solution is to process the row in **multiple passes**: load a chunk, update a
running max and running sum, load the next chunk, and so on. This is called
**online softmax** and is the key idea behind **Flash Attention**, which you will
study in Phase 3. For now, the single-pass approach works well for rows up to a few
thousand elements, which covers most attention head dimensions (64, 128, 256) in
practice.

---

## Step-by-Step Execution Trace

Let's trace through the kernel for a tiny example to solidify understanding.

```
Input:  x = [[2.0, 1.0, 0.1],     (shape: 2 x 3)
              [1.0, 2.0, 3.0]]

n_rows = 2, n_cols = 3, BLOCK_SIZE = next_power_of_2(3) = 4

Grid: (2,) -- two programs, one per row

=== Program 0 (row_idx = 0) ===

  row_start_ptr = input_ptr + 0 * 3 = input_ptr

  col_offsets = [0, 1, 2, 3]
  mask        = [T, T, T, F]     (column 3 does not exist)

  row = tl.load(..., mask=mask, other=-inf)
      = [2.0, 1.0, 0.1, -inf]

  row_max = max(2.0, 1.0, 0.1, -inf) = 2.0

  safe_row = [2.0 - 2.0, 1.0 - 2.0, 0.1 - 2.0, -inf - 2.0]
           = [0.0,       -1.0,       -1.9,       -inf       ]

  numerator = [exp(0.0), exp(-1.0), exp(-1.9), exp(-inf)]
            = [1.0,      0.368,     0.150,     0.0      ]

  denominator = 1.0 + 0.368 + 0.150 + 0.0 = 1.518

  softmax_output = [1.0/1.518, 0.368/1.518, 0.150/1.518, 0.0/1.518]
                 = [0.659,     0.242,       0.099,       0.0       ]

  Store [0.659, 0.242, 0.099] (mask prevents writing the 4th value)

  Check: 0.659 + 0.242 + 0.099 = 1.000  (valid probability distribution)

=== Program 1 (row_idx = 1) ===

  (Runs simultaneously with Program 0)

  row = [1.0, 2.0, 3.0, -inf]
  row_max = 3.0
  safe_row = [-2.0, -1.0, 0.0, -inf]
  numerator = [0.135, 0.368, 1.0, 0.0]
  denominator = 1.503
  softmax_output = [0.090, 0.245, 0.665, 0.0]

  Store [0.090, 0.245, 0.665]

  Check: 0.090 + 0.245 + 0.665 = 1.000
```

Notice how the `-inf` padding values flow through harmlessly at every step.

---

## Comparing Unfused vs Fused: A Summary

```
+-----------------------------------------------------------------------+
|              Unfused (5 kernels)     |     Fused (1 kernel)           |
|--------------------------------------|--------------------------------|
| HBM reads:   ~5n per row            | HBM reads:   n per row         |
| HBM writes:  ~3n per row            | HBM writes:  n per row         |
| Total traffic: ~8n                   | Total traffic: 2n              |
| Kernel launches: 5                   | Kernel launches: 1             |
| Intermediate tensors: 3             | Intermediate tensors: 0        |
| GPU memory for intermediates: O(n)  | GPU memory for intermediates: 0|
|                                      |                                |
| FLOPs: same                          | FLOPs: same                    |
+-----------------------------------------------------------------------+

The fused kernel does the same math with ~4x less memory traffic.
For a memory-bound operation, less memory traffic = faster.
```

---

## Exercises

### Exercise 1: Softmax with Temperature

In language models, softmax is often applied with a **temperature** parameter that
controls how "sharp" or "flat" the distribution is:

```
softmax(x_i / T)
```

- `T < 1` makes the distribution sharper (more confident, less random).
- `T > 1` makes the distribution flatter (less confident, more random).
- `T = 1` is the standard softmax.

Modify the kernel to accept a `temperature` parameter. You only need to add one line
before computing `row_max`: divide the loaded row by the temperature.

Verify your result against `torch.softmax(x / temperature, dim=-1)`.

### Exercise 2: Log-Softmax

`log_softmax` is used in the cross-entropy loss function, which is the standard
training loss for classification and language modeling. The naive approach:

```python
log_softmax(x) = log(softmax(x))
```

is numerically unstable because `softmax` can produce values very close to 0, and
`log(0) = -inf`.

The numerically stable version is:

```
log_softmax(x_i) = (x_i - max(x)) - log(sum_j(exp(x_j - max(x))))
```

Implement this as a fused Triton kernel. Hint: you already compute `safe_row` and
`denominator` — you just need to compute `safe_row - log(denominator)` instead of
`numerator / denominator`.

Verify against `torch.log_softmax(x, dim=-1)`.

### Exercise 3: Performance Crossover

Profile your Triton softmax against `torch.softmax` at various row sizes. At what
`n_cols` does the Triton version start to beat PyTorch? At what `n_cols` does it
start to lose? Create a plot showing both curves. (Hint: the benchmark code above
already does this — study the output.)

### Exercise 4: The Large-Row Problem

Try running the kernel with `n_cols = 32768`. What happens? You may see a
compilation error or very poor performance. This is because `BLOCK_SIZE = 32768`
requires each program to use 32768 registers for the row data alone, which exceeds
what a single SM can provide.

Think about how you would handle this. What if you processed the row in two passes —
first find the max, then compute the softmax using that max? This is the idea behind
**online softmax** (see the Milakov & Gimelshein paper in Resources). You will
implement this multi-pass approach when you study Flash Attention in Phase 3.

---

## Milestone Checklist

You are ready to move on to the next lesson when you can:

- [ ] Explain why subtracting the max before `exp` prevents numerical overflow.
- [ ] Write a fused softmax kernel from scratch without reference.
- [ ] Explain how `tl.max` and `tl.sum` perform parallel reductions within a single
      program (and draw the reduction tree).
- [ ] Explain why `other=-float('inf')` is the correct padding value for softmax
      (and what would go wrong with 0).
- [ ] Explain why a fused kernel is faster than 5 separate kernels, even though the
      same number of FLOPs are performed. (Answer: reduced HBM traffic.)
- [ ] Explain the limitation of requiring `BLOCK_SIZE >= n_cols` and sketch how you
      might work around it.

---

## Resources

- [Triton fused softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) — the official tutorial this lesson builds on.
- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) — Milakov and Gimelshein, 2018. Describes the multi-pass algorithm for softmax that handles rows longer than one block. This is foundational for Flash Attention.
- [Triton language reference: reductions](https://triton-lang.org/main/python-api/triton.language.html) — documentation for `tl.max`, `tl.sum`, and other reduction operations.
