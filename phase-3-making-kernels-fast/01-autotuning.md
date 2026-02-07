# 01 — Autotuning: Let the Compiler Find the Fast Path

> **Prerequisites:** You have completed Phases 1 and 2. You can write Triton kernels
> for vector addition, softmax, matmul, and layernorm. You understand programs, blocks,
> masking, tiling, and reductions. Now you learn to make kernels fast.

---

## Learning Objectives

By the end of this lesson you will be able to:

1. Explain why kernel performance depends on configuration parameters like block size,
   number of warps, and pipeline stages.
2. Use `@triton.autotune` to automatically search for the best configuration.
3. Use `triton.testing.Benchmark` to profile kernels and compare implementations.
4. Apply grouped program ordering (swizzling) for better L2 cache utilization.
5. Interpret profiling results and reason about why one configuration beats another.

---

## Why Autotuning Matters

In Phase 2 you wrote a matmul kernel with `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=128`,
`BLOCK_SIZE_K=32`. Those values were chosen for you. But were they optimal?

The answer is: **it depends.** The best configuration depends on at least three things:

1. **The GPU.** An A100 has different SRAM sizes, register files, and memory bandwidth
   than a T4. A config that saturates an A100's compute units might starve a T4.
2. **The problem size.** A 512x512 matmul has different tiling sweet spots than a
   4096x4096 matmul. Small matrices may not generate enough blocks to keep the GPU busy
   with large tile sizes.
3. **The data type.** FP16 operations can use Tensor Cores, which favor certain tile
   shapes (multiples of 16). FP32 operations cannot, and the optimal tile shape is
   different.

Manual tuning means trying hundreds of combinations by hand, running benchmarks for
each, and recording results in a spreadsheet. This is tedious, error-prone, and
non-portable (the results only apply to the specific GPU you tested on).

**Autotuning** solves this: you specify a list of configurations to try, and Triton
benchmarks each one at runtime, picks the winner, and caches the result for future
calls. It is exhaustive search over a small, curated configuration space.

---

## What Parameters to Tune

There are three categories of tunable parameters in a Triton kernel.

### 1. Block Sizes

These are the `tl.constexpr` parameters that control how much data each program
processes. In a matmul kernel:

- `BLOCK_SIZE_M` — number of rows of the output tile
- `BLOCK_SIZE_N` — number of columns of the output tile
- `BLOCK_SIZE_K` — depth of the inner-product accumulation tile

The tradeoffs:

```
Larger block sizes
  + More data reuse (each loaded element participates in more computations)
  + Higher arithmetic intensity (better roofline position)
  - More register pressure (each program needs more registers to hold the tile)
  - Fewer programs total (less parallelism across SMs)
  - May exceed SRAM capacity

Smaller block sizes
  + Less register pressure
  + More programs (more parallelism, better SM utilization)
  - Less data reuse (lower arithmetic intensity)
  - More global memory traffic relative to compute
```

### 2. Number of Warps (`num_warps`)

Recall from Phase 1 that a **warp** is a group of 32 threads that execute in lockstep.
A thread block (Triton program) is composed of one or more warps. `num_warps` controls
how many warps each program gets.

The tradeoffs:

```
More warps per program (e.g., 8)
  + More parallelism within a single program
  + Better at hiding memory latency (more warps to schedule while others wait)
  - Each warp gets fewer registers (total registers are split across warps)
  - Only helps if there is enough independent work within the program

Fewer warps per program (e.g., 2)
  + Each warp gets more registers
  + Good for compute-heavy kernels with large tiles
  - Less latency hiding
  - May leave the SM's warp scheduler underutilized
```

A common heuristic: use more warps for smaller block sizes (there is less work per
program, so you need more parallelism within it) and fewer warps for larger block sizes.

### 3. Number of Pipeline Stages (`num_stages`)

This controls **software pipelining** — overlapping the memory loads for the next loop
iteration with the computation of the current iteration. Without pipelining, the program
does:

```
Load tile k=0  ->  Compute tile k=0  ->  Load tile k=1  ->  Compute tile k=1  -> ...
         |---- idle GPU compute -----|          |---- idle GPU compute -----|
```

With `num_stages=3`, the program prefetches future tiles while computing the current one:

```
Load k=0  Load k=1  Load k=2  Load k=3  ...
          Compute k=0  Compute k=1  Compute k=2  ...
                       ^^^^^^^^^^^^^^^^^^^^^^^^
                       Memory and compute overlap!
```

The tradeoffs:

```
More stages (e.g., 4 or 5)
  + Better latency hiding (compute never waits for loads)
  - More SRAM consumed (each prefetched tile sits in SRAM until used)
  - May reduce the number of programs that can run concurrently on an SM

Fewer stages (e.g., 1 or 2)
  + Less SRAM pressure
  - More stalls waiting for memory
```

On newer GPUs with more SRAM (like the A100 with 192 KB per SM), you can afford more
stages. On older GPUs (like the T4 with 64 KB per SM), you may need to keep stages low.

---

## The `@triton.autotune` Decorator

Let's add autotuning to the matmul kernel from Phase 2. Here is the complete code.

```python
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64},
            num_stages=3, num_warps=8,
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32},
            num_stages=4, num_warps=4,
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32},
            num_stages=4, num_warps=4,
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32},
            num_stages=4, num_warps=4,
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32},
            num_stages=4, num_warps=4,
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32},
            num_stages=4, num_warps=4,
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32},
            num_stages=5, num_warps=2,
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32},
            num_stages=5, num_warps=2,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides (number of elements to skip to move one row/col)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block sizes (set by autotune)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Computes C = A @ B where A is (M, K) and B is (K, N)."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # Simple row-major program ordering (we will improve this later)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Offsets for this program's tile
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the first tile of A and B
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Accumulator for the output tile
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension in tiles
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)

    # Store the result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
```

### Understanding Every Part of `@triton.autotune`

**`configs` — The search space:**

Each `triton.Config` specifies one complete configuration to try. The dictionary sets
the `tl.constexpr` parameters, and `num_stages` and `num_warps` set the hardware
parameters.

```python
triton.Config(
    {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64},
    num_stages=3,
    num_warps=8,
)
```

This config says: "Try tile size 128x256, accumulate in chunks of 64 along K, use 3
pipeline stages, and assign 8 warps to each program."

You provide 8, 10, or even 20 configs. Triton will try all of them and keep the fastest.
The configs are not random guesses — they are informed by knowledge of the hardware:

- Block sizes should be powers of 2 (for alignment and vectorized loads).
- `BLOCK_SIZE_M * BLOCK_SIZE_N` should not exceed the register file capacity.
- For Tensor Core operations (FP16), block dimensions should be multiples of 16.
- Configs with small blocks and few warps are good for small matrices.
- Configs with large blocks and many warps are good for large matrices.

**`key=['M', 'N', 'K']` — The cache key:**

This tells Triton: "The optimal config depends on the values of M, N, and K. If any
of these change, re-run autotuning."

Why? Because a 512x512 matmul might be fastest with 32x64 blocks (many small programs
to keep the GPU busy), while a 4096x4096 matmul might be fastest with 128x256 blocks
(fewer, larger programs that maximize data reuse).

The key must list kernel arguments that affect which config is optimal. For a matmul,
the matrix dimensions are the obvious choice. You would NOT include things like
`stride_am` in the key, because the stride does not change the optimal tiling.

### How Autotuning Works at Runtime

When you call the autotuned kernel for the first time with specific values of M, N, K:

```
Call matmul(A, B) with M=1024, N=1024, K=1024

1. Triton checks its cache: "Have I seen key (1024, 1024, 1024) before?"
   -> No. Start autotuning.

2. For each config in configs:
   a. Compile the kernel with this config's parameters
   b. Run several warmup iterations (not timed — lets caches and TLBs settle)
   c. Run several timed iterations
   d. Record the median execution time

3. Pick the config with the lowest median time.

4. Cache the result: key (1024, 1024, 1024) -> winning config

5. Run the kernel with the winning config and return the result.

Subsequent calls with M=1024, N=1024, K=1024:
   -> Cache hit. Skip autotuning. Use the cached winner immediately.

Call with M=2048, N=2048, K=2048:
   -> Cache miss (new key). Repeat steps 1-5.
```

The first call with new dimensions is slow (it runs the kernel 8+ times). All subsequent
calls with the same dimensions are fast (single kernel launch). In practice, ML training
loops call the same shapes repeatedly, so autotuning happens once at the start and then
the cache is always hit.

---

## Grouped Program Ordering (Swizzling)

Before we profile, there is one more optimization to add. It concerns how programs are
**ordered** — which program gets assigned to which output tile.

### The Problem with Row-Major Ordering

The simple approach assigns programs to output tiles in row-major order:

```python
pid_m = pid // num_pid_n
pid_n = pid % num_pid_n
```

For a 4x4 grid of output tiles, the programs are numbered:

```
             Column of output tiles
             0     1     2     3
Row 0:    [ P0    P1    P2    P3  ]
Row 1:    [ P4    P5    P6    P7  ]
Row 2:    [ P8    P9    P10   P11 ]
Row 3:    [ P12   P13   P14   P15 ]
```

Programs that run close in time (e.g., P0, P1, P2, P3) process tiles in the same row
of the output. This means:

- They all read the **same rows** of matrix A (good — reuse in L2 cache).
- They each read **different columns** of matrix B (bad — each program loads fresh B
  data, and by the time P4 runs and needs the same B columns as P0, the data has been
  evicted from L2).

### The Solution: Grouped Ordering

Instead of processing an entire row before moving to the next, we process tiles in
small **groups** that form a roughly square patch:

```python
GROUP_SIZE_M: tl.constexpr = 8  # Process 8 rows of tiles at a time

num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

# How many programs in one "super-column" group?
num_pid_in_group = GROUP_SIZE_M * num_pid_n

# Which group is this program in?
group_id = pid // num_pid_in_group

# The first row of tiles in this group
first_pid_m = group_id * GROUP_SIZE_M

# Handle the last group (may have fewer than GROUP_SIZE_M rows)
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

# Map pid to (row, col) within the group
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

The resulting program ordering (with GROUP_SIZE_M=2 for readability):

```
             Column of output tiles
             0     1     2     3
Row 0:    [ P0    P2    P4    P6  ]
Row 1:    [ P1    P3    P5    P7  ]
Row 2:    [ P8    P10   P12   P14 ]
Row 3:    [ P9    P11   P13   P15 ]
```

Now look at which programs run close together in time:

- P0 (row 0, col 0) and P1 (row 1, col 0): They read different rows of A, but the
  **same columns of B**. B data loaded by P0 is still in L2 when P1 runs.
- P2 (row 0, col 1) and P3 (row 1, col 1): Again, same columns of B.

The group creates a pattern where consecutive programs share B data:

```
A matrix                    B matrix
┌───────────────┐           ┌───────────────┐
│ ███ row for P0│           │ ██  ██        │
│ ███ row for P1│           │ ██  ██        │
│               │           │ col  col      │
│               │           │ for  for      │
│               │           │ P0,1 P2,3     │
└───────────────┘           └───────────────┘

P0 and P1 share B columns  ->  L2 cache hit for B data
P0 and P1 use different A  ->  A data loaded fresh (unavoidable)
```

The net effect: significantly more L2 cache hits for B, which reduces global memory
traffic. The speedup is substantial for large matrices — often 10-20%.

### Adding GROUP_SIZE_M to Autotuning

You can make `GROUP_SIZE_M` a tunable parameter by adding it to your configs:

```python
triton.Config(
    {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
    num_stages=3, num_warps=8,
),
```

---

## Complete Autotuned Matmul with Grouped Ordering

Here is the full, production-style kernel combining autotuning and swizzling.

```python
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
            num_stages=3, num_warps=8,
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
            num_stages=4, num_warps=4,
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
            num_stages=4, num_warps=4,
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
            num_stages=4, num_warps=4,
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
            num_stages=4, num_warps=4,
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
            num_stages=4, num_warps=4,
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
            num_stages=5, num_warps=2,
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
            num_stages=5, num_warps=2,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Autotuned matmul: C = A @ B with grouped program ordering."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # --- Grouped ordering for L2 cache optimization ---
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # --- Compute tile offsets ---
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # --- Accumulate over K dimension ---
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)

    # --- Store the result ---
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    """Launcher function for the autotuned matmul kernel."""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_cuda and b.is_cuda, "Inputs must be on GPU"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c
```

Notice how the grid lambda uses `META['BLOCK_SIZE_M']` and `META['BLOCK_SIZE_N']`.
During autotuning, the grid size changes with each config — a config with 128x256
blocks produces fewer programs than one with 32x64 blocks. The lambda lets the grid
adapt.

---

## Profiling with `triton.testing.Benchmark`

Now let's measure how our autotuned kernel compares to PyTorch's built-in `torch.matmul`
(which calls cuBLAS, NVIDIA's hand-optimized matrix multiply library).

```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],
        x_vals=[128 * i for i in range(2, 33)],  # 256 to 4096
        line_arg='provider',
        line_vals=['cublas', 'triton'],
        line_names=['cuBLAS (torch.matmul)', 'Triton (autotuned)'],
        styles=[('green', '-'), ('blue', '-')],
        ylabel='TFLOPS',
        plot_name='matmul-performance',
        args={},
    )
)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.matmul(a, b), quantiles=quantiles
        )
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matmul(a, b), quantiles=quantiles
        )
    # Compute TFLOPS: matmul does 2*M*N*K FLOPs (multiply + accumulate)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(print_data=True, show_plots=True)
```

### Understanding the Benchmark Code

**`x_names` and `x_vals`:** The benchmark sweeps over matrix sizes. We set M=N=K (square
matrices) from 256 to 4096 in steps of 128. This tests both small and large regimes.

**`line_arg` and `line_vals`:** We compare two implementations: cuBLAS (via
`torch.matmul`) and our Triton kernel.

**`ylabel='TFLOPS'`:** For matmul, the right metric is compute throughput, not memory
bandwidth. A matmul of (M, K) x (K, N) performs 2*M*N*K floating point operations
(each output element requires K multiplies and K-1 additions, which we approximate as
2*K FLOPs).

**`quantiles=[0.5, 0.2, 0.8]`:** Returns the median, 20th, and 80th percentile
execution times. The benchmark uses these to plot error bars, showing performance
variability.

### Reading the Results

You will see a table and a plot. Here is how to interpret them:

```
        M=N=K    cuBLAS (TFLOPS)    Triton (TFLOPS)
        256      2.1                1.8
        512      12.4               11.9
        1024     45.2               42.8
        2048     62.1               59.3
        4096     65.0               63.5
```

(These are illustrative numbers — your actual results will vary by GPU.)

What to look for:

1. **At small sizes (256-512):** cuBLAS usually wins because it has hand-tuned kernels
   for small matrices, including special cases that skip the tiling overhead entirely.
   Triton's autotuning overhead also hurts here.

2. **At medium sizes (1024-2048):** The gap narrows. Triton's autotuner finds good
   configs, and the kernel is well-structured.

3. **At large sizes (4096+):** Triton should approach cuBLAS performance (within 5-15%).
   Both are compute-bound and approaching the GPU's theoretical TFLOPS limit.

4. **Peak TFLOPS vs. theoretical max:** The T4's theoretical FP16 Tensor Core peak is
   ~65 TFLOPS. If you see 60+ TFLOPS, your kernel is excellent. If you see 30 TFLOPS,
   something is wrong (probably a bad config or missing grouped ordering).

---

## Using NVIDIA Nsight Compute (Bonus)

For deep performance analysis beyond what Triton's benchmarks show, NVIDIA provides
Nsight Compute (`ncu`). This is not required for this curriculum, but knowing it exists
is valuable for interviews.

Nsight Compute can tell you:
- What percentage of peak memory bandwidth your kernel achieves
- What percentage of peak compute throughput your kernel achieves
- Whether your kernel is memory-bound or compute-bound
- Specific stall reasons (waiting for memory, register bank conflicts, etc.)

To profile a Triton kernel:

```bash
# Save your kernel to a script (e.g., matmul_profile.py)
ncu --set full python matmul_profile.py
```

This generates a detailed report. The most useful sections are:
- **Speed of Light** — what percentage of the GPU's theoretical peak you are hitting
- **Memory Workload Analysis** — global memory traffic, L2 hit rates, SRAM utilization
- **Compute Workload Analysis** — SM utilization, warp occupancy

For interview purposes, being able to say "I've used Nsight Compute to identify that
my kernel was limited by L2 cache miss rate, which I fixed with grouped ordering" is
a strong signal.

---

## Exercises

### Exercise 1: Expand the Search Space

Add 4 more configs to the autotune decorator. Try:
- A very large tile: `BLOCK_SIZE_M=256, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64`
- A very small tile: `BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32`
- Different warp counts for the same tile: `BLOCK_SIZE_M=128, BLOCK_SIZE_N=128` with
  both `num_warps=4` and `num_warps=8`

Run the benchmark. Do any of your new configs win for any matrix size?

### Exercise 2: Autotune Your Softmax Kernel

Take the softmax kernel from Phase 2 and add `@triton.autotune`. The tunable parameters
are:
- `BLOCK_SIZE` (how many columns each program processes)
- `num_warps`

Create at least 4 configs. Use `key=['n_cols']` since the optimal config depends on the
row length. Benchmark against `torch.softmax` at various column counts (128, 512, 2048,
8192).

### Exercise 3: Config Archaeology

Run the autotuned matmul at two sizes:
- 512x512 x 512x512
- 4096x4096 x 4096x4096

After each run, inspect which config won. You can do this by adding a print statement
in the launcher or by checking `matmul_kernel.best_config`. Questions:
- Which config wins for the small matrix? Why? (Hint: think about how many programs
  each config generates and whether that is enough to fill the GPU.)
- Which config wins for the large matrix? Why? (Hint: think about data reuse.)

### Exercise 4: Pipeline Stages

Create two sets of configs: one with `num_stages=1` (no pipelining) and one with
`num_stages=4`. Keep everything else the same. Benchmark both on a T4 GPU.
- Is there a measurable difference?
- At what matrix size does pipelining start to help?
- Why might a T4 (with 64 KB SRAM) benefit less from pipelining than an A100
  (with 192 KB SRAM)?

---

## Milestone Checklist

You are ready to move on to the next lesson when you can:

- [ ] Add `@triton.autotune` to any kernel with an appropriate set of configs.
- [ ] Explain what `key` does and choose the right key parameters.
- [ ] Explain the tradeoffs of block sizes, num_warps, and num_stages.
- [ ] Implement grouped program ordering and explain why it helps L2 cache utilization.
- [ ] Use `triton.testing.Benchmark` to compare your kernel against PyTorch.
- [ ] Read benchmark results and diagnose whether a kernel is underperforming.

---

## Resources

- [Triton autotuning tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) — the official matmul tutorial with autotuning.
- [NVIDIA Nsight Compute documentation](https://docs.nvidia.com/nsight-compute/) — for deep GPU profiling.
- [Triton Config API reference](https://triton-lang.org/main/python-api/triton.html) — documentation for `triton.Config` and `triton.autotune`.
- [CUTLASS tiling documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md) — NVIDIA's guide to efficient GEMM tiling, which explains the same L2 cache optimization concepts from the CUDA perspective.
