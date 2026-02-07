# 03 — Matrix Multiplication: The Most Important Kernel

> **Prerequisites:** You have completed lessons 01 (vector addition) and 02 (fused
> softmax). You can write Triton kernels with `tl.load`, `tl.store`, masking,
> reductions (`tl.max`, `tl.sum`), and launcher functions. You understand programs,
> the grid, and `BLOCK_SIZE`.

---

## Learning Objectives

By the end of this module, you will be able to:

1. Explain why matrix multiplication is the dominant operation in deep learning
2. Understand **tiling** — what it is, why it exists, and how it maps to the memory hierarchy
3. Write a tiled matrix multiply kernel in Triton from scratch
4. Read and reason about 2D pointer arithmetic with broadcasting

---

## Why Matrix Multiplication Is King

If you only master one GPU kernel, make it this one.

In a transformer model, nearly every operation that matters is a matrix multiply:

- **Linear layers:** `output = input @ weight + bias`
- **Attention QKV projections:** `Q = input @ W_q`, `K = input @ W_k`, `V = input @ W_v`
- **Attention scores:** `scores = Q @ K^T`
- **Attention output:** `output = scores @ V`
- **Feedforward layers:** two linear layers = two matmuls

In GPT-3 (175B parameters), roughly **99% of all floating-point operations** are matrix multiplications. The remaining 1% is everything else: softmax, LayerNorm, activation functions, residual additions. Those operations matter for memory bandwidth, but in terms of raw FLOPs, matmul dominates by two orders of magnitude.

This is why GPU hardware is designed around matmul. NVIDIA's Tensor Cores are special-purpose circuits that compute small matrix multiplies (e.g., 16x16x16) in a single clock cycle. When you write `tl.dot(a, b)` in Triton, it maps down to Tensor Core instructions on supported GPUs (Volta and newer, including the T4 you are using in Colab).

**Matmul is also unique because it is compute-bound, not memory-bound.** We will see exactly why in the arithmetic intensity analysis below. This means that unlike vector addition or softmax — where performance is limited by how fast you can read/write memory — matmul performance is limited by how fast the hardware can multiply and add numbers. This is good news: it means there are enough FLOPs per byte loaded to keep the GPU's compute units busy.

---

## The Naive Approach and Why It Fails

Here is the standard triple-loop matrix multiplication in Python:

```python
# C = A @ B
# A is (M, K), B is (K, N), C is (M, N)
for i in range(M):
    for j in range(N):
        for k in range(K):
            C[i, j] += A[i, k] * B[k, j]
```

Each output element `C[i, j]` is the dot product of row `i` of A with column `j` of B. That dot product touches `K` elements from A and `K` elements from B.

Now count the total memory accesses:

- There are `M * N` output elements.
- Each one requires loading `K` elements from A and `K` elements from B.
- Total memory accesses: `M * N * 2K`

And count the total FLOPs:

- Each output element requires `K` multiply-add operations.
- Total FLOPs: `M * N * K * 2` (one multiply and one add per iteration)

The **arithmetic intensity** — FLOPs per byte of memory traffic — is:

```
Arithmetic Intensity = FLOPs / Bytes
                     = (2 * M * N * K) / (2K * M * N * bytes_per_element)
                     = 1 / bytes_per_element
```

For float32, that is `1/4 = 0.25` FLOPs per byte. For float16, `1/2 = 0.5` FLOPs per byte.

To put this in perspective: the T4 GPU can do ~65 TFLOPS (with Tensor Cores on float16) but can only move ~300 GB/s from global memory. To keep the compute units busy, you need at least `65000 / 300 = ~217` FLOPs per byte. The naive approach gives you 0.5. **You are 434x short of what the hardware needs.** The GPU's compute units are starving — waiting idle most of the time while data trickles in from memory.

This is the problem tiling solves.

---

## Tiling — The Key Insight

The waste in the naive approach comes from **redundant memory loads**. Consider element `A[i, k]`:

- It is needed to compute `C[i, 0]`, `C[i, 1]`, `C[i, 2]`, ..., `C[i, N-1]`
- In the naive version, it gets loaded from global memory **N separate times**

Similarly, `B[k, j]` is loaded **M separate times** (once for each row of C that uses column j).

Tiling eliminates this redundancy by loading a chunk of data once, then reusing it many times from fast memory (SRAM/registers) before loading the next chunk.

### The Tiling Strategy

Instead of computing one element of C at a time, we compute a **tile** — a rectangular block of output elements — all at once.

```
Matrix A (M x K)          Matrix B (K x N)          Matrix C (M x N)
+-----------------+        +-----------------+        +-----------------+
|                 |        |    |BLOCK_N|    |        |                 |
|  +-------+     |        |   +-------+     |        |  +-------+     |
|  | Tile  |     |        |   | Tile  |     |        |  |Output |     |
|  |   A   |BLOCK|        |   |   B   |     |        |  | Tile  |     |
|  | BM x  |_M   |        |   |BK x  |     |        |  |BM x BN|     |
|  |  BK   |     |        |   | BN   |     |        |  |       |     |
|  +-------+     |        |   +-------+     |        |  +-------+     |
|                 |        |                 |        |                 |
+-----------------+        +-----------------+        +-----------------+
```

Here is the plan:

1. **Each Triton program** computes one `BLOCK_SIZE_M x BLOCK_SIZE_N` tile of C.
2. To compute that output tile, we need a strip of A (rows `BLOCK_SIZE_M`, all `K` columns) and a strip of B (all `K` rows, columns `BLOCK_SIZE_N`).
3. We do not load the entire strips at once — that would require too much memory. Instead, we **loop along the K dimension** in chunks of `BLOCK_SIZE_K`.
4. In each iteration: load a `BLOCK_SIZE_M x BLOCK_SIZE_K` tile of A and a `BLOCK_SIZE_K x BLOCK_SIZE_N` tile of B.
5. Multiply those tiles together and add the result to a running accumulator.
6. After processing all K-chunks, the accumulator holds the final output tile.

### The K-Dimension Loop Visualized

Suppose `BLOCK_SIZE_M = BLOCK_SIZE_N = BLOCK_SIZE_K = BK`, and `K = 3 * BK`:

```
Iteration 0:              Iteration 1:              Iteration 2:

A: rows [0:BM]            A: rows [0:BM]            A: rows [0:BM]
   cols [0:BK]               cols [BK:2BK]             cols [2BK:3BK]

   +---------+               +---------+               +---------+
   |  Load   |               |  Load   |               |  Load   |
   | A tile  |               | A tile  |               | A tile  |
   +---------+               +---------+               +---------+
       x                         x                         x
   +---------+               +---------+               +---------+
   |  Load   |               |  Load   |               |  Load   |
   | B tile  |               | B tile  |               | B tile  |
   +---------+               +---------+               +---------+

B: rows [0:BK]            B: rows [BK:2BK]          B: rows [2BK:3BK]
   cols [0:BN]               cols [0:BN]               cols [0:BN]

       =                         =                         =
   acc += A_tile @ B_tile    acc += A_tile @ B_tile    acc += A_tile @ B_tile

Final result: acc holds the complete output tile C[0:BM, 0:BN]
```

Each iteration slides the "window" along the K dimension by `BLOCK_SIZE_K` positions. The A tile slides right; the B tile slides down. The output tile stays fixed — we are just accumulating more partial results into it.

### Why Tiling Helps — The Math

Let us count memory accesses with tiling, using block sizes `BM`, `BN`, `BK`:

**How many times is each element loaded from global memory?**

Consider one element `A[i, k]`:
- It lives in a tile of A that is `BM x BK` in size.
- That tile is loaded by one specific program (the one computing the output tile containing row `i`).
- But the element is used in the dot product for **all BN output columns** in that program's output tile.
- Without tiling, it would have been loaded separately for each of those `BN` output columns.
- With tiling, it is loaded **once** and reused `BN` times.

Similarly, each element of B is loaded once and reused `BM` times.

**The data reuse ratio is BLOCK_SIZE.** If `BM = BN = 64`, each element is reused 64 times instead of being loaded 64 separate times. That is a 64x reduction in global memory traffic.

**New arithmetic intensity:**

Total data loaded from global memory per output tile:
- A tiles: `BM * BK * (K/BK) = BM * K` elements (one strip of A)
- B tiles: `BK * BN * (K/BK) = K * BN` elements (one strip of B)
- Total: `(BM + BN) * K` elements

Total FLOPs per output tile: `2 * BM * BN * K`

```
Arithmetic Intensity = (2 * BM * BN * K) / ((BM + BN) * K * bytes_per_element)
                     = 2 * BM * BN / ((BM + BN) * bytes_per_element)
```

For `BM = BN = 64` and float16 (2 bytes):
```
= 2 * 64 * 64 / ((64 + 64) * 2)
= 8192 / 256
= 32 FLOPs per byte
```

That is **64x better** than the naive approach. With larger block sizes (128), you get 64 FLOPs/byte — approaching the ~217 FLOPs/byte needed to fully saturate the T4's compute. The remaining gap is closed by Tensor Cores, which perform multiple FLOPs per instruction.

### Connection to the Memory Hierarchy

The reason tiling works comes down to physics: different memories have different speeds and sizes.

```
                        Speed        Size
                     +----------+----------+
    Registers        | ~20 TB/s |  ~256 KB |  <-- Accumulator lives here
    L1/Shared Memory | ~12 TB/s |   64 KB  |  <-- Tiles of A, B loaded here
    L2 Cache         |  ~4 TB/s |    4 MB  |  <-- Tiles may be cached here
    Global Memory    |  300 GB/s |   16 GB  |  <-- Full matrices A, B, C
                     +----------+----------+
```

When we load a `BM x BK` tile of A from global memory, it goes into fast on-chip memory (L1/shared memory/registers). Then the `tl.dot()` operation multiplies it against the B tile, reusing each element `BN` times — all at register/L1 speed, which is 40-60x faster than global memory.

**Tiling is the fundamental technique that converts a memory-bound problem into a compute-bound problem.** This is not specific to GPUs — CPU matrix libraries (like Intel MKL) do the same thing with CPU caches. The principle is universal: load data into fast memory, reuse it as much as possible, then move on.

---

## Program-to-Tile Mapping

Before writing the kernel, we need to understand how Triton programs map to output tiles.

We have a 2D grid of output tiles: `ceil(M / BM)` tiles along the row dimension and `ceil(N / BN)` tiles along the column dimension. But Triton programs have a 1D program ID. So we need to convert a 1D `pid` into 2D tile coordinates `(pid_m, pid_n)`.

The simplest approach is **row-major ordering** — tiles are numbered left-to-right, top-to-bottom, like reading a book:

```
Output matrix C (M x N), divided into tiles:

         col tile 0    col tile 1    col tile 2    col tile 3
        +-----------+-----------+-----------+-----------+
row     |           |           |           |           |
tile 0  |  pid = 0  |  pid = 1  |  pid = 2  |  pid = 3  |
        |           |           |           |           |
        +-----------+-----------+-----------+-----------+
row     |           |           |           |           |
tile 1  |  pid = 4  |  pid = 5  |  pid = 6  |  pid = 7  |
        |           |           |           |           |
        +-----------+-----------+-----------+-----------+
row     |           |           |           |           |
tile 2  |  pid = 8  |  pid = 9  |  pid = 10 |  pid = 11 |
        |           |           |           |           |
        +-----------+-----------+-----------+-----------+

num_pid_n = 4  (number of column tiles)

pid_m = pid // num_pid_n    pid_n = pid % num_pid_n

Examples:
  pid = 0  -> pid_m = 0, pid_n = 0  (top-left tile)
  pid = 5  -> pid_m = 1, pid_n = 1  (second row, second column)
  pid = 11 -> pid_m = 2, pid_n = 3  (bottom-right tile)
```

This is the same idea as converting a 1D array index to 2D matrix coordinates — something you may recognize from NumPy's `np.unravel_index`. We will use a more optimized tile ordering (swizzling) in Phase 3; this simple version is easier to reason about.

---

## The Triton Kernel

Here is the complete tiled matmul kernel. Read through it once to get the big picture, then we will go through it line by line.

```python
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides (number of elements to skip to move one position along a dimension)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block sizes (compile-time constants)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # -----------------------------------------------------------
    # Step 1: Figure out which output tile this program computes.
    # -----------------------------------------------------------
    pid = tl.program_id(axis=0)

    # Convert 1D program ID to 2D tile coordinates (row-major)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # -----------------------------------------------------------
    # Step 2: Compute the offsets for this tile.
    # -----------------------------------------------------------
    # Row offsets for the output tile (and the A tile strip)
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # Column offsets for the output tile (and the B tile strip)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # K-dimension offsets (starting position — will advance in the loop)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # -----------------------------------------------------------
    # Step 3: Build 2D pointer arrays for the first tiles of A and B.
    # -----------------------------------------------------------
    # a_ptrs has shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
    # Each element is the memory address of one entry in the A tile.
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)

    # b_ptrs has shape (BLOCK_SIZE_K, BLOCK_SIZE_N)
    # Each element is the memory address of one entry in the B tile.
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Step 4: Accumulate partial results by looping over K chunks.
    # -----------------------------------------------------------
    # The accumulator is float32 for numerical precision, even if inputs are float16.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Bounds-check masks: make sure we don't read past the edges of A or B.
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_bn[None, :] < N)

        # Load tiles from global memory. Out-of-bounds elements become 0.0,
        # which is safe for addition (0 contributes nothing to the dot product).
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # The core operation: multiply A tile by B tile and accumulate.
        # tl.dot maps to Tensor Core instructions on supported hardware.
        accumulator += tl.dot(a, b)

        # Advance pointers and K-offsets for the next chunk.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        offs_k += BLOCK_SIZE_K

    # -----------------------------------------------------------
    # Step 5: Write the output tile to C.
    # -----------------------------------------------------------
    # Convert the accumulator from float32 back to float16 for storage.
    c = accumulator.to(tl.float16)

    # Compute output pointers and mask.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, c, mask=c_mask)
```

---

## Line-by-Line Explanation

### Step 1: Which Tile Am I?

```python
pid = tl.program_id(axis=0)
num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
pid_m = pid // num_pid_n
pid_n = pid % num_pid_n
```

Every Triton program gets a unique `pid`. We launch one program per output tile. `tl.cdiv(N, BLOCK_SIZE_N)` is ceiling division — the number of tiles along the column dimension. The integer division and modulo convert from a flat 1D ID to 2D tile coordinates, exactly as diagrammed in the section above.

### Step 2: Computing Offsets

```python
offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
offs_k = tl.arange(0, BLOCK_SIZE_K)
```

`offs_am` is a 1D vector of row indices. If `pid_m = 2` and `BLOCK_SIZE_M = 64`, then `offs_am = [128, 129, 130, ..., 191]` — the 64 rows of A (and C) that this program is responsible for.

`offs_bn` is a 1D vector of column indices into B (and C).

`offs_k` starts at `[0, 1, 2, ..., BLOCK_SIZE_K - 1]`. It tracks our current position along the K dimension and gets advanced by `BLOCK_SIZE_K` in each loop iteration.

### Step 3: 2D Pointer Arithmetic with Broadcasting

This is the trickiest part, so let us go through it carefully.

```python
a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
```

Let us break down what `[:, None]` and `[None, :]` do. If you are familiar with NumPy broadcasting, this is the exact same mechanism.

`offs_am` has shape `(BLOCK_SIZE_M,)` — it is a 1D vector of row indices.
`offs_k` has shape `(BLOCK_SIZE_K,)` — it is a 1D vector of column indices.

We want a 2D grid of pointers with shape `(BLOCK_SIZE_M, BLOCK_SIZE_K)`, where entry `[i, j]` points to `A[offs_am[i], offs_k[j]]`.

```
offs_am[:, None]  has shape (BLOCK_SIZE_M, 1)     — a column vector
offs_k[None, :]   has shape (1, BLOCK_SIZE_K)      — a row vector

When we multiply and add:
  offs_am[:, None] * stride_am    shape: (BLOCK_SIZE_M, 1)
  + offs_k[None, :] * stride_ak  shape: (1, BLOCK_SIZE_K)
  = result                        shape: (BLOCK_SIZE_M, BLOCK_SIZE_K)  <- broadcast!
```

Visually, for `BLOCK_SIZE_M = 4` and `BLOCK_SIZE_K = 3`:

```
offs_am = [128, 129, 130, 131]    (which rows of A)
offs_k  = [0, 1, 2]               (which columns of A, initially)

           offs_k * stride_ak
           col 0    col 1    col 2
         +--------+--------+--------+
row 128  | A[128,0]| A[128,1]| A[128,2]|   offs_am * stride_am
row 129  | A[129,0]| A[129,1]| A[129,2]|
row 130  | A[130,0]| A[130,1]| A[130,2]|
row 131  | A[131,0]| A[131,1]| A[131,2]|
         +--------+--------+--------+

Each entry is a pointer (memory address) to the corresponding element of A.
```

The same logic applies to `b_ptrs`, but with shape `(BLOCK_SIZE_K, BLOCK_SIZE_N)`:

```python
b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
```

Here `offs_k` becomes the row dimension (K rows of B) and `offs_bn` becomes the column dimension (N columns of B).

**What are strides?** A stride tells you how many elements to skip in memory to move one position along a dimension. For a row-major `(M, K)` matrix A:
- `stride_am = K` (skip one full row of K elements to move down one row)
- `stride_ak = 1` (move one element to go to the next column)

PyTorch computes these for you with `tensor.stride()`.

### Step 4: The Accumulation Loop

```python
accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
```

**Why float32 when the inputs are float16?** Because we are accumulating many partial products. If you accumulate in float16, the limited precision (about 3 decimal digits) causes errors to compound over the K-dimension loop. For `K = 4096`, you might be adding 128 partial products (with `BLOCK_SIZE_K = 32`). Doing this in float32 (about 7 decimal digits) gives much more accurate results. This is a standard practice in mixed-precision training: compute in low precision, accumulate in high precision.

```python
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
```

This loops over the K dimension in chunks. If `K = 4096` and `BLOCK_SIZE_K = 32`, we have 128 iterations. Each iteration processes one "slice" of the dot product.

```python
    a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K)
    b_mask = (offs_k[:, None] < K) & (offs_bn[None, :] < N)
```

These masks handle boundary conditions. When `M` is not a multiple of `BLOCK_SIZE_M`, the bottom-edge tiles extend past the matrix. When `K` is not a multiple of `BLOCK_SIZE_K`, the last K-chunk extends past the matrix. The masks ensure we load zeros for out-of-bounds elements — which is safe because adding zero to the accumulator does not change the result.

```python
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)
    b = tl.load(b_ptrs, mask=b_mask, other=0.0)
```

Load the current tiles. `other=0.0` means out-of-bounds positions get zero, which is correct for matmul (zero contributes nothing to a dot product).

```python
    accumulator += tl.dot(a, b)
```

**This is the most important line in the kernel.** `tl.dot(a, b)` computes the matrix product of the A tile `(BLOCK_SIZE_M, BLOCK_SIZE_K)` and the B tile `(BLOCK_SIZE_K, BLOCK_SIZE_N)`, producing a result of shape `(BLOCK_SIZE_M, BLOCK_SIZE_N)`, which is accumulated into the output tile.

On GPUs with Tensor Cores (including the T4), `tl.dot` maps to special hardware instructions that can multiply small matrices (e.g., 16x16x16) in a single clock cycle. This is why matmul on GPUs is so fast — the hardware has dedicated circuitry for exactly this operation.

```python
    a_ptrs += BLOCK_SIZE_K * stride_ak
    b_ptrs += BLOCK_SIZE_K * stride_bk
    offs_k += BLOCK_SIZE_K
```

Advance the pointers to the next K-chunk:

```
Before:  a_ptrs points to A[rows, 0:BK]
After:   a_ptrs points to A[rows, BK:2BK]

Before:  b_ptrs points to B[0:BK, cols]
After:   b_ptrs points to B[BK:2BK, cols]
```

For A, we move right by `BLOCK_SIZE_K` columns: adding `BLOCK_SIZE_K * stride_ak` shifts every pointer in `a_ptrs` to the next K-chunk. For B, we move down by `BLOCK_SIZE_K` rows: adding `BLOCK_SIZE_K * stride_bk` shifts every pointer in `b_ptrs` down by `BLOCK_SIZE_K` rows.

`offs_k` is updated to keep the boundary masks accurate.

### Step 5: Storing the Result

```python
c = accumulator.to(tl.float16)
```

Convert from the float32 accumulator back to float16 for storage. This is the output dtype.

```python
offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
tl.store(c_ptrs, c, mask=c_mask)
```

This follows the exact same broadcasting pattern as the A and B pointers. We compute a 2D grid of pointers into C, mask for boundary conditions, and store the result.

---

## The Launcher Function

The launcher is the Python function that sets up and calls the kernel:

```python
def matmul(a, b):
    # Validate inputs
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # Calculate the grid: one program per output tile.
    # The lambda is needed because block sizes might be tuned by @triton.autotune later.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),   # stride_am, stride_ak
        b.stride(0), b.stride(1),   # stride_bk, stride_bn
        c.stride(0), c.stride(1),   # stride_cm, stride_cn
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32,
    )
    return c
```

A few things to note:

**`grid` is a lambda.** It takes a `META` dictionary containing the `tl.constexpr` arguments. This is Triton's convention: it allows the grid size to depend on compile-time constants, which is useful when using `@triton.autotune` to search over different block sizes (Phase 3). For now, the block sizes are fixed, so this lambda is equivalent to writing `grid = (triton.cdiv(M, 64) * triton.cdiv(N, 64),)`.

**`.stride(0)` and `.stride(1)`.** PyTorch tensors know their own strides. For a contiguous `(M, K)` tensor, `.stride(0) = K` (skip K elements to go to the next row) and `.stride(1) = 1` (skip 1 element to go to the next column). Passing strides explicitly makes the kernel work with transposed or non-contiguous tensors too.

**Block size choices.** `BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32` is a reasonable starting point for the T4. The optimal values depend on the matrix sizes and the GPU — we will explore autotuning in Phase 3.

---

## Complete Runnable Example

Copy this into a Colab cell with a T4 GPU:

```python
import torch
import triton
import triton.language as tl


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
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_bn[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        accumulator += tl.dot(a, b)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        offs_k += BLOCK_SIZE_K

    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
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
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32,
    )
    return c


# --- Correctness test ---
torch.manual_seed(0)
a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
b = torch.randn((512, 512), device='cuda', dtype=torch.float16)

triton_output = matmul(a, b)
torch_output = torch.matmul(a, b)

print(f"triton_output={triton_output}")
print(f"torch_output={torch_output}")

if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=1e-2):
    print("Correctness check passed!")
else:
    print("Correctness check FAILED")
    diff = (triton_output - torch_output).abs()
    print(f"Max difference: {diff.max().item():.6f}")
    print(f"Mean difference: {diff.mean().item():.6f}")
```

Note: we use `atol=1e-2` for the correctness check because float16 matmul has limited precision. Both Triton and PyTorch accumulate in float32, but the intermediate rounding can cause small differences. A max difference under 0.01 is normal for float16.

---

## Benchmarking

Let us compare our Triton kernel against PyTorch's `torch.matmul`, which calls NVIDIA's cuBLAS library under the hood. cuBLAS is an extremely optimized matrix multiplication library that has been tuned by NVIDIA engineers for years — it is the gold standard.

```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],
        x_vals=[(size, size, size) for size in [128, 256, 512, 1024, 2048, 4096]],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'PyTorch (cuBLAS)'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='TFLOPS',
        plot_name='matmul-performance',
        args={},
    )
)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

benchmark.run(show_plots=True, print_data=True)
```

**Expected results on a T4:**

- For small matrices (128x128): Triton and cuBLAS are both fast, but kernel launch overhead dominates. Results may be noisy.
- For medium matrices (1024x1024): Our basic Triton kernel typically achieves **~15-25 TFLOPS**, which is about 60-80% of cuBLAS.
- For large matrices (4096x4096): The gap narrows. Triton may get within 70-85% of cuBLAS.

The T4's theoretical peak with Tensor Cores on float16 is ~65 TFLOPS. cuBLAS gets close to this peak (~50-60 TFLOPS) because it uses many advanced optimizations: swizzled tile ordering, software pipelining, double buffering, and hand-tuned assembly. Our basic kernel does not have any of these yet — we will add autotuning and swizzled ordering in Phase 3.

The fact that a ~50-line Triton kernel gets 70%+ of a hand-tuned NVIDIA library is remarkable. This is Triton's value proposition: you get most of the performance with a fraction of the effort.

---

## Common Pitfalls

**"My results are slightly wrong."** Check these in order:
1. Are you accumulating in float32? If you accumulate in float16, precision degrades.
2. Are your masks correct? A wrong mask loads garbage values instead of zeros.
3. Did you advance both `a_ptrs` and `offs_k`? If you advance pointers but not `offs_k`, the mask becomes wrong.

**"My kernel is very slow."** Check these:
1. Are your block sizes powers of 2? Non-powers-of-2 prevent Tensor Core usage.
2. Is `BLOCK_SIZE_K` at least 16? Tensor Cores on the T4 need at least 16 for the K dimension.
3. Are your input tensors contiguous? Non-contiguous tensors have stride patterns that can tank performance.

**"I get an illegal memory access error."** This usually means:
1. Your mask logic has a bug, allowing out-of-bounds loads/stores.
2. Your pointer arithmetic computes an address outside the tensor's allocated memory.
3. One of your matrix dimensions is 0.

---

## Exercises

### Exercise 1: Block Size Exploration

Try these combinations on 1024x1024 matrices and benchmark each:
- `BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32`
- `BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32`
- `BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32`
- `BLOCK_SIZE_M=128, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32`
- `BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64`

Which is fastest? Why might larger block sizes help (more data reuse) but also hurt (more register pressure, fewer programs running concurrently)?

### Exercise 2: What Happens Without `other=0.0`?

Change `tl.load(a_ptrs, mask=a_mask, other=0.0)` to `tl.load(a_ptrs, mask=a_mask)` (removing the `other` argument). Run the correctness test. What happens? Why?

Hint: without `other`, masked-off elements contain undefined values from whatever was in memory. These garbage values get multiplied and accumulated, corrupting the result. The effect is worst for matrices whose dimensions are not multiples of the block size, because that is when the mask actually prevents some loads.

### Exercise 3: Column-Major Tile Ordering

Change the pid mapping from row-major to column-major:

```python
# Row-major (original):
num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
pid_m = pid // num_pid_n
pid_n = pid % num_pid_n

# Column-major (try this):
num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
pid_m = pid % num_pid_m
pid_n = pid // num_pid_m
```

Benchmark both. Does performance change? Think about which elements of A and B are accessed by programs with adjacent `pid` values in each ordering. (We will revisit this in Phase 3, where a more sophisticated "swizzled" ordering gives real performance gains by improving L2 cache hit rates.)

### Exercise 4: Matrix Size Sweep

Benchmark at sizes: 128, 256, 512, 768, 1000, 1024, 2048, 3000, 4096. Two things to look for:
1. At what size does Triton start matching or beating PyTorch? (Hint: Triton has higher kernel launch overhead, so it loses on small matrices.)
2. How does performance differ between "clean" sizes (powers of 2 like 1024, 2048) and "awkward" sizes (like 768, 1000, 3000)? Why?

---

## Milestone

You can now:

- [ ] Explain why matmul dominates deep learning compute
- [ ] Explain what tiling is and why it improves arithmetic intensity
- [ ] Trace through the K-dimension loop and explain what each iteration does
- [ ] Read 2D pointer arithmetic with broadcasting (`[:, None]` and `[None, :]`)
- [ ] Write a tiled matmul kernel in Triton from scratch (close the file and try it)
- [ ] Explain why the accumulator must be float32 even when inputs are float16

If you can close this file and re-implement the matmul kernel in Colab from memory — getting the pointer arithmetic, the loop, and the masks right — you have genuinely internalized this material. It will probably take two or three tries before you can do it cleanly. That is normal.

---

## What Comes Next

In the next module, **04 — Fused LayerNorm**, you will implement a kernel entirely on your own. It is a solo exercise: you get the specification and hints, and you try to write it before looking at the reference solution. This will test whether you have truly internalized the patterns from vector addition, softmax, and matmul.

---

## Resources

- [Triton Matrix Multiplication Tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) — The official tutorial, which includes additional optimizations like swizzled tile ordering
- [PMPP Chapter 4: Tiled Matrix Multiplication](https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands/dp/0323912311) — The textbook treatment with CUDA, but the tiling concept is identical
- [Matrix Multiplication Background (GPU Gems)](https://developer.nvidia.com/gpugems/gpugems2/part-iv-general-purpose-computation-gpus-primer/chapter-44-gpu-computing-era) — Historical perspective on why matmul maps so well to GPUs
