# 01 — Vector Addition: Your First Triton Kernel

> **Prerequisites:** You have completed Phase 1 and understand threads, blocks, warps,
> the GPU memory hierarchy (registers, SRAM/shared memory, global memory/HBM), and the
> roofline model. You have Python and PyTorch experience but have never written GPU
> kernel code.

---

## Learning Objectives

By the end of this lesson you will be able to:

1. Write a complete Triton kernel and launch it from Python.
2. Explain the relationship between Triton *programs* and GPU thread blocks.
3. Use masking to prevent out-of-bounds memory accesses.
4. Benchmark a Triton kernel against PyTorch and interpret the results using the
   roofline model from Phase 1.

---

## Why Start with Vector Addition?

Vector addition is the "Hello, World!" of GPU programming. Given two vectors
**x** and **y** of length *n*, we want to compute **output[i] = x[i] + y[i]** for
every *i*.

This is the simplest possible kernel for three reasons:

- **No reductions.** Every output element depends on exactly one element from each
  input. There is no cross-element communication (no max, no sum, no attention).
- **No shared memory.** Each element is independent, so we never need to coordinate
  between threads or cache data in SRAM.
- **No tiling.** There is only one dimension to iterate over.

This simplicity lets you focus entirely on *how the Triton programming model works*
without being distracted by algorithmic complexity. Once you understand the
scaffolding here, every future kernel builds on the same patterns.

---

## Triton's Programming Model — Explained from Scratch

### From CUDA Threads to Triton Programs

In Phase 1 you learned that GPUs launch thousands of threads organized into blocks.
In CUDA, you write code from the perspective of a single thread:

```
// CUDA pseudocode — one thread, one element
int i = blockIdx.x * blockDim.x + threadIdx.x;
output[i] = x[i] + y[i];
```

Triton raises the abstraction level. Instead of thinking about individual threads,
you think about **programs** that each operate on a **block of data**. The Triton
compiler handles mapping your program onto the actual hardware threads and warps
inside each block.

### The Python Analogy

The best way to understand Triton's execution model is to imagine splitting your
work in plain Python:

```python
# Python equivalent of what Triton does
BLOCK_SIZE = 1024

num_blocks = math.ceil(n / BLOCK_SIZE)

for pid in range(num_blocks):
    start = pid * BLOCK_SIZE
    end = min(start + BLOCK_SIZE, n)
    output[start:end] = x[start:end] + y[start:end]
```

Every iteration of that loop is what Triton calls one **program instance**. The
critical difference is that on a GPU every program instance runs **in parallel**
across different Streaming Multiprocessors instead of sequentially in a `for` loop.

Here is the mapping between the Python analogy and Triton concepts:

| Python loop           | Triton concept              | CUDA equivalent      |
|-----------------------|-----------------------------|----------------------|
| `pid` (loop index)    | `tl.program_id(axis=0)`    | `blockIdx.x`         |
| `BLOCK_SIZE`          | `BLOCK_SIZE: tl.constexpr` | `blockDim.x`         |
| One loop iteration    | One program instance        | One thread block     |
| `num_blocks`          | The *grid* size             | `gridDim.x`          |

### What Triton Generates For You

When you write a Triton kernel, the compiler:

1. Takes your program-level code (which looks like it operates on vectors/blocks).
2. Figures out how to map those vector operations onto the 32-thread warps inside
   each hardware thread block.
3. Generates optimized PTX/SASS machine code for your specific GPU.
4. Handles memory coalescing, register allocation, and instruction scheduling.

You never write thread-level code. You never worry about `threadIdx.x`. Triton
handles it.

---

## The Kernel — Line by Line

Here is the complete vector addition kernel. We will dissect every line after the
listing.

```python
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,      # Pointer to first input vector
    y_ptr,      # Pointer to second input vector
    output_ptr, # Pointer to output vector
    n_elements, # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program processes
):
    # Which program am I? (like: which chunk of work am I responsible for?)
    pid = tl.program_id(axis=0)

    # Calculate the starting index for this program's chunk
    block_start = pid * BLOCK_SIZE

    # Create offsets: [block_start, block_start+1, ..., block_start+BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask: don't load/store beyond the end of the array
    mask = offsets < n_elements

    # Load data from global memory (with mask for safety)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Compute
    output = x + y

    # Store result back to global memory
    tl.store(output_ptr + offsets, output, mask=mask)
```

Now let's walk through every piece.

---

### `@triton.jit`

```python
@triton.jit
def add_kernel(...):
```

This decorator tells Triton: "This function is a GPU kernel. Do not run it on the
CPU — compile it to GPU machine code instead."

When you call the kernel for the first time, Triton's Just-In-Time (JIT) compiler
kicks in. It compiles your Python-like code into **PTX** (NVIDIA's intermediate
assembly language) and then into **SASS** (the final machine code for your specific
GPU architecture). This compilation happens once and is cached, so subsequent calls
are fast.

This is different from normal Python functions, which the CPU interpreter executes.
After `@triton.jit`, the function body lives on the GPU.

---

### The Parameters

```python
def add_kernel(
    x_ptr,      # Pointer to first input vector
    y_ptr,      # Pointer to second input vector
    output_ptr, # Pointer to output vector
    n_elements, # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program processes
):
```

**Pointers (`x_ptr`, `y_ptr`, `output_ptr`):**
In PyTorch you work with tensor objects that hide their memory layout from you. On
the GPU side, Triton works with **pointers**. A pointer is the memory address of the
first element of a tensor. Think of it as a street address — it tells the GPU
*where in memory* the data lives, but nothing about its shape or dtype. When you
pass a PyTorch tensor to a Triton kernel, Triton automatically extracts its data
pointer.

**`n_elements`:**
A plain integer telling the kernel the total length of the vectors. Since pointers
carry no size information, we must pass the length explicitly.

**`BLOCK_SIZE: tl.constexpr`:**
The `tl.constexpr` annotation means "this value is a compile-time constant." Triton
does not treat it as a regular runtime variable — it bakes the value directly into
the generated machine code. This matters because the compiler can make much better
optimization decisions (loop unrolling, register allocation, vectorized loads) when
it knows the block size at compile time. You always pass `BLOCK_SIZE` as a keyword
argument from the launcher, and Triton recompiles (and caches) a specialized kernel
for each distinct value.

---

### `tl.program_id(axis=0)` — "Which program am I?"

```python
pid = tl.program_id(axis=0)
```

Every program instance needs to know *which chunk* of the data it is responsible
for. `tl.program_id(axis=0)` returns this program's index along the first (and in
this case, only) axis of the launch grid.

If we launch 4 programs, then:

```
Program 0:  pid = 0
Program 1:  pid = 1
Program 2:  pid = 2
Program 3:  pid = 3
```

All four programs execute the same kernel code simultaneously, but each one gets a
different `pid`, so each one processes a different chunk of the array. This is the
fundamental pattern of GPU parallelism — the same program, different data (this is
what NVIDIA calls SIMT: Single Instruction, Multiple Threads).

The `axis=0` argument selects the first dimension of the grid. For 1D problems like
vector addition, there is only one axis. In later kernels (like matrix
multiplication) you will use `axis=0` and `axis=1` for a 2D grid.

---

### Computing Offsets

```python
block_start = pid * BLOCK_SIZE
offsets = block_start + tl.arange(0, BLOCK_SIZE)
```

**`block_start`:** Each program is responsible for `BLOCK_SIZE` consecutive elements.
Program 0 starts at index 0, program 1 starts at index `BLOCK_SIZE`, program 2 at
`2 * BLOCK_SIZE`, and so on.

**`tl.arange(0, BLOCK_SIZE)`:** This is Triton's equivalent of Python's `range()`,
but instead of producing a lazy iterator, it creates a *tensor* of integers:
`[0, 1, 2, ..., BLOCK_SIZE-1]`. This tensor lives in GPU registers.

**`offsets`:** Adding `block_start` to that range gives us the global indices this
program is responsible for. For example, with `BLOCK_SIZE = 4`:

```
Program 0 (pid=0):  block_start = 0   ->  offsets = [0, 1, 2, 3]
Program 1 (pid=1):  block_start = 4   ->  offsets = [4, 5, 6, 7]
Program 2 (pid=2):  block_start = 8   ->  offsets = [8, 9, 10, 11]
```

Notice that `offsets` is a vector, not a scalar. This is the key difference from
CUDA: in Triton, each program works on a *block* of elements at once using vector
operations.

---

### Masking — Preventing Out-of-Bounds Accesses

```python
mask = offsets < n_elements
```

This is one of the most important safety mechanisms in Triton. Here is the problem
it solves:

Suppose you have `n_elements = 10` and `BLOCK_SIZE = 4`. You need
`ceil(10 / 4) = 3` programs. Program 2 computes `offsets = [8, 9, 10, 11]`. But
indices 10 and 11 are **past the end of the array** (valid indices are 0 through 9).

```
Array:     [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9]  <- only 10 elements
                                                 ^
Program 2 offsets: [8, 9, 10, 11]
                          ^^  ^^  <- out of bounds!
```

Without masking, the GPU would try to read (and later write) memory that does not
belong to our array. This is **undefined behavior** — it could read garbage, corrupt
other data, or crash the kernel.

The mask is a boolean tensor: `[True, True, False, False]` in this example. When
passed to `tl.load` and `tl.store`, it tells Triton: "Only perform the memory
operation where the mask is `True`. Skip the `False` positions."

This pattern appears in essentially every Triton kernel. Always ask yourself: "Could
the last program go past the end of my data?" If yes, you need a mask.

---

### Loading Data from Global Memory

```python
x = tl.load(x_ptr + offsets, mask=mask)
y = tl.load(y_ptr + offsets, mask=mask)
```

In PyTorch, when you write `x + y`, memory access is implicit — PyTorch handles it
behind the scenes. In Triton, you must explicitly tell the GPU to load data from
global memory (HBM) into registers using `tl.load`.

**Pointer arithmetic (`x_ptr + offsets`):**
Remember that `x_ptr` is the memory address of the first element of the tensor. When
you add an offset to a pointer, you move forward in memory by that many elements.
So `x_ptr + offsets` produces a vector of memory addresses:

```
x_ptr + offsets = [x_ptr + 0, x_ptr + 1, x_ptr + 2, ..., x_ptr + BLOCK_SIZE-1]
                    ^           ^           ^
                    address     address     address
                    of x[0]    of x[1]     of x[2]
```

`tl.load` then reads the values at all of these addresses in parallel, returning a
vector of values that now live in the fast registers of the GPU's Streaming
Multiprocessor.

The `mask=mask` argument tells `tl.load` to skip loading at positions where the mask
is `False`. For those positions, the result defaults to 0 (you can change this
default with the `other=` argument, which we will use in the softmax kernel).

---

### The Computation

```python
output = x + y
```

This is pleasantly straightforward. `x` and `y` are both vectors of `BLOCK_SIZE`
elements sitting in registers. The `+` operation adds them element-wise, producing
another vector. This is where the actual arithmetic happens, and it executes
entirely within the Streaming Multiprocessor's compute units — no memory traffic
needed.

From the roofline model in Phase 1, recall that this operation does very little
arithmetic relative to the amount of memory it moves (we load two elements and store
one just to do a single addition). This makes vector addition firmly
**memory-bound**. The GPU's compute units are barely breaking a sweat — they spend
most of their time waiting for data to arrive from global memory.

---

### Storing the Result

```python
tl.store(output_ptr + offsets, output, mask=mask)
```

The mirror image of `tl.load`. This writes the computed values from registers back
to global memory (HBM). The pointer arithmetic works the same way:
`output_ptr + offsets` produces a vector of destination addresses, and `tl.store`
writes one value to each address in parallel.

The `mask=mask` ensures we only write to valid positions. Without it, the last
program could write garbage to memory addresses beyond our output tensor.

---

## The Launcher Function

The kernel defines *what* each program does. The launcher defines *how many*
programs to run and passes the arguments. This is regular Python code that runs on
the CPU.

```python
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Validate inputs
    assert x.is_cuda and y.is_cuda, "Inputs must be on GPU"
    assert x.shape == y.shape, "Inputs must have the same shape"

    output = torch.empty_like(x)
    n_elements = output.numel()

    # Calculate grid size — how many programs to launch
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Launch the kernel
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output
```

Let's break this down.

### `torch.empty_like(x)`

We allocate the output tensor with `empty_like` rather than `zeros_like`. Why?
Because our kernel will write to every single element of the output. Pre-filling it
with zeros would be wasted work — a pointless extra kernel launch that writes zeros
only for us to immediately overwrite them. `empty_like` allocates the memory without
initializing it, which is faster.

### `triton.cdiv` — Ceiling Division

```python
triton.cdiv(n_elements, meta['BLOCK_SIZE'])
# Equivalent to: math.ceil(n_elements / BLOCK_SIZE)
```

We need to figure out how many programs to launch. If we have 1025 elements and
`BLOCK_SIZE = 1024`, regular integer division gives `1025 // 1024 = 1`. But one
program only covers elements 0 through 1023, leaving element 1024 unprocessed. We
need ceiling division to get 2 programs. The second program will have a mask that
disables processing for indices 1025 through 2047 (which do not exist), and it will
only process the single remaining element at index 1024.

### The Grid Lambda

```python
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
```

The grid must be a function (lambda) that takes a `meta` dictionary and returns a
tuple of integers. Why a function instead of a plain tuple? Because when we use
Triton's **autotuning** feature (covered in a later lesson), `BLOCK_SIZE` is not
known until Triton picks the best configuration at runtime. The lambda lets Triton
pass in the chosen `BLOCK_SIZE` through `meta['BLOCK_SIZE']` and compute the correct
grid size.

For now, since we hard-code `BLOCK_SIZE=1024`, this lambda always returns the same
value. But using the lambda pattern from the start is a good habit.

The return value is a tuple. For our 1D problem it is a 1-tuple like `(1024,)`. For
2D problems (matrices) it would be a 2-tuple like `(rows, cols)`.

### The `[grid]` Launch Syntax

```python
add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
```

This is Triton's way of launching a kernel. It looks unusual at first, but here is
what it means:

- `add_kernel` — the compiled GPU kernel.
- `[grid]` — how many program instances to launch (the grid configuration).
- `(x, y, output, n_elements, BLOCK_SIZE=1024)` — the arguments to pass to each
  program instance. PyTorch tensors are automatically converted to pointers.

All program instances start executing in parallel on the GPU. The CPU does not wait
for them to finish (kernel launch is asynchronous). However, when you later use
`output` in PyTorch, PyTorch's CUDA stream synchronization ensures the kernel has
completed.

---

## Putting It All Together

Here is the complete, runnable code. You can paste this into a Google Colab notebook
with a T4 GPU runtime.

```python
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda, "Inputs must be on GPU"
    assert x.shape == y.shape, "Inputs must have the same shape"

    output = torch.empty_like(x)
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output


# ---- Verify correctness ----
size = 2**20  # ~1M elements
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')

triton_output = add(x, y)
torch_output = x + y

assert torch.allclose(triton_output, torch_output), "Results don't match!"
print("Correctness verified")
```

---

## Visualizing the Execution

Here is what happens when we call `add(x, y)` with 10 elements and `BLOCK_SIZE = 4`:

```
Input x:  [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9]
Input y:  [y0, y1, y2, y3, y4, y5, y6, y7, y8, y9]

Grid: ceil(10 / 4) = 3 programs

Program 0 (pid=0)          Program 1 (pid=1)          Program 2 (pid=2)
  offsets = [0, 1, 2, 3]     offsets = [4, 5, 6, 7]     offsets = [8, 9, 10, 11]
  mask    = [T, T, T, T]     mask    = [T, T, T, T]     mask    = [T, T, F,  F ]
  loads x[0:4], y[0:4]       loads x[4:8], y[4:8]       loads x[8:10], skips rest
  computes x+y                computes x+y                computes x+y (2 elements)
  stores output[0:4]         stores output[4:8]         stores output[8:10]

                  All three programs run in PARALLEL on the GPU.

Output: [x0+y0, x1+y1, x2+y2, ..., x8+y8, x9+y9]
```

---

## Benchmarking: Triton vs PyTorch

Now for the exciting part — how does our hand-written kernel compare to PyTorch?

```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],                              # Parameter to vary
        x_vals=[2**i for i in range(12, 28)],          # 4K to 128M elements
        line_arg='provider',                           # Argument that selects the implementation
        line_vals=['triton', 'torch'],                 # Implementations to compare
        line_names=['Triton', 'PyTorch'],              # Legend labels
        styles=[('blue', '-'), ('red', '-')],          # Line styles
        ylabel='GB/s',                                 # Y-axis label
        plot_name='vector-addition-performance',       # Output file name
        args={},                                       # Extra arguments (none here)
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    # Calculate throughput in GB/s
    gbps = lambda ms: 3 * x.numel() * x.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(print_data=True, show_plots=True)
```

### Understanding the Benchmark

**Why GB/s?**
Recall from the roofline model in Phase 1: every operation lives somewhere on the
roofline chart. Vector addition has an arithmetic intensity of approximately
**1 FLOP per 12 bytes** (we read two 4-byte float32 values, write one 4-byte
float32 value, and perform 1 addition). That places it deep in the
**memory-bound** region. The bottleneck is not compute — it is how fast we can
shuttle data between HBM and the SMs. So we measure throughput in GB/s
(gigabytes per second), not FLOPS.

**The formula:**
```
3 * x.numel() * x.element_size() / ms * 1e-6
```

Let's unpack this:
- `3 * x.numel() * x.element_size()` = total bytes moved. We read two vectors and
  write one, so 3 memory operations per element. Each float32 is 4 bytes.
- `/ ms * 1e-6` converts from bytes/millisecond to GB/s.

**`triton.testing.do_bench`:**
This utility runs the kernel many times, performs warm-up runs, and returns the
median, 20th percentile, and 80th percentile execution times. The quantiles help
you understand performance variability.

**What to expect on a T4 GPU:**
The T4 has approximately 320 GB/s of HBM bandwidth. For large vectors (millions of
elements), both Triton and PyTorch should approach this limit — perhaps 250-300
GB/s. They will be close in performance because PyTorch's built-in addition kernel
is already well-optimized and this operation is so simple there is little room for
improvement. The real benefits of Triton come from **fusing** multiple operations,
as you will see in the next lesson.

For small vectors (a few thousand elements), you may see low GB/s because the
kernel launch overhead dominates: it takes a few microseconds to launch a kernel
regardless of how little work it does.

---

## Key Concepts Summary

```
+------------------------------------------------------------------+
|  TRITON KERNEL ANATOMY                                            |
|                                                                   |
|  @triton.jit          <-- "This is a GPU kernel"                  |
|  def kernel(                                                      |
|      ptrs,            <-- Pointers to tensor data in HBM          |
|      sizes,           <-- Dimensions (kernel has no shape info)    |
|      BLOCK: constexpr <-- Compile-time constant for optimization  |
|  ):                                                               |
|      pid = program_id()     <-- "Which chunk am I?"               |
|      offsets = compute()    <-- "Which elements am I handling?"    |
|      mask = bounds_check()  <-- "Are all my elements valid?"      |
|      data = tl.load()      <-- Explicit read from HBM             |
|      result = compute()    <-- Math in registers (fast!)          |
|      tl.store()            <-- Explicit write to HBM              |
+------------------------------------------------------------------+

+------------------------------------------------------------------+
|  LAUNCHER ANATOMY                                                 |
|                                                                   |
|  def wrapper(tensors):                                            |
|      output = allocate()                                          |
|      grid = how_many_programs()                                   |
|      kernel[grid](ptrs, sizes, BLOCK_SIZE=...)                    |
|      return output                                                |
+------------------------------------------------------------------+
```

---

## Exercises

### Exercise 1: Experiment with BLOCK_SIZE

Change `BLOCK_SIZE` to 512 and then to 2048. Run the benchmark again. Questions to
answer:

- Does performance change significantly?
- Why might there be a difference? (Hint: think about the number of programs
  launched and how much work each one does. Also think about register pressure —
  larger blocks use more registers per program.)

### Exercise 2: Three-Vector Addition (Kernel Fusion)

Write a Triton kernel that computes `output = x + y + w` in a single kernel. This
is your first taste of **kernel fusion**: instead of launching two separate kernels
(one for `x + y`, one for `result + w`), you do everything in one pass. The fused
version reads global memory once and writes once. The unfused version would read and
write twice.

```python
# Unfused (what PyTorch does):
temp = x + y    # Kernel 1: reads x, y from HBM -> writes temp to HBM
out = temp + w   # Kernel 2: reads temp, w from HBM -> writes out to HBM
# Total: 4 reads + 2 writes = 6 memory passes

# Fused (your Triton kernel):
out = x + y + w  # Kernel 1: reads x, y, w from HBM -> writes out to HBM
# Total: 3 reads + 1 write = 4 memory passes
```

### Exercise 3: Element-wise Multiply

Modify the kernel to compute `output = x * y` instead. How much code needs to
change? (Answer: one character.)

### Exercise 4: Fused Multiply-Add (FMA)

Implement `output = x * y + z` as a single kernel. This is called a **fused
multiply-add** (FMA). Modern NVIDIA GPUs have hardware FMA units that can compute
`a * b + c` in a single clock cycle — the same time it takes to do just a multiply
or just an add. By writing `x * y + z` in a single Triton kernel, you allow the
compiler to emit FMA instructions.

```python
# Hint: the kernel body is just:
output = x * y + z
```

Verify that your result matches `torch.addcmul(z, x, y, value=1)` or simply
`x * y + z` in PyTorch.

---

## Milestone Checklist

You are ready to move on to the next lesson when you can:

- [ ] Write a Triton kernel from scratch (without looking at the reference).
- [ ] Explain what `pid` is and why each program gets a different one.
- [ ] Explain why `BLOCK_SIZE` is a `tl.constexpr`.
- [ ] Explain why we need a `mask` and what would happen without it.
- [ ] Explain what `tl.load` and `tl.store` do and why memory access is explicit.
- [ ] Explain why vector addition is memory-bound and what that means for
      benchmarking.

---

## Resources

- [Triton official vector addition tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html) — the official tutorial this lesson is based on.
- [Triton language reference](https://triton-lang.org/main/python-api/triton.language.html) — documentation for `tl.load`, `tl.store`, `tl.arange`, and other primitives.
