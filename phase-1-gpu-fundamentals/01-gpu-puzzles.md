# 01 — GPU Puzzles: Building Your Mental Model

## Learning Objectives

By the end of this module, you will be able to:

1. Explain what a **thread**, **block**, and **grid** are in GPU programming
2. Understand how threads map to data elements using index calculations
3. Reason about **guard conditions** — why threads sometimes need to "do nothing"
4. Work with 2D thread indexing for matrix operations
5. Explain what **shared memory** is and why it exists
6. Solve all 14 GPU Puzzles in Sasha Rush's GPU Puzzles notebook

---

## Why Start with GPU Puzzles?

If you are coming from Python and PyTorch, jumping straight into Triton or CUDA is
overwhelming. There are too many new concepts hitting you at once: kernel syntax,
memory management, compilation, hardware details.

**GPU Puzzles strips all of that away.** It uses a tiny, simplified GPU simulator
(built on Numba) that lets you focus purely on the *thinking pattern* of GPU
programming:

- How do I split work across thousands of threads?
- How does each thread know which piece of data to work on?
- What happens when there are more threads than data elements?
- How do threads within a group cooperate?

Think of it like learning chess on a 4x4 board before playing on the full 8x8
board. The rules are the same, but the complexity is manageable.

**Every concept you learn in GPU Puzzles maps directly to Triton.** When you later
write `tl.arange(0, BLOCK_SIZE)` in a Triton kernel, you will already understand
why you need it — because you practiced the same idea in GPU Puzzles with
`cuda.threadIdx.x`.

---

## Setup

GPU Puzzles runs in Google Colab (free GPU required) or any environment with a
CUDA-capable GPU and Numba installed.

### Option A: Google Colab (Recommended for Beginners)

1. Go to the [GPU Puzzles repo](https://github.com/srush/GPU-Puzzles) and click the **"Open In Colab"** badge, or open the notebook directly from the repo
2. **Make a copy** of the notebook: **File > Save a copy in Drive**
3. Enable GPU: **Runtime > Change runtime type > T4 GPU**
4. Run the first setup cell in the notebook. It will install dependencies:

```python
!pip install -qqq git+https://github.com/danoneata/chalk@srush-patch-1
!wget -q https://github.com/srush/GPU-Puzzles/raw/main/robot.png https://github.com/srush/GPU-Puzzles/raw/main/lib.py
```

Then import the library:

```python
import numba
import numpy as np
from lib import CudaProblem, Coord
```

If you see output without errors, you are ready. The library provides a visual
test harness that shows you:
- The input data
- What your kernel produced
- What the correct output should be

### Option B: Local Setup

If you have a local NVIDIA GPU:

```bash
pip install numba numpy
git clone https://github.com/srush/GPU-Puzzles.git
cd GPU-Puzzles
jupyter notebook GPU_puzzles.ipynb
```

The notebook handles its own imports — just run the cells in order.

---

## Core Concepts You Need Before Starting

Before you touch the first puzzle, you need two ideas.

### Concept 1: What Is a Kernel?

In normal Python, you write a function and call it once:

```python
def add_one(x):
    return [v + 1 for v in x]

result = add_one([1, 2, 3, 4])  # One call, processes all elements
```

In GPU programming, you write a function (called a **kernel**) that describes what
**one thread** does. Then you launch it across many threads simultaneously:

```python
# Pseudocode — this is the GPU mental model
def add_one_kernel(output, input):
    i = my_thread_id()       # "Which thread am I?"
    output[i] = input[i] + 1  # Each thread handles one element

# Launch 4 threads, all running at the same time
launch(add_one_kernel, num_threads=4)
```

The key shift: **you stop thinking about loops and start thinking about "what does
one worker do?"**

### Concept 2: Threads and Blocks

GPUs organize threads into groups called **blocks**. Think of it this way:

```
Analogy:
  - A BLOCK is a team of workers sitting in the same room.
  - A THREAD is one worker on that team.
  - The GRID is the entire workforce — all teams combined.

Example: You need to process 1000 elements.
  - You create 10 blocks, each with 100 threads.
  - Block 0's threads handle elements 0–99.
  - Block 1's threads handle elements 100–199.
  - ... and so on.
```

Every thread can figure out its **global position** using two pieces of information:

```
global_index = block_id * block_size + thread_position_within_block
```

In GPU Puzzles (and CUDA), this is written as:

```python
i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
```

Where:
- `cuda.threadIdx.x` — "Which worker am I within my team?" (0 to block_size - 1)
- `cuda.blockIdx.x` — "Which team am I on?" (0 to num_blocks - 1)
- `cuda.blockDim.x` — "How many workers are on each team?" (the block size)

You will use this formula in almost every puzzle.

---

## Puzzle-by-Puzzle Guide

### Puzzle 1: Map

**What it teaches:** The most basic GPU operation — each thread reads one input
element, does something to it, and writes one output element.

**The key insight:** Each thread must use its index to figure out *which* element
it is responsible for. The pattern is:

```python
i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
out[i] = fn(input[i])
```

**Common mistakes:**
- Writing a Python loop. There is no loop — each thread IS one iteration.
- Forgetting that `cuda.threadIdx.x` starts at 0 within each block.

**How this maps to Triton:** In Triton, you will write `offsets = pid * BLOCK_SIZE
+ tl.arange(0, BLOCK_SIZE)` — the exact same idea, but operating on a *block* of
elements rather than one element per thread. Triton promotes you from thinking about
individual threads to thinking about blocks of data, but the indexing logic is
identical.

---

### Puzzle 2: Zip

**What it teaches:** Combining two inputs element-wise, like `a[i] + b[i]`.

**The key insight:** A single thread can read from multiple input arrays. The
index `i` is the same for all of them — you are just zipping inputs together:

```python
i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
out[i] = fn(input_a[i], input_b[i])
```

**Common mistakes:**
- Thinking you need separate threads for each input array. You do not. One thread
  reads from both arrays, computes, and writes.

**How this maps to Triton:** Elementwise operations like `x + y`, `relu(x)`, or
`x * scale + bias` all follow this pattern. This is the simplest fused kernel
pattern: read multiple inputs, compute, write output — all in one kernel.

---

### Puzzle 3: Guards

**What it teaches:** The fact that the number of threads launched does not always
match the data size, and threads beyond the end must do nothing.

**The key insight:** GPU blocks come in fixed sizes (e.g., 128, 256). If your data
has 1000 elements and your block size is 256, you launch `ceil(1000/256) = 4`
blocks for 1024 threads. But threads 1000 through 1023 have no data — they must
**not** write to memory.

```python
i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
if i < data_size:        # <-- The guard
    out[i] = fn(input[i])
# Threads with i >= data_size do nothing
```

**Common mistakes:**
- Forgetting the guard entirely, causing out-of-bounds memory writes (a common
  source of silent data corruption in real GPU code).
- Using `<=` instead of `<` (off-by-one errors).

**How this maps to Triton:** In Triton, guards become **masks**:
```python
mask = offsets < n_elements
tl.store(output_ptr + offsets, result, mask=mask)
```
This is one of the most important patterns in all of Triton programming. You will
use masks in every single kernel you write. If you only internalize one thing from
GPU Puzzles, let it be guards/masks.

---

### Puzzle 4: Map 2D

**What it teaches:** Working with 2D data (matrices) using 2D thread indexing.

**The key insight:** Threads can be organized in a 2D grid. Each thread has both
an `x` and a `y` index, corresponding to a row and column:

```python
row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
out[row, col] = fn(input[row, col])
```

But in memory, a 2D array is stored as a flat 1D sequence (row-major). So
`matrix[row, col]` is really at address `row * num_cols + col`. Understanding
this is critical for memory performance later.

```
Matrix (logical view):       Memory (physical view):
  col 0  col 1  col 2        Address: 0  1  2  3  4  5  6  7  8
  [  a      b      c  ]       Data:   a  b  c  d  e  f  g  h  i
  [  d      e      f  ]
  [  g      h      i  ]       matrix[1][2] = memory[1*3 + 2] = memory[5] = f
```

**Common mistakes:**
- Mixing up rows and columns (which is `x` and which is `y`).
- Forgetting that 2D also needs guards for both dimensions.

**How this maps to Triton:** Triton kernels for matrix operations (matmul, softmax
over rows) always work with 2D indexing. You will compute `row_offsets` and
`col_offsets` in the same way.

---

### Puzzle 5: Broadcast

**What it teaches:** How to apply a 1D value across a 2D structure — the same
idea as NumPy broadcasting.

**The key insight:** Not every thread reads from a unique input position. In
broadcasting, many threads share the same input value:

```python
# Adding a bias vector to every row of a matrix
row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
out[row, col] = matrix[row, col] + bias[col]  # bias is 1D, indexed only by col
```

**Common mistakes:**
- Indexing the bias with both row and col (treating it as 2D when it is 1D).
- Forgetting that the broadcast dimension doesn't use the row index.

**How this maps to Triton:** Broadcasting shows up everywhere in real kernels —
adding bias vectors, scaling by per-channel weights, applying attention masks.
Understanding which index to use for which tensor is essential.

---

### Puzzle 6–8: Blocks (1D)

**What it teaches:** This is where the real mental shift happens. Instead of every
thread working independently, you now have **groups of threads (blocks) that must
cooperate** to process a chunk of data.

**The key insight:** A block of threads can load a chunk of data into **shared
memory**, work on it together, and then write results out. The critical idea is
that each thread loads one piece, but then can read *any* piece that *any* thread
in the block loaded.

Here is the pattern:

```
Step 1: Each thread loads one element from global memory into shared memory
Step 2: Synchronize (wait for ALL threads in the block to finish loading)
Step 3: Each thread reads from shared memory to compute its result
Step 4: Write result back to global memory
```

The synchronization step (`cuda.syncthreads()`) is crucial. Without it, thread 5
might try to read what thread 12 loaded before thread 12 has finished loading it.

**Common mistakes:**
- Forgetting `cuda.syncthreads()` — this causes **race conditions** where you
  read data that has not been written yet.
- Confusing the shared memory index with the global memory index. Shared memory
  starts at 0 for each block, but global memory positions depend on which block
  you are in.

**How this maps to Triton:** Triton handles a lot of this automatically. When you
write `tl.load(ptr + offsets)`, Triton loads a block of data in a coordinated way.
But understanding the load-sync-compute-store pattern helps you reason about what
Triton does under the hood.

---

### Puzzle 9–11: Blocks 2D

**What it teaches:** The same block cooperation idea, but now for 2D data. This is
the foundation of how matrix multiplication works on GPUs.

**The key insight:** A 2D block of threads cooperatively loads a **tile** (a small
rectangular patch) of a matrix into shared memory. Each thread loads one element
of the tile.

```
Full Matrix (too big for shared memory):
  ┌─────────────────────────────┐
  │ ░░░░░░░░░ │                 │
  │ ░ Tile 0 ░ │                │
  │ ░░░░░░░░░ │                 │
  │───────────┼─────────────────│
  │           │ ░░░░░░░░░       │
  │           │ ░ Tile 3 ░      │
  │           │ ░░░░░░░░░       │
  └─────────────────────────────┘

Each block processes one tile.
Each thread in the block handles one element of the tile.
```

**Common mistakes:**
- Getting the index math wrong between local (within-tile) and global (within-matrix)
  coordinates. This is the hardest part of 2D block puzzles.
- Forgetting guards at the matrix edges where a tile might extend beyond the
  matrix boundary.

**How this maps to Triton:** Tiled matrix operations are the core of high-performance
matmul in Triton. Phase 3 of this curriculum covers this in depth, and the
intuition you build here will be essential.

---

### Puzzle 12–14: Shared Memory (Reductions)

**What it teaches:** How to compute a single result from many elements (e.g., sum,
max) using threads working together — this is called a **reduction**.

**The key insight:** You cannot just have one thread loop through all the data —
that wastes the GPU's parallelism. Instead, threads cooperate in a tree pattern:

```
Step 1: 8 values in shared memory
  [a0] [a1] [a2] [a3] [a4] [a5] [a6] [a7]

Step 2: 4 threads add pairs
  [a0+a1]  [a2+a3]  [a4+a5]  [a6+a7]

Step 3: 2 threads add pairs
  [a0+a1+a2+a3]  [a4+a5+a6+a7]

Step 4: 1 thread adds the final pair
  [a0+a1+a2+a3+a4+a5+a6+a7]  ← Result!
```

This takes `log2(N)` steps instead of `N` steps. For 1024 elements, that is 10
steps instead of 1024.

Shared memory is essential here because threads need to **read each other's
intermediate results** between steps. Each step requires a `syncthreads()` barrier.

**The shared memory analogy:**

> Think of shared memory as a **whiteboard in a meeting room**. All workers (threads)
> in the same team (block) can read and write to this whiteboard. But workers in a
> different team, in a different room, cannot see it. It is fast to use (it is right
> there on the wall), but it is small (you can only fit so much on a whiteboard).
>
> Global memory, by contrast, is like a **filing cabinet in the hallway**. Everyone
> can access it, but you have to walk out of the room to get to it, which is slow.

**Common mistakes:**
- Not synchronizing between reduction steps (reading a value before another thread
  has finished updating it).
- Off-by-one errors in the reduction tree (which threads are active at each step).
- Forgetting to handle the case where block size is larger than the data to reduce.

**How this maps to Triton:** Triton provides `tl.sum()`, `tl.max()`, and other
reductions that handle the tree pattern for you within a block. But when you need
cross-block reductions (e.g., computing the mean of an entire row for LayerNorm),
you still need to understand the multi-step pattern: each block reduces its chunk,
writes a partial result, then a second kernel combines the partial results. Knowing
the tree reduction pattern helps you understand *why* some operations need multiple
kernel launches.

---

## Summary of GPU Puzzles Concepts

| Puzzle Section | Core Concept | Triton Equivalent |
|---|---|---|
| Map | One thread per element, index calculation | `pid * BLOCK_SIZE + tl.arange()` |
| Zip | Multiple inputs, same index | Loading multiple tensors in one kernel |
| Guards | Bounds checking for partial blocks | `mask = offsets < n_elements` |
| Map 2D | Row/column indexing, row-major layout | 2D offset calculations |
| Broadcast | Shared values across dimensions | Loading 1D tensor, broadcasting in 2D op |
| Blocks 1D | Cooperative loading, syncthreads | `tl.load()` with block-sized ranges |
| Blocks 2D | Tiled processing of matrices | Tiled matmul in Phase 3 |
| Shared Memory | Parallel reductions | `tl.sum()`, `tl.max()` |

---

## Practice Exercises After Completing GPU Puzzles

Once you have solved all 14 puzzles, try these variations to deepen your
understanding. You do not need to code them — just think through the solution
on paper or in pseudocode.

### Exercise 1: Non-power-of-2 Sizes
The puzzles use clean sizes (powers of 2). What changes if the array has 1000
elements and your block size is 256? Write out which threads are active in the
last block. Which threads need guards?

### Exercise 2: Reverse an Array
Each thread reads `input[i]` and writes to `output[n - 1 - i]`. What is the
index calculation? Do you need shared memory for this?

### Exercise 3: Running Sum (Prefix Sum)
Given `[3, 1, 4, 1, 5]`, produce `[3, 4, 8, 9, 14]`. Why is this much harder
than a simple sum reduction? Why can you not just have each thread add up
everything before it? (Hint: that would be O(N^2) total work, defeating the
purpose of parallelism.) This is a preview of a classic GPU algorithm called
**scan** that you will encounter later.

### Exercise 4: Matrix Transpose
To transpose a matrix, element `[row][col]` must move to `[col][row]`. Why is a
naive approach (each thread reads from `[row][col]` and writes to `[col][row]`)
slow? (Hint: think about which memory accesses are contiguous and which are
scattered. This connects to **memory coalescing**, covered in the next module.)

### Exercise 5: Sliding Window Average
Given an array, compute the average of each element and its two neighbors:
`out[i] = (in[i-1] + in[i] + in[i+1]) / 3`. Why does shared memory help here?
(Hint: without shared memory, `in[i]` gets loaded from global memory three
times — once by each of its neighboring threads.)

---

## Milestone Checklist

You are ready to move on to the next module (02 — Memory and Compute) when you
can confidently answer all of the following:

- [ ] **What is a thread?** It is one unit of execution on the GPU. Each thread
      runs the kernel function independently with its own index.

- [ ] **What is a block?** A group of threads that are scheduled together and can
      share data through shared memory. Typical sizes: 128, 256, 512 threads.

- [ ] **How does a thread know which data to work on?** Through the formula:
      `global_index = blockIdx * blockDim + threadIdx`

- [ ] **Why do we need guards/masks?** Because the number of threads launched is
      rounded up to a multiple of the block size, so some threads may have no
      valid data to process.

- [ ] **What is shared memory?** A small, fast memory space visible to all
      threads in the same block. Used for inter-thread communication and data
      reuse within a block.

- [ ] **What is a reduction?** Combining many values into one (like sum or max)
      using a parallel tree pattern that takes `log2(N)` steps.

If any of these feel shaky, revisit the relevant puzzle. The puzzles are short
enough that re-solving them takes minutes, and repetition builds real intuition.

---

## What Comes Next

The next module, **02 — Memory and Compute**, explains the hardware behind
everything you just practiced. You will learn:

- Why shared memory is fast and global memory is slow (the memory hierarchy)
- What a **warp** is (a hardware detail GPU Puzzles hides from you)
- Why **memory coalescing** matters for performance
- The **roofline model** for understanding whether your kernel is bottlenecked by
  memory or compute

With the mental model from GPU Puzzles and the hardware understanding from the
next module, you will be fully prepared to write real Triton kernels in Phase 2.
