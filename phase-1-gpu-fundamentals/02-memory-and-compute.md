# 02 — Memory Hierarchy and the Compute Model

## Learning Objectives

By the end of this module, you will be able to:

1. Explain the architectural difference between CPUs and GPUs and why GPUs are
   faster for ML workloads
2. Describe the full GPU execution hierarchy: threads, warps, thread blocks, grids
3. Explain **warp divergence** and why it hurts performance
4. Name all four levels of the GPU memory hierarchy and their relative speeds
5. Explain **memory coalescing** and why scattered memory access is slow
6. Calculate the **arithmetic intensity** of a simple operation
7. Determine whether an operation is **memory-bound** or **compute-bound**
8. Explain why **kernel fusion** is the primary motivation behind Triton

This module is dense. Take your time. These concepts are the foundation for
every performance decision you will make when writing Triton kernels.

---

## GPU vs CPU Architecture

### Why GPUs Are Fast for Machine Learning

A CPU is designed to run **one complex task** very fast. It has sophisticated
features like branch prediction, out-of-order execution, and large caches that
make a single thread of execution blazingly quick.

A GPU is designed to run **thousands of simple tasks** simultaneously. It
sacrifices single-thread performance for massive parallelism.

**The Chef Analogy:**

> A **CPU** is like **one world-class chef** in a kitchen. This chef can follow
> any recipe, improvise, handle complex multi-step techniques, and make decisions
> on the fly. But no matter how skilled, one chef can only cook one dish at a time.
>
> A **GPU** is like **1,000 line cooks**, each following simple instructions:
> "chop this onion," "stir this pot," "plate this dish." Individually, none of
> them can match the expert chef. But when you need to prepare 1,000 identical
> plates for a banquet, the 1,000 line cooks finish in a fraction of the time.

Machine learning is the banquet. Operations like "multiply every element by a
weight" or "add a bias to every neuron" are the same simple instruction repeated
millions of times. GPUs are purpose-built for this pattern.

### Architecture at a Glance

```
CPU (e.g., Intel i9, 24 cores)            GPU (e.g., NVIDIA T4, 2560 cores)
┌─────────────────────────────┐           ┌──────────────────────────────────────┐
│  ┌──────┐  ┌──────┐        │           │  ┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐  │
│  │ Core │  │ Core │  ...   │           │  │c││c││c││c││c││c││c││c││c││c│  │
│  │ (big)│  │ (big)│        │           │  └─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘  │
│  └──────┘  └──────┘        │           │  ┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐  │
│       6-24 cores total     │           │  │c││c││c││c││c││c││c││c││c││c│  │
│                             │           │  └─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘  │
│  ┌─────────────────────┐   │           │  ┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐  │
│  │   Large L3 Cache    │   │           │  │c││c││c││c││c││c││c││c││c││c│  │
│  │    (30+ MB)         │   │           │  └─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘  │
│  └─────────────────────┘   │           │       ... hundreds more rows ...    │
│                             │           │         2560 cores total            │
│  Each core: complex,       │           │                                      │
│  out-of-order execution,   │           │  Each core: simple ALU,              │
│  branch prediction,        │           │  in-order execution,                 │
│  speculative execution     │           │  no branch prediction                │
└─────────────────────────────┘           └──────────────────────────────────────┘
        16-128 GB RAM                              16 GB VRAM (HBM/GDDR)
     ~50 GB/s bandwidth                          320 GB/s bandwidth (T4)
```

Key differences:

| Feature | CPU | GPU (T4) |
|---|---|---|
| Core count | 6–24 | 2,560 |
| Clock speed per core | ~5 GHz | ~1.6 GHz |
| Single-thread performance | Excellent | Poor |
| Parallel throughput | Low | Massive |
| Memory bandwidth | ~50 GB/s | 320 GB/s |
| Best for | Complex branchy code | Uniform data-parallel work |

---

## The GPU Execution Model

In the GPU Puzzles module, you learned about threads and blocks. Now we go
deeper into the hardware reality.

### Threads

A **thread** is the smallest unit of execution on a GPU. One thread runs your
kernel function once, typically processing one element (or a small chunk) of data.

When you write a kernel, you write the code for **one thread**. The GPU then runs
thousands or millions of copies of that code simultaneously, each with a different
thread index.

```
Your kernel code (one thread's perspective):
    "I am thread 42. I load input[42], add 1, and store to output[42]."

What the GPU actually does:
    Thread 0:    load input[0],  add 1, store output[0]
    Thread 1:    load input[1],  add 1, store output[1]
    Thread 2:    load input[2],  add 1, store output[2]
    ...
    Thread 999:  load input[999], add 1, store output[999]
    (all happening simultaneously)
```

### Warps: The True Unit of Execution

Here is something GPU Puzzles did not tell you: the GPU does **not** execute
threads individually. It executes them in groups of **32** called **warps**.

> **Analogy:** Imagine 32 soldiers marching in lockstep. They all lift their left
> foot at the same time, then their right foot at the same time. They all execute
> the **same instruction** simultaneously — but each soldier operates on
> **different data**. This execution model is called **SIMT** (Single Instruction,
> Multiple Threads).

```
One Warp = 32 Threads in Lockstep
┌────────────────────────────────────────────────────────┐
│  T0   T1   T2   T3   T4  ...  T29  T30  T31           │
│  ↓    ↓    ↓    ↓    ↓        ↓    ↓    ↓             │
│ LOAD LOAD LOAD LOAD LOAD ... LOAD LOAD LOAD  (cycle 1)│
│  ↓    ↓    ↓    ↓    ↓        ↓    ↓    ↓             │
│ ADD  ADD  ADD  ADD  ADD  ... ADD  ADD  ADD   (cycle 2) │
│  ↓    ↓    ↓    ↓    ↓        ↓    ↓    ↓             │
│ STORE STORE STORE ...  STORE STORE STORE     (cycle 3) │
└────────────────────────────────────────────────────────┘
All 32 threads execute the SAME instruction each cycle.
```

**Why should you care?** Because of **warp divergence**.

### Warp Divergence

If threads within a warp take **different paths** through an `if/else`, the warp
must execute **both** paths, disabling the threads that should not participate in
each one. This effectively halves (or worse) your performance.

```
Code:
    if thread_id % 2 == 0:
        do_path_A()      # Even threads
    else:
        do_path_B()      # Odd threads

What the warp actually does:

  Step 1: Execute path A (odd threads sit idle)
    T0 ✓  T1 ✗  T2 ✓  T3 ✗  T4 ✓  T5 ✗ ...
    A     -     A     -     A     -

  Step 2: Execute path B (even threads sit idle)
    T0 ✗  T1 ✓  T2 ✗  T3 ✓  T4 ✗  T5 ✓ ...
    -     B     -     B     -     B

  Total time: time(A) + time(B), NOT max(time(A), time(B))
```

In the worst case (every other thread diverges), you lose 50% of your compute.
The takeaway: **write code where all 32 threads in a warp follow the same path
whenever possible.**

> In Triton, you rarely deal with warp divergence directly because Triton operates
> at the block level and uses masks instead of branches. But understanding warps
> helps you reason about performance.

### Thread Blocks (Cooperative Thread Arrays / CTAs)

A **thread block** is a group of threads that:
1. Execute on the **same Streaming Multiprocessor** (SM) — a physical compute unit
2. Can communicate through **shared memory**
3. Can synchronize with **barriers** (`__syncthreads()`)

Typical block sizes in practice: **128**, **256**, or **512** threads.

A block of 256 threads contains 256 / 32 = **8 warps**.

```
Thread Block (256 threads)
┌──────────────────────────────────────────┐
│  Warp 0:  T0   T1   T2  ... T31         │
│  Warp 1:  T32  T33  T34 ... T63         │
│  Warp 2:  T64  T65  T66 ... T95         │
│  Warp 3:  T96  T97  T98 ... T127        │
│  Warp 4:  T128 T129 T130... T159        │
│  Warp 5:  T160 T161 T162... T191        │
│  Warp 6:  T192 T193 T194... T223        │
│  Warp 7:  T224 T225 T226... T255        │
│                                          │
│  [====== Shared Memory (48-96 KB) =====] │
│  (visible to all threads in this block)  │
└──────────────────────────────────────────┘
```

### Grid

The **grid** is the entire collection of thread blocks launched for one kernel
invocation. A grid can be 1D, 2D, or 3D.

```
Grid (1D example: processing a vector of 2048 elements, block_size=256)
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│Block 0 │ │Block 1 │ │Block 2 │ │Block 3 │ │Block 4 │ │Block 5 │ │Block 6 │ │Block 7 │
│T0..T255│ │T0..T255│ │T0..T255│ │T0..T255│ │T0..T255│ │T0..T255│ │T0..T255│ │T0..T255│
│elem    │ │elem    │ │elem    │ │elem    │ │elem    │ │elem    │ │elem    │ │elem    │
│0-255   │ │256-511 │ │512-767 │ │768-1023│ │1024-   │ │1280-   │ │1536-   │ │1792-   │
└────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘
```

### The Full Hierarchy

```
                    ┌─────────────────────┐
                    │        GRID         │
                    │  (all blocks for    │
                    │   one kernel call)  │
                    └────────┬────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ Block 0  │  │ Block 1  │  │ Block 2  │  ...
        │          │  │          │  │          │
        │ Warp 0   │  │ Warp 0   │  │ Warp 0   │
        │  T0..T31 │  │  T0..T31 │  │  T0..T31 │
        │ Warp 1   │  │ Warp 1   │  │ Warp 1   │
        │  T32..63 │  │  T32..63 │  │  T32..63 │
        │ ...      │  │ ...      │  │ ...      │
        │ Warp 7   │  │ Warp 7   │  │ Warp 7   │
        │ T224..255│  │ T224..255│  │ T224..255│
        └──────────┘  └──────────┘  └──────────┘

  Grid:  Launched by the host (CPU). Can have millions of blocks.
  Block: Group of threads on one SM. Can share memory. Can sync.
  Warp:  32 threads in lockstep. The hardware's true scheduling unit.
  Thread: One execution of the kernel. Has private registers.
```

---

## Memory Hierarchy

This is the most important section in this entire curriculum. **The vast majority
of GPU performance problems are memory problems, not compute problems.** If you
understand only one thing about GPU programming, understand this section.

### Overview

GPUs have a layered memory system. Faster memories are smaller and closer to the
compute cores. Slower memories are larger and farther away.

```
NVIDIA T4 Memory Hierarchy

                Speed          Size            Scope
               (cycles)
  ┌──────────┐
  │Registers │   1 cycle      64 KB/SM        Per thread (private)
  └────┬─────┘
       │
  ┌────▼─────┐
  │  Shared  │   ~5 cycles    64 KB/SM*       Per block (shared within block)
  │  Memory  │                                 (also serves as L1 cache)
  └────┬─────┘
       │
  ┌────▼─────┐
  │ L2 Cache │   ~50 cycles   4 MB            All blocks (hardware-managed)
  └────┬─────┘
       │
  ┌────▼─────┐
  │  Global  │   ~400 cycles  16 GB (GDDR6)   All blocks, host-accessible
  │  Memory  │
  │  (VRAM)  │
  └──────────┘

  * On the T4 (Turing architecture), 96 KB per SM is split between
    L1 cache and shared memory, configurable up to 64 KB shared.
```

Let us look at each level.

### Registers (Fastest, Smallest)

Every thread has a set of private **registers** — the fastest storage on the chip.
When you declare a local variable in your kernel, it lives in a register.

```python
# In a kernel:
x = input[i]      # x is stored in a register
y = x * 2 + 1     # y is also in a register
output[i] = y     # writing from register to global memory
```

Key facts:
- Access time: **1 cycle** (essentially free)
- Size: about **255 registers per thread** on modern GPUs
- Scope: completely **private** to one thread — no other thread can see your registers
- If your kernel uses too many registers, the GPU **spills** them to slower local
  memory, which destroys performance

### Shared Memory / L1 Cache (Fast, Small)

**Shared memory** is a small, fast scratchpad memory visible to all threads in the
same block. You, the programmer, control what goes into it and when.

> **Analogy:** Shared memory is a **whiteboard in a team's meeting room**. Everyone
> on the team (all threads in the block) can read and write to it. It is fast
> because it is right there in the room. But it is small (you can only fit so much),
> and other teams (other blocks) cannot see it — they have their own whiteboards
> in their own rooms.

Key facts:
- Access time: **~5 cycles**
- Size: **up to 64 KB per block** on T4 (shared with L1 cache, configurable)
- Scope: all threads **within one block** — invisible to other blocks
- You control it: you decide when to load data in and when to read it out
- Requires explicit **synchronization** (`__syncthreads()`) between reads and writes

Common uses:
- **Data reuse**: Load once from global memory, use many times from shared memory
- **Inter-thread communication**: Thread A computes something, puts it in shared
  memory, thread B reads it (after synchronization)
- **Reductions**: Sum/max across threads in a block

The L1 cache is hardware-managed and shares the same physical memory as shared
memory. On the T4, you can configure the split (e.g., 32 KB shared + 64 KB L1,
or 64 KB shared + 32 KB L1).

### L2 Cache (Medium)

The **L2 cache** sits between shared memory and global memory. It is
**hardware-managed** — you do not control it directly. When data is fetched from
global memory, a copy is kept in L2 in case it is needed again soon.

Key facts:
- Access time: **~50 cycles**
- Size: **4 MB** on T4 (shared across all SMs)
- Scope: all blocks on the GPU
- You have no direct control over it (the hardware decides what to cache)

### Global Memory / VRAM (Slowest, Largest)

**Global memory** is the GPU's main memory — the VRAM you see in specs (16 GB on
T4). This is where your PyTorch tensors live.

Key facts:
- Access time: **~400 cycles** (200-800x slower than registers)
- Size: **16 GB** on T4
- Scope: accessible by all threads, all blocks, and the host CPU
- This is where `torch.tensor` data lives on the GPU

To put the speed difference in perspective:

```
If accessing a register takes 1 second (like opening a book on your desk),
then accessing global memory takes 6-7 minutes (like driving to the library).

That is why you want to minimize global memory accesses and maximize reuse
through shared memory and registers.
```

### Memory Coalescing

**Memory coalescing** is one of the most important performance concepts in GPU
programming. Here is the idea:

When 32 threads in a warp access **consecutive** addresses in global memory, the
GPU hardware combines all 32 requests into a single, efficient memory transaction.
If they access **scattered** addresses, each request becomes a separate slow
transaction.

```
Coalesced Access (FAST):
  Thread 0 reads address 0
  Thread 1 reads address 1
  Thread 2 reads address 2
  ...
  Thread 31 reads address 31

  → GPU combines into ONE 128-byte memory transaction


Non-Coalesced Access (SLOW):
  Thread 0 reads address 7
  Thread 1 reads address 100
  Thread 2 reads address 42
  ...
  Thread 31 reads address 999

  → GPU issues up to 32 SEPARATE memory transactions
```

**The NumPy analogy:**

```python
import numpy as np
a = np.random.randn(10000)

# Coalesced: One contiguous slice — fast
result = a[0:32]

# Non-coalesced: Random fancy indexing — slow
indices = np.array([7, 100, 42, 891, 3, ...])
result = a[indices]
```

Just as NumPy is faster with contiguous slices than random access, GPUs are
dramatically faster with coalesced memory access.

**Practical impact:** A kernel with perfect coalescing can run **10-20x faster**
than the same kernel with scattered access. This is not a minor optimization — it
is the difference between a fast kernel and a useless one.

**Example: Matrix Transpose**

When reading a matrix row-by-row, consecutive threads access consecutive memory
addresses (coalesced). But when writing the transpose, consecutive threads must
write to addresses that are a full row apart (non-coalesced). This is why a naive
matrix transpose is slow and why shared memory is used as an intermediary to fix
the access pattern:

```
Row-major matrix in memory:
  [a00 a01 a02 a03 | a10 a11 a12 a13 | a20 a21 a22 a23 | ...]
   ↑    ↑    ↑    ↑
   T0   T1   T2   T3   ← Threads read consecutive addresses: COALESCED

Writing the transpose naively:
  output[col][row] for each thread
  T0 writes to address 0, T1 writes to address 4, T2 writes to address 8...
  ↑ Stride of 4 (num_rows) between threads: NON-COALESCED

Fix: Load a tile into shared memory (coalesced read), then write from
shared memory with a transposed access pattern (coalesced write).
```

> **In Triton**, the compiler handles many coalescing details for you when you use
> `tl.load` and `tl.store` with contiguous offset patterns. But understanding
> coalescing is essential for knowing why certain access patterns are fast and
> others are slow.

---

## Bandwidth vs Compute: The Roofline Model

Now that you understand the memory hierarchy, the crucial question becomes:
**What actually limits your kernel's speed?**

There are only two possibilities:
1. **Memory-bound**: The kernel spends most of its time waiting for data to arrive
   from memory. The compute units are idle, waiting.
2. **Compute-bound**: The kernel has enough data but spends most of its time
   doing arithmetic. Memory is not the bottleneck.

### Arithmetic Intensity

The key metric that determines which category a kernel falls into is
**arithmetic intensity**: the number of floating-point operations (FLOPs) performed
per byte of data loaded from memory.

```
Arithmetic Intensity = FLOPs / Bytes Loaded
```

**Example 1: Vector Addition** (`c[i] = a[i] + b[i]`)
- Load 2 floats (a[i] and b[i]) = 8 bytes
- Perform 1 addition = 1 FLOP
- Store 1 float (c[i]) = 4 bytes
- Arithmetic intensity = 1 FLOP / 12 bytes = **0.083 FLOP/byte**
- This is extremely memory-bound.

**Example 2: Matrix Multiplication** (`C = A @ B`, both NxN)
- Total FLOPs: 2 * N^3 (each element of C requires N multiplications and N additions)
- Total bytes loaded: 2 * N^2 * 4 bytes (matrices A and B)
- Arithmetic intensity = 2N^3 / (8N^2) = **N/4 FLOP/byte**
- For N=1024: intensity = 256 FLOP/byte — very compute-bound.

### The Roofline

The **roofline model** visualizes the maximum achievable performance as a function
of arithmetic intensity.

```
Performance                NVIDIA T4 Roofline (FP32)
(TFLOPS)
    │
8.1 │─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  Compute ceiling (8.1 TFLOPS)
    │                      ╱─────────────────────────
    │                    ╱
    │                  ╱
    │                ╱
    │              ╱
    │            ╱     ← Bandwidth ceiling (320 GB/s)
    │          ╱          slope = memory bandwidth
    │        ╱
    │      ╱
    │    ╱
    │  ╱
    │╱
    └──────────────────────────────────────────────── Arithmetic Intensity
    0    5   10   15   20   25   30                  (FLOP/byte)
              ↑
         Ridge point: 8100 / 320 ≈ 25.3 FLOP/byte
         (below this: memory-bound, above: compute-bound)

    Where common operations fall:

    Vector add           (0.08 FLOP/byte)  ← DEEP memory-bound
    Softmax              (~0.5 FLOP/byte)  ← Memory-bound
    LayerNorm            (~1 FLOP/byte)    ← Memory-bound
    GELU activation      (~1 FLOP/byte)    ← Memory-bound
    Matmul (large N)     (>>25 FLOP/byte)  ← Compute-bound
```

**Key insight:** Almost every operation in a transformer *except* matrix
multiplication is **memory-bound**. This means the compute cores are sitting idle
most of the time, waiting for data. **Making these operations faster is about
reducing memory traffic, not about doing math faster.**

### T4 Specs — What the Numbers Mean

| Spec | Value | What It Means |
|---|---|---|
| FP32 Compute | 8.1 TFLOPS | Can do 8.1 trillion single-precision floating-point operations per second |
| FP16 Compute | 65 TFLOPS | With Tensor Cores, 8x faster for half-precision (used in mixed-precision training) |
| Memory Bandwidth | 320 GB/s | Can move 320 billion bytes per second between VRAM and compute cores |
| VRAM | 16 GB GDDR6 | Total memory for model weights, activations, optimizer states |

**Practical meaning:** The ratio of compute to bandwidth determines the ridge
point: 8100 GFLOPS / 320 GB/s = **~25 FLOP/byte**. Any operation with arithmetic
intensity below 25 cannot fully utilize the T4's compute — it is bottlenecked by
how fast data can be fetched.

Since most ML operations have arithmetic intensity well below 25, **memory bandwidth
is usually the bottleneck**, and optimizing memory access patterns is more important
than optimizing arithmetic.

---

## Why Kernel Fusion Matters

This section explains the core motivation for Triton's existence.

### The Problem: Launch Overhead and Redundant Memory Traffic

In PyTorch, each operation launches a separate GPU kernel. Consider **LayerNorm**:

```python
# PyTorch LayerNorm — what actually happens under the hood:
def layer_norm(x, gamma, beta):
    mean = x.mean(dim=-1, keepdim=True)    # Kernel 1: read x, write mean
    var = x.var(dim=-1, keepdim=True)       # Kernel 2: read x, write var
    x_norm = (x - mean) / (var + eps).sqrt() # Kernel 3: read x, mean, var; write x_norm
    return gamma * x_norm + beta            # Kernel 4: read x_norm, gamma, beta; write output
```

Each kernel launch:
1. **Reads** input tensors from global memory (slow, ~400 cycles per access)
2. **Computes** the result (fast)
3. **Writes** the result back to global memory (slow again)

The next kernel then **reads that same result right back** from global memory.

```
Without Fusion (4 separate kernels):

  Global Memory    Compute    Global Memory    Compute    Global Memory
  ┌──────┐         ┌────┐     ┌──────┐         ┌────┐     ┌──────┐
  │ Read │ ──────→ │Mean│ ──→ │Write │         │    │     │      │
  │  x   │         └────┘     │ mean │         │    │     │      │
  └──────┘                    └──┬───┘         │    │     │      │
                                 │              │    │     │      │
                              ┌──▼───┐         │    │     │      │
                              │ Read │ ──────→ │Var │ ──→ │Write │
                              │x,mean│         └────┘     │ var  │
                              └──────┘                    └──┬───┘
                                                             │
                        ... and so on for each step ...      │

  Total global memory round-trips: ~8 reads + 4 writes = 12 memory operations
```

```
With Fusion (1 kernel):

  Global Memory         Compute (all in registers/shared mem)        Global Memory
  ┌──────┐         ┌─────────────────────────────────────────┐       ┌──────┐
  │ Read │ ──────→ │ mean → var → normalize → scale + shift  │ ────→ │Write │
  │  x   │         │     (all done using registers)          │       │output│
  └──────┘         └─────────────────────────────────────────┘       └──────┘

  Total global memory round-trips: 1 read + 1 write = 2 memory operations
```

**The fused version does 6x fewer memory operations.** For a memory-bound operation
like LayerNorm, this translates almost directly to a 3-6x speedup.

### Why This Is THE Reason Triton Exists

PyTorch's eager mode launches a separate kernel for every operation. The PyTorch
JIT and `torch.compile` can fuse some patterns, but they are limited in what
they can combine.

**Triton lets you write a single kernel that does everything.** You read from
global memory once, do all your computation while the data sits in fast
registers and shared memory, and write back once.

This is not just a nice optimization — for memory-bound operations (which is most
of a transformer), it is the **single most important optimization available.**

### Concrete Impact

| Operation | PyTorch (separate kernels) | Fused Triton Kernel | Speedup |
|---|---|---|---|
| Vector elementwise (x * w + b) | 2 kernels, 4 memory ops | 1 kernel, 2 memory ops | ~2x |
| LayerNorm | 4-5 kernels, ~12 memory ops | 1 kernel, 2 memory ops | 3-6x |
| Softmax | 3 kernels, ~6 memory ops | 1 kernel, 2 memory ops | 2-4x |
| Fused Attention (FlashAttention) | Many kernels, O(N^2) memory | 1 kernel, O(N) memory | 2-10x+ |

---

## How This Connects to Triton

Let us map everything in this module to what you will actually do in Phase 2.

**Threads → Triton Programs:**
Triton abstracts away individual threads. Instead of thinking "thread 42 processes
element 42," you think "program 3 processes elements 768 through 1023." Each Triton
"program" corresponds to one block of threads.

**Warps → Handled Automatically:**
Triton schedules warps within a block for you. You do not write warp-level code.
But understanding warps helps you choose good block sizes (multiples of 32) and
understand performance characteristics.

**Memory Hierarchy → Still Your Responsibility:**
Triton handles some memory optimizations automatically, but you still decide:
- **Block size**: How much data each program processes (determines shared memory usage)
- **Access patterns**: Whether your `tl.load` and `tl.store` calls are coalesced
- **Fusion strategy**: Which operations to combine into one kernel

**Memory Coalescing → Built Into Triton's Design:**
When you write `tl.load(ptr + offsets)` where `offsets = tl.arange(0, BLOCK_SIZE)`,
Triton generates coalesced memory accesses automatically. This is one of Triton's
biggest advantages over hand-written CUDA.

**Preview of Phase 2:**
In the next phase, you will write a vector addition kernel where each Triton
"program" processes one block of data:

```python
# Preview — you will understand every line of this after Phase 2
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)                          # Which block am I?
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # My elements
    mask = offsets < n_elements                       # Guard for partial blocks
    x = tl.load(x_ptr + offsets, mask=mask)          # Coalesced load
    y = tl.load(y_ptr + offsets, mask=mask)          # Coalesced load
    tl.store(output_ptr + offsets, x + y, mask=mask) # Coalesced store
```

Notice the direct mapping from GPU Puzzles:
- `pid` is the block index (like `cuda.blockIdx.x`)
- `offsets` computes global positions (like `blockIdx * blockDim + threadIdx`)
- `mask` is the guard condition (like `if i < n_elements`)

---

## Exercises (Conceptual — No Code Required)

### Exercise 1: Arithmetic Intensity

Calculate the arithmetic intensity of the following operations. Assume 32-bit
(4-byte) floats.

**(a) ReLU**: `out[i] = max(0, x[i])`
- Bytes loaded: ?
- Bytes stored: ?
- FLOPs: ?
- Arithmetic intensity: ?

<details>
<summary>Answer</summary>

- Load: 4 bytes (one float)
- Store: 4 bytes (one float)
- FLOPs: 1 (one comparison/max)
- Intensity: 1 / 8 = 0.125 FLOP/byte — deeply memory-bound

</details>

**(b) Dot Product of Two Vectors of Length N**: `result = sum(a[i] * b[i])`
- Total bytes loaded: ?
- Total FLOPs: ?
- Arithmetic intensity: ?

<details>
<summary>Answer</summary>

- Load: 2 * N * 4 = 8N bytes
- FLOPs: N multiplications + N additions = 2N
- Intensity: 2N / 8N = 0.25 FLOP/byte — memory-bound

</details>

**(c) Matrix Multiply (NxN @ NxN)**:
- Total bytes loaded: ?
- Total FLOPs: ?
- Arithmetic intensity: ?

<details>
<summary>Answer</summary>

- Load: 2 * N^2 * 4 = 8N^2 bytes
- FLOPs: 2 * N^3 (N multiplications + N additions for each of N^2 output elements)
- Intensity: 2N^3 / 8N^2 = N/4 FLOP/byte
- For N=1024: 256 FLOP/byte — very compute-bound

</details>

### Exercise 2: Memory Access Patterns

For each scenario, state whether the memory access is coalesced or non-coalesced:

**(a)** 32 threads in a warp read elements `a[0], a[1], a[2], ..., a[31]`

**(b)** 32 threads in a warp read elements `a[0], a[32], a[64], ..., a[992]`

**(c)** 32 threads in a warp read the SAME element `a[0]`

<details>
<summary>Answers</summary>

(a) **Coalesced.** Consecutive threads access consecutive addresses. One memory
transaction.

(b) **Non-coalesced (strided).** Each thread accesses an element 32 positions
apart. This results in up to 32 separate memory transactions — extremely slow.
This pattern commonly appears in column-wise access of a row-major matrix.

(c) **Broadcast.** This is actually handled efficiently by the hardware — it
recognizes that all threads want the same address and serves it with a single
transaction. (This is a special case, not the same as coalescing.)

</details>

### Exercise 3: Draw the Memory Access Pattern

On paper, draw the memory layout of a 4x4 matrix stored in row-major order.
Then draw which memory addresses are accessed by a warp of 4 threads (simplified)
when:

**(a)** Each thread reads one element from the same row
**(b)** Each thread reads one element from the same column

Which access pattern is coalesced? Which is strided?

### Exercise 4: Kernel Fusion

Consider the GELU activation: `gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`

In PyTorch, this involves multiple operations: power, multiply, add, tanh,
multiply, add, multiply. If each were a separate kernel:

- How many global memory round-trips would occur?
- How many would a fused kernel need?
- Is GELU memory-bound or compute-bound?

### Exercise 5: Roofline Reasoning

An operation runs at 200 GFLOPS on a T4.

- If its arithmetic intensity is 2 FLOP/byte, is it memory-bound or compute-bound?
  (The T4 memory bandwidth ceiling at 2 FLOP/byte is 320 * 2 = 640 GFLOPS, and
  the compute ceiling is 8100 GFLOPS. Which ceiling is lower?)
- What is the maximum theoretical GFLOPS it could achieve at this arithmetic
  intensity?
- What fraction of peak memory bandwidth is it using? (200 / 640)

---

## Key Terms Glossary

| Term | Definition |
|---|---|
| **Thread** | The smallest unit of execution on a GPU. Runs one instance of the kernel function. |
| **Warp** | A group of 32 threads that execute in lockstep (SIMT). The hardware's scheduling unit. |
| **Warp Divergence** | When threads in a warp take different code paths, forcing serial execution of both paths. |
| **Thread Block (CTA)** | A group of threads (128-512 typical) that can share memory and synchronize. Runs on one SM. |
| **Grid** | The full collection of thread blocks for one kernel launch. |
| **Streaming Multiprocessor (SM)** | A physical compute unit on the GPU. Each SM runs one or more thread blocks. |
| **Global Memory (VRAM)** | The GPU's main memory (16 GB on T4). Large but slow (~400 cycles). |
| **Shared Memory (SMEM)** | Fast on-chip memory (~5 cycles) shared among threads in one block. Up to 64 KB per block. |
| **Registers** | Fastest per-thread storage (~1 cycle). Limited to ~255 per thread. |
| **L1/L2 Cache** | Hardware-managed caches. L1 shares physical space with shared memory. L2 is 4 MB on T4. |
| **Memory Coalescing** | When a warp accesses consecutive addresses, the GPU combines requests into one transaction. |
| **Arithmetic Intensity** | FLOPs per byte of memory traffic. Determines if an operation is memory- or compute-bound. |
| **Memory-Bound** | Performance limited by memory bandwidth, not compute. Most ML operations. |
| **Compute-Bound** | Performance limited by compute throughput, not memory. Mainly matmul. |
| **Roofline Model** | A visual model showing max achievable performance as a function of arithmetic intensity. |
| **Kernel Fusion** | Combining multiple operations into one kernel to reduce global memory round-trips. |
| **SIMT** | Single Instruction, Multiple Threads. GPU execution model where a warp of 32 threads executes the same instruction on different data. |
| **Tensor Cores** | Specialized hardware units for matrix multiply-accumulate at FP16/BF16. Present on T4 (Turing) and later. |

---

## Milestone Checklist

You are ready for **Phase 2: Triton Basics** when you can confidently explain all
of the following:

- [ ] **What is a warp?** 32 threads that execute in lockstep. The actual
      scheduling unit of the GPU hardware.

- [ ] **Why does memory coalescing matter?** Because 32 coalesced accesses become
      one fast transaction, while 32 scattered accesses become 32 slow
      transactions. This can cause a 10-20x performance difference.

- [ ] **Why does kernel fusion help?** It reduces the number of global memory
      round-trips. Instead of writing intermediate results to slow global memory
      and reading them back, fused kernels keep data in fast registers and shared
      memory.

- [ ] **Is softmax memory-bound or compute-bound?** Memory-bound. It has low
      arithmetic intensity (~0.5 FLOP/byte) — there are only a few operations
      (exp, sum, divide) for each element loaded. The GPU compute cores sit idle
      waiting for data.

- [ ] **Bonus: What is the arithmetic intensity of vector addition?** About
      0.083 FLOP/byte (1 FLOP / 12 bytes). Deeply memory-bound.

---

## Recommended Resources

For deeper exploration of the concepts in this module:

### Books
- **"Programming Massively Parallel Processors" (PMPP)** by Hwu, Kirk, and El Hajj
  — Chapters 1-4 cover the execution model and memory hierarchy in full detail.
  This is the standard textbook for GPU programming.

### NVIDIA Documentation
- [CUDA C++ Programming Guide — Memory Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy)
- [CUDA Best Practices — Memory Optimizations](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)
- [T4 (Turing) Architecture Whitepaper](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf)

### Interactive Resources
- [GPU Puzzles](https://github.com/srush/GPU-Puzzles) — What you completed in Module 01
- [Roofline Model explained (Berkeley)](https://crd.lbl.gov/divisions/amcr/computer-science-amcr/par/research/roofline/)

### Videos
- "How GPU Computing Works" — GTC presentation by Stephen Jones (NVIDIA)
- "GPU Architecture and CUDA Programming" — Lecture series by Wen-mei Hwu (UIUC)

---

## What Comes Next

In **Phase 2: Triton Basics**, you will put all of this theory into practice:

1. **Vector Addition** — Your first real Triton kernel. You will see `tl.program_id`,
   `tl.arange`, `tl.load`, `tl.store`, and masks — all mapping directly to what
   you learned in GPU Puzzles.
2. **Softmax** — Your first fused kernel. You will see why keeping data in
   registers instead of writing back to global memory gives a 2-4x speedup.
3. **Benchmarking** — You will measure your kernels against PyTorch and see the
   roofline model in action.

The hardware knowledge from this module is what separates someone who can *write*
Triton kernels from someone who can write *fast* Triton kernels.
