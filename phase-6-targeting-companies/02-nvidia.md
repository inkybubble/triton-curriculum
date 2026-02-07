# Phase 6 — Company Targeting: NVIDIA

## Learning Objectives

- Understand NVIDIA's role in the AI ecosystem
- Bridge from Triton to CUDA concepts
- Know TensorRT, cuDNN, and the NVIDIA software stack
- Prepare for NVIDIA-specific technical interviews

---

## What NVIDIA Does (Beyond GPUs)

- **Hardware:** GPUs (A100, H100, B200), DGX systems, networking (NVLink, NVSwitch, InfiniBand)
- **Software:** CUDA, cuDNN, TensorRT, Triton Inference Server (not the language), NCCL, cuBLAS
- **Frameworks:** NeMo (LLM training), Megatron-LM (distributed training)
- **The entire AI infrastructure stack** -- from silicon to software

NVIDIA's business is not just selling GPUs. It's selling a platform. The software stack (CUDA ecosystem) is the moat. Understanding that stack is what makes you valuable.

## Why They'd Want Someone with Triton Kernel Skills

- You understand GPU performance at a deep level
- You can bridge the gap between ML researchers and hardware
- NVIDIA needs people who can optimize their software stack
- The compiler team works on things similar to what Triton does automatically
- You can evaluate and improve the tools that other developers use

---

## Bridging Triton to CUDA

You've written Triton kernels. Here's how CUDA differs:

| Triton | CUDA | Notes |
|--------|------|-------|
| Program | Thread block | Triton abstracts away individual threads |
| `tl.program_id()` | `blockIdx.x` | Same concept |
| `tl.arange(0, BLOCK_SIZE)` | `threadIdx.x` | Triton uses block-level, CUDA uses thread-level |
| `tl.load()` / `tl.store()` | Pointer dereference | CUDA: `output[i] = input[i]` |
| `tl.constexpr` | Template parameter | Compile-time constants |
| `@triton.autotune` | Manual tuning or autotuning frameworks | |
| Implicit shared memory | `__shared__` keyword | CUDA requires explicit management |
| Implicit synchronization | `__syncthreads()` | CUDA requires explicit barriers |

The fundamental difference: Triton operates at the **block level** (you think about tiles of data), CUDA operates at the **thread level** (you think about individual threads). Triton's compiler decides how to map your block-level operations to threads and shared memory. In CUDA, you make those decisions yourself.

---

## CUDA Basics (For Interview Context)

A simple CUDA kernel for comparison:

```cuda
// CUDA vector addition -- compare to your Triton version
__global__ void add_kernel(float* x, float* y, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = x[idx] + y[idx];
    }
}

// Launch
int block_size = 256;
int grid_size = (n + block_size - 1) / block_size;
add_kernel<<<grid_size, block_size>>>(x, y, output, n);
```

How this maps to the Triton version:

- `blockIdx.x` = `tl.program_id(0)` -- which block am I?
- `threadIdx.x` = the index within the block -- Triton handles this with `tl.arange`
- `blockDim.x` = BLOCK_SIZE -- threads per block
- `<<<grid_size, block_size>>>` = Triton's `kernel[grid](...)`

In Triton, you wrote something like:

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)
```

Same computation. Triton gives you the mask pattern and block-level loads/stores. CUDA gives you raw thread-level control. Know both.

---

## Key NVIDIA Technologies

### 1. Tensor Cores

- Specialized hardware for matrix multiply-accumulate (MMA)
- Operate on small tiles (e.g., 16x16x16 for FP16)
- Triton's `tl.dot()` maps to Tensor Core instructions automatically
- H100 Tensor Cores: 4th generation, support FP8
- To use Tensor Cores in CUDA, you use WMMA (Warp Matrix Multiply Accumulate) intrinsics or CUTLASS
- Key interview point: Tensor Cores deliver an order of magnitude more throughput than CUDA cores for matrix operations, but only for specific shapes and data types

### 2. TensorRT

- NVIDIA's inference optimization engine
- Pipeline: trained model -> graph optimization -> kernel generation -> optimized engine
- Key optimizations: layer fusion, precision calibration (FP32 -> FP16 -> INT8), kernel auto-tuning
- Similar in spirit to what you've been doing manually with Triton, but automated
- TensorRT-LLM: the newer version specifically for LLM inference, combines TensorRT with custom attention kernels and quantization

### 3. cuDNN

- Library of optimized primitives for deep learning (convolutions, RNNs, attention)
- When you call `torch.nn.functional.conv2d`, cuDNN runs under the hood
- cuDNN's attention implementation competes with Flash Attention
- It's a "black box" of hand-tuned kernels -- you can't see the source, but you can benchmark against it

### 4. NCCL (Pronounced "Nickel")

- You used this in Phase 4 -- it's the communication library for multi-GPU training
- Optimized for NVLink (GPU-to-GPU within a node) and InfiniBand (across nodes)
- Implements: Ring AllReduce, tree AllReduce, and more
- NCCL is why `torch.distributed` works efficiently on NVIDIA hardware

### 5. NVLink and NVSwitch

- **NVLink:** Direct GPU-to-GPU connection, 900 GB/s on H100
- **NVSwitch:** Connects all 8 GPUs in a DGX node with full bisection bandwidth
- This is why tensor parallelism works well within a node (Phase 4)
- Across nodes, you rely on InfiniBand (~400 Gb/s per port) -- much slower than NVLink
- This bandwidth hierarchy is why parallelism strategy matters: tensor parallel within node, pipeline/data parallel across nodes

### 6. Compiler Stack (Advanced)

- **PTX:** NVIDIA's intermediate representation (like assembly for the GPU)
- **SASS:** The actual GPU machine code
- Compilation paths:
  - Triton: Triton IR -> LLVM IR -> PTX -> SASS
  - CUDA: CUDA C++ -> PTX -> SASS
- PTX provides forward compatibility (same PTX runs on newer GPUs)
- SASS is architecture-specific (different for A100 vs. H100)
- You can inspect PTX with `cuobjdump` or look at Triton's generated PTX for debugging

---

## GPU Architecture Deep Dive (For Interviews)

**Streaming Multiprocessor (SM):** The basic compute unit.

- H100: 132 SMs, each with 128 CUDA cores + 4 Tensor Cores
- A100: 108 SMs, each with 64 CUDA cores + 4 Tensor Cores

**Occupancy:** The ratio of active warps to maximum warps per SM.

- Higher occupancy helps hide memory latency (more warps to switch to while waiting)
- But maximum occupancy doesn't always mean maximum performance
- Sometimes using more registers per thread (lower occupancy) gives better performance because each thread does more useful work

**Register pressure:** Using too many registers per thread reduces occupancy.

- Each SM has a fixed register file (e.g., 65536 registers on H100)
- If your kernel uses 64 registers per thread and a warp has 32 threads, that's 2048 registers per warp
- The SM can only fit 32 warps -- occupancy is capped

**Shared memory vs. L1 cache:** Configurable split on most GPUs.

- You can allocate more shared memory (for explicit data reuse) or more L1 cache (for implicit caching)
- In Triton, this is handled automatically. In CUDA, you choose with `cudaFuncSetCacheConfig`

**Memory hierarchy (refresh from Phase 1):**

```
Registers (~0 cycles latency)
    |
Shared Memory / L1 (~20-30 cycles)
    |
L2 Cache (~200 cycles)
    |
HBM / Global Memory (~400-600 cycles)
```

---

## Interview Prep -- What to Expect

- **"Explain the GPU memory hierarchy"** -- you know this from Phase 1. Be specific about latencies and bandwidths.
- **"How would you optimize this CUDA kernel?"** -- apply your Triton optimization knowledge: coalesced memory access, tiling, occupancy analysis, Tensor Core utilization.
- **"What is warp divergence and why does it matter?"** -- when threads in a warp take different branches, both paths execute serially. Avoid data-dependent branching within a warp.
- **"Design a system for efficient LLM inference"** -- combine your Phase 3 (optimization) and Phase 4 (distributed) knowledge.
- Possibly a CUDA coding exercise (not common, but possible).
- Strong focus on computer architecture and systems knowledge. NVIDIA interviews go deeper on hardware than other companies.

---

## Reading List (Prioritized)

### Must-Read

1. [CUDA C++ Programming Guide (first 5 chapters)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
2. H100 whitepaper -- search "NVIDIA H100 Tensor Core GPU Architecture"
3. PMPP Book chapters 1-8 (you started this in Phase 1)

### Should-Read

4. TensorRT documentation (overview sections)
5. Megatron-LM paper (you saw this in Phase 4)
6. CUTLASS documentation -- NVIDIA's template library for CUDA matrix multiply

### Nice-to-Have

7. GTC talks on YouTube (NVIDIA's conference -- search for talks on topics you care about)
8. [NVIDIA Developer Blog](https://developer.nvidia.com/blog/)
9. "Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking" paper

---

## Exercises

1. Read a CUDA vector addition tutorial and compare it line-by-line to your Triton version. Write down every difference.
2. Install the CUDA toolkit on Colab and compile a simple CUDA program. Get "Hello from GPU" working.
3. Look at NVIDIA's current ML-related job openings. List the skills that overlap with your Triton knowledge and identify gaps.
4. Write a 1-paragraph explanation of Tensor Cores for a non-technical person. Focus on what they do, not how they work.

---

## Milestone

You can explain GPU architecture beyond what Triton abstracts, you understand the NVIDIA software stack, and you can map between Triton and CUDA concepts fluently. You can answer "how does a GPU actually work?" at multiple levels of detail.
