# 03 — Reading Real Kernels: Navigating Production Triton Code

> **Prerequisites:** You have completed the autotuning and Flash Attention lessons. You
> can write, autotune, and benchmark Triton kernels. Now you learn to read and understand
> kernels written by others.

---

## Learning Objectives

By the end of this lesson you will be able to:

1. Navigate a real-world kernel codebase (vLLM, Unsloth, torchtune) and find the
   relevant kernel files.
2. Apply a systematic 5-step process to understand any unfamiliar Triton kernel.
3. Identify common optimization patterns used in production kernels.
4. Read CUDA C++ kernel signatures and map concepts to their Triton equivalents.
5. Write a concise "kernel review" summarizing what a kernel does and how.

---

## Why Read Real Kernels?

Writing your own kernels from tutorials is necessary but not sufficient. Production
kernel code is different from tutorial code in several ways:

- **It is not annotated.** Tutorial kernels have a comment on every line. Real kernels
  have sparse comments, if any. You must infer intent from the code structure.
- **It handles edge cases.** Real kernels deal with arbitrary dtypes, non-power-of-2
  dimensions, different GPU architectures, and backward passes.
- **It uses advanced tricks.** Production kernels exploit hardware-specific features
  (Tensor Core layouts, async copy, warp-level primitives) that tutorials skip.
- **It is structured for a larger system.** The kernel is one file in a codebase of
  thousands. Understanding how it integrates with the Python-level API, the model
  definition, and the training loop is itself a skill.

Reading real kernels builds three concrete skills:

1. **Interview readiness.** "I've read the PagedAttention kernel in vLLM and can
   explain how it handles non-contiguous KV cache blocks" is a strong signal.
2. **Contribution readiness.** You cannot submit a PR to a project whose code you
   cannot read. Reading comes before writing.
3. **Pattern recognition.** After reading 5-10 production kernels, you start
   recognizing recurring patterns (tiling strategies, reduction trees, online
   algorithms) that make new kernels easier to understand.

---

## A Systematic Approach to Reading Kernels

When you open a new kernel file for the first time, resist the urge to read top to
bottom. Instead, follow these five steps in order.

### Step 1: Start with the Python Wrapper (Launcher Function)

The launcher function is the entry point that users call. It is regular Python code,
not GPU code, and it tells you:

- What tensors go in and come out
- How the grid is configured (1D, 2D, 3D)
- What arguments are passed to the kernel
- What `tl.constexpr` values are set

**What to look for:**

```python
def some_operation(input_tensor, weight, bias, ...):
    # 1. Input validation and shape extraction
    batch, seq_len, hidden = input_tensor.shape

    # 2. Output allocation
    output = torch.empty_like(input_tensor)

    # 3. Grid configuration — THIS TELLS YOU HOW WORK IS DIVIDED
    grid = (triton.cdiv(seq_len, BLOCK_SIZE), batch)

    # 4. Kernel launch — LOOK AT WHAT ARGUMENTS ARE PASSED
    some_kernel[grid](
        input_tensor, weight, bias, output,
        seq_len, hidden,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2),
        BLOCK_SIZE=128,
        HIDDEN_DIM=hidden,
    )
    return output
```

From this launcher alone, you can already answer:
- This kernel operates on 3D tensors (batch, seq_len, hidden).
- Work is parallelized over (seq_len_blocks, batch) — each program handles one block
  of sequence positions for one batch element.
- The hidden dimension is a compile-time constant, so the entire hidden dim is processed
  within a single program.

### Step 2: Read the Kernel Signature

```python
@triton.jit
def some_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    seq_len, hidden,
    stride_batch, stride_seq, stride_hidden,
    BLOCK_SIZE: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
):
```

Classify each parameter:
- **Pointers:** `input_ptr`, `weight_ptr`, `bias_ptr`, `output_ptr` — data locations
- **Dimensions:** `seq_len`, `hidden` — runtime shape information
- **Strides:** `stride_batch`, etc. — memory layout information
- **Constants:** `BLOCK_SIZE`, `HIDDEN_DIM` — compiled into the kernel

Also check: is there an `@triton.autotune` decorator? If so, read the configs to
understand what parameters are being searched.

### Step 3: Identify the Program Mapping

The first lines of the kernel body establish which piece of work this program handles:

```python
pid_seq = tl.program_id(0)
pid_batch = tl.program_id(1)
```

Ask yourself:
- How many dimensions does the grid have?
- What does each program compute? (One row? One tile? One block of rows?)
- How many programs are launched total?

### Step 4: Trace the Data Flow

Now read the kernel body. Focus on three phases:

**Load phase:** What data is loaded from HBM? Look for `tl.load()` calls.
```python
# What is being loaded? From where? With what mask?
x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
```

**Compute phase:** What operations are performed? Look for `tl.dot()`, arithmetic,
reductions (`tl.sum`, `tl.max`), and control flow.
```python
# What math is happening?
normalized = (x - mean) / tl.sqrt(var + eps)
output = normalized * weight + bias
```

**Store phase:** What is written back to HBM? Look for `tl.store()` calls.
```python
# What is the output? Where does it go?
tl.store(output_ptr + offsets, output, mask=mask)
```

### Step 5: Identify Optimizations

After understanding the basic data flow, look for performance tricks:

- **Tiling:** Is there a loop that iterates over tiles? What dimension is tiled?
- **Fusion:** Does the kernel combine multiple operations (e.g., layernorm + dropout +
  residual) that PyTorch would do in separate kernels?
- **Online algorithms:** Is there a running max/sum (like Flash Attention's online
  softmax)?
- **Grouped ordering:** Is there program reordering for L2 cache optimization?
- **Mixed precision:** Does the kernel accumulate in FP32 but load/store in FP16?
- **Async operations:** Does the kernel use `tl.async_copy` or pipeline stages?

---

## Case Study 1: vLLM PagedAttention

### What vLLM Does

vLLM is the most widely used open-source LLM inference engine. Its core innovation is
**PagedAttention**, which manages the KV cache using a paging mechanism inspired by
operating system virtual memory.

Repository: [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)

### The Problem PagedAttention Solves

During LLM inference (text generation), the model generates tokens one at a time. Each
new token requires the key and value vectors from ALL previous tokens (the KV cache).
This cache grows with each token generated.

The naive approach pre-allocates a contiguous block of memory for the maximum possible
sequence length. If your max sequence length is 8192 but most requests only use 200
tokens, you waste 97% of the memory. With many concurrent requests, this waste makes it
impossible to serve many users simultaneously.

PagedAttention solves this by splitting the KV cache into fixed-size **pages** (called
"blocks" in vLLM — not to be confused with GPU thread blocks). Each page holds a few
tokens' worth of KV vectors. Pages are allocated on demand and can be non-contiguous in
memory.

```
Standard KV cache:
  Request 1: [K₁ K₂ K₃ K₄ ... K₈₁₉₂]   <- contiguous, mostly empty
  Request 2: [K₁ K₂ K₃ K₄ ... K₈₁₉₂]   <- contiguous, mostly empty

PagedAttention KV cache:
  Request 1: Page 7 -> Page 3 -> Page 12    (only 3 pages needed for 48 tokens)
  Request 2: Page 1 -> Page 9               (only 2 pages needed for 32 tokens)
  Free pages: 0, 2, 4, 5, 6, 8, 10, 11, 13, ...
```

### Finding the Kernel Files

The vLLM codebase is large. Here is how to navigate to the attention kernels:

```
vllm/
  vllm/
    attention/
      backends/               <- Different attention backend implementations
        flash_attn.py         <- Flash Attention backend (wraps flash_attn library)
        torch_sdpa.py         <- PyTorch SDPA backend
      ops/
        paged_attn.py         <- PagedAttention ops (Python interface)
    _custom_ops.py            <- Bindings to C++/CUDA custom ops
  csrc/
    attention/
      paged_attention_v1.cu   <- The CUDA kernel (not Triton — CUDA C++)
      paged_attention_v2.cu   <- Optimized version for long sequences
      attention_kernels.cuh   <- Shared kernel code
```

vLLM's PagedAttention is implemented in CUDA C++, not Triton. This is common for
performance-critical inference kernels — CUDA gives more control over warp-level
operations and shared memory layout. However, the concepts are the same as what you
learned in Triton.

### Reading the PagedAttention Structure

Without reproducing the full CUDA code, here is the architecture:

**The Python layer** (`paged_attn.py`):
- Takes the query tensor, the KV cache (as a pair of tensors), and a **block table**
  (a mapping from sequence position to page index).
- Passes the block table to the kernel so it knows where to find each token's KV data.

**The kernel's key innovation:**
- Instead of loading K and V from contiguous memory, the kernel uses the block table to
  look up which page each token's KV data lives in.
- The inner loop iterates over pages, not contiguous positions.
- Within each page, the tokens are contiguous (so loads are efficient).
- Across pages, there is indirection (the block table lookup), but pages are large
  enough that the indirection cost is amortized.

**Applying the 5-step process:**

1. **Launcher:** Look for the Python function that calls `ops.paged_attention_v1`.
   Note the block table argument — this is the page mapping.

2. **Signature:** The CUDA kernel takes query, key_cache, value_cache, block_tables,
   context_lens, and the usual dimension/stride parameters.

3. **Program mapping:** Each thread block handles one query head. Within the block,
   threads cooperate to process key/value tokens.

4. **Data flow:** Load query. For each page in the block table: load K/V tokens from
   that page, compute attention scores, accumulate. Store output.

5. **Optimizations:** Shared memory is used to store the query vector (broadcast to all
   threads). Warp-level reductions compute the softmax. The block table indirection is
   the key structural difference from standard attention.

### What to Take Away

PagedAttention is a good example of how kernel design is driven by a systems-level
insight (memory fragmentation in KV caches), not just low-level GPU optimization. The
kernel is more complex than standard attention because of the indirection, but the
basic structure (load Q, iterate over K/V, accumulate attention output) is the same
pattern you implemented in Flash Attention.

---

## Case Study 2: Unsloth Fused Kernels

### What Unsloth Does

Unsloth is a library that accelerates LLM fine-tuning by fusing operations that PyTorch
executes as separate kernels. The result is faster training with lower memory usage.

Repository: [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth)

### Finding the Kernel Files

Unsloth's Triton kernels are typically located in:

```
unsloth/
  unsloth/
    kernels/
      cross_entropy_loss.py    <- Fused cross-entropy loss
      rms_layernorm.py         <- Fused RMS LayerNorm
      rope_embedding.py        <- Fused RoPE (Rotary Position Embedding)
      swiglu.py                <- Fused SwiGLU activation
      utils.py                 <- Shared utilities
```

### Fused Cross-Entropy Loss — A Walkthrough

The standard cross-entropy loss in PyTorch looks like this:

```python
# PyTorch's approach (simplified):
logits = model(input)                           # (batch, seq_len, vocab_size)
logits = logits.view(-1, vocab_size)            # (batch*seq_len, vocab_size)
loss = F.cross_entropy(logits, targets)
```

The problem: `logits` has shape (batch*seq_len, vocab_size), and vocab_size is often
32,000 to 128,000. PyTorch's cross-entropy first computes `log_softmax(logits)` —
which requires materializing a (batch*seq_len, vocab_size) intermediate tensor — and
then picks the target entries.

Unsloth fuses the entire computation: log-sum-exp, target lookup, and loss reduction
happen in a single kernel. The vocab-sized intermediate is never written to HBM.

**Applying the 5-step process:**

1. **Launcher:** Look for the wrapper function. It takes logits and targets, allocates
   an output tensor for per-token losses, and launches the kernel with a grid over
   (num_tokens,).

2. **Signature:** The kernel takes logits_ptr, targets_ptr, loss_ptr, and the vocab
   size as a constexpr.

3. **Program mapping:** Each program handles one token (one row of the logits matrix).

4. **Data flow:**
   - Load the target index for this token.
   - Iterate over the vocab dimension in tiles (BLOCK_SIZE chunks).
   - For each tile: load logits, compute max (for numerical stability), accumulate
     the sum of exponentials.
   - After all tiles: compute `log_sum_exp = max + log(sum_exp)`.
   - Load the logit corresponding to the target.
   - Loss for this token = `log_sum_exp - target_logit`.
   - Store the loss.

5. **Optimizations:**
   - The vocab dimension is tiled, so the full (vocab_size,) row is never in SRAM at
     once.
   - Online reduction for max and sum (same pattern as Flash Attention's online softmax).
   - The target logit is loaded in a second pass (or tracked during the first pass if
     it falls in the current tile).
   - FP32 accumulation for numerical precision, even if logits are FP16.

### LoRA-Aware Kernels

Unsloth also has kernels that are aware of LoRA (Low-Rank Adaptation). In standard
fine-tuning with LoRA, the forward pass computes:

```python
output = base_linear(x) + lora_B(lora_A(x))
```

This requires three separate kernel launches (base linear, LoRA A, LoRA B) plus an
addition. Unsloth fuses these into fewer kernel launches by incorporating the LoRA
decomposition directly into the kernel. The key insight is that the LoRA matrices are
small (rank 8-64), so they can be loaded once and reused.

### What to Take Away

Unsloth demonstrates the power of **kernel fusion** — the same algorithmic idea applied
to different operations (cross-entropy, RMS norm, RoPE, SwiGLU). The pattern is always:
take N separate PyTorch operations, identify the intermediate tensors that get written
to HBM between operations, and fuse everything into one kernel that keeps intermediates
in registers/SRAM.

---

## Case Study 3: torchtune

### What torchtune Does

torchtune is PyTorch's official library for fine-tuning LLMs. It is more conservative
than Unsloth — fewer custom kernels, more reliance on PyTorch's built-in optimizations.
But it provides a clean, well-documented codebase that shows how custom ops integrate
into a training pipeline.

Repository: [github.com/pytorch/torchtune](https://github.com/pytorch/torchtune)

### Finding Relevant Code

```
torchtune/
  torchtune/
    modules/
      attention.py             <- Attention implementation
      rms_norm.py              <- RMS LayerNorm
      rotary_positional_embeddings.py  <- RoPE
      loss/
        ce_chunked_output_loss.py      <- Memory-efficient cross-entropy
    training/
      _distributed.py          <- Distributed training utilities
```

### The Chunked Cross-Entropy Loss

torchtune takes a different approach to the cross-entropy memory problem than Unsloth.
Instead of writing a custom Triton kernel, torchtune **chunks** the computation in
Python:

```python
# torchtune's approach (simplified concept):
total_loss = 0
for chunk_start in range(0, seq_len, chunk_size):
    chunk_logits = logits[:, chunk_start:chunk_start+chunk_size, :]
    chunk_targets = targets[:, chunk_start:chunk_start+chunk_size]
    total_loss += F.cross_entropy(chunk_logits, chunk_targets, reduction='sum')
total_loss = total_loss / num_tokens
```

This avoids materializing the full (batch, seq_len, vocab_size) logits tensor at once.
Instead, only one chunk's worth of logits is in memory at any time.

This is a useful comparison point: torchtune achieves memory savings through **Python-
level chunking** rather than a custom kernel. The Unsloth approach (custom Triton
kernel) is faster because it also avoids the intermediate softmax tensor within each
chunk. But the torchtune approach is simpler, more maintainable, and "good enough" for
many use cases.

### Attention Implementation

torchtune's attention module is a clean PyTorch implementation that dispatches to
`torch.nn.functional.scaled_dot_product_attention`. This is worth reading because it
shows:

- How attention is structured in a production training codebase (handling GQA, RoPE
  integration, KV cache for inference).
- How custom ops are called: the Python module handles shape manipulation and
  dispatching, while the actual compute is delegated to optimized backends.
- How causal masking is handled at the Python level before being passed to the kernel.

### What to Take Away

torchtune shows the opposite end of the spectrum from Unsloth: fewer custom kernels,
more reliance on PyTorch's standard library. This is a valid engineering choice — custom
kernels add maintenance burden and debugging complexity. Reading torchtune helps you
understand when a custom kernel is worth writing (large performance gap, critical
bottleneck) versus when Python-level optimizations suffice.

---

## Reading CUDA Kernels (Bonus)

Some production kernels (like vLLM's PagedAttention) are written in CUDA C++, not
Triton. If you are targeting NVIDIA or need to read CUDA code in other projects, here
is a quick mapping.

### CUDA to Triton Concept Mapping

| CUDA C++ | Triton | Notes |
|---|---|---|
| `__global__ void kernel(...)` | `@triton.jit def kernel(...)` | Kernel function declaration |
| `blockIdx.x` | `tl.program_id(0)` | Which block/program |
| `threadIdx.x` | (implicit) | Triton abstracts away individual threads |
| `blockDim.x` | `BLOCK_SIZE` (constexpr) | Size of each block |
| `__shared__ float smem[N]` | (automatic) | Triton manages shared memory |
| `__syncthreads()` | (automatic) | Triton inserts barriers as needed |
| `atomicAdd(&x, val)` | `tl.atomic_add(ptr, val)` | Atomic operations |
| `#pragma unroll` | (automatic) | Triton unrolls loops over constexpr bounds |
| `reinterpret_cast<float4*>(ptr)` | (automatic) | Triton handles vectorized loads |

### Key Differences

**Thread-level vs. block-level programming:** In CUDA, you write code from the
perspective of a single thread. The thread computes its own index, loads its own data
element, and writes its own result. In Triton, you write code from the perspective of a
block/program that processes a tile of data using vector operations.

**Explicit vs. implicit shared memory:** In CUDA, you declare shared memory arrays,
load data into them explicitly, and call `__syncthreads()` before reading. In Triton,
the compiler decides when to use shared memory and inserts synchronization automatically.

**Warp-level primitives:** CUDA exposes warp-level operations like `__shfl_sync` (warp
shuffle), `__ballot_sync`, and cooperative matrix operations (WMMA/MMA). Triton does
not expose these directly — the compiler generates them under the hood from higher-level
operations like `tl.dot`. If you see these in CUDA code, understand what they do
conceptually, but know that Triton handles them automatically.

### Reading a CUDA Kernel Signature

```cpp
__global__ void paged_attention_v1_kernel(
    scalar_t* __restrict__ out,          // Output tensor (pointer + restrict hint)
    const scalar_t* __restrict__ q,      // Query tensor (const = read-only)
    const scalar_t* __restrict__ k_cache, // Key cache
    const scalar_t* __restrict__ v_cache, // Value cache
    const int* __restrict__ block_tables, // Page table (integer indices)
    const int* __restrict__ context_lens, // Length of each sequence
    const int num_heads,                  // Dimension info
    const int head_size,
    const int block_size,                 // Page size (KV tokens per page)
    const int max_num_blocks_per_seq      // Max pages per sequence
)
```

This maps to a Triton kernel with the same pointers and dimension arguments. The
`__restrict__` keyword tells the compiler that pointers do not alias (do not point to
overlapping memory), which enables more aggressive optimization. The `const` keyword
marks read-only pointers.

---

## Patterns to Watch For

After reading several production kernels, you will notice these recurring patterns:

### Pattern 1: Two-Phase Reduction

When a reduction (sum, max) spans more data than one program can handle:
- Phase 1: Each program reduces its local tile and writes a partial result.
- Phase 2: A second kernel reduces the partial results into the final answer.

This appears in layernorm (reduce over hidden dim), cross-entropy (reduce over vocab),
and attention (reduce over sequence length).

### Pattern 2: Fused Elementwise Operations

Multiple elementwise operations chained together in one kernel:
```
output = dropout(gelu(linear(x)))
```
Instead of three kernels (linear, gelu, dropout), one kernel loads x, computes all
three, and stores the result. The intermediates never touch HBM.

### Pattern 3: Online Accumulation

When computing a statistic (max, sum, mean) over a dimension that is too large to fit
in one tile, the kernel maintains running statistics and updates them tile by tile. You
saw this in Flash Attention. It also appears in online layernorm and streaming softmax.

### Pattern 4: Block Table Indirection

Instead of loading data from contiguous memory, the kernel loads from non-contiguous
locations using a lookup table. PagedAttention is the primary example, but the pattern
also appears in sparse attention and mixture-of-experts routing.

### Pattern 5: Mixed-Precision Accumulation

Loading data in FP16 (or BF16) for memory efficiency but accumulating in FP32 for
numerical precision:
```python
x = tl.load(ptr, ...).to(tl.float32)   # Load FP16, cast to FP32
acc += x                                 # Accumulate in FP32
tl.store(ptr, acc.to(tl.float16))       # Cast back to FP16 for storage
```

This is universal in production kernels. The `tl.dot` operation in Triton does this
automatically when the accumulator is FP32.

---

## Exercises

### Exercise 1: vLLM Kernel Exploration

Clone the vLLM repository and find the PagedAttention implementation.

```bash
git clone https://github.com/vllm-project/vllm.git
```

1. Find the Python-level PagedAttention wrapper. What arguments does it take?
2. Find the CUDA kernel file. What is the grid configuration (how are thread blocks
   assigned to work)?
3. How does the kernel use the block table to look up KV cache pages?
4. What modifications does vLLM make compared to the standard Flash Attention algorithm?

### Exercise 2: Unsloth Kernel Summary

Clone the Unsloth repository and find one Triton kernel (cross-entropy, RMS norm, RoPE,
or SwiGLU).

```bash
git clone https://github.com/unslothai/unsloth.git
```

Apply the 5-step process and write a one-paragraph summary:
- What does the kernel compute?
- How is work divided across programs?
- What fusion does it perform (what separate PyTorch ops does it replace)?
- What is the key optimization insight?

### Exercise 3: torchtune Architecture

Clone the torchtune repository and read the attention module.

```bash
git clone https://github.com/pytorch/torchtune.git
```

1. How does torchtune implement grouped query attention (GQA)?
2. Where is RoPE applied — before or after the attention computation?
3. Could the RoPE + attention be fused into a single Triton kernel? What would the
   kernel signature look like?

### Exercise 4: Kernel Review (Open-Ended)

Find ANY Triton kernel in a public GitHub repository (search for `@triton.jit` on
GitHub). Apply the 5-step reading process and write a one-page "kernel review"
containing:

- **What:** One-sentence description of what the kernel computes.
- **Grid:** How work is divided (1D, 2D, 3D grid; what each axis represents).
- **Algorithm:** The data flow (load, compute, store) in 3-5 bullet points.
- **Optimizations:** What performance tricks are used.
- **Assessment:** Is the kernel well-written? Any improvements you would suggest?

This exercise is excellent interview preparation. Being able to read, understand, and
critique a kernel on the spot demonstrates deep comprehension.

---

## Milestone Checklist

You are ready to move on to Phase 4 (Distributed Training) when you can:

- [ ] Open any Triton kernel and identify the program mapping, data flow, and
      optimizations within 30 minutes.
- [ ] Navigate the vLLM codebase and explain PagedAttention at a high level.
- [ ] Navigate the Unsloth codebase and explain one fused kernel.
- [ ] Read a CUDA kernel signature and map it to Triton concepts.
- [ ] Write a concise kernel review for an unfamiliar kernel.

---

## Resources

- [vLLM GitHub](https://github.com/vllm-project/vllm) — the most widely deployed LLM inference engine.
- [Unsloth GitHub](https://github.com/unslothai/unsloth) — fast LLM fine-tuning with fused Triton kernels.
- [torchtune GitHub](https://github.com/pytorch/torchtune) — PyTorch's official LLM fine-tuning library.
- [vLLM PagedAttention paper](https://arxiv.org/abs/2309.06180) — the paper describing the PagedAttention algorithm.
- [Triton GitHub repository](https://github.com/triton-lang/triton) — the Triton compiler source, including its own test kernels.
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — reference for reading CUDA kernel code.
