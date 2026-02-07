# Triton Kernel Programming Curriculum

A structured self-study curriculum to learn GPU kernel programming with Triton, targeting roles at OpenAI, Anthropic, and NVIDIA.

## Who This Is For

You know Python and PyTorch. You've built transformers from scratch and fine-tuned models on Modal. Your GPU experience is limited to `device="cuda"` — you don't know what threads, blocks, or warps are yet. This curriculum fixes that.

## Setup

### Google Colab (Primary)

All kernel code in this curriculum runs on a free Colab T4 GPU.

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Create a new notebook
3. Runtime → Change runtime type → **T4 GPU**
4. Install Triton (it comes pre-installed on Colab, but to be safe):

```python
!pip install triton
import triton
print(triton.__version__)
```

5. Verify GPU access:

```python
import torch
print(torch.cuda.is_available())        # True
print(torch.cuda.get_device_name(0))    # Tesla T4
```

### Modal (For Distributed Training — Phase 4)

Phase 4 requires multi-GPU setups. Use Modal for this:

```bash
pip install modal
modal setup  # one-time auth
```

```python
import modal
app = modal.App("distributed-training")

@app.function(gpu="A100", cpu=8)
def train():
    import torch
    print(torch.cuda.device_count())
```

### Local Machine (M4 Max)

Your M4 Max cannot run Triton or CUDA. Use it for:
- Reading these `.md` files
- Writing code before pasting into Colab
- Running PyTorch CPU-only experiments
- Reading papers and documentation

## How to Use This Curriculum

Each `.md` file is a **study guide**, not a notebook. The workflow:

1. **Read** the `.md` file on your laptop
2. **Understand** the concepts and code
3. **Open Colab** and re-implement everything from scratch (don't copy-paste)
4. **Experiment** — change parameters, break things, observe what happens
5. **Do the exercises** at the bottom of each file
6. **Check the milestone** — can you do the thing it describes? If yes, move on

## Curriculum Overview

### Phase 1: GPU Fundamentals (Week 1–2)

*"What is a GPU actually doing?"*

Before writing any Triton code, you need a mental model of how GPUs work. This phase is text-heavy and concept-heavy. No Triton yet.

| File | What You'll Learn |
|------|-------------------|
| `01-gpu-puzzles.md` | Hands-on warmup with GPU Puzzles (Sasha Rush) |
| `02-memory-and-compute.md` | Threads, blocks, warps, memory hierarchy, SIMT, roofline |

### Phase 2: Triton Basics (Week 2–4)

*"Writing your first GPU kernels."*

Now you write real Triton kernels, starting simple and building up. Every line of code is explained.

| File | What You'll Learn |
|------|-------------------|
| `01-vector-addition.md` | Your first kernel — programs, blocks, masking |
| `02-fused-softmax.md` | Reductions, numerical stability, what "fused" means |
| `03-matrix-multiplication.md` | Tiling, shared memory, the most important kernel |
| `04-fused-layernorm.md` | Solo exercise with reference code |

### Phase 3: Making Kernels Fast (Week 4–6)

*"Going from correct to fast."*

You know how to write kernels. Now make them competitive with cuBLAS.

| File | What You'll Learn |
|------|-------------------|
| `01-autotuning.md` | `@triton.autotune`, profiling, benchmarking |
| `02-flash-attention.md` | Paper-to-kernel walkthrough |
| `03-reading-real-kernels.md` | Navigating vLLM, Unsloth, torchtune source code |

### Phase 4: Distributed Training (Week 6–8)

*"Scaling beyond one GPU."*

| File | What You'll Learn |
|------|-------------------|
| `01-ddp-from-scratch.md` | Data parallelism, all-reduce, ring all-reduce |
| `02-fsdp.md` | Fully Sharded Data Parallel, ZeRO stages |
| `03-tensor-parallelism.md` | Splitting layers across GPUs, Megatron-LM style |

### Phase 5: Build in Public (Week 8–9)

*"Make your work visible."*

| File | What You'll Learn |
|------|-------------------|
| `01-blog-post-guide.md` | How to write a kernel walkthrough blog post |
| `02-open-source-contribution.md` | Making your first PR to vLLM or torchtune |

### Phase 6: Targeting Companies (Week 9–10)

*"Tailoring your prep."*

| File | What You'll Learn |
|------|-------------------|
| `01-anthropic.md` | Interpretability, RLHF, Constitutional AI |
| `02-nvidia.md` | CUDA, TensorRT, compiler internals |
| `03-openai.md` | Scale, infra, speculative decoding |

## Timeline

This is a **10-week plan** at ~2 hours/day. Adjust based on your pace.

```
Week  1-2:  Phase 1 — GPU fundamentals (read, think, draw diagrams)
Week  2-4:  Phase 2 — First Triton kernels (code every day)
Week  4-6:  Phase 3 — Optimization (profile, tune, iterate)
Week  6-8:  Phase 4 — Distributed training (multi-GPU on Modal)
Week  8-9:  Phase 5 — Blog post + open source PR
Week  9-10: Phase 6 — Company-specific prep + applications
```

## Prerequisites Checklist

Before starting, make sure you can:

- [x] Write Python fluently
- [x] Use PyTorch (tensors, autograd, training loops)
- [x] Understand transformers (attention, feedforward, residuals)
- [x] Use NumPy broadcasting and vectorized operations
- [x] Use Google Colab with a GPU runtime

## Key Resources (Referenced Throughout)

- [Triton Official Tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- [GPU Puzzles by Sasha Rush](https://github.com/srush/GPU-Puzzles)
- [PMPP Book (Programming Massively Parallel Processors)](https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands/dp/0323912311)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [Triton Paper (Tillet et al.)](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)
