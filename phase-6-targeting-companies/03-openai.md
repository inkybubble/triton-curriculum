# Phase 6 — Company Targeting: OpenAI

## Learning Objectives

- Understand OpenAI's infrastructure challenges
- Know key concepts: scale, efficiency, serving optimization
- Understand speculative decoding and other inference techniques
- Prepare for OpenAI-specific technical interviews

---

## What OpenAI Does (Infra Perspective)

- Trains the largest models in the world (GPT-4, etc.)
- Serves millions of users via the API -- inference at massive scale
- Pushes the frontier on training efficiency and scaling
- Built Triton (the programming language you've been using!)
- Key infra challenges: training stability at scale, efficient serving, cost optimization

OpenAI's infra problems are defined by scale. Everything is bigger, faster, and more expensive than anywhere else. The questions are always: how do we train faster, serve cheaper, and scale further?

## Why They'd Want Someone with Kernel Skills

- OpenAI built Triton -- they want people who can use AND improve it
- Training GPT-scale models requires custom kernel optimization
- Serving at their scale requires every inference optimization possible
- They care about the full stack: from kernels to distributed systems to serving infrastructure
- If you can save 5% on inference cost, that's millions of dollars at their scale

---

## Key Areas to Study

### 1. Speculative Decoding

**Paper:** [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)

**The problem:** LLM inference is slow because it's sequential. Each token depends on the previous one. You generate one token at a time, and each token requires a full forward pass through the model.

**The insight:** Use a smaller, faster "draft" model to generate K candidate tokens. Then verify them all in parallel with the large model in a single forward pass.

**How it works:**

```
Draft model generates: [token1, token2, token3, token4, token5]
                            |
Large model verifies all 5 in ONE forward pass
                            |
Result: [accept, accept, accept, reject, --]
                            |
Output: 3 tokens generated in ~1 large model forward pass
         (instead of 3 separate forward passes)
```

**Why it's clever:** It's mathematically guaranteed to produce the SAME distribution as the large model alone. No quality loss. Pure speedup.

**Systems implications:**
- Need to run two models efficiently (draft model + target model)
- Draft model should be small enough that its cost is negligible
- Acceptance rate depends on how well the draft model approximates the target
- Batching draft and verify steps requires careful memory management
- Typical speedup: 2-3x for well-matched draft models

### 2. Continuous Batching

**Traditional batching:** Wait for a batch of requests, process all together, return all together.

**The problem:** Some sequences are short (10 tokens), some are long (2000 tokens). Short ones wait for long ones. GPU sits idle on padding.

**Continuous batching:** As sequences finish, immediately add new ones to the batch. The batch is always full.

```
Time ->
Request A: [=========]
Request B: [====]
                [Request D starts immediately]
Request C: [=======]
                   [Request E starts]
```

- vLLM implements this -- you studied it in Phase 3
- Key kernel challenge: variable-length sequences in the same batch require careful attention masking and memory layout
- PagedAttention (from vLLM) enables this by managing KV cache memory flexibly

### 3. Quantization for Inference

The progression: FP32 -> FP16 -> INT8 -> INT4. Each step roughly halves memory and can double throughput.

**Weight quantization:** Store weights in low precision, compute in higher precision. The key insight is that not all weights are equally important -- you can quantize most of them aggressively.

**Key papers:**

- [GPTQ](https://arxiv.org/abs/2210.17323): Post-training quantization using approximate second-order information. Fast to apply, good quality at INT4.
- [AWQ (Activation-Aware Weight Quantization)](https://arxiv.org/abs/2306.00978): Protects salient weights based on activation magnitudes. Better quality than naive quantization.
- [SmoothQuant](https://arxiv.org/abs/2211.10438): Migrates quantization difficulty from activations to weights. Enables INT8 for both weights and activations.

**Triton connection:** Many quantization kernels are written in Triton. The pattern is: dequantize weights on-the-fly during matmul. You load INT4 weights, convert to FP16, and multiply -- all in one fused kernel. This avoids storing the full FP16 weights in memory.

### 4. Training at Scale

- **Mixed precision training:** You know this from Phase 4. FP16/BF16 compute with FP32 master weights.
- **Gradient checkpointing:** Trade compute for memory. Don't store intermediate activations; recompute them during the backward pass. Reduces memory by O(sqrt(n)) for n layers.
- **Efficient data loading:** Don't let the data pipeline be the bottleneck. Tokenization, shuffling, and batching must keep up with GPU throughput.
- **Training stability:** At GPT-4 scale, loss spikes happen. Learning rate scheduling, gradient clipping, and careful initialization matter enormously.
- **Scaling laws:** Predicting the right model size for a compute budget. The Chinchilla paper changed how everyone thinks about this.

### 5. Infrastructure Architecture

- **Large GPU clusters:** Thousands of GPUs coordinated for a single training run
- **Fault tolerance:** GPUs die. Nodes go down. Network links fail. What happens to a training run that takes months?
- **Checkpointing strategies:** How often to checkpoint (too often wastes time, too rarely loses work), where to store checkpoints (local SSD vs. distributed storage), async checkpointing
- **Network topology:** How GPUs are connected affects parallelism strategy. Fat-tree, rail-optimized, etc.
- **Scheduling:** How do you allocate GPU resources across training runs, experiments, and inference serving?

---

## Triton (The Language) Connection

OpenAI created Triton to make kernel development accessible. This is directly relevant:

- Understanding Triton deeply shows alignment with their approach to infrastructure
- Be ready to discuss: **"What do you think are Triton's limitations? How would you improve it?"**

Good answers to that question:

- **Warp-level primitives:** Triton abstracts away warps, which limits some optimizations (e.g., warp-level reductions, warp shuffle)
- **Limited GPU feature support:** Some hardware features (e.g., async copy, TMA on H100) are hard to access from Triton
- **Compilation overhead:** Triton kernels compile at runtime, which adds startup cost. JIT caching helps but isn't perfect
- **Debugging tools:** printf debugging works but there's no real debugger. Profiling requires external tools (Nsight Compute)
- **Maturity:** The ecosystem of examples, documentation, and community is smaller than CUDA's

If you can critique Triton thoughtfully, it shows you understand it deeply -- not just as a user, but as someone who could contribute to it.

---

## Interview Prep -- What to Expect

- **Systems design:** "Design an LLM serving system that handles 10K requests/second"
- **Coding:** Python, possibly kernel optimization, distributed systems problems
- **ML fundamentals:** Transformer details, training dynamics, scaling behavior
- **Performance:** "How would you make inference 2x faster for a 70B model?" -- have multiple answers (quantization, speculative decoding, better batching, kernel optimization, model parallelism)
- **Scale thinking:** Everything at OpenAI is about doing things at unprecedented scale. Don't propose solutions that work for 1 GPU but not 1000.
- **Possible Triton-specific questions:** "Write a Triton kernel for X" or "How would you optimize this kernel?" -- you've trained for this in Phases 2-3

For the systems design question, a strong answer structure:

1. Requirements: throughput, latency, model size
2. Hardware: how many GPUs, what parallelism strategy
3. Serving architecture: load balancer -> routing -> model replicas
4. Optimizations: continuous batching, KV cache management, quantization
5. Monitoring: how to detect degraded performance, SLO violations

---

## Reading List (Prioritized)

### Must-Read

1. [Speculative Decoding paper](https://arxiv.org/abs/2211.17192)
2. [Triton paper: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)
3. Flash Attention papers (you know these from Phase 3)
4. [GPT-3 paper: Language Models are Few-Shot Learners (systems sections)](https://arxiv.org/abs/2005.14165)

### Should-Read

5. [Scaling Laws paper (Kaplan et al.)](https://arxiv.org/abs/2001.08361)
6. [Chinchilla paper: Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
7. One quantization paper -- [GPTQ](https://arxiv.org/abs/2210.17323) or [AWQ](https://arxiv.org/abs/2306.00978)

### Nice-to-Have

8. "Efficiently Scaling Transformer Inference" (Google, but relevant to anyone doing inference)
9. [OpenAI's research blog](https://openai.com/research)
10. Andrej Karpathy's "Let's reproduce GPT-2" video

---

## Exercises

1. **Implement a toy speculative decoding loop in Python** (no GPU needed). Simulate a draft model (random acceptance with probability p) and a target model. Measure the expected tokens per "forward pass" for different values of p and K (number of draft tokens).

2. **Calculate:** If a draft model accepts 70% of tokens and generates 5 candidates, what's the expected speedup? Work through the math: expected accepted tokens = sum over i from 1 to K of p^i, plus 1 for the first rejection or completion. Compare wall-clock time to sequential generation.

3. **Design a serving system on paper.** You need to serve a 70B model to 10K users. How many GPUs? What parallelism? How do you handle variable-length requests? Draw the architecture.

4. **Read the Triton paper** and write a 1-paragraph summary of how the compiler works. Focus on: what abstraction does Triton provide, and how does the compiler map that to GPU hardware?

5. **Look at OpenAI's current job openings.** What skills do they emphasize for infrastructure roles? How do those map to what you've learned in Phases 1-5?

---

## Phase 6 Summary

Each company has different strengths and focus areas, but the core skills overlap:

| Skill | Anthropic | NVIDIA | OpenAI |
|-------|-----------|--------|--------|
| GPU kernel programming (Triton/CUDA) | Valued | Core requirement | Valued (they built Triton) |
| Distributed training | Important | Important | Critical |
| Inference optimization | Important | Important | Critical |
| Systems thinking | High | High | High |
| Research awareness | High (safety focus) | Medium | High (scaling focus) |
| Hardware knowledge | Medium | Very high | Medium |

The differentiators:

- **Anthropic:** Show you care about safety and understand their research. Demonstrate that efficient infra enables responsible AI development.
- **NVIDIA:** Go deep on hardware. Know CUDA, know the memory hierarchy, know the compiler stack. Bridge the gap between Triton and CUDA.
- **OpenAI:** Think at scale. Every problem is 100x bigger than you'd expect. Know Triton deeply and be ready to critique it constructively.

---

## Milestone

You have a tailored reading list for your target company, can discuss their key technical challenges, and can explain how your skills -- Triton kernels, distributed training, optimization -- directly apply to their work. You know what to study, what to practice, and what to say in an interview.
