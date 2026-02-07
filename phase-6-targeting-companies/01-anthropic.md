# Phase 6 — Company Targeting: Anthropic

## Learning Objectives

- Understand what Anthropic works on and cares about
- Know the key research areas relevant to ML infra roles
- Build a reading list tailored to Anthropic interviews
- Understand how your Triton/kernel skills apply at Anthropic

---

## What Anthropic Does

- Builds frontier AI models (Claude) with a focus on AI safety
- Key research pillars: Constitutional AI, RLHF, interpretability, scaling
- Infrastructure: training and serving large language models at scale
- The infra team makes training and inference FAST and EFFICIENT

## Why They'd Want Someone with Kernel Skills

- Training Claude requires custom kernels for efficiency
- Serving Claude at scale requires optimized inference (attention, decoding)
- Interpretability research requires custom GPU operations
- Someone who understands GPU performance can optimize the entire stack

---

## Key Areas to Study

### 1. Constitutional AI (CAI)

**Paper:** [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)

**What it is:** Training AI to be helpful and harmless using AI-generated feedback instead of only human feedback.

**Why it matters for infra:** CAI training pipelines are complex. They involve multiple model generations, filtering, and retraining. Each step is a GPU workload that needs to be fast.

**Key concept:** The AI critiques and revises its own outputs based on a set of principles (the "constitution"). This means the training pipeline has stages where the model generates text, evaluates it, and regenerates -- all of which hit the GPU.

### 2. RLHF (Reinforcement Learning from Human Feedback)

**Paper:** [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

**What it is:** Fine-tuning models using human preference data.

**The training pipeline:**

```
SFT (Supervised Fine-Tuning)
    |
    v
Reward Model Training (learns human preferences)
    |
    v
PPO / DPO (optimizes the model against the reward model)
```

**Why it matters for infra:**

- Reward model inference runs during training -- it must be fast
- PPO requires multiple forward passes per training step (policy model + reward model + value model)
- Memory management is critical: you're holding multiple models in GPU memory simultaneously
- DPO simplifies the pipeline (no separate reward model) but still requires paired preference data

**Be able to explain the full RLHF pipeline at a systems level.** Know which parts are compute-bound vs. memory-bound, and where custom kernels help.

### 3. Interpretability / Mechanistic Interpretability

**Papers:**
- [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
- [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)

**What it is:** Understanding what's happening inside neural networks at the level of individual features.

**Key concepts:**

- **Sparse autoencoders:** Trained on model activations to decompose them into interpretable features
- **Feature visualization:** Understanding what activates specific neurons or features
- **Superposition:** Models represent more features than they have dimensions, compressing information

**Why it matters for infra:** Interpretability research requires running many experiments, extracting activations from intermediate layers, and training auxiliary models (like sparse autoencoders). All of this is GPU-intensive. Efficient activation extraction and SAE training are real kernel optimization targets.

The Anthropic interpretability team is one of the most respected in the field. Knowing their work shows you've done your homework.

### 4. Scaling Laws

**Paper:** [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)

**What it is:** Predicting model performance based on compute, data, and parameter count.

**Why it matters:** Anthropic makes compute allocation decisions based on scaling laws. If you can train 10% faster with a better kernel, that's equivalent to 10% more compute -- which translates directly to better models according to scaling laws.

Being able to discuss scaling laws shows you think about efficiency at a macro level, not just micro-optimization.

### 5. Inference Optimization

Topics you should be fluent in:

- **KV Cache management:** You studied this in Phase 3. Know how memory grows with sequence length and batch size.
- **PagedAttention:** From your vLLM study. Virtual memory for KV caches.
- **Speculative decoding:** Use a small model to draft tokens, verify with the large model.
- **Quantization:** INT8, INT4, FP8. Trade precision for speed and memory.
- **Batched inference:** Continuous batching for serving many users.

---

## Interview Prep -- What to Expect

- **Systems design:** "Design a training pipeline for RLHF"
- **Coding:** Python, possibly implementing a simple kernel or distributed training component
- **ML fundamentals:** Transformer architecture, attention mechanism, training dynamics
- **Performance analysis:** "This kernel is running at X GB/s on an A100. Is that good? How would you improve it?"
- **Research discussion:** Be ready to discuss papers from their research page

For the performance analysis question: know the theoretical bandwidth of an A100 (2 TB/s for HBM, ~19.5 TFLOPS for FP32, ~312 TFLOPS for FP16 Tensor Cores). If a kernel runs at 1.5 TB/s on a memory-bound workload, that's 75% of peak -- good but improvable. Be able to do this math on the spot.

---

## Anthropic's Culture

- **Safety-focused** -- they genuinely care about building AI responsibly. This is not a talking point; it's the reason the company exists.
- **Research-oriented** -- even infra roles involve understanding the research. You won't just be optimizing kernels in isolation; you'll need to understand what the researchers need and why.
- **Collaborative** -- small teams, high trust.
- **Show that you care about AI safety, not just performance optimization.** If you're asked "why Anthropic?", having a thoughtful answer about safety matters more than saying "you have the best GPUs."

---

## Reading List (Prioritized)

### Must-Read

1. [Anthropic's research page](https://www.anthropic.com/research)
2. [Constitutional AI paper](https://arxiv.org/abs/2212.08073)
3. [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) -- Anthropic's interpretability breakthrough
4. Flash Attention paper -- you already know this from Phase 3

### Should-Read

5. "Challenges in Deploying Machine Learning" -- practical systems challenges
6. [RLHF paper (InstructGPT)](https://arxiv.org/abs/2203.02155)
7. [DPO paper: Direct Preference Optimization](https://arxiv.org/abs/2305.18290)

### Nice-to-Have

8. Anthropic's blog posts on Claude's development
9. [Chris Olah's blog](https://colah.github.io/) -- Anthropic cofounder, interpretability pioneer
10. [Transformer Circuits Thread](https://transformer-circuits.pub/)

---

## Exercises

1. Read Constitutional AI and write a 1-paragraph summary of how it works. Focus on the training pipeline, not the philosophy.
2. Draw the RLHF training pipeline as a systems diagram. Label each stage with: what model(s) are active, whether it's compute-bound or memory-bound, and where custom kernels would help.
3. Look at Anthropic's current job openings. List the skills they mention that you already have from Phases 1-5, and the skills you still need to build.
4. Practice answering: "Walk me through how you'd optimize inference for a 70B parameter model." Cover: quantization, KV cache management, batching strategy, and hardware utilization.

---

## Milestone

You can discuss Anthropic's research at a systems level and explain how your kernel/infra skills would contribute to their mission. You can articulate why safety matters to you and how efficient infrastructure enables safety research.
