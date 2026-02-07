# Writing a Technical Blog Post About Your GPU Kernel Work

## Learning Objectives

- Write a technical blog post about a GPU kernel you have implemented
- Structure a post that demonstrates deep understanding to both peers and hiring managers
- Choose a topic that stands out in the ML infrastructure job market

---

## Why Blog?

A well-written kernel walkthrough is worth more than 10 LeetCode problems for ML infrastructure roles. Here is why:

- **It proves depth.** Anyone can list "Triton" on a resume. A blog post that walks through your fused softmax kernel, explains the tiling strategy, and shows benchmarks against PyTorch proves you actually understand the material deeply enough to teach it.
- **Hiring managers read these.** Engineers at Anthropic, OpenAI, NVIDIA, and Meta actively read technical blogs. A post that shows up when someone searches "Triton Flash Attention implementation" is a passive job application that works while you sleep.
- **It becomes a portfolio piece.** You link it on your resume, pin it on your GitHub profile, and reference it in interviews. When an interviewer asks "tell me about a challenging project," you have a detailed, public artifact to point to.
- **Writing clarifies your own thinking.** You will discover gaps in your understanding the moment you try to explain something in writing. That is a feature, not a bug.

---

## Choosing Your Topic

Not all blog post topics are equal. Here is a rough ranking based on how much impact they will have on your career.

### Tier 1 — Highest Impact

These topics signal serious GPU programming skill. They are the posts that get bookmarked and shared.

| Topic | Why It Works |
|-------|-------------|
| "Implementing Flash Attention in Triton from Scratch" | Flash Attention is the single most important kernel innovation in modern LLMs. Implementing it from scratch shows you understand online softmax, tiling, and memory hierarchy. |
| "How I Made My Triton Matmul 80% as Fast as cuBLAS" | Matrix multiplication is the canonical GPU benchmark. Getting close to cuBLAS performance requires mastering shared memory, tiling, and pipeline scheduling. |
| "Fusing LayerNorm + Dropout in a Single Triton Kernel" | Kernel fusion is what companies actually need. This shows you can identify fusion opportunities and execute on them. |
| "Understanding Online Softmax: The Trick Behind Flash Attention" | Deep concept explanations that make hard ideas accessible are rare and valuable. |

### Tier 2 — Good

These are solid topics that demonstrate competence, especially if paired with original benchmarks or insights.

| Topic | Why It Works |
|-------|-------------|
| "A Beginner's Guide to Triton: From Vector Add to Fused Softmax" | Good for establishing yourself as a teacher. Works best if you add insights the official tutorial does not cover. |
| "What I Learned Reading vLLM's PagedAttention Kernel" | Code reading posts are underrated. They show you can navigate production codebases. |
| "Memory-Bound vs Compute-Bound: A Practical Guide with Triton Benchmarks" | Practical performance analysis with real numbers is always useful. |
| "Quantized Matmul in Triton: INT8 and Beyond" | Quantization is increasingly important for inference. |

### What to Avoid

- **Pure tutorial rewrites.** The Triton documentation already has a vector add tutorial. Rewriting it adds nothing. You need to bring your own angle, experiments, or insights.
- **Posts without code or benchmarks.** Theory alone is not convincing. Readers want to see that your kernel actually runs and how it performs.
- **Posts without your own perspective.** "Here is how Flash Attention works" is a summary. "Here is how I implemented Flash Attention, where I got stuck, and what I learned" is a blog post.

---

## Blog Post Structure — The Template

Use this structure as a starting point. Not every post needs every section, but this is a proven format that works.

```
Title Format: [Action Verb] + [Specific Thing] + [Optional: "in Triton" / "from Scratch"]

Examples:
  "Implementing Flash Attention in Triton — A Step-by-Step Guide"
  "Fusing LayerNorm and Dropout into a Single Triton Kernel"
  "Beating PyTorch's Softmax with a Custom Triton Kernel"
```

### Section 1: The Hook (2-3 sentences)

State the problem and why it matters. Be concrete and specific.

**Good:**
> Attention is O(N^2) in memory. For a sequence length of 8192, that is 256MB just for the
> attention matrix — per head, per layer, per batch element. Flash Attention brings this
> down to O(N) by never materializing the full matrix. Here is how to implement it in Triton.

**Bad:**
> In this post, I will discuss GPU programming and how to use Triton to write kernels.
> GPUs are important for machine learning.

The hook should make a reader who works on ML infrastructure think: "I want to read this."

### Section 2: Background (1-2 paragraphs, keep it SHORT)

Assume the reader knows PyTorch but has limited Triton exposure. Do not spend 500 words explaining what a GPU is. Link to resources for deeper background.

```markdown
## Background

If you are not familiar with Triton, it is a Python-like language for writing GPU kernels
developed by OpenAI. The key idea: you write programs that operate on **blocks** of data
rather than individual threads, and the compiler handles the low-level details.

For a primer on Triton basics, see [the official tutorials](https://triton-lang.org/main/getting-started/tutorials/).
This post assumes you can read a simple Triton kernel.
```

### Section 3: The Approach (60% of the post)

This is the core of your post. Walk through the algorithm and implementation step by step.

**Guidelines:**

- **Show code incrementally.** Do not dump 200 lines of kernel code upfront. Build it piece by piece.
- **Explain the "why" before the "what."** Before showing a code block, explain what problem it solves.
- **Use diagrams.** Even simple ASCII diagrams help enormously.

```
Example ASCII diagram for tiling:

    Global Memory (HBM)
    ┌─────────────────────────────┐
    │  Full A matrix (M x K)      │
    │  ┌─────┐                    │
    │  │Block│ ← Load this tile   │
    │  │(BM x│   into SRAM        │
    │  │ BK) │                    │
    │  └─────┘                    │
    └─────────────────────────────┘
           │
           ▼
    SRAM (on-chip, fast)
    ┌─────────┐
    │ Tile of  │ ← Compute on this
    │ A: BM×BK │
    └─────────┘
```

- **Annotate your code.** Comments in the kernel should explain the non-obvious parts.

```python
@triton.jit
def fused_softmax_kernel(
    output_ptr, input_ptr, input_row_stride, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    # Each program instance handles one row of the input
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride

    # Load the entire row into SRAM — this only works if BLOCK_SIZE >= n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))

    # Numerically stable softmax: subtract max before exp
    # Without this, exp() overflows for large values
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # Write back to HBM
    output_row_start_ptr = output_ptr + row_idx * input_row_stride
    tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)
```

### Section 4: Benchmarks (CRITICAL — Do Not Skip This)

Benchmarks are what separate a good blog post from a great one. Without numbers, readers have no reason to believe your kernel is useful.

**What to include:**

- Comparison against at least one baseline (usually PyTorch)
- Performance at multiple input sizes
- A table or chart (or both)
- Honest discussion of where your implementation falls short

```markdown
## Benchmarks

All benchmarks run on an NVIDIA A100-80GB, CUDA 12.1, PyTorch 2.1, Triton 2.1.

| Sequence Length | PyTorch (ms) | My Triton Kernel (ms) | Speedup |
|-----------------|-------------|----------------------|---------|
| 512             | 0.12        | 0.09                 | 1.33x   |
| 1024            | 0.41        | 0.22                 | 1.86x   |
| 2048            | 1.53        | 0.71                 | 2.15x   |
| 4096            | 6.02        | 2.14                 | 2.81x   |
| 8192            | 23.8        | 7.91                 | 3.01x   |

**Note:** My implementation does not yet support causal masking, which adds overhead.
The PyTorch baseline uses `torch.nn.functional.scaled_dot_product_attention` with
the default backend.
```

**Benchmarking code to include:**

```python
import triton

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512, 1024, 2048, 4096, 8192],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'PyTorch'],
        styles=[('blue', '-'), ('red', '-')],
        ylabel='ms',
        plot_name='fused-softmax-performance',
        args={'M': 4096},
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: fused_softmax(x))
    return ms

benchmark.run(show_plots=True, print_data=True)
```

### Section 5: What I Learned (2-3 paragraphs)

This section is what makes your post personal and memorable. Be honest.

```markdown
## What I Learned

The hardest part was not the algorithm — it was understanding **why** certain block sizes
performed dramatically differently. My first implementation used BLOCK_SIZE=128 for everything,
and performance was terrible for small matrices. It took me two days of reading the Triton
compiler source to understand that the compiler generates different PTX instructions depending
on block size, and that alignment matters more than I expected.

I also underestimated how much time I would spend on numerical correctness. My initial
implementation passed basic tests but had subtle precision issues at large sequence lengths.
The fix was switching from a naive two-pass softmax to the online algorithm — which ironically
is the key insight behind Flash Attention itself.

If I were to do this again, I would start by profiling the PyTorch baseline with
`torch.profiler` to understand exactly where time is spent before writing any Triton code.
Understanding the bottleneck first would have saved me from optimizing the wrong thing.
```

### Section 6: Resources and Code

```markdown
## Code and Resources

- **Full code:** [GitHub repo link] or [Google Colab link]
- **Run it yourself:** [![Open In Colab](badge-url)](colab-link)
- **References:**
  - [Flash Attention paper (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)
  - [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/)
  - [CUDA C++ Programming Guide — Memory Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
```

---

## Writing Tips

**Voice and tone:**
- Write like you are explaining to a smart friend over coffee, not writing a textbook
- Use "I" freely — this is your experience and your perspective
- Contractions are fine, jargon is fine (your audience knows what HBM is), but define anything Triton-specific

**Show your mistakes:**
- "My first version was 10x slower because I forgot about memory coalescing" is more interesting and more instructive than a perfect walkthrough
- Readers learn from your debugging process, not just your final code
- Mistakes make you relatable and credible

**Code quality:**
- All code should be copy-pasteable and runnable
- Include import statements
- Specify your environment (GPU, CUDA version, Triton version, PyTorch version)
- Add a "Run this yourself" section with a Colab link if possible

**Length:**
- Aim for 1500-2500 words for a substantial post
- Under 1000 words feels thin; over 3000 words and you should consider splitting into a series

---

## Where to Publish

| Platform | Pros | Cons | Best For |
|----------|------|------|----------|
| **GitHub Pages** (Jekyll/Hugo) | Free, you own it, custom domain possible, no paywall | Setup required, no built-in audience | Long-term portfolio |
| **Medium** | Large audience, good SEO | Paywall frustrates readers, you do not own the platform | Reaching a broad audience |
| **Dev.to** | Developer-focused, no paywall, good community | Smaller ML audience | General developer content |
| **Substack** | Good for building a following, email list | Not code-focused, formatting can be clunky | Building a newsletter |
| **Twitter/X threads** | Maximum visibility, easy to share | Not suitable for long-form, content gets buried | Promotion and summaries |

**Recommended approach:** Publish on GitHub Pages (you own it forever) and cross-post a summary thread on Twitter/X to drive traffic.

### Quick GitHub Pages Setup

```bash
# Create a blog repository
gh repo create my-gpu-blog --public --clone
cd my-gpu-blog

# Option 1: Simple markdown (GitHub renders it automatically)
# Just push .md files and enable GitHub Pages in repo settings

# Option 2: Jekyll (more control over styling)
# Add a _config.yml and GitHub Pages will build it for you
```

```yaml
# _config.yml (minimal Jekyll setup)
title: "Your Name — GPU Kernel Engineering"
description: "Notes on Triton, CUDA, and ML infrastructure"
theme: minima
markdown: kramdown
highlighter: rouge
```

---

## Promoting Your Post

Publishing is only half the job. You need to get it in front of the right people.

1. **Twitter/X thread.** Write a 5-7 tweet thread summarizing the key insights, with a link to the full post at the end. Tag relevant people if you reference their work (Tri Dao for Flash Attention, the Triton team for Triton-related posts). Do not spam, but genuine engagement is welcome.

2. **Reddit.** Post on r/MachineLearning (for research-oriented posts), r/LocalLLaMA (for inference optimization), or r/CUDA (for low-level GPU work). Follow each subreddit's rules about self-promotion.

3. **Discord and Slack communities.** The PyTorch Discord, the MLOps Community Slack, and various GPU programming Discord servers are good places to share.

4. **Hacker News.** If your post is genuinely novel or interesting, submit it to HN. GPU kernel posts do well there.

5. **LinkedIn.** Less technical audience, but recruiters read LinkedIn. A short post linking to your blog can surface you to hiring managers.

---

## Exercises

These exercises are designed to be completed in order. By the end, you will have a published blog post.

### Exercise 1: Write a Draft

Pick one kernel you implemented during Phase 2 or Phase 3. Write a 1000-word draft blog post following the template above. Focus on:
- A clear hook
- Incremental code walkthrough
- At least one benchmark comparison

Do not worry about polish yet. Get the content down.

### Exercise 2: Create a Code Repository

Create a GitHub repository for your kernel code:
- Clean, well-commented kernel implementation
- A benchmarking script that readers can run
- A README that explains what the kernel does and how to run it
- Requirements file or environment specification

```bash
gh repo create triton-flash-attention --public --clone
# Add your code, benchmarks, and README
```

### Exercise 3: Get Feedback

Share your draft with someone and ask these specific questions:
- Can you follow the explanation without prior GPU programming experience?
- Where did you get lost or confused?
- Is there anything you wanted to know that I did not cover?

Incorporate their feedback and publish.

---

## Milestone

You have completed this section when:

- [x] You have a published blog post about a Triton kernel you implemented
- [x] The post includes working code and benchmark results
- [x] The code is in a public GitHub repository
- [x] You have shared the post on at least one platform (Twitter, Reddit, etc.)

This blog post will be one of your strongest assets when applying to ML infrastructure roles. It is concrete, public, and demonstrates exactly the skills these teams need.
