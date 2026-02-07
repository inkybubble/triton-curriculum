# Contributing to Open Source ML Infrastructure Projects

## Learning Objectives

- Find approachable first issues in ML infrastructure projects
- Navigate large, unfamiliar codebases like vLLM, torchtune, and Triton
- Make your first meaningful pull request and get it merged
- Build lasting relationships with project maintainers

---

## Why Contribute to Open Source?

Open source contributions are the strongest signal you can send to ML infrastructure hiring managers, short of actual job experience. Here is why:

- **It is verifiable.** Unlike resume bullet points, your PR history is public. A hiring manager can click your GitHub profile, read your code, see how you respond to review feedback, and evaluate your communication skills — all before an interview.
- **You work alongside your future colleagues.** The maintainers of vLLM, torchtune, and Triton work at the exact companies you are targeting. A merged PR means someone at that company has already reviewed and approved your work.
- **It mirrors the actual job.** Reading unfamiliar code, understanding design decisions, writing tests, responding to code review — this is what ML infrastructure engineers do every day. Open source lets you practice in public.
- **It compounds.** Your first PR is the hardest. After that, you know the codebase, the maintainers know you, and the next contribution is easier. Consistent contributors often get invited to join as maintainers.

If you have completed Phases 1 through 4 of this curriculum, you have the technical skills to contribute. The main barrier is process, not knowledge. This guide will walk you through that process.

---

## Target Repositories

These are the repositories where your contributions will have the most career impact. They are ranked by approachability, not importance — start where you are most likely to succeed.

### 1. torchtune (Most Approachable)

| | |
|---|---|
| **GitHub** | https://github.com/pytorch/torchtune |
| **Maintained by** | PyTorch team (Meta) |
| **Language** | Python (PyTorch) |
| **Why contribute here** | Well-organized codebase, clear issue labels, active and welcoming maintainers, thorough contributing guide |

**Good first contributions:**
- Adding new fine-tuning recipes or model configurations
- Improving documentation and docstrings
- Expanding test coverage for existing functionality
- Adding support for new model architectures (when a new model is released)
- Performance profiling and optimization of existing recipes

**What to expect:** torchtune has a clear review process and maintainers typically respond within a few days. The codebase is pure Python and follows standard PyTorch patterns, so it will feel familiar.

### 2. vLLM (Moderate Difficulty)

| | |
|---|---|
| **GitHub** | https://github.com/vllm-project/vllm |
| **Maintained by** | Originally UC Berkeley, now an independent company |
| **Language** | Python + CUDA + Triton |
| **Why contribute here** | The most widely deployed LLM serving framework. Kernel contributions here are directly visible to hiring managers. |

**Good first contributions:**
- Bug fixes (often well-scoped and clearly described)
- Documentation improvements
- Adding support for new model architectures
- Improving error messages and logging

**More advanced contributions (after your first PR):**
- Optimizing existing Triton kernels (benchmark before and after)
- Adding new fused kernels for common operations
- Performance profiling and bottleneck analysis

**What to expect:** vLLM moves fast and has many contributors. Issues can get claimed quickly. The review process can take longer due to volume. Be patient, be responsive, and keep your PRs small.

### 3. Triton (Advanced)

| | |
|---|---|
| **GitHub** | https://github.com/triton-lang/triton |
| **Maintained by** | OpenAI |
| **Language** | Python + C++ + MLIR |
| **Why contribute here** | Contributing to the Triton compiler itself is impressive and rare. It signals deep understanding of the stack. |

**Good first contributions:**
- Adding or improving tutorials (your blog post from section 01, polished!)
- Documentation fixes and improvements
- Test cases, especially for edge cases and corner cases
- Bug fixes in the Python frontend layer

**More advanced contributions (requires MLIR knowledge):**
- Compiler optimization passes
- Backend code generation improvements
- New language features

**What to expect:** The Triton codebase is complex and the bar for compiler changes is high. Start with Python-level contributions (tutorials, tests, documentation) and work your way deeper as you learn the codebase.

### 4. Unsloth (Community-Driven)

| | |
|---|---|
| **GitHub** | https://github.com/unslothai/unsloth |
| **Maintained by** | Community / Unsloth AI |
| **Language** | Python + Triton |
| **Why contribute here** | Focused specifically on fine-tuning optimization with heavy Triton usage. Your Phase 3 kernel skills are directly applicable. |

**Good first contributions:**
- Adding support for newly released models
- Kernel optimizations with benchmarks
- Documentation and usage examples
- Bug reports and fixes

### 5. Other Notable Repositories

- **SGLang** (https://github.com/sgl-project/sglang) — LLM serving, similar to vLLM, growing community
- **llm.c** (https://github.com/karpathy/llm.c) — Andrej Karpathy's C/CUDA LLM training, good for learning
- **ThunderKittens** (https://github.com/HazyResearch/ThunderKittens) — GPU kernel DSL from Hazy Research, research-oriented

---

## Finding Your First Issue

This is the step where most people get stuck. Here is a concrete, repeatable process.

### Step 1: Browse Issue Labels

Every well-maintained project uses labels to categorize issues. Look for:

| Label | What It Means |
|-------|--------------|
| `good first issue` | Explicitly marked as beginner-friendly. Start here. |
| `help wanted` | Maintainers want outside contributors to work on this. |
| `documentation` | Low barrier to entry, high value to the project. |
| `bug` | Often well-defined scope — reproduce, find cause, fix. |
| `enhancement` | Feature requests. Some are small, some are large. Read carefully. |

### Step 2: Search Efficiently

Use GitHub's search or the `gh` CLI to find issues:

```bash
# Find good first issues in vLLM
gh issue list --repo vllm-project/vllm --label "good first issue" --state open

# Find help wanted issues in torchtune
gh issue list --repo pytorch/torchtune --label "help wanted" --state open

# Find documentation issues in Triton
gh issue list --repo triton-lang/triton --label "documentation" --state open
```

You can also browse directly:
- https://github.com/vllm-project/vllm/labels/good%20first%20issue
- https://github.com/pytorch/torchtune/labels/good%20first%20issue
- https://github.com/triton-lang/triton/labels/good%20first%20issue

### Step 3: Evaluate the Issue

Before committing to an issue, ask yourself:

1. **Is someone already working on it?** Read the comments. If someone posted "I will work on this" two weeks ago and there is no follow-up, it might still be available — but ask first.
2. **Do I understand the problem?** If you cannot explain the issue in your own words, keep reading or ask a clarifying question.
3. **Can I reproduce it?** For bug reports, try to reproduce the bug locally before claiming the issue.
4. **Is the scope clear?** Vague issues like "improve performance" are risky for a first contribution. Look for issues with clear acceptance criteria.

### Step 4: Claim the Issue (Before Writing Code)

This is important. Do not silently start working on something. Comment on the issue first:

```
Hi, I'd like to work on this issue. Here's my understanding of the problem:

[1-2 sentence summary of what's wrong and why]

My planned approach:
- [Step 1]
- [Step 2]
- [Step 3]

Does this approach sound reasonable? Happy to adjust based on feedback.
```

This accomplishes three things:
1. It prevents duplicate work (no one else will claim the issue)
2. It gets early feedback on your approach (before you spend hours coding)
3. It introduces you to the maintainers

Wait for a response before writing code. Most maintainers respond within a few days.

---

## Making Your First Pull Request

### Setting Up Your Development Environment

```bash
# 1. Fork and clone the repository
gh repo fork vllm-project/vllm --clone
cd vllm

# 2. Read the contributing guide — this is not optional
# Every major project has one and reviewers expect you to follow it
cat CONTRIBUTING.md

# 3. Set up the development environment
# (follow the project's specific instructions — they vary)
pip install -e ".[dev]"  # common pattern

# 4. Make sure existing tests pass BEFORE you change anything
pytest tests/ -x --timeout=60  # run tests, stop on first failure

# 5. Create a feature branch
git checkout -b fix/issue-1234-brief-description
```

### Writing Your Changes

**Follow existing patterns.** This is the most important rule for a first contribution. Do not introduce new patterns, new libraries, or new code organization. Match the style of the surrounding code exactly.

```python
# If the project uses this pattern:
def process_tokens(tokens: List[int], *, max_length: int = 512) -> Tensor:
    ...

# Then write your code the same way:
def process_embeddings(embeddings: List[Tensor], *, max_length: int = 512) -> Tensor:
    ...

# Do NOT introduce a different style, even if you prefer it:
def processEmbeddings(embeddings, max_length=512):  # Wrong — different naming convention
    ...
```

**Run the project's linters and formatters:**

```bash
# Common tools you will encounter
ruff check .          # Python linting (increasingly common)
ruff format .         # Python formatting
black .               # Python formatting (older projects)
isort .               # Import sorting
mypy .                # Type checking
```

**Write tests for your changes:**

```python
# If you fixed a bug, add a test that would have caught it
def test_softmax_handles_empty_input():
    """Regression test for issue #1234: softmax crashed on empty tensor."""
    x = torch.empty(0, 10, device='cuda')
    result = fused_softmax(x)
    assert result.shape == (0, 10)
```

### Submitting the PR

```bash
# 1. Run tests one more time
pytest tests/relevant_test_file.py -v

# 2. Commit with a clear message
git add -A
git commit -m "Fix: handle empty tensor input in fused softmax (#1234)

The fused_softmax kernel crashed when given a tensor with a zero-sized
dimension. This adds an early return for empty inputs and a regression test.

Fixes #1234"

# 3. Push to your fork
git push -u origin fix/issue-1234-brief-description

# 4. Create the pull request
gh pr create \
  --title "Fix: handle empty tensor input in fused softmax (#1234)" \
  --body "$(cat <<'EOF'
## Summary

Fixes #1234. The `fused_softmax` kernel crashed when given a tensor with a
zero-sized dimension because the grid calculation divided by zero.

## Changes

- Added an early return in `fused_softmax` for empty input tensors
- Added a regression test in `tests/test_softmax.py`

## Testing

- Ran the full softmax test suite: `pytest tests/test_softmax.py -v` (all passing)
- Manually verified the fix with the reproduction script from the issue
EOF
)"
```

### PR Description Template

A good PR description answers three questions: **what** changed, **why** it changed, and **how** you tested it.

```markdown
## Summary
[1-2 sentences: what does this PR do and why?]

Fixes #[issue number]

## Changes
- [Bullet point for each meaningful change]
- [Be specific: "Added X" not "Made some changes"]

## Testing
- [How did you test this?]
- [What commands did you run?]
- [Any manual testing steps?]

## Benchmarks (if applicable)
| Metric | Before | After |
|--------|--------|-------|
| ...    | ...    | ...   |
```

---

## Navigating the Code Review Process

Your PR will almost certainly receive review feedback. This is normal and expected — even experienced maintainers get review comments on their PRs.

### Responding to Reviews

**Do:**
- Respond to every comment, even if just to say "Done" after making the change
- Ask clarifying questions if you do not understand a suggestion
- Push fixes as new commits (do not force-push during review — it makes it hard for reviewers to see what changed)
- Thank reviewers for their time

**Do not:**
- Argue with maintainers about code style — their project, their rules
- Get discouraged by a long list of comments — it means the reviewer is engaged
- Disappear for weeks without responding — if you need time, say so
- Take review feedback personally — it is about the code, not about you

### Common Review Feedback (and How to Handle It)

| Feedback | What It Means | What to Do |
|----------|--------------|------------|
| "Can you add a test for this?" | Your change needs test coverage | Write a test that specifically covers the changed behavior |
| "Nit: [minor style issue]" | Non-blocking style suggestion | Fix it — shows you care about details |
| "What about the case where X?" | Reviewer found an edge case | Either handle it or explain why it is out of scope |
| "Could you benchmark this?" | Reviewer wants performance data | Run benchmarks and include results |
| "Let's discuss the approach" | Reviewer has concerns about the design | Engage in the discussion before making more changes |

---

## Contribution Ideas That Showcase Kernel Skills

Once you have your first PR merged, here are ideas for contributions that directly demonstrate the skills you built in Phases 1-4.

### For vLLM

- **Benchmark an existing kernel** at different input sizes and hardware configurations. Document the results in an issue or discussion post. This is valuable even if you do not change any code.
- **Profile a specific operation** using `torch.profiler` or `nsys` and identify bottlenecks. File an issue with your findings.
- **Optimize a Triton kernel** by improving tiling, memory access patterns, or reducing register pressure. Include before/after benchmarks.
- **Add Triton implementations** for operations currently using PyTorch fallbacks.

### For torchtune

- **Add a custom Triton kernel** for a training operation that is currently handled by a sequence of PyTorch ops.
- **Profile an existing recipe** and identify the top time-consuming operations.
- **Benchmark quantized vs full-precision training** for a specific model configuration.

### For Triton

- **Turn your blog post into an official tutorial.** This is a natural progression — you already have the content.
- **Add test cases for edge cases** in existing operations (empty inputs, single-element inputs, very large inputs).
- **Document an underdocumented feature** by reading the source code and writing a clear explanation.

---

## Building Relationships With Maintainers

Open source is a community, not just a code repository. Building relationships is how one-off contributions become career opportunities.

### Be a Consistent Presence

Do not make one PR and disappear. Sustainable engagement looks like:

- **Aim for one contribution per month** — it does not have to be code. Issue triage, code review, and documentation all count.
- **Review other people's PRs.** You do not need to be a maintainer to leave helpful comments. Reading PRs teaches you the codebase and shows the maintainers you are engaged.
- **Participate in issue discussions.** If you have insight on a bug report or feature request, share it. Even "I can reproduce this on X hardware with Y configuration" is useful.

### Join Community Channels

Most major projects have additional communication channels beyond GitHub:

- **PyTorch:** PyTorch Discord, PyTorch Dev Discuss forums
- **vLLM:** vLLM Discord, community meetings
- **Triton:** Triton Slack (check the repo for an invite link)

These are where design decisions happen, where new contributors get help, and where maintainers get to know you as a person, not just a GitHub handle.

### The Long Game

The goal is not just to get a PR merged. It is to build a relationship where maintainers recognize your name, trust your judgment, and think of you when a role opens up on their team.

This happens gradually:
1. First PR merged — they know your name
2. Third PR merged — they trust your code quality
3. You review other PRs thoughtfully — they trust your judgment
4. You help other newcomers — they see you as a community member
5. A role opens up — they reach out to you directly

This is not hypothetical. This is how a significant number of ML infrastructure engineers get hired.

---

## Suggested Timeline

This timeline assumes you are dedicating a few hours per week alongside other activities.

| Week | Goal |
|------|------|
| **Week 1** | Pick a repository. Read the contributing guide. Clone the repo and get the development environment and test suite running locally. Read through 5 recently merged PRs to understand the project's conventions. |
| **Week 2** | Find 3 open issues you could tackle. Evaluate each one. Comment on the most promising issue expressing interest and outlining your approach. |
| **Week 3** | Implement your fix or feature. Write tests. Run linters. Submit your PR. |
| **Week 4** | Respond to review feedback. Iterate on your PR until it is approved and merged. Start looking for your next issue. |
| **Ongoing** | Maintain a cadence of roughly one contribution per month. Diversify: code, reviews, documentation, issue triage. |

---

## Exercises

### Exercise 1: Set Up Your Environment

Fork one of the target repositories and get the development environment running locally. Verify that the existing test suite passes on your machine.

```bash
# Example with torchtune (adjust for your chosen repo)
gh repo fork pytorch/torchtune --clone
cd torchtune
pip install -e ".[dev]"
pytest tests/ -x --timeout=120
```

Document any setup issues you encounter — that itself could become a documentation PR.

### Exercise 2: Study Recent PRs

Read through 5 recently merged PRs in your chosen repository. For each one, note:

- How is the PR title formatted?
- What does the description include?
- How detailed are the tests?
- How long did the review process take?
- What kind of feedback did the reviewer give?

This gives you a template for your own PR.

### Exercise 3: Find Candidate Issues

Find 3 open issues you could potentially tackle. For each one, write:

- A one-sentence summary of the problem
- Your planned approach (2-3 bullet points)
- An estimate of how long it would take
- Any questions you would need to ask before starting

### Exercise 4: Make Contact

Comment on one issue expressing interest, using the template from the "Claim the Issue" section above. This is the step that matters most — everything before this is preparation, and everything after flows from it.

---

## Milestone

You have completed this section when:

- [x] You have at least one merged PR in a recognizable ML infrastructure project (vLLM, torchtune, Triton, Unsloth, or similar)
- [x] Or: you have a PR under active review with positive engagement from maintainers
- [x] You have read and commented on at least one other contributor's PR
- [x] You have a working development environment for at least one target repository

Combined with your blog post from section 01, you now have two powerful pieces of public evidence that you can do ML infrastructure work: original technical writing and production-quality code contributions. These are exactly what hiring managers look for when evaluating candidates who do not yet have years of industry experience.
