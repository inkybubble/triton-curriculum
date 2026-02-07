# Phase 4, Lesson 2: Fully Sharded Data Parallel (FSDP)

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain the memory limitations of DDP and why we need something better
- Describe the three ZeRO optimization stages and what each one shards
- Use PyTorch FSDP to train models too large for DDP
- Choose between DDP and FSDP for a given model size and GPU count

---

## Prerequisites

You should have completed Lesson 1 (DDP from Scratch) and understand:
- How data parallelism works (full model on every GPU, split the batch)
- What AllReduce does (every GPU ends up with the averaged gradients)
- `rank`, `world_size`, `DistributedSampler`, `torchrun`

---

## The Problem with DDP

In Lesson 1, we saw that DDP puts a **full copy** of the model on every GPU. Let's work through the memory math to see why this becomes a problem.

### Memory breakdown for a single GPU in DDP

Consider training a 7B parameter model in mixed precision (fp16 forward/backward, fp32 optimizer):

| Component | Formula | Size |
|-----------|---------|------|
| **Parameters** (fp16) | 7B * 2 bytes | 14 GB |
| **Gradients** (fp16) | 7B * 2 bytes | 14 GB |
| **Optimizer state: momentum** (fp32) | 7B * 4 bytes | 28 GB |
| **Optimizer state: variance** (fp32) | 7B * 4 bytes | 28 GB |
| **Master weights** (fp32 copy for optimizer) | 7B * 4 bytes | 28 GB |
| **Total (excluding activations)** | | **112 GB** |

An A100 has 80 GB. The model alone needs 112 GB per GPU -- it does not fit, and we have not even counted activations yet.

Let's also look at smaller models:

| Model | Params | DDP memory per GPU | A100 (80GB) |
|-------|--------|-------------------|-------------|
| 1B | 1B | ~16 GB | Fits easily |
| 3B | 3B | ~48 GB | Fits, tight |
| 7B | 7B | ~112 GB | Does NOT fit |
| 13B | 13B | ~208 GB | Does NOT fit |
| 70B | 70B | ~1120 GB | Does NOT fit |

DDP works well for models up to about 3-4B parameters. Beyond that, you need a smarter approach.

### The redundancy problem

Look at what DDP stores across 4 GPUs:

```
DDP with 4 GPUs (7B model):

GPU 0:  [All 7B params] [All 7B grads] [All optimizer states]  = 112 GB
GPU 1:  [All 7B params] [All 7B grads] [All optimizer states]  = 112 GB
GPU 2:  [All 7B params] [All 7B grads] [All optimizer states]  = 112 GB
GPU 3:  [All 7B params] [All 7B grads] [All optimizer states]  = 112 GB

Total cluster memory used: 448 GB
But unique information:     112 GB  (everything is duplicated!)
```

Every GPU has an **identical** copy of everything. This is massively redundant. What if, instead of every GPU storing everything, we split (sharded) the storage across GPUs?

---

## ZeRO: Zero Redundancy Optimizer

ZeRO is a set of memory optimization techniques developed by Microsoft Research (published in 2019). The core insight is simple:

> In DDP, every GPU stores the same parameters, gradients, and optimizer states. Instead of replicating everything, **shard** these across GPUs so each GPU only stores a portion.

ZeRO has three stages, each more aggressive than the last.

### ZeRO Stage 1: Shard Optimizer States

The optimizer states (Adam's momentum and variance, plus the fp32 master copy of weights) are the biggest memory consumers -- about 12 bytes per parameter for Adam with mixed precision.

In Stage 1, instead of every GPU storing all optimizer states, we split them:

```
DDP (no ZeRO) -- 4 GPUs, 7B params:

GPU 0: params(14GB) + grads(14GB) + optim_states(84GB) = 112 GB
GPU 1: params(14GB) + grads(14GB) + optim_states(84GB) = 112 GB
GPU 2: params(14GB) + grads(14GB) + optim_states(84GB) = 112 GB
GPU 3: params(14GB) + grads(14GB) + optim_states(84GB) = 112 GB


ZeRO Stage 1 -- 4 GPUs, 7B params:

GPU 0: params(14GB) + grads(14GB) + optim_states_shard(21GB) = 49 GB
GPU 1: params(14GB) + grads(14GB) + optim_states_shard(21GB) = 49 GB
GPU 2: params(14GB) + grads(14GB) + optim_states_shard(21GB) = 49 GB
GPU 3: params(14GB) + grads(14GB) + optim_states_shard(21GB) = 49 GB

Memory saved per GPU: 63 GB  (from 112 GB to 49 GB)
```

**How it works:**

1. Forward and backward proceed normally -- each GPU has all parameters and all gradients.
2. After backward, instead of AllReduce on gradients, we use **ReduceScatter**: each GPU ends up with only the gradients it needs for its shard of the optimizer.
3. Each GPU updates only its shard of the optimizer states and parameters.
4. Before the next forward pass, we use **AllGather** to reconstruct the full parameters on all GPUs.

### ZeRO Stage 2: Shard Optimizer States + Gradients

Stage 2 extends Stage 1 by also sharding the gradients:

```
ZeRO Stage 2 -- 4 GPUs, 7B params:

GPU 0: params(14GB) + grads_shard(3.5GB) + optim_states_shard(21GB) = 38.5 GB
GPU 1: params(14GB) + grads_shard(3.5GB) + optim_states_shard(21GB) = 38.5 GB
GPU 2: params(14GB) + grads_shard(3.5GB) + optim_states_shard(21GB) = 38.5 GB
GPU 3: params(14GB) + grads_shard(3.5GB) + optim_states_shard(21GB) = 38.5 GB

Memory saved per GPU: 73.5 GB  (from 112 GB to 38.5 GB)
```

**How it works:**

1. Forward proceeds normally (all GPUs have all parameters).
2. During backward, gradients are computed and immediately **ReduceScattered** -- each GPU accumulates and keeps only the gradient shard it owns.
3. Gradients for parameters this GPU does not own are discarded after ReduceScatter.
4. Each GPU updates its shard of the optimizer.
5. AllGather to reconstruct full parameters for the next forward pass.

### ZeRO Stage 3: Shard Everything

Stage 3 is the most aggressive -- it shards parameters, gradients, AND optimizer states:

```
ZeRO Stage 3 -- 4 GPUs, 7B params:

GPU 0: params_shard(3.5GB) + grads_shard(3.5GB) + optim_states_shard(21GB) = 28 GB
GPU 1: params_shard(3.5GB) + grads_shard(3.5GB) + optim_states_shard(21GB) = 28 GB
GPU 2: params_shard(3.5GB) + grads_shard(3.5GB) + optim_states_shard(21GB) = 28 GB
GPU 3: params_shard(3.5GB) + grads_shard(3.5GB) + optim_states_shard(21GB) = 28 GB

Memory saved per GPU: 84 GB  (from 112 GB to 28 GB)
```

A 7B model that needed 112 GB per GPU with DDP now needs only 28 GB per GPU! It easily fits on an A100.

**How it works:**

1. Before each forward layer, **AllGather** the full parameters for that layer from all GPUs. Each GPU only stores 1/N of the parameters at rest.
2. Run the forward pass for that layer.
3. Discard the non-local parameters (the ones gathered from other GPUs).
4. During backward, AllGather the parameters again (they are needed for gradient computation).
5. Compute gradients and ReduceScatter them.
6. Discard non-local parameters again.
7. Each GPU updates its shard of optimizer states and parameters.

### Visual comparison

```
What each GPU stores at rest:

              Parameters     Gradients      Optimizer States
              ──────────     ─────────      ────────────────
DDP:          [████████]     [████████]     [████████████████████]
              full           full           full

ZeRO-1:      [████████]     [████████]     [█████]
              full           full           1/N

ZeRO-2:      [████████]     [██]           [█████]
              full           1/N            1/N

ZeRO-3:      [██]           [██]           [█████]
              1/N            1/N            1/N
```

### The tradeoff: memory vs communication

There is no free lunch. Each ZeRO stage saves memory but adds communication:

| Stage | Memory per GPU | Communication volume | Communication operations |
|-------|---------------|---------------------|------------------------|
| DDP (no ZeRO) | Full | 2x params (AllReduce) | 1 AllReduce/step |
| ZeRO-1 | Optim/N | 2x params | AllReduce + AllGather |
| ZeRO-2 | Optim/N + Grads/N | 2x params | ReduceScatter + AllGather |
| ZeRO-3 | Everything/N | 3x params | 2x AllGather + ReduceScatter |

ZeRO-3 requires ~1.5x the communication of DDP. You are trading **bandwidth** for **memory**. This tradeoff is worth it when the model does not fit in memory with DDP, but if the model does fit, DDP will be faster due to lower communication overhead.

---

## Communication Primitives: ReduceScatter and AllGather

In Lesson 1, you learned about AllReduce. FSDP uses two additional primitives that are worth understanding.

### AllGather

AllGather collects chunks from all GPUs and gives everyone the full result:

```
Before AllGather:
  GPU 0 has: [A]
  GPU 1 has: [B]
  GPU 2 has: [C]
  GPU 3 has: [D]

After AllGather:
  GPU 0 has: [A, B, C, D]
  GPU 1 has: [A, B, C, D]
  GPU 2 has: [A, B, C, D]
  GPU 3 has: [A, B, C, D]
```

FSDP uses AllGather to reconstruct full layer parameters before forward/backward. Each GPU stores 1/N of the parameters and AllGathers the rest.

### ReduceScatter

ReduceScatter is the complement of AllGather. It reduces (sums) data across GPUs and gives each GPU only its shard of the result:

```
Before ReduceScatter (sum):
  GPU 0 has: [a0, a1, a2, a3]
  GPU 1 has: [b0, b1, b2, b3]
  GPU 2 has: [c0, c1, c2, c3]
  GPU 3 has: [d0, d1, d2, d3]

After ReduceScatter:
  GPU 0 has: [a0+b0+c0+d0]          <- sum of all chunk-0s
  GPU 1 has: [a1+b1+c1+d1]          <- sum of all chunk-1s
  GPU 2 has: [a2+b2+c2+d2]          <- sum of all chunk-2s
  GPU 3 has: [a3+b3+c3+d3]          <- sum of all chunk-3s
```

FSDP uses ReduceScatter on gradients: each GPU computes full gradients, then ReduceScatter gives each GPU only the gradient shard it needs for its optimizer shard.

### The relationship

You can think of AllReduce as ReduceScatter followed by AllGather:
```
AllReduce = ReduceScatter + AllGather
```

ZeRO-2 replaces AllReduce with just ReduceScatter (since each GPU only needs its own gradient shard). This does not save bandwidth (ReduceScatter is half of AllReduce in terms of data volume, but you still need AllGather for parameters), but it saves memory by not storing full gradients.

---

## FSDP in PyTorch

PyTorch's `FullyShardedDataParallel` (FSDP) implements ZeRO-style sharding. Let's build up from a minimal example to a full training script.

### Minimal FSDP example

```python
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy


def setup():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank


def main():
    local_rank = setup()

    # Create model
    model = nn.Sequential(
        nn.Linear(1024, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 1024),
    ).to(local_rank)

    # Wrap with FSDP -- this is the key change from DDP
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO Stage 3
        device_id=local_rank,
    )

    # Training proceeds exactly like DDP from here
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    x = torch.randn(16, 1024, device=local_rank)

    output = model(x)
    loss = output.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
```

Launch with:
```bash
torchrun --nproc_per_node=4 train_fsdp.py
```

### Sharding strategies

FSDP supports multiple sharding levels, corresponding to the ZeRO stages:

```python
from torch.distributed.fsdp import ShardingStrategy

# ZeRO Stage 3: shard parameters, gradients, and optimizer states
# Maximum memory savings, highest communication
ShardingStrategy.FULL_SHARD

# ZeRO Stage 2: shard gradients and optimizer states (keep full params)
# Good balance of memory and communication
ShardingStrategy.SHARD_GRAD_OP

# No sharding: essentially the same as DDP
# Use this as a baseline for comparison
ShardingStrategy.NO_SHARD

# Hybrid sharding: full shard within a node, replicate across nodes
# Good for multi-node training (keeps heavy communication on fast NVLink)
ShardingStrategy.HYBRID_SHARD
```

`HYBRID_SHARD` deserves special mention. In multi-node setups, the network between machines is much slower than NVLink within a machine. Hybrid sharding does ZeRO-3 among the GPUs within each machine (fast NVLink communication) and DDP-style replication across machines (less communication over the slow network).

```
Hybrid sharding with 2 nodes, 4 GPUs each:

Node 0 (GPUs 0-3): FSDP (full shard) within this node
Node 1 (GPUs 4-7): FSDP (full shard) within this node

Between nodes: DDP-style gradient AllReduce
```

---

## FSDP Wrapping Policies

For small models, wrapping the entire model with one FSDP call works fine. For large transformer models, you want **finer-grained sharding** -- wrapping individual transformer layers separately.

### Why wrapping matters

FSDP shard boundaries determine the granularity of AllGather and ReduceScatter. When you wrap the whole model as one FSDP unit:

- Before forward: AllGather ALL parameters at once (huge memory spike)
- After forward: discard all non-local parameters

When you wrap each transformer layer individually:

- Before layer 0 forward: AllGather only layer 0's parameters
- After layer 0 forward: discard layer 0's non-local parameters
- Before layer 1 forward: AllGather only layer 1's parameters
- ...

This keeps peak memory much lower because only one layer's full parameters are materialized at a time.

```
Whole-model wrapping (bad for large models):

Memory  ▲
        │   ┌──── ALL parameters gathered ────┐
        │   │                                  │
        │   │                                  │
        │───┘                                  └───
        └──────────────────────────────────────────►  Time
            forward pass

Per-layer wrapping (good for large models):

Memory  ▲
        │   ┌─┐   ┌─┐   ┌─┐   ┌─┐
        │   │ │   │ │   │ │   │ │     Only one layer at a time
        │   │ │   │ │   │ │   │ │
        │───┘ └───┘ └───┘ └───┘ └───
        └──────────────────────────────────────────►  Time
            L0    L1    L2    L3
```

### Automatic wrapping for transformers

PyTorch provides `transformer_auto_wrap_policy` for the common case of wrapping individual transformer blocks:

```python
import functools
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


# Define which module class represents one transformer layer
# This depends on your model -- look at the model's source code
auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={
        TransformerBlock,  # Replace with your model's layer class
    },
)

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    auto_wrap_policy=auto_wrap_policy,
    device_id=local_rank,
)
```

For HuggingFace models, the layer class is usually something like:
- LLaMA: `LlamaDecoderLayer`
- GPT-2: `GPT2Block`
- BERT: `BertLayer`

You can find it by inspecting the model:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
print(model)  # Look for the repeated layer class
```

### Size-based wrapping

An alternative to specifying layer classes is wrapping based on parameter count:

```python
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

auto_wrap_policy = functools.partial(
    size_based_auto_wrap_policy,
    min_num_params=1_000_000,  # Wrap any module with > 1M params
)
```

---

## Full FSDP Training Script

Here is a complete, production-ready FSDP training script:

```python
import os
import functools
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


# ---------- Model ----------
class TransformerBlock(nn.Module):
    """A single transformer block. FSDP will shard at this granularity."""
    def __init__(self, d_model=1024, n_heads=16, d_ff=4096):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Pre-norm transformer block
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed)[0]
        x = x + self.ff(self.norm2(x))
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, n_layers=24, d_model=1024, n_heads=16, vocab_size=32000):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.head(x)
        return logits


# ---------- Setup ----------
def setup():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


# ---------- Mixed Precision Policy ----------
# FSDP has its own mixed precision handling, separate from torch.cuda.amp
mp_policy = MixedPrecision(
    param_dtype=torch.float16,      # Parameters stored in fp16
    reduce_dtype=torch.float16,     # Gradient reduction in fp16
    buffer_dtype=torch.float16,     # Buffers in fp16
)


# ---------- Training ----------
def main():
    local_rank, rank, world_size = setup()

    # 1. Create model
    #    For very large models, you can create on CPU first to save GPU memory
    #    during initialization, or use meta device + materialize
    model = SimpleTransformer(n_layers=24, d_model=1024).to(local_rank)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model has {total_params / 1e6:.1f}M parameters")

    # 2. Define wrapping policy -- wrap each TransformerBlock separately
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

    # 3. Wrap with FSDP
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        auto_wrap_policy=auto_wrap_policy,
        device_id=local_rank,
        # Limit AllGather memory usage
        limit_all_gathers=True,
    )

    # 4. Optimizer -- create AFTER wrapping with FSDP
    #    FSDP reshapes parameters, so the optimizer must see the FSDP-wrapped params
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # 5. Dataset and DataLoader
    #    Using dummy data here -- replace with your real dataset
    dataset = TensorDataset(torch.randint(0, 32000, (10000, 512)))

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,          # Per-GPU batch size (keep small for large models)
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )

    # 6. Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        model.train()

        total_loss = 0.0
        num_batches = 0

        for batch_idx, (input_ids,) in enumerate(dataloader):
            input_ids = input_ids.to(local_rank)

            # Teacher forcing: input is tokens[:-1], target is tokens[1:]
            logits = model(input_ids[:, :-1])
            targets = input_ids[:, 1:]

            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

            loss.backward()

            # Gradient clipping -- works normally with FSDP
            model.clip_grad_norm_(1.0)

            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1

            if rank == 0 and batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        if rank == 0:
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch} complete. Average loss: {avg_loss:.4f}")

        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, rank)

    dist.destroy_process_group()


# ---------- Checkpointing ----------
def save_checkpoint(model, optimizer, epoch, rank):
    """
    Save a checkpoint with FSDP. This requires special handling because
    the model parameters are sharded across GPUs.
    """
    # FULL_STATE_DICT: gather all shards to rank 0 and save a regular state dict
    # This produces a checkpoint compatible with non-FSDP loading
    full_state_dict_config = FullStateDictConfig(
        offload_to_cpu=True,  # Save CPU memory on rank 0
        rank0_only=True,      # Only rank 0 materializes the full state dict
    )

    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, full_state_dict_config
    ):
        state_dict = model.state_dict()

        if rank == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': state_dict,
            }, f'checkpoint_epoch_{epoch}.pt')
            print(f"Checkpoint saved for epoch {epoch}")

    # Ensure all ranks wait for the save to complete
    dist.barrier()


if __name__ == '__main__':
    main()
```

---

## Key Concepts Explained

### Mixed precision with FSDP

FSDP has its own mixed precision system, separate from `torch.cuda.amp`. You define a `MixedPrecision` policy that controls:

```python
from torch.distributed.fsdp import MixedPrecision

mp_policy = MixedPrecision(
    # Parameters are stored and computed in this dtype
    param_dtype=torch.float16,

    # Gradients are reduced (AllReduce/ReduceScatter) in this dtype
    # Using fp16 halves communication bandwidth
    reduce_dtype=torch.float16,

    # Buffers (e.g., BatchNorm running stats) use this dtype
    buffer_dtype=torch.float16,
)
```

Why does FSDP need its own mixed precision? Because parameters are sharded and gathered on-the-fly. FSDP can store the shards in fp32 but cast to fp16 during AllGather, reducing communication volume. This is more efficient than the standard `autocast` approach.

For more stable training with newer GPUs (A100, H100), you can use `bfloat16`:

```python
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)
```

`bfloat16` has the same exponent range as `float32` (so no overflow/underflow issues) but lower precision. It is generally preferred over `float16` when available.

### Checkpointing with FSDP

Saving and loading model checkpoints is more involved with FSDP because the parameters are sharded across GPUs. PyTorch offers three approaches:

**1. Full state dict (simplest, most compatible):**

```python
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

# Gather all shards to rank 0 and save a normal state dict
save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
    state_dict = model.state_dict()  # Full model on rank 0
    if rank == 0:
        torch.save(state_dict, 'checkpoint.pt')
```

Pros: Produces a standard checkpoint that can be loaded without FSDP.
Cons: Requires enough CPU memory on rank 0 to hold the full model.

**2. Sharded state dict (scalable):**

```python
from torch.distributed.fsdp import ShardedStateDictConfig, StateDictType

# Each rank saves its own shard
save_policy = ShardedStateDictConfig(offload_to_cpu=True)

with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, save_policy):
    state_dict = model.state_dict()
    # Each rank saves its own file
    torch.save(state_dict, f'checkpoint_rank_{rank}.pt')
```

Pros: No single machine needs to hold the full model. Scales to any model size.
Cons: Must load with the same number of GPUs (same sharding).

### Activation checkpointing with FSDP

For very large models, you may also need to save memory on activations (the intermediate values stored during the forward pass for use in the backward pass). Activation checkpointing (also called gradient checkpointing) trades compute for memory: instead of storing activations, it recomputes them during backward.

```python
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.checkpoint import checkpoint


class TransformerBlockWithCheckpointing(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.block = TransformerBlock(d_model, n_heads, d_ff)

    def forward(self, x):
        # checkpoint() recomputes the forward during backward
        # instead of storing activations
        return checkpoint(self.block, x, use_reentrant=False)
```

Or apply it after wrapping:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)

# After wrapping with FSDP, apply activation checkpointing
# to each FSDP-wrapped transformer layer
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)

apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=checkpoint_wrapper,
    check_fn=lambda module: isinstance(module, TransformerBlock),
)
```

---

## FSDP Execution Flow: A Detailed Walkthrough

Let's trace through exactly what happens during one forward-backward-update cycle with FSDP (ZeRO Stage 3) on 4 GPUs with a 3-layer model:

```
Initial state (each GPU stores 1/4 of each layer's parameters):

GPU 0: [L0_shard0] [L1_shard0] [L2_shard0]
GPU 1: [L0_shard1] [L1_shard1] [L2_shard1]
GPU 2: [L0_shard2] [L1_shard2] [L2_shard2]
GPU 3: [L0_shard3] [L1_shard3] [L2_shard3]


=== FORWARD PASS ===

Step 1: AllGather L0 parameters
  All GPUs now have full L0: [L0_shard0, L0_shard1, L0_shard2, L0_shard3]

Step 2: Forward through L0
  Each GPU computes: activation_0 = L0(input_chunk)

Step 3: Free non-local L0 parameters
  Each GPU keeps only its L0 shard again

Step 4: AllGather L1 parameters
Step 5: Forward through L1: activation_1 = L1(activation_0)
Step 6: Free non-local L1 parameters

Step 7: AllGather L2 parameters
Step 8: Forward through L2: output = L2(activation_1)
Step 9: Free non-local L2 parameters
Step 10: Compute loss


=== BACKWARD PASS ===

Step 11: AllGather L2 parameters (needed for gradient computation)
Step 12: Backward through L2: compute grad_L2
Step 13: ReduceScatter grad_L2: each GPU gets its shard of grad_L2
Step 14: Free non-local L2 parameters

Step 15: AllGather L1 parameters
Step 16: Backward through L1: compute grad_L1
Step 17: ReduceScatter grad_L1
Step 18: Free non-local L1 parameters

Step 19: AllGather L0 parameters
Step 20: Backward through L0: compute grad_L0
Step 21: ReduceScatter grad_L0
Step 22: Free non-local L0 parameters


=== OPTIMIZER STEP ===

Each GPU updates only its shards:
  GPU 0: update L0_shard0, L1_shard0, L2_shard0 using grad shards
  GPU 1: update L0_shard1, L1_shard1, L2_shard1
  GPU 2: update L0_shard2, L1_shard2, L2_shard2
  GPU 3: update L0_shard3, L1_shard3, L2_shard3
```

Notice the pattern: AllGather before use, free after use. Parameters are only fully materialized when they are needed. This is why per-layer wrapping matters -- it controls the granularity of this gather/free cycle.

---

## When to Use What: Decision Guide

### Quick decision flowchart

```
Does the model fit on one GPU (with optimizer states + activations)?
│
├── YES ──► Use DDP
│           (simplest, fastest, no memory overhead from sharding)
│
└── NO
    │
    ├── Does the model fit if you shard optimizer states?
    │   │
    │   ├── YES ──► Use FSDP with SHARD_GRAD_OP (ZeRO Stage 2)
    │   │           (moderate memory savings, moderate communication)
    │   │
    │   └── NO
    │       │
    │       └── Use FSDP with FULL_SHARD (ZeRO Stage 3)
    │           + activation checkpointing if needed
    │
    └── Does even a single layer not fit on one GPU?
        │
        └── YES ──► You need Tensor Parallelism (Lesson 3)
```

### Detailed comparison

| Aspect | DDP | FSDP (SHARD_GRAD_OP) | FSDP (FULL_SHARD) |
|--------|-----|---------------------|-------------------|
| Memory per GPU | Full model | ~60% of DDP | ~25% of DDP |
| Communication | 1x AllReduce | ~1x (ReduceScatter + AllGather) | ~1.5x (2x AllGather + ReduceScatter) |
| Throughput | Highest | ~5-10% slower than DDP | ~10-20% slower than DDP |
| Max model size (4x A100-80GB) | ~3-4B | ~7-10B | ~20-30B |
| Complexity | Low | Medium | Medium-High |
| Checkpoint saving | Simple | Needs FSDP context | Needs FSDP context |

---

## Connection to Triton

In Phases 1-3, you focused on making individual GPU operations faster through custom Triton kernels. FSDP operates at a higher level -- it manages how data is distributed across GPUs. But they interact:

1. **Your Triton kernels still run on each GPU independently.** FSDP manages parameter sharding and communication; the actual compute (matrix multiplies, attention, etc.) still uses whatever kernels you have registered.

2. **Communication-compute overlap.** FSDP tries to overlap AllGather with computation. If your Triton kernels are faster, there is more time for communication to happen in the background, improving overall throughput.

3. **Memory awareness.** FSDP saves memory on parameters, gradients, and optimizer states. Your Triton kernels can save memory on activations (e.g., fused kernels that avoid materializing intermediate tensors). The two are complementary.

4. **Custom backward passes.** If you wrote custom Triton backward kernels in Phase 3, they will produce gradients that FSDP's ReduceScatter operates on. Your kernels need to produce correct gradients, but they do not need to be aware of FSDP.

---

## Common Gotchas

### 1. Creating optimizer before FSDP wrapping

```python
# WRONG -- optimizer sees pre-FSDP parameters
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
model = FSDP(model)  # Parameters are reshaped!

# RIGHT -- optimizer sees FSDP-wrapped parameters
model = FSDP(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

FSDP reshapes and flattens parameters during wrapping. The optimizer must see the post-FSDP parameters.

### 2. Using model.parameters() count with FSDP

```python
# After FSDP wrapping, model.parameters() shows SHARDED parameters
# The count will be different from the original model
total_params = sum(p.numel() for p in model.parameters())
# This gives the LOCAL parameter count, not the full model count

# To get the full count, sum across all ranks or count before wrapping
```

### 3. Gradient clipping

With DDP, you use `torch.nn.utils.clip_grad_norm_`. With FSDP, use the model's built-in method:

```python
# DDP
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# FSDP -- use the model's method (it handles sharded grads correctly)
model.clip_grad_norm_(1.0)
```

### 4. Not using `limit_all_gathers=True`

Without this, FSDP may prefetch all AllGathers aggressively, temporarily using too much memory:

```python
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    limit_all_gathers=True,  # Prevents memory spikes
)
```

### 5. Moving model to GPU before FSDP wrapping with large models

For models that do not fit on one GPU:

```python
# WRONG for very large models -- model must fit on GPU during init
model = HugeModel().to(device)  # OOM!
model = FSDP(model)

# RIGHT -- initialize on CPU or meta device
model = HugeModel()  # On CPU
model = FSDP(model, device_id=local_rank)  # FSDP handles device placement
```

---

## Exercises

### Exercise 1: Memory calculation

Calculate the per-GPU memory usage for a 7B parameter model (fp16 params, fp32 optimizer) with:
- DDP on 4 GPUs
- FSDP Stage 2 on 4 GPUs
- FSDP Stage 3 on 4 GPUs

Assume Adam optimizer (momentum + variance + master weights = 12 bytes/param for optimizer states).

**Solution sketch:**

| Component | DDP | FSDP Stage 2 | FSDP Stage 3 |
|-----------|-----|-------------|-------------|
| Params (fp16) | 14 GB | 14 GB | 14/4 = 3.5 GB |
| Grads (fp16) | 14 GB | 14/4 = 3.5 GB | 14/4 = 3.5 GB |
| Optim (fp32) | 84 GB | 84/4 = 21 GB | 84/4 = 21 GB |
| **Total** | **112 GB** | **38.5 GB** | **28 GB** |

With FSDP Stage 3, the 7B model comfortably fits on an A100 (80 GB), leaving room for activations.

### Exercise 2: Implement FSDP training

Take the DDP training script from Lesson 1 and convert it to use FSDP with `FULL_SHARD`. What lines need to change?

### Exercise 3: DDP vs FSDP speed comparison

If you have access to multiple GPUs, train a model that fits comfortably with DDP using both DDP and FSDP Stage 3. Measure the training throughput (samples/second). FSDP should be slightly slower due to extra communication. How much slower is it?

### Exercise 4: Sharding strategy selection

For each scenario, which strategy would you recommend?

1. Fine-tuning BERT (110M params) on 4 A100s
2. Pre-training a 7B model on 8 A100-80GBs
3. Pre-training a 70B model on 64 A100s across 8 nodes

---

## Milestone

After completing this lesson, you should be able to:

- Explain why DDP's memory usage is redundant and how ZeRO eliminates that redundancy
- Describe what each ZeRO stage shards and the memory-communication tradeoff
- Write an FSDP training script with proper wrapping policies and checkpointing
- Choose between DDP, FSDP Stage 2, and FSDP Stage 3 based on model size and GPU count

---

## Resources

- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) -- official getting-started guide
- [PyTorch FSDP Advanced Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_adv_tutorial.html) -- wrapping policies, mixed precision, checkpointing
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) -- the original ZeRO paper by Rajbhandari et al. (2019)
- [Microsoft DeepSpeed ZeRO Blog](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/) -- accessible explanation with diagrams
- [Gupta et al., 2020 -- Training GPT-3 Like Models on a Single Machine](https://arxiv.org/abs/2101.06840) -- practical memory analysis
