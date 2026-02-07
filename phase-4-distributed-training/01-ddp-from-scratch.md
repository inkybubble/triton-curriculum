# Phase 4, Lesson 1: Distributed Data Parallel (DDP) from Scratch

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain why training on a single GPU is no longer sufficient for modern models
- Describe data parallelism and how it distributes work across GPUs
- Explain the AllReduce communication primitive and the Ring AllReduce algorithm
- Implement a DDP training script using PyTorch's `DistributedDataParallel`
- Identify and avoid common DDP pitfalls

---

## Prerequisites

You should be comfortable writing Python and PyTorch code, and you should have completed Phases 1-3 of this curriculum (writing Triton kernels). No prior knowledge of multi-GPU programming is required. We will build every concept from scratch.

---

## Why Multiple GPUs?

Up to this point, everything you have done has run on a single GPU. You wrote Triton kernels that launch thread blocks on one device, and PyTorch handled all your training on one device. That was fine for small models and datasets, but the world has moved on.

### The model size problem

Models are getting enormous. Here are some real numbers:

| Model | Parameters | Size in fp16 | Size in fp32 |
|-------|-----------|-------------|-------------|
| BERT-base | 110M | ~220 MB | ~440 MB |
| GPT-2 | 1.5B | ~3 GB | ~6 GB |
| LLaMA-7B | 7B | ~14 GB | ~28 GB |
| LLaMA-70B | 70B | ~140 GB | ~280 GB |
| GPT-3 | 175B | ~350 GB | ~700 GB |

An NVIDIA A100 GPU has 80 GB of memory. An H100 also has 80 GB. That means a 7B model in fp16 (14 GB for parameters alone) fits on one GPU, but once you add optimizer states (Adam stores momentum and variance, which are each the same size as the parameters), gradients, and activations, you are looking at 50-70 GB -- barely squeezing onto one GPU with almost no room for batch data.

A 70B model? 140 GB just for the weights. It does not fit on a single GPU at all.

### The training speed problem

Even when a model fits on one GPU, training can be painfully slow. If training a 7B model on a single A100 takes 30 days, that is impractical for iteration. If you could use 8 GPUs and cut that to ~4 days, you have a fundamentally different development workflow.

### Two strategies

There are two broad approaches to using multiple GPUs:

1. **Data parallelism** -- keep the full model on every GPU, but split the training data across GPUs. Each GPU processes a different chunk of the batch. This makes training faster.

2. **Model parallelism** -- split the model itself across GPUs. Each GPU holds only a piece of the model. This lets you train models that do not fit on one GPU.

This lesson covers data parallelism. It is the simplest, most common, and most important multi-GPU strategy. Lessons 2 and 3 cover model parallelism approaches.

---

## Data Parallelism -- The Mental Model

### The exam grading analogy

Imagine you are a professor and you have 1,000 exam papers to grade. You have an answer key. Grading all 1,000 papers yourself would take forever.

Instead, you make 4 copies of the answer key and hand them to 4 teaching assistants. Each TA gets 250 papers. They all grade independently, using the same answer key. When they are done, you collect all the results.

Data parallelism works exactly the same way:

- The **answer key** is the model (its weights)
- The **exam papers** are the training data (the batch)
- The **TAs** are the GPUs
- **"Collecting results"** is averaging the gradients across GPUs

### The concrete steps

Here is what happens in every training step with data parallelism:

1. **Copy**: Every GPU has a full copy of the model (same weights on all GPUs).
2. **Split**: The training batch is divided evenly across GPUs. If you have a batch of 128 samples and 4 GPUs, each GPU gets 32 samples.
3. **Forward**: Each GPU runs forward pass on its 32 samples independently.
4. **Backward**: Each GPU computes gradients on its 32 samples independently.
5. **Communicate**: The gradients from all GPUs are **averaged** together. After this step, every GPU has the same averaged gradients.
6. **Update**: Each GPU applies the averaged gradients to update its model weights. Since all GPUs started with the same weights and apply the same averaged gradients, all models stay in sync.
7. **Repeat** from step 2.

The result: N GPUs process N times as much data per step, so training is approximately N times faster.

### The big picture

```
                        ┌─────────────────────────────────────────────────────┐
    Batch               │                                                     │
   ┌──────────┐         │  GPU 0: model copy  ──  forward(batch_0)            │
   │ batch_0  │────────►│       backward  ──  grads_0  ──┐                    │
   ├──────────┤         │                                 │                    │
   │ batch_1  │────────►│  GPU 1: model copy  ──  forward(batch_1)            │
   ├──────────┤         │       backward  ──  grads_1  ──┤── AllReduce ──►    │
   │ batch_2  │────────►│                                 │   avg_grads        │
   ├──────────┤         │  GPU 2: model copy  ──  forward(batch_2)  ──►       │
   │ batch_3  │────────►│       backward  ──  grads_2  ──┤   update all       │
   └──────────┘         │                                 │   models           │
                        │  GPU 3: model copy  ──  forward(batch_3)            │
                        │       backward  ──  grads_3  ──┘                    │
                        └─────────────────────────────────────────────────────┘
```

The only step where GPUs need to talk to each other is step 5 -- the gradient averaging. Everything else is independent. This communication step is called **AllReduce**, and it is the single most important concept in data parallelism.

---

## AllReduce -- The Key Communication Primitive

### What AllReduce does

AllReduce is a collective communication operation. "Collective" means all GPUs participate. Here is what it does in plain English:

> Every GPU starts with its own array of numbers (gradients). After AllReduce, every GPU ends up with the **same** array, where each element is the **sum** (or average) of the corresponding elements from all GPUs.

Before AllReduce:
```
GPU 0 has: [1, 2, 3, 4]
GPU 1 has: [5, 6, 7, 8]
GPU 2 has: [2, 3, 4, 5]
GPU 3 has: [4, 1, 2, 3]
```

After AllReduce (sum):
```
GPU 0 has: [12, 12, 16, 20]
GPU 1 has: [12, 12, 16, 20]
GPU 2 has: [12, 12, 16, 20]
GPU 3 has: [12, 12, 16, 20]
```

Every GPU ends up with the same result: the element-wise sum across all GPUs. To get the average, you divide by the number of GPUs (4 in this case).

### The naive approach and why it fails

The simplest way to implement AllReduce is:

1. Every GPU sends its gradients to GPU 0.
2. GPU 0 sums them all up.
3. GPU 0 broadcasts the result back to everyone.

This works, but it has a critical problem: **GPU 0 is a bottleneck.** It has to receive data from every other GPU and send data back to every other GPU. If you have 8 GPUs with large gradient tensors, GPU 0's network link is completely saturated while all other GPUs sit idle waiting.

```
          Naive AllReduce -- GPU 0 is the bottleneck

          GPU 1 ──────►  ┌───────┐  ──────► GPU 1
          GPU 2 ──────►  │ GPU 0 │  ──────► GPU 2
          GPU 3 ──────►  │ (sum) │  ──────► GPU 3
                          └───────┘

          Problem: GPU 0's bandwidth is the limit.
          With N GPUs, GPU 0 must receive N-1 messages
          and send N-1 messages.
```

### Ring AllReduce -- The clever algorithm

Ring AllReduce solves the bottleneck problem by distributing the communication evenly across all GPUs. No single GPU does more work than any other.

**Setup:** Arrange the GPUs in a logical ring. Each GPU only communicates with its two neighbors (the one to its left and the one to its right).

```
        ┌───────┐         ┌───────┐
        │ GPU 0 │────────►│ GPU 1 │
        └───────┘         └───────┘
            ▲                  │
            │                  ▼
        ┌───────┐         ┌───────┐
        │ GPU 3 │◄────────│ GPU 2 │
        └───────┘         └───────┘
```

**The algorithm has two phases:**

#### Phase 1: Scatter-Reduce (N-1 rounds)

Each GPU splits its data into N chunks (one per GPU). Over N-1 rounds, chunks are passed around the ring, and each GPU adds (reduces) what it receives to its own chunk.

Let's trace through this with 4 GPUs, each starting with 4 chunks of data. We label each chunk as `gpu_chunk`. For example, `a0` means GPU 0's chunk 0.

**Initial state** (each GPU has its own 4 chunks):
```
GPU 0: [a0, a1, a2, a3]
GPU 1: [b0, b1, b2, b3]
GPU 2: [c0, c1, c2, c3]
GPU 3: [d0, d1, d2, d3]
```

**Round 1:** Each GPU sends one chunk to the right, receives one from the left, and adds the received chunk to its own.

```
GPU 0 sends chunk 0 (a0) to GPU 1,  receives chunk 3 (d3) from GPU 3
GPU 1 sends chunk 1 (b1) to GPU 2,  receives chunk 0 (a0) from GPU 0
GPU 2 sends chunk 2 (c2) to GPU 3,  receives chunk 1 (b1) from GPU 1
GPU 3 sends chunk 3 (d3) to GPU 0,  receives chunk 2 (c2) from GPU 2

After adding received to local:
GPU 0: [ a0,     a1,     a2,     a3+d3  ]
GPU 1: [ a0+b0,  b1,     b2,     b3     ]
GPU 2: [ c0,     b1+c1,  c2,     c3     ]
GPU 3: [ d0,     d1,     c2+d2,  d3     ]
```

**Round 2:** Each GPU sends the chunk it just updated to the right.

```
After round 2:
GPU 0: [ a0,         a1,         a2+c2+d2,  a3+d3    ]
GPU 1: [ a0+b0,      b1,         b2,        a3+b3+d3 ]
GPU 2: [ c0,         a0+b1+c1,   c2,        c3       ]
GPU 3: [ a0+b0+d0,   d1,         c2+d2,     d3       ]
```

**Round 3 (final scatter-reduce round):** After this round, each GPU has one chunk that contains the sum across ALL GPUs:

```
GPU 0: [ a0,              a1+b1+c1+d1,  a2+c2+d2,        a3+d3           ]
                          ▲ complete!
GPU 1: [ a0+b0,           b1,           a2+b2+c2+d2,     a3+b3+d3       ]
                                         ▲ complete!
GPU 2: [ c0,              a0+b1+c1,     c2,              a3+b3+c3+d3    ]
                                                          ▲ complete!
GPU 3: [ a0+b0+c0+d0,     d1,           c2+d2,           d3             ]
         ▲ complete!
```

Each GPU now "owns" one fully-reduced chunk.

#### Phase 2: AllGather (N-1 rounds)

Now each GPU has one complete chunk. In N-1 more rounds, these complete chunks are passed around the ring (this time without adding -- just replacing) until every GPU has all N complete chunks.

```
After AllGather:
GPU 0: [a0+b0+c0+d0, a1+b1+c1+d1, a2+b2+c2+d2, a3+b3+c3+d3]
GPU 1: [a0+b0+c0+d0, a1+b1+c1+d1, a2+b2+c2+d2, a3+b3+c3+d3]
GPU 2: [a0+b0+c0+d0, a1+b1+c1+d1, a2+b2+c2+d2, a3+b3+c3+d3]
GPU 3: [a0+b0+c0+d0, a1+b1+c1+d1, a2+b2+c2+d2, a3+b3+c3+d3]
```

Every GPU now has the full summed result. Divide by N to get the average.

#### Why Ring AllReduce is efficient

- Each GPU sends and receives exactly the same amount of data.
- No single GPU is a bottleneck.
- The total data transferred per GPU is `2 * (N-1)/N * data_size`, which approaches `2 * data_size` as N grows. This is **bandwidth-optimal** -- you cannot do better.
- In practice, NVIDIA's NCCL library implements highly optimized versions of this (and other algorithms like tree-based AllReduce) that take advantage of NVLink and NVSwitch hardware.

### Connection to your Triton knowledge

In Phases 1-3, you learned how to write kernels that maximize compute throughput on a single GPU -- tiling, memory coalescing, shared memory. AllReduce is the multi-GPU analog: it is about maximizing **communication** throughput between GPUs. Just as your Triton kernels need to be bandwidth-aware for global memory, distributed training needs to be bandwidth-aware for GPU-to-GPU links.

---

## Key Distributed Computing Concepts

Before we write code, let's define the foundational terms. These will appear everywhere in distributed training.

### Process, Rank, and World Size

When you do distributed training, you launch **multiple processes** -- typically one per GPU. Each process is independent: it has its own Python interpreter, its own memory space, and its own GPU.

- **World size**: The total number of processes (GPUs) participating. If you have 4 GPUs, `world_size = 4`.

- **Rank**: A unique integer ID assigned to each process, from 0 to `world_size - 1`. Think of it as a name tag. GPU 0 has rank 0, GPU 1 has rank 1, and so on.

- **Local rank**: When you have multiple machines (nodes), each with multiple GPUs, the local rank is the GPU index within one machine. If you have 2 machines with 4 GPUs each, the ranks are 0-7, but local ranks are 0-3 on each machine.

```
Machine 0:                    Machine 1:
  GPU 0  (rank=0, local=0)     GPU 4  (rank=4, local=0)
  GPU 1  (rank=1, local=1)     GPU 5  (rank=5, local=1)
  GPU 2  (rank=2, local=2)     GPU 6  (rank=6, local=2)
  GPU 3  (rank=3, local=3)     GPU 7  (rank=7, local=3)

  world_size = 8
```

### Backend: NCCL

When GPUs need to communicate (e.g., AllReduce), they use a **communication backend**. For GPU-to-GPU communication, the standard is **NCCL** (NVIDIA Collective Communications Library, pronounced "nickel").

NCCL is optimized for NVIDIA GPUs. It automatically detects the best communication path between GPUs:
- **NVLink**: Direct GPU-to-GPU links within a machine (~600 GB/s on modern hardware). Very fast.
- **PCIe**: Slower fallback within a machine (~32 GB/s).
- **InfiniBand / Ethernet**: For communication between machines.

You do not need to manage any of this. You just specify `backend='nccl'` and NCCL handles the rest.

### Process Group

A process group is a set of processes that can communicate with each other. When you call `dist.init_process_group()`, you create the default group containing all processes. You can also create sub-groups for more advanced patterns (we will not need that in this lesson).

---

## Implementing DDP from Scratch (Conceptual)

Let's first understand what DDP does **under the hood** by building a simplified version, then we will use PyTorch's real implementation.

### Step 1: The training function that runs on each GPU

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


def setup(rank, world_size):
    """
    Initialize the distributed environment.

    This must be called at the start of each process. It sets up the
    communication infrastructure so that GPUs can talk to each other.

    Args:
        rank: This process's unique ID (0, 1, 2, ..., world_size-1)
        world_size: Total number of processes (GPUs)
    """
    # Set which GPU this process should use
    torch.cuda.set_device(rank)

    # Initialize the process group
    dist.init_process_group(
        backend='nccl',        # Use NVIDIA's GPU communication library
        init_method='env://',  # Coordinate using environment variables
                               # (torchrun sets these automatically)
        rank=rank,             # This process's ID
        world_size=world_size  # Total number of processes
    )


def cleanup():
    """Tear down the distributed environment."""
    dist.destroy_process_group()
```

Let's break down `init_process_group`:

- `backend='nccl'`: Use NCCL for GPU communication. Other options include `'gloo'` (CPU-based, slower for GPUs) and `'mpi'` (Message Passing Interface, less common in deep learning).

- `init_method='env://'`: This tells PyTorch how to find the other processes. When using `torchrun` (the standard launcher), it sets environment variables (`MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`) that all processes read to find each other. One process acts as a "rendezvous" point -- like a meeting spot where all processes check in before training begins.

- `rank` and `world_size`: Identify this process and the total count.

### Step 2: Manual gradient synchronization (the DIY approach)

Before using PyTorch's DDP wrapper, let's see what it does manually:

```python
def train_manual_ddp(rank, world_size):
    """
    A simplified DDP training loop, doing gradient sync manually.
    This shows you what DDP does under the hood.
    """
    setup(rank, world_size)

    # Create model -- EVERY GPU gets a full copy
    model = SimpleModel().to(rank)

    # IMPORTANT: All GPUs must start with the same weights!
    # Broadcast GPU 0's weights to all other GPUs.
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    # Create dataset and distributed sampler
    dataset = make_dataset()

    # DistributedSampler ensures each GPU gets DIFFERENT data
    # With 4 GPUs and 1000 samples:
    #   GPU 0 sees samples [0, 4, 8, 12, ...]
    #   GPU 1 sees samples [1, 5, 9, 13, ...]
    #   GPU 2 sees samples [2, 6, 10, 14, ...]
    #   GPU 3 sees samples [3, 7, 11, 15, ...]
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,  # Total number of GPUs
        rank=rank,                # This GPU's ID
        shuffle=True
    )
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        # Tell the sampler which epoch it is so it shuffles differently
        # each epoch (explained below)
        sampler.set_epoch(epoch)

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(rank)
            batch_y = batch_y.to(rank)

            # Forward pass -- each GPU processes its own chunk
            output = model(batch_x)
            loss = torch.nn.functional.cross_entropy(output, batch_y)

            # Backward pass -- each GPU computes its own gradients
            loss.backward()

            # ===== THIS IS THE KEY STEP =====
            # Average gradients across all GPUs
            for param in model.parameters():
                if param.grad is not None:
                    # AllReduce: sum gradients across all GPUs, in-place
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    # Divide by world_size to get the average
                    param.grad.data /= world_size
            # After this loop, all GPUs have identical averaged gradients
            # ================================

            # Update weights -- same gradients + same weights = same update
            optimizer.step()
            optimizer.zero_grad()

    cleanup()
```

This works, but it is inefficient. The gradient synchronization (the AllReduce loop) happens **after** the entire backward pass completes. That means:

1. Run full backward pass.
2. Wait.
3. Synchronize all gradients.
4. Update.

PyTorch's DDP does something smarter: it **overlaps** communication with computation.

### Step 3: How PyTorch DDP optimizes this

PyTorch's `DistributedDataParallel` registers **hooks** on each parameter. When a gradient is computed during `backward()`, the hook fires immediately and starts the AllReduce for that gradient **while the rest of backward is still running**.

```
Timeline WITHOUT overlap (our manual version):

backward:     |====== compute all gradients ======|
allreduce:                                          |====== sync all gradients ======|
update:                                                                                |== step ==|

Timeline WITH overlap (PyTorch DDP):

backward:     |====== compute grads for layers N...1 ======|
allreduce:         |==== sync early grads ====|==== sync more ====|== sync last ==|
update:                                                                            |== step ==|
```

DDP also groups (buckets) small gradient tensors together to reduce the number of AllReduce calls, since each call has some fixed overhead.

---

## Using PyTorch's DistributedDataParallel

Now let's write the real version -- the way you would actually do it:

```python
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


# ---------- Model ----------
class SimpleModel(nn.Module):
    """A simple model for demonstration. Replace with your real model."""
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------- Setup/Cleanup ----------
def setup(rank, world_size):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size,
    )

def cleanup():
    dist.destroy_process_group()


# ---------- Training ----------
def train(rank, world_size):
    setup(rank, world_size)

    # 1. Create model on this GPU
    model = SimpleModel().to(rank)

    # 2. Wrap with DDP
    #    This single line handles:
    #    - Broadcasting weights from rank 0 to all other ranks
    #    - Registering gradient hooks for AllReduce
    #    - Bucketing small gradients together for efficiency
    ddp_model = DDP(model, device_ids=[rank])

    # 3. Create dataset with distributed sampling
    #    In a real scenario, this would be your actual dataset
    dataset = TensorDataset(
        torch.randn(10000, 784),
        torch.randint(0, 10, (10000,))
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=64,        # Per-GPU batch size
        sampler=sampler,
        num_workers=2,
        pin_memory=True,      # Speeds up CPU-to-GPU transfer
    )

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 4. Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        # IMPORTANT: set_epoch ensures different shuffling each epoch
        sampler.set_epoch(epoch)

        ddp_model.train()
        epoch_loss = 0.0

        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(rank)
            target = target.to(rank)

            output = ddp_model(data)
            loss = criterion(output, target)

            loss.backward()  # DDP handles gradient sync automatically here

            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

        # Only print from rank 0 to avoid duplicate output
        if rank == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch}: loss = {avg_loss:.4f}")

    # 5. Save checkpoint -- only from rank 0
    if rank == 0:
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': ddp_model.module.state_dict(),  # .module!
            'optimizer_state_dict': optimizer.state_dict(),
        }, 'checkpoint.pt')

    cleanup()


# ---------- Entry point ----------
if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
```

Let's examine the key parts in detail.

### `DDP(model, device_ids=[rank])`

This is where the magic happens. When you wrap your model with `DDP`:

1. It **broadcasts** the model's state from rank 0 to all other ranks, ensuring all GPUs start with identical weights.
2. It **registers backward hooks** on every parameter. When `loss.backward()` computes a gradient, the hook fires and queues that gradient for AllReduce.
3. It **buckets** parameters together. Instead of doing one AllReduce per parameter (which would have high overhead for small parameters), it groups parameters into buckets (default 25 MB) and does one AllReduce per bucket.

### `DistributedSampler`

Without this, every GPU would iterate over the **same** data in the **same** order. That would be equivalent to training on one GPU -- no speedup at all.

`DistributedSampler` divides the dataset indices across GPUs:

```
Dataset has 1000 samples, 4 GPUs:

Without DistributedSampler (wrong):
  GPU 0: [0, 1, 2, 3, 4, 5, ..., 999]
  GPU 1: [0, 1, 2, 3, 4, 5, ..., 999]  <- same data!
  GPU 2: [0, 1, 2, 3, 4, 5, ..., 999]  <- same data!
  GPU 3: [0, 1, 2, 3, 4, 5, ..., 999]  <- same data!

With DistributedSampler (correct):
  GPU 0: [  0,   4,   8,  12, ...,  996]  <- 250 unique samples
  GPU 1: [  1,   5,   9,  13, ...,  997]  <- 250 different samples
  GPU 2: [  2,   6,  10,  14, ...,  998]  <- 250 different samples
  GPU 3: [  3,   7,  11,  15, ...,  999]  <- 250 different samples
```

### `sampler.set_epoch(epoch)`

This is easy to forget but important. The sampler uses a random number generator to shuffle the data. The seed for this RNG is `seed + epoch`. If you do not call `set_epoch(epoch)`, the seed is the same every epoch, and every GPU gets the same ordering every epoch. Calling `set_epoch` ensures the data is shuffled differently each epoch while still ensuring no GPU overlap.

### `ddp_model.module`

DDP wraps your model inside itself. To access the original model (e.g., for saving state), use `ddp_model.module`. This is a common source of bugs:

```python
# WRONG -- saves the DDP wrapper, not the model
torch.save(ddp_model.state_dict(), 'checkpoint.pt')

# RIGHT -- saves the actual model
torch.save(ddp_model.module.state_dict(), 'checkpoint.pt')
```

---

## Launching Distributed Training

### Option 1: `mp.spawn` (simple, single machine)

```python
if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
```

`mp.spawn` creates `world_size` child processes, each calling `train(rank, world_size)` where `rank` goes from 0 to `world_size - 1`.

### Option 2: `torchrun` (recommended)

`torchrun` is PyTorch's built-in distributed launcher. It is the standard way to launch distributed training:

```bash
# Single machine, 4 GPUs
torchrun --nproc_per_node=4 train_script.py

# Multiple machines (run on each machine)
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=10.0.0.1 \
    --master_port=29500 \
    train_script.py
```

With `torchrun`, you do not need `mp.spawn`. Instead, you read rank and world_size from environment variables:

```python
def main():
    # torchrun sets these automatically
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(local_rank)

    # ... rest of training ...

if __name__ == '__main__':
    main()
```

`torchrun` is preferred because:
- It handles environment variable setup.
- It supports elastic training (adding/removing GPUs during training).
- It handles error propagation and worker restarts.

### Option 3: Running on Modal

[Modal](https://modal.com/) is a cloud platform that makes it easy to run GPU workloads. Here is how to run DDP training on Modal:

```python
import modal

app = modal.App("ddp-training")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision")
)

@app.function(gpu="A10G:4", image=image, timeout=3600)
def train_ddp():
    """Launch DDP training on 4 GPUs using torchrun."""
    import subprocess

    # torchrun handles all the distributed setup
    result = subprocess.run(
        [
            "torchrun",
            "--nproc_per_node=4",        # Use all 4 GPUs
            "--master_addr=localhost",    # Single machine
            "--master_port=29500",
            "/path/to/train_script.py",
        ],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Training failed")
```

---

## Effective Batch Size and Learning Rate

This is a subtlety that trips up many people. When you use DDP with N GPUs, the **effective batch size** is:

```
effective_batch_size = per_gpu_batch_size * world_size
```

If your per-GPU batch size is 32 and you have 4 GPUs, your effective batch size is 128. This matters because:

1. **The gradient average is different from single-GPU training.** With a single GPU and batch size 128, the gradient is the average over 128 samples. With 4 GPUs and batch size 32 each, the gradient is also the average over 128 samples (since AllReduce averages the 4 partial averages). So **mathematically, DDP with N GPUs gives the same gradients as single-GPU with N times the batch size.**

2. **You may need to adjust the learning rate.** The "linear scaling rule" (from [Goyal et al., 2017](https://arxiv.org/abs/1706.02677)) suggests: if you multiply the batch size by N, multiply the learning rate by N. So if your single-GPU learning rate was `1e-3` with batch 32, use `4e-3` with 4 GPUs. In practice, this is a starting point -- you may need warmup and tuning.

```
Single GPU:  batch_size=32,  lr=1e-3
4 GPU DDP:   batch_size=32,  lr=4e-3  (linear scaling)
             effective batch = 128
```

---

## Gradient Accumulation with DDP

Sometimes you want a large effective batch size but do not have enough GPU memory for a large per-GPU batch. Gradient accumulation lets you simulate a larger batch by accumulating gradients over multiple mini-steps.

With DDP, there is an important optimization: you only need to synchronize gradients on the **last** accumulation step, not every step. DDP provides a `no_sync()` context manager for this:

```python
accumulation_steps = 4

for batch_idx, (data, target) in enumerate(dataloader):
    data = data.to(rank)
    target = target.to(rank)

    # Only sync gradients on the last accumulation step
    if (batch_idx + 1) % accumulation_steps != 0:
        # no_sync() skips the AllReduce -- saves communication
        with ddp_model.no_sync():
            output = ddp_model(data)
            loss = criterion(output, target) / accumulation_steps
            loss.backward()
    else:
        # Last accumulation step -- sync gradients normally
        output = ddp_model(data)
        loss = criterion(output, target) / accumulation_steps
        loss.backward()  # AllReduce happens here

        optimizer.step()
        optimizer.zero_grad()
```

Without `no_sync()`, DDP would run AllReduce on every `loss.backward()`, wasting communication bandwidth on intermediate gradients that you are going to accumulate anyway.

---

## Common Gotchas and Debugging

### 1. Forgetting `sampler.set_epoch()`

**Symptom:** Training works but converges slightly worse than expected.

**Cause:** The data is shuffled the same way every epoch. All epochs see the same order.

**Fix:** Always call `sampler.set_epoch(epoch)` before iterating.

### 2. Not using `DistributedSampler`

**Symptom:** Training runs but is not faster than single GPU.

**Cause:** All GPUs process the same data. You are doing 4x the work for the same result.

**Fix:** Use `DistributedSampler`.

### 3. Deadlocks from conditional code

**Symptom:** Training hangs forever.

**Cause:** AllReduce is a **collective** operation -- all GPUs must participate. If one GPU takes a different code path that skips the forward/backward pass, the other GPUs wait forever.

```python
# WRONG -- can deadlock if not all GPUs enter this branch
if some_condition(data):
    loss = model(data)
    loss.backward()

# RIGHT -- all GPUs must call backward on every step
loss = model(data)
loss.backward()
```

### 4. Saving checkpoints from all ranks

**Symptom:** All 4 GPUs write to the same file simultaneously, corrupting it. Or you get 4 identical checkpoint files wasting disk space.

**Fix:** Only save from rank 0:

```python
if rank == 0:
    torch.save(model.module.state_dict(), 'checkpoint.pt')

# Add a barrier so other ranks wait until the save is complete
dist.barrier()
```

### 5. Accessing `model` instead of `model.module`

**Symptom:** `state_dict()` has unexpected key prefixes like `module.layer1.weight` instead of `layer1.weight`.

**Fix:** Use `model.module` to access the underlying model.

### 6. Mismatched operations across ranks

**Symptom:** Cryptic NCCL errors or hangs.

**Cause:** All ranks must execute the same collective operations in the same order. If rank 0 calls `all_reduce` but rank 1 calls `broadcast`, the program will hang or crash.

---

## When to Use DDP vs Other Strategies

| Scenario | Strategy | Reason |
|----------|----------|--------|
| Model fits on 1 GPU | DDP | Simplest, fastest, most mature |
| Model barely fits on 1 GPU | DDP + gradient checkpointing | Saves activation memory |
| Model + optimizer don't fit on 1 GPU | FSDP (Lesson 2) | Shards optimizer/gradients/params |
| A single layer doesn't fit on 1 GPU | Tensor Parallelism (Lesson 3) | Splits individual layers |

DDP should be your **default** choice. Only move to FSDP or tensor parallelism when DDP cannot handle your model size.

---

## Connection to Triton

In Phases 1-3, you wrote custom Triton kernels for operations like matrix multiplication, softmax, and attention. Those kernels run on a single GPU. With DDP:

- Your Triton kernels still run as before on each GPU independently.
- DDP only affects the gradient synchronization step, which uses NCCL (not your kernels).
- However, in later phases, when you write custom backward passes in Triton, you need to ensure your gradient kernels produce correct gradients that DDP can synchronize.

The single-GPU performance optimizations you learned (memory coalescing, tiling, occupancy) are still critical. DDP just multiplies your throughput by the number of GPUs. If each GPU is slow, 4 slow GPUs are still slow.

---

## Exercises

### Exercise 1: Calculate training time

If training a model takes 24 hours on 1 GPU, estimate the time on 4 GPUs with DDP. Assume communication overhead is 10% of compute time.

**Solution:** Compute per GPU is 24h / 4 = 6h. Communication overhead adds 10%, so 6h * 1.1 = 6.6 hours. In practice, you might see 6.5-7 hours depending on the model and hardware.

### Exercise 2: Gradient accumulation with DDP

Write a training loop that uses gradient accumulation with 8 accumulation steps and DDP. Use `no_sync()` correctly.

### Exercise 3: Synchronous vs asynchronous

What happens if one GPU is significantly slower than the others (a "straggler")? In DDP, all GPUs must synchronize at AllReduce. The fast GPUs will wait for the slow GPU. This is called **synchronous** training.

Think about: what would asynchronous training look like? What are the tradeoffs? (Hint: stale gradients.)

### Exercise 4: Memory calculation

A model has 1B parameters in fp16. Calculate the memory footprint per GPU with DDP:
- Parameters: ? GB
- Gradients: ? GB
- Adam optimizer states (momentum + variance, in fp32): ? GB
- Total: ? GB

---

## Milestone

After completing this lesson, you should be able to:

- Explain AllReduce and the Ring AllReduce algorithm to someone unfamiliar with distributed systems
- Write a DDP training script from scratch using PyTorch
- Launch distributed training with `torchrun` or `mp.spawn`
- Identify and fix common DDP bugs (missing sampler, set_epoch, rank 0 saves)
- Know when DDP is the right tool and when you need something else

---

## Resources

- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) -- the official tutorial with more advanced examples
- [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html) -- high-level map of all distributed features
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html) -- low-level details of the communication library
- [Goyal et al., 2017 -- Accurate, Large Minibatch SGD](https://arxiv.org/abs/1706.02677) -- the linear scaling rule paper
- [Ring AllReduce explained (Baidu, 2017)](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/) -- excellent deep dive into the algorithm
