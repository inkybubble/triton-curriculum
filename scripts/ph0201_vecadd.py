# %%
import os

# Enable interpreter mode if no CUDA GPU is available
if not os.environ.get("TRITON_INTERPRET") and not __import__("torch").cuda.is_available():
    os.environ["TRITON_INTERPRET"] = "1"

import torch
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU found — running in Triton interpreter mode")
print(f"Triton version: {triton.__version__}")

# %%
# The full kernel
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,      # Pointer to first input vector
    y_ptr,      # Pointer to second input vector
    output_ptr, # Pointer to output vector
    n_elements, # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program processes
):
    # Which program am I? (like: which chunk of work am I responsible for?)
    pid = tl.program_id(axis=0)

    # Calculate the starting index for this program's chunk
    block_start = pid * BLOCK_SIZE

    # Create offsets: [block_start, block_start+1, ..., block_start+BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask: don't load/store beyond the end of the array
    mask = offsets < n_elements

    # Load data from global memory (with mask for safety)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Compute
    output = x + y

    # Store result back to global memory
    tl.store(output_ptr + offsets, output, mask=mask)


# %%
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr, # pointer to the first input vector
    y_ptr, # pointer to the second input vector
    output_ptr, # pointer to the output vector
    n_elements, # total number of elements
    BLOCK_SIZE: tl.constexpr, # number of elements each program process
):
    # which program am i?
    pid=tl.program_id(axis=0) # returns the program's index along the first (and in this case only) axis of the launch grid

    # calculate the starting index for this program's chunk
    # if each program is responsible for BLOCK_SIZE consecutive elemmnts, this is the start of the elements for which program pid-th is responsible
    block_start=pid*BLOCK_SIZE
    # tl.arange creates a tensor of integers (not an iterator) to be added to block_start - offset, in the end, is a vector (constant + vector)
    offsets=block_start+tl.arange(0, BLOCK_SIZE)

    # mask (to prevent out-of-bounds access)
    mask=offsets<n_elements
    # For a certain size of array, I might need a certain number of blocks. The last might go beyond the size of the array. the mask is an array of Trues and Falses

    # loading the data fro global memory:
    x=tl.load(x_ptr+offsets, mask=mask)
    y=tl.load(y_ptr+offsets, mask=mask)

    # addition
    output=x+y
    # little math vs amount of moved memory - vector addition is memory bound (the gpu compute units barely work. They spend most time waiting for data)

    # store the result (writes back to global memory):
    tl.store(output_ptr+offsets, output, mask=mask)

# LAUNCHER FUNCTION
def add(x: torch.Tensor, y: torch.Tensor)-> torch.Tensor:
    # validate inputs
    assert x.shape==y.shape, "Inputs must have the same shape"

    # we use empty_like because the kernel will write to every single element of the output - prefilling with zero is wasted work (a pointless extra kernel launch) - empty_like allocates memory without initializing it
    output=torch.empty_like(x)
    n_elements=output.numel()

    # Calculate grid size - how many programs to launch
    #triton.cdiv is triton's ceiling division. It will create a grid with enough blocks to cover and exceed the vector (hence the maskign)
    # grid lambda: the grid is a function (lambda) that takes a meta dictionary and returns a tuple of integers. It's a function because Triton's autotuning is such that when using it BLOCK_SIZE is not known until triton picks the best config at runtime. lambda lets triton pass block_size through meta['BLOCK_SIZE'] and compute the correct grid size.
    # For now we have hardcoded BLOCK_SIZE=1024, so the lambda returns always the same result, but good habit to introduce the grid lambda pattern from the start.


    grid=lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Launch the kernel
    # the grid launch syntax. This is triton's way to launch a kernel.
    # [grid] says how many programs to launch (the grid configuration)
    # the rest are the arguments to pass to each instance. Pytorch tensors are automatically converted to pointers.
    # all program instances start in parallel working on the GPU. The cpu doesn't wait for them to finish (kernel launch is asynchronous), However, when use output in Pytorch, cuda's stream synchronization ensures that the kernel has completed.
    add_kernel[grid](x,y, output, n_elements, BLOCK_SIZE=1024)
    return output


# %%
# putting it all together
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid=tl.program_id(axis=0)
    block_start=BLOCK_SIZE*pid
    offsets=block_start+tl.arange(0, BLOCK_SIZE)
    mask=offsets<n_elements

    x=tl.load(x_ptr+offsets, mask=mask)
    y=tl.load(y_ptr+offsets, mask=mask)

    output=x+y

    tl.store(output_ptr+offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape==y.shape, "Inputs must have the same shape"

    output=torch.empty_like(x)
    n_elements=output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x,y, output, n_elements, BLOCK_SIZE=1024)
    return output

# --verify correctness --#
size=2**20
x=torch.rand(size, device=DEVICE)
y=torch.rand(size, device=DEVICE)

triton_output=add(x,y)
torch_output=x+y
assert torch.allclose(triton_output, torch_output), "Results don't match!"
print("Correctness verified")

# %%
# Benchmarking (Triton vs pytorch) — requires a real GPU
if torch.cuda.is_available():
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['size'], #Parameter to vary,
            x_vals=[2**i for i in range(12, 28)], #4k to 128m elements
            line_arg="provider", # argument that selects the implementation
            line_vals=['triton', 'torch'],# Implementations to compare
            line_names=["Triton", "Pytorch"], # Legend Labels
            styles=[("blue", "-"), ("red", "-")], # Line styles
            ylabel='GB/s', #Y-axis label
            plot_name="vector-addition-performance", #Output file name
            args={}, # extra arguments (none here)

        )
    )
    def benchmark(size, provider):
        x = torch.rand(size, device=DEVICE, dtype=torch.float32)
        y = torch.rand(size, device=DEVICE, dtype=torch.float32)
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'torch':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
        # Calculate throughput in GB/s
        gbps = lambda ms: 3 * x.numel() * x.element_size() / ms * 1e-6
        return gbps(ms), gbps(max_ms), gbps(min_ms)

    benchmark.run(print_data=True, show_plots=True)
else:
    print("Skipping benchmark — requires a CUDA GPU")

# %%
# Block Size Experiments
# %%
# Triple sum
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    w_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid=tl.program_id(axis=0)
    block_start=BLOCK_SIZE*pid
    offsets=block_start+tl.arange(0, BLOCK_SIZE)
    mask=offsets<n_elements

    x=tl.load(x_ptr+offsets, mask=mask)
    y=tl.load(y_ptr+offsets, mask=mask)
    w=tl.load(w_ptr+offsets, mask=mask)


    output=x+y+w

    tl.store(output_ptr+offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    assert x.device == y.device == w.device, "Inputs must be on the same device"
    assert x.shape==y.shape==w.shape, "Inputs must have the same shape"

    output=torch.empty_like(x)
    n_elements=output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x,y,w, output, n_elements, BLOCK_SIZE=8192)
    return output

# --verify correctness --#
size=2**20
x=torch.rand(size, device=DEVICE)
y=torch.rand(size, device=DEVICE)
w=torch.rand(size, device=DEVICE)


triton_output=add(x,y,w)
torch_output=x+y+w
assert torch.allclose(triton_output, torch_output), "Results don't match!"
print("Correctness verified")

# Benchmarking (Triton vs pytorch) — requires a real GPU
if torch.cuda.is_available():
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['size'], #Parameter to vary,
            x_vals=[2**i for i in range(12, 28)], #4k to 128m elements
            line_arg="provider", # argument that selects the implementation
            line_vals=['triton', 'torch'],# Implementations to compare
            line_names=["Triton", "Pytorch"], # Legend Labels
            styles=[("blue", "-"), ("red", "-")], # Line styles
            ylabel='GB/s', #Y-axis label
            plot_name="vector-addition-performance", #Output file name
            args={}, # extra arguments (none here)

        )
    )
    def benchmark(size, provider):
        x = torch.rand(size, device=DEVICE, dtype=torch.float32)
        y = torch.rand(size, device=DEVICE, dtype=torch.float32)
        w = torch.rand(size, device=DEVICE, dtype=torch.float32)
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'torch':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y + w, quantiles=quantiles)
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y, w), quantiles=quantiles)
        # Calculate throughput in GB/s
        gbps = lambda ms: 3 * x.numel() * x.element_size() / ms * 1e-6
        return gbps(ms), gbps(max_ms), gbps(min_ms)

    benchmark.run(print_data=True, show_plots=True)
else:
    print("Skipping benchmark — requires a CUDA GPU")
# %%
# Exercise 3 - Fused Multiply-Add (FMA)
import torch
import triton
import triton.language as tl

@triton.jit
def fma_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid=tl.program_id(axis=0)
    block_start=BLOCK_SIZE*pid
    offsets=block_start+tl.arange(0, BLOCK_SIZE)
    mask=offsets<n_elements

    # 3 global reads
    x=tl.load(x_ptr+offsets, mask=mask)
    y=tl.load(y_ptr+offsets, mask=mask)
    z=tl.load(z_ptr+offsets, mask=mask)


    output=x*y+z

    # 1 global write
    tl.store(output_ptr+offsets, output, mask=mask)

def fma(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    assert x.device == y.device == z.device, "Inputs must be on the same device"
    assert x.shape==y.shape==z.shape, "Inputs must have the same shape"

    output=torch.empty_like(x)
    n_elements=output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fma_kernel[grid](x,y,z, output, n_elements, BLOCK_SIZE=8192)
    return output

# --verify correctness --#
size=2**20
x=torch.rand(size, device=DEVICE)
y=torch.rand(size, device=DEVICE)
z=torch.rand(size, device=DEVICE)


triton_output=fma(x,y,z)
torch_output=x*y+z
assert torch.allclose(triton_output, torch_output), "Results don't match!"
print("Correctness verified")

# Benchmarking (Triton vs pytorch) — requires a real GPU
if torch.cuda.is_available():
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['size'], #Parameter to vary,
            x_vals=[2**i for i in range(12, 28)], #4k to 128m elements
            line_arg="provider", # argument that selects the implementation
            line_vals=['triton', 'torch'],# Implementations to compare
            line_names=["Triton", "Pytorch"], # Legend Labels
            styles=[("blue", "-"), ("red", "-")], # Line styles
            ylabel='GB/s', #Y-axis label
            plot_name="fused-multiplication-addition-performance", #Output file name
            args={}, # extra arguments (none here)

        )
    )
    def benchmark(size, provider):
        x = torch.rand(size, device=DEVICE, dtype=torch.float32)
        y = torch.rand(size, device=DEVICE, dtype=torch.float32)
        z = torch.rand(size, device=DEVICE, dtype=torch.float32)
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'torch':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: x * y + z, quantiles=quantiles)
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: fma(x, y, z), quantiles=quantiles)
        # Calculate throughput in GB/s
        gbps = lambda ms: 3 * x.numel() * x.element_size() / ms * 1e-6
        return gbps(ms), gbps(max_ms), gbps(min_ms)

    benchmark.run(print_data=True, show_plots=True)
else:
    print("Skipping benchmark — requires a CUDA GPU")
# %%
