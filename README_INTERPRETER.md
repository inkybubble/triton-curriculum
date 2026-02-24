# Triton Interpreter Mode on macOS (Apple Silicon)

Run [OpenAI Triton](https://github.com/triton-lang/triton) kernels on macOS without an NVIDIA GPU using **Interpreter Mode**. Triton's interpreter mode executes kernels in pure Python (via NumPy), bypassing the need for CUDA/ROCm hardware entirely.

Tested on: **Apple M4 Max, macOS 15, Python 3.12**

## Prerequisites

- macOS (Intel or Apple Silicon)
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Git
- A C++ compiler (Xcode Command Line Tools: `xcode-select --install`)

## Setup

### 1. Create the project and virtual environment

```bash
mkdir triton_interpreter && cd triton_interpreter
uv init --python=3.12
uv sync
```

### 2. Clone Triton and install build dependencies

```bash
git clone https://github.com/triton-lang/triton.git triton-src
cd triton-src
uv pip install -r python/requirements.txt
```

### 3. Build Triton from source

There are no PyPI wheels for macOS — you must build from source. The `setup.py` and `pyproject.toml` are at the **repo root** (not in `python/`).

```bash
uv pip install .
```

This will download LLVM automatically if you don't have it and compile the C++ backend. Expect the first build to take several minutes.

### 4. Install runtime dependencies

```bash
uv pip install numpy torch
```

### 5. Remove or rename the clone (important)

The cloned directory is named `triton-src` to avoid shadowing the installed `triton` package. If you named it `triton`, rename or remove it now:

```bash
# If your clone is named 'triton', rename it:
mv triton triton-src

# Or remove it entirely (it's already installed):
rm -rf triton-src
```

If a directory named `triton/` exists in your working directory, Python will import it as a namespace package instead of the installed Triton, causing `ImportError: cannot import name '__version__'`.

## Usage

Always set `TRITON_INTERPRET=1` when running scripts:

```bash
TRITON_INTERPRET=1 uv run main.py
```

Or export it for your session:

```bash
export TRITON_INTERPRET=1
uv run main.py
```

The included `main.py` runs two kernels — vector addition and matrix multiplication — and validates each against PyTorch:

```
$ TRITON_INTERPRET=1 uv run main.py
PASSED [add]:    triton output matches torch (max error: 0.0e+00)
PASSED [matmul]: triton output matches torch (max error: 9.5e-06)
```

The small matmul error (~1e-05) is expected due to float32 accumulation order differences between the tiled Triton kernel and PyTorch's `@` operator.

## Troubleshooting

### `ImportError: cannot import name '__version__' from 'triton'`

A directory named `triton/` in your working directory is shadowing the installed package. Rename or delete it (see step 5 above), or run your script from a different directory.

### `No solution found when resolving dependencies` (pip/uv install triton)

PyPI only has Linux wheels. You must build from source (step 3).

### `does not appear to be a Python project`

You ran `uv pip install -e python` — the repo was restructured and `setup.py` is now at the repo root. Use `uv pip install .` from the repo root instead.

## How it works

When `TRITON_INTERPRET=1` is set, Triton replaces its GPU JIT compiler with a Python interpreter that:

1. Simulates the GPU grid/block execution model
2. Runs each "program" (block) sequentially in Python
3. Uses NumPy for the underlying tensor operations

This means kernels run **much slower** than on a real GPU, but it's useful for:

- Developing and debugging kernel logic on macOS
- Writing and running Triton unit tests without GPU access
- Learning the Triton programming model
