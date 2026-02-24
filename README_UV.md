# Setting up uv on a Linux Remote Box

## Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

This installs uv to `~/.local/bin/`. Add it to your PATH if not already:

```bash
source $HOME/.local/bin/env
```

To make it permanent, add that line to your `~/.bashrc` or `~/.zshrc`.

## Clone and set up the project

```bash
git clone <repo-url> triton-curriculum && cd triton-curriculum
uv sync
```

## Install Triton (from PyPI)

On Linux with a CUDA GPU, Triton has prebuilt wheels — no need to build from source:

```bash
uv pip install triton
```

If you need the latest development version, build from source instead:

```bash
git clone https://github.com/triton-lang/triton.git triton-src
cd triton-src
uv pip install -r python/requirements.txt
uv pip install .
cd ..
```

## Install runtime dependencies

```bash
uv add numpy pandas torch matplotlib ipykernel ipython
```

## Verify

```bash
uv run python -c "import triton; print(f'Triton {triton.__version__}')"
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Run scripts

With a GPU available, run directly:

```bash
uv run scripts/ph0201_vecadd.py
```

Without a GPU (interpreter mode):

```bash
TRITON_INTERPRET=1 uv run scripts/ph0201_vecadd.py
```
