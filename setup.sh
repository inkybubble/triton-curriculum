#!/bin/bash
set -e

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

# Create environment with torch and triton
uv venv /workspace/.venv --python 3.12
source /workspace/.venv/bin/activate

uv pip install torch triton

echo ""
echo "Done! Activate with: source /workspace/.venv/bin/activate"
