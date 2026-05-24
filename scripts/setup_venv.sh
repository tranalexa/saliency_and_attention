#!/usr/bin/env bash
# Create project venv with PyTorch + Modal CLI dependencies.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  echo "Creating .venv ..."
  python3 -m venv .venv
fi

# shellcheck source=/dev/null
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements-pytorch.txt
pip install -r requirements-modal.txt

echo ""
echo "Virtual environment ready."
echo "  source .venv/bin/activate"
echo ""
echo "Modal (one-time auth):"
echo "  modal setup"
echo ""
echo "Local notebooks also need:"
echo "  export IMAGENET_ROOT=/path/to/imagenet"
