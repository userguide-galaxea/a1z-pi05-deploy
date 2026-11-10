#!/usr/bin/env bash
# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$(uname -s)" == Linux ]] || { echo 'Model setup requires Linux with NVIDIA CUDA support.' >&2; exit 2; }
command -v uv >/dev/null || { echo 'Install uv first; see README.' >&2; exit 2; }
python3 "$ROOT/scripts/bootstrap.py" --openpi-only
export GIT_LFS_SKIP_SMUDGE=1
uv sync --project "$ROOT/openpi" --frozen --python 3.11
uv pip install --python "$ROOT/openpi/.venv/bin/python" -e "$ROOT"
"$ROOT/openpi/.venv/bin/python" -c 'import jax; print(jax.devices()); assert any(d.platform == "gpu" for d in jax.devices()), "No JAX GPU backend"'
