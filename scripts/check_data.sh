#!/usr/bin/env bash
# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PI05_PYTHON:-$ROOT/openpi/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo 'Install the model environment first: bash scripts/setup_model.sh' >&2; exit 2; }
exec "$PYTHON_BIN" "$ROOT/scripts/model.py" data-check "$@"
