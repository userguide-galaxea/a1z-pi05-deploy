#!/usr/bin/env bash
# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PI05_PYTHON:-$ROOT/openpi/.venv/bin/python}"
export PYTHONPATH="$ROOT/src:$ROOT/openpi/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" -m pi05_a1z.policy_bridge "$@"
