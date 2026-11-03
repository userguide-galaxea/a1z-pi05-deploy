#!/usr/bin/env bash
# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

set -euo pipefail
A1Z_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "$A1Z_ROOT/scripts/bootstrap.py"
python -m pip install -r "$A1Z_ROOT/requirements-robot.txt"
python -m pip install -e "$A1Z_ROOT/a1z-data-collect/a1z-t/GALAXEA-A1Z" -e "$A1Z_ROOT/a1z-data-collect"
# Install the nested Leader without replacing workspace dependencies.
python -m pip install --no-deps -e "$A1Z_ROOT/a1z-data-collect/a1z-t"
python - "$A1Z_ROOT" <<'PY'
from pathlib import Path
import sys
import a1z, a1z_t
from teleop_app.config import load_config
root = Path(sys.argv[1]).resolve()
sdk = root / "a1z-data-collect/a1z-t/GALAXEA-A1Z"
if not Path(a1z.__file__).resolve().is_relative_to(sdk):
    raise SystemExit(f"Unexpected SDK import: {a1z.__file__}; reinstall the workspace SDK")
print("SDK:", a1z.__file__)
print("Leader:", a1z_t.__file__)
print("Profile:", load_config()["a1z_t_version"])
PY
