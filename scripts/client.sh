#!/usr/bin/env bash
# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${A1Z_CLIENT_PYTHON:-python3}" "$ROOT/scripts/deploy.py" "$@"
