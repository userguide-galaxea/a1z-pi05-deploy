#!/usr/bin/env python3
# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Check source revisions and patch without a GPU, weights or hardware."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    errors = []
    for name, expected in json.loads((ROOT / "versions.json").read_text()).items():
        if name == "$comment":
            continue
        target = ROOT / name
        if not (target / ".git").exists():
            errors.append(f"Uninitialized submodule: {name}")
            continue
        actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
        if actual != expected:
            errors.append(f"Version mismatch: {name}")
        else:
            print(f"OK {name} {actual[:12]}")
    patch = ROOT / "patches/a1z-data-collect-integration.patch"
    result = subprocess.run(["git", "-C", str(ROOT / "a1z-data-collect"),
                             "apply", "--reverse", "--check", str(patch)], capture_output=True)
    if result.returncode:
        errors.append("Missing or modified collection integration patch")
    if (ROOT / "GalaxeaVLA").exists():
        errors.append("Obsolete GalaxeaVLA folder; this project uses openpi")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Source integration OK. GPU training and physical execution are separate checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
