#!/usr/bin/env python3
# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Initialize pinned source only, then apply the collection integration patch."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def git(folder, *args, check=True):
    return subprocess.run(["git", "-c", "fetch.recurseSubmodules=false", "-C", str(folder), *args],
                          text=True, capture_output=True, check=check)


def main():
    versions = json.loads((ROOT / "versions.json").read_text())
    for name in ("openpi", "a1z-data-collect", "a1z-data-collect/a1z-t",
                 "a1z-data-collect/a1z-t/GALAXEA-A1Z"):
        parent, _, child = name.rpartition("/")
        folder = ROOT / parent
        git(folder, "submodule", "sync", "--", child)
        git(folder, "submodule", "update", "--init", "--", child)
        target = ROOT / name
        if git(target, "rev-parse", "HEAD").stdout.strip() != versions[name]:
            if git(target, "status", "--porcelain").stdout.strip():
                raise RuntimeError(f"Preserve local edits in {name} before updating")
            if git(target, "cat-file", "-e", versions[name] + "^{commit}", check=False).returncode:
                git(target, "fetch", "origin", versions[name])
            git(target, "checkout", "--detach", versions[name])
    patch = str(ROOT / "patches/a1z-data-collect-integration.patch")
    collector = ROOT / "a1z-data-collect"
    if git(collector, "apply", "--reverse", "--check", patch, check=False).returncode == 0:
        print("Collection patch already applied")
    else:
        git(collector, "apply", "--check", patch)
        git(collector, "apply", patch)
        print("Collection patch applied")
    print("Pinned source ready; no model weights downloaded.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(exc.stderr or str(exc), file=sys.stderr)
        sys.exit(1)
    except (OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
