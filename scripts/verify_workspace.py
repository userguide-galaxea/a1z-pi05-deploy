#!/usr/bin/env python3
# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Check source revisions and patch without a GPU, weights or hardware."""
import json
import subprocess
import sys

from bootstrap import COLLECTOR, LEADER, LEADER_MAIN, MAIN_REF, ROOT, SDK, git, head


def main():
    errors = []
    versions = json.loads((ROOT / "versions.json").read_text())
    for name in ("openpi", "a1z_data_collection", LEADER, SDK):
        target = ROOT / name
        if not (target / ".git").exists():
            errors.append(f"Source not initialized: {name}; run python3 scripts/bootstrap.py")
            continue
        actual = head(target)
        expected = versions[name]
        if name == LEADER:
            if actual == expected:
                print(f"OK {name} {actual} (tested fallback)")
                continue
            fetched = git(target, "rev-parse", "--verify", LEADER_MAIN, check=False)
            if fetched.returncode or actual != fetched.stdout.strip():
                errors.append(f"Version mismatch: {name} is neither fetched main nor the tested fallback")
                continue
        elif name == COLLECTOR:
            if actual == expected:
                print(f"OK {name} {actual} (tested fallback)")
                continue
            fetched = git(target, "rev-parse", "--verify", MAIN_REF, check=False)
            if fetched.returncode or actual != fetched.stdout.strip():
                errors.append(f"Version mismatch: {name} is neither fetched main nor the tested fallback")
                continue
        elif name == SDK:
            if actual == expected:
                print(f"OK {name} {actual} (tested fallback)")
                continue
            fetched = git(target, "rev-parse", "--verify", MAIN_REF, check=False)
            if fetched.returncode or actual != fetched.stdout.strip():
                errors.append(f"Version mismatch: {name} is neither fetched main nor the tested fallback")
                continue
        elif actual != expected:
            errors.append(f"Version mismatch: {name}")
            continue
        print(f"OK {name} {actual}")
    patch = ROOT / "patches/a1z-data-collect-integration.patch"
    result = git(ROOT / "a1z_data_collection", "apply", "--reverse", "--check", str(patch), check=False)
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
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as exc:
        print(exc.stderr or str(exc), file=sys.stderr)
        sys.exit(1)
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
