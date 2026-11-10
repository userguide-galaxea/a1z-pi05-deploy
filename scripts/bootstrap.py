#!/usr/bin/env python3
# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Initialize source and apply the collection integration patch."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LEADER = "a1z-data-collect/a1z-t"
SDK_NAME = "galaxea_a1z"
SDK = f"{LEADER}/{SDK_NAME}"
LEADER_MAIN = "refs/remotes/origin/main"


def git(folder, *args, check=True):
    return subprocess.run(
        ["git", "-c", "fetch.recurseSubmodules=false", "-c", "submodule.recurse=false",
         "-C", str(folder), *args],
        text=True, capture_output=True, check=check,
    )


def head(folder):
    return git(folder, "rev-parse", "HEAD").stdout.strip()


def require_clean(name):
    if git(ROOT / name, "status", "--porcelain", "--ignore-submodules=all").stdout.strip():
        raise RuntimeError(f"Preserve local edits in {name} before updating")


def init_submodule(folder, child):
    git(folder, "submodule", "sync", "--", child)
    git(folder, "submodule", "update", "--init", "--checkout", "--", child)


def init_pinned(name, revision):
    target = ROOT / name
    if (target / ".git").exists() and head(target) != revision:
        require_clean(name)
    init_submodule(ROOT, name)
    if head(target) != revision:
        raise RuntimeError(f"Version mismatch: {name} must match versions.json")


def sdk_gitlink(leader, revision):
    entry = git(leader, "ls-tree", revision, "--", SDK_NAME).stdout.split()
    if len(entry) == 4 and entry[:2] == ["160000", "commit"]:
        return entry[2]
    return None


def init_leader(versions, tested):
    leader = ROOT / LEADER
    sdk = ROOT / SDK
    collector = leader.parent
    if not (leader / ".git").exists():
        init_submodule(collector, leader.name)
    else:
        git(collector, "submodule", "sync", "--", leader.name)

    if tested:
        revision = versions[LEADER]
        if git(leader, "cat-file", "-e", revision + "^{commit}", check=False).returncode:
            git(leader, "fetch", "--no-tags", "origin", revision)
    else:
        fetched = git(leader, "fetch", "--no-tags", "origin",
                      f"+refs/heads/main:{LEADER_MAIN}", check=False)
        if fetched.returncode:
            raise RuntimeError(
                f"Cannot fetch A1Z-T main; retry with --tested-a1z-t.\n{fetched.stderr.strip()}"
            )
        revision = git(leader, "rev-parse", LEADER_MAIN).stdout.strip()

    sdk_revision = sdk_gitlink(leader, revision)
    if sdk_revision is None:
        raise RuntimeError("Selected A1Z-T lacks the expected SDK gitlink; retry with --tested-a1z-t")
    if tested and sdk_revision != versions[SDK]:
        raise RuntimeError("The fallback A1Z-T and SDK commits in versions.json do not match")

    leader_changed = head(leader) != revision
    if leader_changed:
        require_clean(LEADER)
    if (sdk / ".git").exists() and (leader_changed or head(sdk) != sdk_revision):
        require_clean(SDK)
    if leader_changed:
        git(leader, "checkout", "--detach", revision)
    init_submodule(leader, SDK_NAME)
    if head(sdk) != sdk_revision:
        raise RuntimeError(f"SDK version mismatch: expected {sdk_revision} from A1Z-T")
    mode = "tested fallback" if tested else "origin/main"
    print(f"A1Z-T ({mode}): {revision}\nA1Z SDK: {sdk_revision}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--tested-a1z-t", action="store_true", help="Use the tested A1Z-T fallback")
    modes.add_argument("--openpi-only", action="store_true", help="Initialize only openpi for model setup")
    args = parser.parse_args()
    versions = json.loads((ROOT / "versions.json").read_text())
    init_pinned("openpi", versions["openpi"])
    if args.openpi_only:
        print("Pinned openpi source ready; no model weights downloaded.")
        return
    init_pinned("a1z-data-collect", versions["a1z-data-collect"])
    init_leader(versions, args.tested_a1z_t)
    patch = str(ROOT / "patches/a1z-data-collect-integration.patch")
    collector = ROOT / "a1z-data-collect"
    if git(collector, "apply", "--reverse", "--check", patch, check=False).returncode == 0:
        print("Collection patch already applied")
    else:
        git(collector, "apply", "--check", patch)
        git(collector, "apply", patch)
        print("Collection patch applied")
    print("Source ready; no model weights downloaded.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(exc.stderr or str(exc), file=sys.stderr)
        sys.exit(1)
    except (OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
