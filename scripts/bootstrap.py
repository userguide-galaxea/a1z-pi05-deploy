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
COLLECTOR = "a1z-data-collect"
LEADER = f"{COLLECTOR}/a1z-t"
SDK_NAME = "galaxea_a1z"
SDK = f"{LEADER}/{SDK_NAME}"
MAIN_REF = "refs/remotes/origin/main"
LEADER_MAIN = MAIN_REF
COLLECTOR_PATCH = "patches/a1z-data-collect-integration.patch"
COLLECTOR_URL = "https://github.com/userguide-galaxea/a1z-data-collect.git"
LEADER_URL = "https://github.com/userguide-galaxea/a1z-t.git"
SDK_URL = "https://github.com/userguide-galaxea/galaxea_a1z.git"


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


def ensure_clone(path, url):
    """Clone a first-party dependency once; these are plain clones, not submodules."""
    if (ROOT / path / ".git").exists():
        return False
    git(ROOT, "clone", "--branch", "main", url, str(ROOT / path))
    return True


def collector_patch_applied():
    return git(ROOT / COLLECTOR, "apply", "--reverse", "--check",
               str(ROOT / COLLECTOR_PATCH), check=False).returncode == 0


def init_collector(versions, tested):
    """Follow a1z-data-collect main, or fall back to the tested commit."""
    ensure_clone(COLLECTOR, COLLECTOR_URL)
    folder = ROOT / COLLECTOR
    if tested:
        revision = versions[COLLECTOR]
        if git(folder, "cat-file", "-e", revision + "^{commit}", check=False).returncode:
            git(folder, "fetch", "--no-tags", "origin", revision)
    else:
        fetched = git(folder, "fetch", "--no-tags", "origin", f"+refs/heads/main:{MAIN_REF}", check=False)
        if fetched.returncode:
            raise RuntimeError(
                f"Cannot fetch A1Z data-collect main; retry with --tested.\n{fetched.stderr.strip()}"
            )
        revision = git(folder, "rev-parse", MAIN_REF).stdout.strip()
    if head(folder) != revision:
        # The integration patch is reapplied below, so undo it before switching.
        patch_was_applied = collector_patch_applied()
        if patch_was_applied:
            git(folder, "apply", "--reverse", str(ROOT / COLLECTOR_PATCH))
        try:
            require_clean(COLLECTOR)
        except RuntimeError:
            # Restore the workspace we touched before reporting the conflict.
            if patch_was_applied:
                git(folder, "apply", str(ROOT / COLLECTOR_PATCH))
            raise
        git(folder, "checkout", "--detach", revision)
    mode = "tested commit" if tested else "origin/main"
    print(f"A1Z data-collect ({mode}): {revision}")
    return revision


def init_leader(versions, tested):
    ensure_clone(LEADER, LEADER_URL)
    leader = ROOT / LEADER
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
    if head(leader) != revision:
        require_clean(LEADER)
        git(leader, "checkout", "--detach", revision)
    mode = "tested fallback" if tested else "origin/main"
    print(f"A1Z-T ({mode}): {revision}")
    return revision


def init_sdk(versions, tested):
    """Clone and follow galaxea_a1z main, or fall back to the tested commit."""
    ensure_clone(SDK, SDK_URL)
    sdk = ROOT / SDK
    if tested:
        revision = versions[SDK]
        if git(sdk, "cat-file", "-e", revision + "^{commit}", check=False).returncode:
            git(sdk, "fetch", "--no-tags", "origin", revision)
    else:
        fetched = git(sdk, "fetch", "--no-tags", "origin", f"+refs/heads/main:{MAIN_REF}", check=False)
        if fetched.returncode:
            raise RuntimeError(
                f"Cannot fetch A1Z SDK main; retry with --tested.\n{fetched.stderr.strip()}"
            )
        revision = git(sdk, "rev-parse", MAIN_REF).stdout.strip()
    if head(sdk) != revision:
        require_clean(SDK)
        git(sdk, "checkout", "--detach", revision)
    mode = "tested commit" if tested else "origin/main"
    print(f"A1Z SDK ({mode}): {revision}")
    return revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--tested", action="store_true",
                       help="Use the tested commits from versions.json for a1z-data-collect and A1Z-T")
    modes.add_argument("--tested-a1z-t", action="store_true", help="Use only the tested A1Z-T fallback")
    modes.add_argument("--openpi-only", action="store_true", help="Initialize only openpi for model setup")
    args = parser.parse_args()
    versions = json.loads((ROOT / "versions.json").read_text())
    init_pinned("openpi", versions["openpi"])
    if args.openpi_only:
        print("Pinned openpi source ready; no model weights downloaded.")
        return
    init_collector(versions, args.tested)
    init_leader(versions, args.tested or args.tested_a1z_t)
    init_sdk(versions, args.tested or args.tested_a1z_t)
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
