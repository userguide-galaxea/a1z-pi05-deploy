# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Local dataset binding before importing LeRobot's environment constants."""
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def bind_dataset(mode):
    if mode not in ("single", "dual"):
        raise ValueError("mode must be single or dual")
    value = os.environ.get("A1Z_DATASET", "")
    if not value:
        raise ValueError("Set A1Z_DATASET to your checked LeRobot v2.1 dataset directory")
    dataset = Path(value).expanduser().resolve(strict=True)
    if (dataset / "INCOMPLETE").exists():
        raise ValueError("Dataset conversion is INCOMPLETE")
    info = json.loads((dataset / "meta/info.json").read_text())
    if info.get("codebase_version") != "v2.1":
        raise ValueError("Expected LeRobot v2.1; do not change the version label manually")
    if any(not isinstance(info.get(key), int) or isinstance(info[key], bool) or info[key] < 1
           for key in ("total_frames", "total_episodes")):
        raise ValueError("Dataset must contain at least one frame and one episode")
    parts = ("left_arm", "left_gripper") + (("right_arm", "right_gripper") if mode == "dual" else ())
    actual = {k for k in info["features"] if k.startswith("action.")}
    if actual != {"action." + p for p in parts}:
        raise ValueError(f"Dataset action fields do not match {mode}")
    for group in ("observation.state", "action"):
        for part in parts:
            f = info["features"][group + "." + part]
            if f["shape"] != ([1] if part.endswith("gripper") else [6]):
                raise ValueError(f"Unexpected dimensions: {group}.{part}")
    cameras = ("head_rgb", "left_wrist_rgb") + (("right_wrist_rgb",) if mode == "dual" else ())
    for name in cameras:
        if "observation.images." + name not in info["features"]:
            raise ValueError(f"Missing camera: {name}")
    for name in ("episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl"):
        if not (dataset / "meta" / name).is_file():
            raise ValueError(f"Missing v2.1 metadata: {name}")
    tasks = {row["task_index"]: row["task"] for row in
             (json.loads(s) for s in (dataset / "meta/tasks.jsonl").read_text().splitlines() if s.strip())}
    if not tasks or any(not isinstance(t, str) or not t.strip() or t.strip().lower() in
                        ("null", "none", "galaxea_a1z", "describe the task here") for t in tasks.values()):
        raise ValueError("Supply real task descriptions in the dataset training copy")
    repo_id = os.environ.get("A1Z_REPO_ID", f"local/a1z_{mode}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo_id) or any(
            part in (".", "..") for part in repo_id.split("/")):
        raise ValueError("A1Z_REPO_ID must be namespace/dataset")
    cache = Path(os.environ.get("HF_LEROBOT_HOME", ROOT / ".cache/lerobot")).expanduser().resolve()
    os.environ["HF_LEROBOT_HOME"] = str(cache)
    os.environ["A1Z_REPO_ID"] = repo_id
    target = cache / repo_id
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        if target.resolve() != dataset:
            raise ValueError(f"{repo_id} is bound to another dataset. Set a new A1Z_REPO_ID; no data overwritten.")
    else:
        target.symlink_to(dataset, target_is_directory=True)
    return dataset, info


def dataset_fingerprint(dataset):
    digest = hashlib.sha256()
    for name in ("info.json", "episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl"):
        digest.update(name.encode() + b"\0")
        digest.update((Path(dataset) / "meta" / name).read_bytes())
    return digest.hexdigest()
