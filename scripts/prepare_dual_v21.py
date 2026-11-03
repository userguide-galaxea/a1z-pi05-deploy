# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Prepare an external dual-arm LeRobot v2.1 dataset for A1Z LoRA.

Copies videos with hardlinks, writes normalized float32 arm/gripper tables and
matching metadata into a new directory. The source is never modified.
"""
import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PART_DIMS = {"left_arm": 6, "left_gripper": 1, "right_arm": 6, "right_gripper": 1}
VECTORS = {f"{group}.{part}": n for group in ("observation.state", "action")
           for part, n in PART_DIMS.items()}
GRIPPERS = {key for key in VECTORS if key.endswith("_gripper")}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def replace_text(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_jsonl(path, rows):
    replace_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original extracted LeRobot v2.1 dataset")
    parser.add_argument("destination", type=Path, help="New training dataset directory")
    parser.add_argument("--task", required=True, help="Task text for frames' task_index")
    parser.add_argument("--gripper-scale", type=float, default=100.0,
                        help="Raw gripper range maximum; default 100 means 0..100 -> 0..1")
    args = parser.parse_args()
    source, dest = args.source.expanduser().resolve(), args.destination.expanduser().resolve()
    if source == dest or dest.is_relative_to(source) or dest.exists():
        parser.error("destination must be a new, separate directory")
    if not args.task.strip() or not np.isfinite(args.gripper_scale) or args.gripper_scale <= 0:
        parser.error("task must be nonempty and gripper scale positive")
    info = json.loads((source / "meta/info.json").read_text())
    if info.get("codebase_version") != "v2.1" or any(k not in info["features"] for k in VECTORS):
        parser.error("source must be dual-arm LeRobot v2.1 with 6+1 actions and states")
    tasks = read_jsonl(source / "meta/tasks.jsonl")
    episodes = read_jsonl(source / "meta/episodes.jsonl")
    stats = read_jsonl(source / "meta/episodes_stats.jsonl")
    if len(episodes) != len(stats) or len(episodes) != info["total_episodes"]:
        parser.error("episode and stats counts do not match")
    shutil.copytree(source, dest, copy_function=os.link)
    (dest / "INCOMPLETE").write_text("Preparing normalized dataset\n")
    try:
        used_task_ids = set()
        for ep, row in zip(episodes, stats):
            i = ep["episode_index"]
            fmt = {"episode_index": i, "episode_chunk": i // info["chunks_size"]}
            path = dest / info["data_path"].format(**fmt)
            table = pq.read_table(path)
            if table.num_rows != ep["length"]:
                raise ValueError(f"episode {i}: length mismatch")
            ids = {int(x) for x in table["task_index"].to_pylist()}
            used_task_ids.update(ids)
            ep["_actual_task_ids"] = sorted(ids)
            new_stats = {k: v for k, v in row["stats"].items() if not k.endswith(".velocities")}
            for key, dim in VECTORS.items():
                if key not in table.schema.names:
                    raise ValueError(f"episode {i}: missing {key}")
                values = np.asarray(table[key].to_pylist(), dtype=np.float64).reshape(-1, dim)
                if not np.isfinite(values).all():
                    raise ValueError(f"episode {i}: nonfinite {key}")
                if key in GRIPPERS:
                    if values.min() < -1e-5 or values.max() > args.gripper_scale + 1e-5:
                        raise ValueError(f"episode {i}: {key} outside [0, {args.gripper_scale}]")
                    values /= args.gripper_scale
                values = values.astype(np.float32)
                column = pa.array(values[:, 0] if dim == 1 else values.tolist(),
                                  type=pa.float32() if dim == 1 else pa.list_(pa.float32()))
                pos = table.schema.get_field_index(key)
                table = table.set_column(pos, key, column)
                new_stats[key] = {
                    "min": values.min(axis=0).astype(float).tolist(),
                    "max": values.max(axis=0).astype(float).tolist(),
                    "mean": values.mean(axis=0, dtype=np.float64).tolist(),
                    "std": values.std(axis=0, dtype=np.float64).tolist(),
                    "count": [len(values)],
                }
            velocities = [name for name in table.schema.names if name.endswith(".velocities")]
            table = table.drop(velocities)
            tmp = path.with_suffix(".tmp.parquet")
            pq.write_table(table, tmp)
            os.replace(tmp, path)
            row["stats"] = new_stats
        if len(used_task_ids) != 1:
            raise ValueError(f"expected one frame task_index, got {sorted(used_task_ids)}")
        task_id = used_task_ids.pop()
        task_map = {int(row["task_index"]): row for row in tasks}
        if task_id not in task_map:
            raise ValueError(f"missing task_index {task_id}")
        task_map[task_id]["task"] = args.task.strip()
        for ep in episodes:
            ep["tasks"] = [task_map[i]["task"] for i in ep.pop("_actual_task_ids")]
        info["features"] = {k: v for k, v in info["features"].items() if not k.endswith(".velocities")}
        for key in VECTORS:
            info["features"][key]["dtype"] = "float32"
        replace_text(dest / "meta/info.json", json.dumps(info, ensure_ascii=False, indent=2) + "\n")
        write_jsonl(dest / "meta/tasks.jsonl", tasks)
        write_jsonl(dest / "meta/episodes.jsonl", episodes)
        write_jsonl(dest / "meta/episodes_stats.jsonl", stats)
        (dest / "INCOMPLETE").unlink()
        print(json.dumps({"dataset": str(dest), "episodes": len(episodes),
                          "frames": info["total_frames"], "task_index": task_id,
                          "task": args.task.strip(), "gripper_scale": args.gripper_scale},
                         ensure_ascii=False, indent=2))
    except Exception:
        print(f"Incomplete output retained for inspection: {dest}")
        raise


if __name__ == "__main__":
    main()
