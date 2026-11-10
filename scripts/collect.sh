#!/usr/bin/env bash
# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

# HDF5 collection with a1z-data-collect and its nested a1z-t leader.
set -euo pipefail

A1Z_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-dual}"
if [[ "$MODE" == dual || "$MODE" == single || "$MODE" == single-left ]]; then
  [[ $# -eq 0 ]] || shift
else
  echo "Usage: bash scripts/collect.sh {dual|single} [Hydra overrides]" >&2
  exit 2
fi
CHECK_ONLY=false
if [[ $# -gt 0 && "$1" == --check ]]; then
  CHECK_ONLY=true
  shift
fi

if [[ "$MODE" == dual ]]; then
  CONFIG="${A1Z_COLLECT_CONFIG:-$A1Z_ROOT/configs/collect.local.yaml}"
elif [[ "$MODE" == single ]]; then
  CONFIG="${A1Z_COLLECT_CONFIG:-$A1Z_ROOT/configs/collect_single.local.yaml}"
else
  CONFIG="${A1Z_COLLECT_CONFIG:-$A1Z_ROOT/configs/collect_single_left.local.yaml}"
fi
MODULE=scripts.record_data
[[ -f "$CONFIG" ]] || {
  echo "Missing local collection config: $CONFIG" >&2
  echo "Copy the matching configs/collect*.example.yaml and fill in your devices." >&2
  exit 2
}
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
SOURCE_CONFIG="$CONFIG"
PYTHON_BIN="${A1Z_PYTHON:-python}"
for override in "$@"; do
  case "$override" in
    config_file=*|arm_cfg=*|arm_cfg.*|camera_cfg=*|camera_cfg.*|collector_cfg=*|collector_cfg.*|hardware_checks=*|hardware_checks.*)
      echo 'Edit hardware settings in the local YAML; hardware overrides would bypass identity checks.' >&2
      exit 2 ;;
  esac
done
CONFIG_ADAPTED=""
CONFIG_ADAPTED="$("$PYTHON_BIN" "$A1Z_ROOT/scripts/normalize_collect_config.py" "$CONFIG")"
trap 'rm -f -- "$CONFIG_ADAPTED"' EXIT
CONFIG="$CONFIG_ADAPTED"

"$PYTHON_BIN" - "$CONFIG" "$MODE" "$SOURCE_CONFIG" "$A1Z_ROOT" <<'PY'
import os
import subprocess
import sys
import yaml

path, mode, source_path, root = sys.argv[1:]
sys.path.insert(0, os.path.join(root, 'a1z_runtime', 'client'))
from device_checks import check_can
cfg = yaml.safe_load(open(path, encoding="utf-8"))
arm = cfg["arm_cfg"]
camera = cfg["camera_cfg"]
single = mode in ("single", "single-left")
expected = "one_master_slave" if single else "two_master_slave"
if arm["collection_type"] != expected:
    raise SystemExit(
        f"Mode {mode} requires collection_type={expected}; "
        f"{source_path} contains {arm['collection_type']!r}. "
        "Set A1Z_COLLECT_CONFIG to the matching configuration before retrying."
    )
if not cfg["task_description"].strip() or cfg["task_description"] == "Describe the task here":
    raise SystemExit("Set a real task_description in the local config")
if not cfg["hdf5_cfg"]["task_name"] or cfg["hdf5_cfg"]["task_name"] == "your_task":
    raise SystemExit("Set hdf5_cfg.task_name in the local config")
roles = ("cam_high", "cam_left_wrist") if single else ("cam_high", "cam_left_wrist", "cam_right_wrist")
if tuple(camera["camera_names"]) != roles:
    raise SystemExit(f"camera_names must be {list(roles)}")
paths = [camera["topics"].get(role) for role in roles]
if any(not path or not os.path.exists(path) for path in paths) or len(set(paths)) != len(paths):
    raise SystemExit("Set distinct, existing camera device paths for every role")
sides = ("left",) if single else ("left", "right")
ports = [arm.get(f"leader_port_{side}") for side in sides]
cans = [arm.get(f"follower_can_{side}") for side in sides]
if len({os.path.realpath(port) for port in ports if port}) != len(ports):
    raise SystemExit("a1z-t requires distinct Leader serial ports for dual collection")
if any(not port or not os.path.exists(port) for port in ports):
    raise SystemExit("Set existing a1z-t Leader serial port(s)")
leader_config = arm.get("leader_config_path")
if leader_config and (not os.path.isabs(leader_config) or not os.path.isfile(leader_config)):
    raise SystemExit("leader_config_path must be an existing absolute config file")
from teleop_app.config import load_config
leader_settings = load_config(leader_config)
print("A1Z-T profile:", leader_settings["a1z_t_version"], flush=True)
if any(not can for can in cans) or len(set(cans)) != len(cans):
    raise SystemExit("Set distinct follower CAN channels")
check_can(yaml.safe_load(open(source_path, encoding='utf-8')))
print(f"{mode} collection: Leader={ports}, follower={cans}, cameras={paths}", flush=True)
PY

if [[ "$CHECK_ONLY" == true ]]; then
  echo "Collection configuration OK; no hardware opened."
  exit 0
fi
[[ -t 0 ]] || { echo "Interactive terminal required before enabling the follower arm(s)" >&2; exit 2; }
echo "Collection may move the follower arm(s). Confirm the workspace is clear and emergency stop is available." >&2
"$PYTHON_BIN" "$A1Z_ROOT/scripts/check_devices.py" --config "$SOURCE_CONFIG"
cd "$A1Z_ROOT/a1z-data-collect"
export PYTHONPATH="$A1Z_ROOT/a1z-data-collect/a1z-t/GALAXEA-A1Z:$A1Z_ROOT/a1z-data-collect/a1z-t:$PWD${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -m "$MODULE" "config_file=$CONFIG" "$@"
