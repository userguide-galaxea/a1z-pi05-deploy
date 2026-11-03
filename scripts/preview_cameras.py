# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Labelled camera preview with bounded capture; no CAN, Leader or arm SDK."""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "a1z_runtime/client"))
from device_checks import cameras_from_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--duration", type=float, default=10)
    args = parser.parse_args()
    if not 0 < args.duration <= 600:
        parser.error("duration must be within (0, 600] seconds")
    cameras = None
    failed = False
    try:
        cameras = cameras_from_config(yaml.safe_load(args.config.read_text()))
        print("Camera only; no CAN/Leader opened. Close with Q/Esc or Ctrl+C.", flush=True)
        cameras.start()
        labels = {"head_rgb": "Head", "left_wrist_rgb": "Wrist" if len(cameras.specs) == 2 else "Left Wrist",
                  "right_wrist_rgb": "Right Wrist"}
        started = time.monotonic()
        while True:
            frames = cameras.frames()
            if args.check_only:
                if time.monotonic() - started >= args.duration:
                    break
                time.sleep(.02)
                continue
            tiles = []
            for spec in cameras.specs:
                tile = cv2.resize(frames[spec.name], (426, 240))
                cv2.putText(tile, labels[spec.name], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2)
                tiles.append(tile)
            cv2.imshow("A1Z cameras", np.hstack(tiles))
            if cv2.waitKey(10) & 255 in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    except (ValueError, RuntimeError, TimeoutError, OSError, KeyError, TypeError) as exc:
        print(f"Camera check failed: {exc}", file=sys.stderr)
        failed = True
    finally:
        if cameras is not None:
            cameras.close()
            print(json.dumps(cameras.statistics(), indent=2), flush=True)
        if not args.check_only:
            cv2.destroyAllWindows()
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
