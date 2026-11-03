# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Optional low-rate dual-camera and action trace for live diagnosis."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np


CAMERA_KEYS = ("head_rgb", "left_wrist_rgb")


class DiagnosticRecorder:
    def __init__(self, directory: Path, every: int = 4, max_frames: int = 2400):
        if every < 1 or max_frames < 1:
            raise ValueError("Recording interval and frame limit must be positive")
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        self.every = every
        self.max_frames = max_frames
        self.frames = 0
        self.dropped = 0
        self.pending = deque()
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="a1z-recorder")
        self.events = (directory / "events.jsonl").open("w", buffering=1)
        self.events.write(json.dumps({
            "type": "session", "time": time.time(), "every": every,
            "max_frames": max_frames, "camera_order": CAMERA_KEYS,
        }) + "\n")

    @staticmethod
    def _write_image(path: Path, images: dict, step: int, timestamp: float) -> None:
        import cv2
        rgb = np.concatenate(
            [np.transpose(images[key], (1, 2, 0)) for key in CAMERA_KEYS], axis=1
        )
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, f"step={step} time={timestamp:.3f}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise RuntimeError("Diagnostic JPEG encoding failed")
        temp = path.with_suffix(".jpg.tmp")
        temp.write_bytes(encoded.tobytes())
        temp.replace(path)

    def frame(self, step: int, images: dict, model_observation: bool) -> str | None:
        if self.frames >= self.max_frames:
            return None
        if not model_observation and step % self.every:
            return None
        while self.pending and self.pending[0].done():
            self.pending.popleft().result()
        if len(self.pending) >= 3:
            self.dropped += 1
            return None
        name = f"frame_{step:06d}.jpg"
        self.pending.append(self.worker.submit(
            self._write_image, self.directory / name, images, step, time.time()
        ))
        self.frames += 1
        return name

    def action(self, step: int, target: np.ndarray, previous: np.ndarray | None,
               frame: str | None, model_observation: bool,
               observed_state: dict | None, response_seconds: float) -> None:
        self.events.write(json.dumps({
            "type": "action", "step": step, "time": time.time(),
            "frame": frame, "model_observation": model_observation,
            "target": target.tolist(),
            "previous_command": previous.tolist() if previous is not None else None,
            "observed_state": (
                {key: np.asarray(value).tolist() for key, value in observed_state.items()}
                if observed_state is not None else None
            ),
            "response_seconds": response_seconds,
        }) + "\n")

    def close(self) -> None:
        self.worker.shutdown(wait=True)
        while self.pending:
            self.pending.popleft().result()
        self.events.write(json.dumps({
            "type": "end", "time": time.time(),
            "frames": self.frames, "dropped": self.dropped,
        }) + "\n")
        self.events.close()
