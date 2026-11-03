# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Latest-frame V4L2 capture with bounded startup, freshness and shutdown."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from multiprocessing.connection import Connection
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import numpy as np


@dataclass(frozen=True)
class CameraSpec:
    name: str
    device: str
    width: int
    height: int
    fps: float


def _worker_command(fd, spec):
    return [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', str(fd),
            spec.device, str(spec.width), str(spec.height), str(spec.fps)]


def _capture(fd, device, width, height, fps):
    # The parent owns cancellation; no robot libraries are imported here.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    sender = Connection(fd, readable=False)
    cap = None
    try:
        import cv2
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open RGB camera: {device}')
        for prop, value in ((cv2.CAP_PROP_FRAME_WIDTH, width),
                            (cv2.CAP_PROP_FRAME_HEIGHT, height),
                            (cv2.CAP_PROP_FPS, fps), (cv2.CAP_PROP_BUFFERSIZE, 1)):
            cap.set(prop, value)
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f'Camera read failed: {device}')
            if frame.dtype != np.uint8 or frame.shape != (height, width, 3):
                raise RuntimeError(f'Expected RGB {width}x{height}, got {frame.shape}: {device}')
            sender.send(('frame', time.monotonic(), frame))
    except (BrokenPipeError, EOFError):
        pass
    except Exception as exc:
        try:
            sender.send(('error', f'{type(exc).__name__}: {exc}'))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        if cap is not None:
            cap.release()
        sender.close()


class _Stream:
    def __init__(self, spec):
        self.spec = spec
        self.process = self.connection = self.thread = None
        self.lock = threading.Lock()
        self.frame = None
        self.stamp = self.first_stamp = 0.0
        self.count = 0
        self.max_gap = 0.0
        self.error = None
        self.closed = False

    def start(self):
        read_fd, write_fd = os.pipe()
        self.connection = Connection(read_fd, writable=False)
        try:
            self.process = subprocess.Popen(_worker_command(write_fd, self.spec),
                                            pass_fds=(write_fd,), start_new_session=True)
        except BaseException:
            self.connection.close()
            raise
        finally:
            os.close(write_fd)
        self.thread = threading.Thread(target=self._receive, daemon=True)
        self.thread.start()

    def _receive(self):
        try:
            while True:
                message = self.connection.recv()
                with self.lock:
                    if message[0] == 'error':
                        self.error = message[1]
                        return
                    _, stamp, frame = message
                    if frame.shape != (self.spec.height, self.spec.width, 3) or frame.dtype != np.uint8:
                        raise RuntimeError('Invalid camera worker frame')
                    if self.count:
                        self.max_gap = max(self.max_gap, stamp - self.stamp)
                    else:
                        self.first_stamp = stamp
                    self.stamp, self.frame = stamp, frame
                    self.count += 1
        except (EOFError, OSError, ValueError) as exc:
            if not self.closed:
                self.error = f'Camera worker disconnected: {exc}'
        except Exception as exc:
            self.error = str(exc)

    def close(self):
        self.closed = True
        if self.process is not None and self.process.poll() is None:
            for action in (self.process.terminate, self.process.kill):
                try:
                    action()
                except ProcessLookupError:
                    break
                try:
                    self.process.wait(timeout=0.4)
                    break
                except subprocess.TimeoutExpired:
                    continue
            else:
                print(f'Camera {self.spec.name} remains blocked in the kernel; check USB connection.', flush=True)
        if self.thread is not None:
            self.thread.join(timeout=0.3)
        if self.connection is not None:
            self.connection.close()


class CameraGroup:
    def __init__(self, specs, startup_timeout=12.0, frame_timeout=1.0):
        self.specs = list(specs)
        for name, value in (('camera_startup_timeout', startup_timeout),
                            ('camera_frame_timeout', frame_timeout)):
            if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        self.startup_timeout, self.frame_timeout = float(startup_timeout), float(frame_timeout)
        paths = [Path(s.device).resolve() for s in self.specs if s.device]
        if not self.specs or len(paths) != len(self.specs) or len(set(paths)) != len(paths):
            raise ValueError('Set distinct camera devices, including resolved symlinks')
        if len({s.name for s in self.specs}) != len(self.specs):
            raise ValueError('Camera names must differ')
        for spec, path in zip(self.specs, paths):
            if not path.exists():
                raise ValueError(f'Camera device missing: {spec.name}: {spec.device}')
            if spec.width < 1 or spec.height < 1 or not math.isfinite(spec.fps) or spec.fps <= 0:
                raise ValueError(f'Invalid resolution/frequency: {spec.name}')
        self.streams = [_Stream(s) for s in self.specs]

    def start(self):
        try:
            deadline = time.monotonic() + self.startup_timeout
            for stream in self.streams:
                print(f'Opening camera {stream.spec.name}: {stream.spec.device}', flush=True)
                stream.start()
            while True:
                for stream in self.streams:
                    if stream.error:
                        raise RuntimeError(f'{stream.spec.name}: {stream.error}')
                waiting = [s.spec.name for s in self.streams if s.frame is None]
                if not waiting:
                    self.frames()
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'Camera startup timed out after {self.startup_timeout:g}s: {waiting}')
                time.sleep(0.01)
        except BaseException:
            self.close()
            raise

    def frames(self):
        result = {}
        for stream in self.streams:
            with stream.lock:
                if stream.error:
                    raise RuntimeError(f'{stream.spec.name}: {stream.error}')
                if stream.process is None or stream.process.poll() is not None:
                    raise RuntimeError(f'Camera worker stopped: {stream.spec.name}')
                age = time.monotonic() - stream.stamp
                if stream.frame is None or age > self.frame_timeout:
                    raise TimeoutError(f'Camera {stream.spec.name} frame stale ({age:.3f}s); check USB/capture')
                result[stream.spec.name] = stream.frame.copy()
        return result

    def statistics(self):
        result = {}
        for s in self.streams:
            with s.lock:
                elapsed = s.stamp - s.first_stamp
                result[s.spec.name] = {'frames': s.count,
                    'fps': (s.count - 1) / elapsed if elapsed > 0 else 0,
                    'max_gap_s': s.max_gap, 'error': s.error}
        return result

    def close(self):
        for stream in self.streams:
            stream.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', nargs=5, required=True)
    args = parser.parse_args().worker
    _capture(int(args[0]), args[1], int(args[2]), int(args[3]), float(args[4]))
