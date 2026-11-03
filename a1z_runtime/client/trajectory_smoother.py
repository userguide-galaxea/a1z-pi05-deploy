# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Bounded 250 Hz joint target interpolation for single-arm and dual-arm PC clients.

The stream holds the last target during a policy recompute. It never predicts
future motion from stale observations.

Limits apply to the reference state on the integration clock, not measured
motor motion. Scheduler stalls are not replayed: each advance uses at most 20 ms.
"""
from __future__ import annotations

import threading
import time

import numpy as np


class BoundedJointInterpolator:
    def __init__(self, initial, max_velocity: float, max_acceleration: float,
                 response_time: float = 0.04):
        self.position = np.asarray(initial, dtype=np.float64).copy()
        if self.position.shape != (6,) or not np.isfinite(self.position).all():
            raise ValueError("Initial joint state must be six finite radians")
        if not all(np.isfinite(x) and x > 0 for x in
                   (max_velocity, max_acceleration, response_time)):
            raise ValueError("Smoothing parameters must be finite and positive")
        self.target = self.position.copy()
        self.velocity = np.zeros(6, dtype=np.float64)
        self.max_velocity = float(max_velocity)
        self.max_acceleration = float(max_acceleration)
        self.response_time = float(response_time)

    def set_target(self, target) -> None:
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (6,) or not np.isfinite(target).all():
            raise ValueError("Target must be six finite radians")
        # Retarget without resetting velocity: reversals must brake first.
        self.target = target.copy()

    def advance(self, dt: float) -> np.ndarray:
        if not np.isfinite(dt) or dt <= 0:
            return self.position.copy()
        dt = min(float(dt), 0.02)  # No large catch-up jump after scheduler stalls.
        error = self.target - self.position
        desired = np.clip(error / self.response_time,
                          -self.max_velocity, self.max_velocity)
        # Reserve the next position step as well as the braking distance.
        # The rational form avoids cancellation near the target.
        brake_step = self.max_acceleration * dt
        stopping_term = 2 * self.max_acceleration * np.abs(error)
        braking_speed = stopping_term / (
            np.sqrt(brake_step * brake_step + stopping_term) + brake_step
        )
        desired = np.sign(error) * np.minimum(np.abs(desired), braking_speed)
        self.velocity += np.clip(
            desired - self.velocity,
            -self.max_acceleration * dt,
            self.max_acceleration * dt,
        )
        # An abrupt retarget can be inside the current stopping distance.
        # Continue bounded braking instead of snapping position or velocity.
        self.position += self.velocity * dt
        return self.position.copy()


class ArmCommandStreamer:
    def __init__(self, arm, initial, *, frequency: float = 250.0,
                 max_velocity: float = 3.0, max_acceleration: float = 20.0,
                 response_time: float = 0.04, stale_timeout: float = 1.0):
        if not np.isfinite(frequency) or frequency <= 0:
            raise ValueError("Stream frequency must be finite and positive")
        if not np.isfinite(stale_timeout) or stale_timeout <= 0:
            raise ValueError("Stale timeout must be finite and positive")
        self.arm = arm
        self.interpolator = BoundedJointInterpolator(
            initial, max_velocity, max_acceleration, response_time
        )
        self.period = 1.0 / float(frequency)
        self.stale_timeout = float(stale_timeout)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="a1z-arm-stream",
                                       daemon=True)
        self.last_target_time = None
        self.error = None
        self.ticks = 0
        self.max_lateness = 0.0

    def start(self) -> None:
        self.thread.start()

    def set_target(self, target) -> None:
        self.check()
        with self.lock:
            self.interpolator.set_target(target)
            self.last_target_time = time.monotonic()

    def check(self) -> None:
        if self.error is not None:
            raise RuntimeError(f"Arm command streamer stopped: {self.error}")
        if self.thread.ident is not None and not self.thread.is_alive():
            raise RuntimeError("Arm command streamer exited unexpectedly")

    def _run(self) -> None:
        deadline = time.monotonic()
        previous = deadline
        try:
            while not self.stop_event.is_set():
                now = time.monotonic()
                with self.lock:
                    last_target = self.last_target_time
                    position = self.interpolator.advance(now - previous)
                previous = now
                if last_target is not None and now - last_target > self.stale_timeout:
                    raise RuntimeError(
                        f"policy actions stale for {now - last_target:.2f}s"
                    )
                if not self.arm.is_running or self.arm.is_estopped:
                    raise RuntimeError(f"SDK fault: {self.arm.get_fault_status()}")
                if not self.arm.command_joint_pos(position):
                    raise RuntimeError(f"SDK refused target: {self.arm.get_fault_status()}")
                self.ticks += 1
                deadline += self.period
                lateness = time.monotonic() - deadline
                self.max_lateness = max(self.max_lateness, lateness)
                if lateness > self.period:
                    deadline = time.monotonic() + self.period
                self.stop_event.wait(max(0.0, deadline - time.monotonic()))
        except Exception as exc:
            self.error = exc
            print(f"[A1Z] smoothing stopped; holding last command: {exc}", flush=True)
            try:
                self.arm.latch_last_command(reason=f"A1Z stream stopped: {exc}")
            except Exception:
                pass
            self.stop_event.set()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                raise RuntimeError("Arm command streamer did not stop in 2s")


class DualArmCommandStreamer:
    """Stream both arms from one clock and stop both if either side fails."""

    def __init__(self, arms, initial, *, frequency: float = 250.0,
                 max_velocity: float = 3.0, max_acceleration: float = 20.0,
                 response_time: float = 0.04, stale_timeout: float = 1.0):
        if len(arms) != 2 or len(initial) != 2 or arms[0] is arms[1]:
            raise ValueError("Two distinct arms and initial joint states are required")
        if not np.isfinite(frequency) or frequency <= 0:
            raise ValueError("Stream frequency must be finite and positive")
        if not np.isfinite(stale_timeout) or stale_timeout <= 0:
            raise ValueError("Stale timeout must be finite and positive")
        self.arms = tuple(arms)
        self.interpolators = tuple(
            BoundedJointInterpolator(joints, max_velocity, max_acceleration,
                                     response_time) for joints in initial
        )
        self.period = 1.0 / float(frequency)
        self.stale_timeout = float(stale_timeout)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="a1z-dual-arm-stream",
                                       daemon=True)
        self.last_target_time = None
        self.error = None
        self.ticks = 0
        self.max_lateness = 0.0

    def start(self) -> None:
        self.thread.start()

    def set_targets(self, targets) -> None:
        self.check()
        if len(targets) != 2:
            raise ValueError("Two joint targets are required")
        # Validate both before updating either interpolator.
        vectors = tuple(np.asarray(target, dtype=np.float64) for target in targets)
        if any(v.shape != (6,) or not np.isfinite(v).all() for v in vectors):
            raise ValueError("Both joint targets must contain six finite radians")
        with self.lock:
            for interpolator, target in zip(self.interpolators, vectors):
                interpolator.set_target(target)
            self.last_target_time = time.monotonic()

    def check(self) -> None:
        if self.error is not None:
            raise RuntimeError(f"Dual-arm command streamer stopped: {self.error}")
        if self.thread.ident is not None and not self.thread.is_alive():
            raise RuntimeError("Dual-arm command streamer exited unexpectedly")

    def _run(self) -> None:
        deadline = time.monotonic()
        previous = deadline
        try:
            while not self.stop_event.is_set():
                now = time.monotonic()
                with self.lock:
                    last_target = self.last_target_time
                    positions = tuple(interpolator.advance(now - previous)
                                      for interpolator in self.interpolators)
                previous = now
                if last_target is not None and now - last_target > self.stale_timeout:
                    raise RuntimeError(f"policy actions stale for {now - last_target:.2f}s")
                for arm in self.arms:
                    if not arm.is_running or arm.is_estopped:
                        raise RuntimeError(f"SDK fault: {arm.get_fault_status()}")
                for arm, position in zip(self.arms, positions):
                    if not arm.command_joint_pos(position):
                        raise RuntimeError(f"SDK refused target: {arm.get_fault_status()}")
                self.ticks += 1
                deadline += self.period
                lateness = time.monotonic() - deadline
                self.max_lateness = max(self.max_lateness, lateness)
                if lateness > self.period:
                    deadline = time.monotonic() + self.period
                self.stop_event.wait(max(0.0, deadline - time.monotonic()))
        except Exception as exc:
            self.error = exc
            print(f"[A1Z] dual-arm smoothing stopped; holding both arms: {exc}", flush=True)
            for arm in self.arms:
                try:
                    arm.latch_last_command(reason=f"A1Z dual stream stopped: {exc}")
                except Exception as hold_exc:
                    print(f"[A1Z] dual-arm hold failed: {hold_exc}", flush=True)
            self.stop_event.set()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                raise RuntimeError("Dual-arm command streamer did not stop in 2s")
