# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Dual A1Z + three cameras, current A1Z wire protocol, safety SDK.

Without --execute this connects to the policy using synthetic observations,
validates responses and NEVER imports the SDK or opens cameras/CAN.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

import msgpack
import numpy as np
import websockets

from trajectory_smoother import DualArmCommandStreamer
from device_checks import cameras_from_config, check_can

DEFAULT_GRIPPER_BOUNDARY_TOLERANCE = 0.02

PARTS = {"left_arm": 6, "left_gripper": 1, "right_arm": 6, "right_gripper": 1}
CAMERAS = {"head_rgb": "camera_head", "left_wrist_rgb": "camera_left_wrist", "right_wrist_rgb": "camera_right_wrist"}


def encode(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf":
            raise ValueError("Unsupported array dtype")
        return {"__ndarray__": True, "data": value.tobytes(), "dtype": value.dtype.str, "shape": value.shape}
    if isinstance(value, np.generic):
        return {"__npgeneric__": True, "data": value.item(), "dtype": value.dtype.str}
    raise TypeError(type(value).__name__)


def decode(value):
    if value.get("__ndarray__"):
        dtype = np.dtype(value["dtype"])
        if dtype.kind not in "biuf":
            raise ValueError("Unsupported array dtype")
        return np.frombuffer(value["data"], dtype=dtype).reshape(value["shape"])
    if value.get("__npgeneric__"):
        return np.dtype(value["dtype"]).type(value["data"])
    return value


def pack(value):
    return msgpack.packb(value, default=encode, use_bin_type=True)


def unpack(value):
    if not isinstance(value, bytes):
        raise RuntimeError(f"Server returned non-binary response: {value}")
    return msgpack.unpackb(value, object_hook=decode, raw=False)


def validate_action(action, previous=None, max_step=0.10, gripper_tolerance=DEFAULT_GRIPPER_BOUNDARY_TOLERANCE):
    """Validate BOTH arms before sending either command; return L7/R7."""
    gripper_tolerance = float(gripper_tolerance)
    if not math.isfinite(gripper_tolerance) or gripper_tolerance < 0:
        raise ValueError("gripper_boundary_tolerance must be finite and nonnegative")
    values = {}
    for name, size in PARTS.items():
        a = np.asarray(action[name], dtype=np.float64)
        if a.shape != (size,) or not np.isfinite(a).all():
            raise ValueError(f"Invalid {name}: expected finite ({size},), got {a}")
        if "gripper" in name:
            value = float(a[0])
            if not -gripper_tolerance <= value <= 1 + gripper_tolerance:
                raise ValueError(
                    f"{name}={value:.6f} exceeds [0, 1] by more than "
                    f"{gripper_tolerance}; command refused"
                )
            clipped = float(np.clip(value, 0.0, 1.0))
            if clipped != value:
                print(f"{name}={value:.6f} clipped to {clipped:.6f} "
                      f"(boundary tolerance {gripper_tolerance})", flush=True)
            a = np.asarray([clipped], dtype=np.float64)
        values[name] = a
    left = np.concatenate([values["left_arm"], values["left_gripper"]])
    right = np.concatenate([values["right_arm"], values["right_gripper"]])
    if previous is not None:
        for name, target, old in zip(("left", "right"), (left, right), previous):
            delta = np.abs(target[:6] - old[:6])
            joint = int(np.argmax(delta))
            if delta[joint] > max_step:
                raise ValueError(
                    f"{name} joint{joint + 1} step {delta[joint]:.6f} rad exceeds "
                    f"{max_step:.6f}; reference={old[joint]:.6f}, "
                    f"target={target[joint]:.6f}; command refused"
                )
    return left, right


def validate_handshake(meta):
    if not isinstance(meta, dict):
        raise ValueError("Policy metadata must be an object")
    entries = meta.get("shape_meta", {}).get("action")
    if entries is None:
        entries = meta.get("shape_meta_by_embodiment", {}).get("galaxea_a1z", {}).get("action")
    actual = {str(item["key"]): int(item["raw_shape"]) for item in entries or []}
    if actual != PARTS:
        raise ValueError(f"Checkpoint action layout {actual} does not match dual-arm {PARTS}")


def validate_smoothing_config(cfg):
    smoothing = cfg.get("smoothing", {})
    if not isinstance(smoothing, dict):
        raise ValueError("smoothing must be an object")
    if not smoothing.get("enabled", False):
        return
    limits = {"frequency": 250, "max_velocity": 7,
              "max_acceleration": 20, "response_time": None,
              "stale_timeout": None}
    for name, maximum in limits.items():
        try:
            value = float(smoothing[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"smoothing.{name} must be numeric") from exc
        if not math.isfinite(value) or value <= 0 or (
                maximum is not None and value > maximum):
            raise ValueError(f"invalid smoothing.{name}: {value}")
    if float(smoothing["stale_timeout"]) <= 0.6:
        raise ValueError("smoothing.stale_timeout must exceed 0.6s")


class Hardware:
    def __init__(self, cfg, execute=False):
        self.cfg, self.arms, self.cameras = cfg, [], None
        self.execute = execute
        self.previous = None
        self.action_sent = False
        self.streamer = None

    def open(self):
        check_can(self.cfg)
        if self.cfg["left_can"] == self.cfg["right_can"]:
            raise ValueError("Left and right CAN channels must differ")
        devices = [self.cfg[field] for field in CAMERAS.values()]
        if None in devices or len(set(devices)) != 3:
            raise ValueError("Set three distinct camera devices in the local PC config")
        try:
            self.cameras = cameras_from_config(self.cfg)
            self.cameras.start()
            self.images()  # Fail before energizing arms if a camera is missing.
            check_can(self.cfg)
            print("All three camera frames verified; initializing arms", flush=True)
            from a1z.robots.get_robot import get_a1z_robot
            for channel in (self.cfg["left_can"], self.cfg["right_can"]):
                print(f"Starting arm on {channel}...", flush=True)
                arm = get_a1z_robot(can_channel=channel, with_gripper=True, zero_gravity_mode=False)
                self.arms.append(arm)
                arm.start()
            self.previous = tuple(np.asarray(arm.get_joint_pos(), dtype=np.float64).copy()
                                  for arm in self.arms)
            if any(joints.shape != (7,) or not np.isfinite(joints).all()
                   for joints in self.previous):
                raise RuntimeError("Invalid initial dual-arm state")
            smoothing = self.cfg.get("smoothing", {})
            if self.execute and smoothing.get("enabled", False):
                self.streamer = DualArmCommandStreamer(
                    self.arms, [joints[:6] for joints in self.previous],
                    frequency=float(smoothing["frequency"]),
                    max_velocity=float(smoothing["max_velocity"]),
                    max_acceleration=float(smoothing["max_acceleration"]),
                    response_time=float(smoothing["response_time"]),
                    stale_timeout=float(smoothing["stale_timeout"]),
                )
                self.streamer.start()
        except BaseException:
            self.close()
            raise

    def images(self):
        return {name: frame[:, :, ::-1].transpose(2, 0, 1).copy()
                for name, frame in self.cameras.frames().items()}

    def state(self):
        left, right = (arm.get_joint_pos().astype(np.float32) for arm in self.arms)
        return {"left_arm": left[:6], "left_gripper": left[6:7], "right_arm": right[:6], "right_gripper": right[6:7]}

    def send(self, action):
        max_step = (self.cfg.get("max_initial_joint_step", self.cfg["max_joint_step"])
                    if not self.action_sent else self.cfg["max_joint_step"])
        targets = validate_action(
            action, self.previous, float(max_step),
            self.cfg.get("gripper_boundary_tolerance", DEFAULT_GRIPPER_BOUNDARY_TOLERANCE),
        )
        for arm in self.arms:
            if not arm.is_running or arm.is_estopped:
                raise RuntimeError(f"SDK fault: {arm.get_fault_status()}")
        try:
            if self.streamer is not None:
                self.streamer.check()
                for arm, target in zip(self.arms, targets):
                    arm.command_gripper(float(target[6]))
                self.streamer.set_targets([target[:6] for target in targets])
            else:
                for arm, target in zip(self.arms, targets):
                    if not arm.command_joint_pos(target):
                        raise RuntimeError(f"SDK refused action: {arm.get_fault_status()}")
        except BaseException:
            # Dual-arm writes cannot be atomic. Latch both sides immediately if
            # either side rejects a command, before propagating the fault.
            self.hold()
            raise
        self.previous = targets
        self.action_sent = True

    def hold(self):
        if self.streamer is not None:
            try:
                self.streamer.stop()
            except Exception as exc:
                print(f"Dual-arm stream stop failed: {exc}", flush=True)
            self.streamer = None
        for arm in self.arms:
            try:
                arm.latch_last_command(reason="A1Z client stopped or failed")
            except Exception as exc:
                print(f"Hold failed: {exc}")

    def close(self):
        errors = []
        arm_count = len(self.arms)
        if self.action_sent or self.streamer is not None:
            try:
                self.hold()
            except BaseException as exc:
                errors.append(f"dual hold: {type(exc).__name__}: {exc}")
        for index, arm in enumerate(self.arms):
            side = "left" if index == 0 else "right"
            # A repeated Ctrl+C must not skip the other arm's stop transaction.
            for attempt in range(2):
                try:
                    print(f"Stopping {side} arm...", flush=True)
                    arm.stop()
                    break
                except KeyboardInterrupt:
                    print(f"Ctrl+C during {side} shutdown; finishing both arm stops", flush=True)
                    if attempt == 1:
                        errors.append(f"{side}: shutdown interrupted twice")
                except BaseException as exc:
                    errors.append(f"{side}: {type(exc).__name__}: {exc}")
                    break
            try:
                status = arm.get_fault_status()
                print(f"{side} arm shutdown status: {json.dumps(status, ensure_ascii=False)}", flush=True)
                if status.get("state") not in ("STOPPED", "HARD_DISABLED"):
                    errors.append(f"{side}: shutdown not confirmed, state={status.get('state')}")
            except BaseException as exc:
                errors.append(f"{side}: could not read shutdown state: {type(exc).__name__}: {exc}")
        self.arms.clear()
        if self.cameras is not None:
            try:
                self.cameras.close()
            except BaseException as exc:
                errors.append(f"cameras: {type(exc).__name__}: {exc}")
            self.cameras = None
        if errors:
            print("Shutdown needs attention: " + "; ".join(errors), flush=True)
        elif arm_count:
            print(f"SDK stop transactions completed for {arm_count} arm(s); physical motor disable is not independently verified.", flush=True)


async def run(cfg, execute=False, observe_only=False):
    if cfg["control_frequency"] <= 0 or cfg["request_timeout"] <= 0 or cfg["max_joint_step"] <= 0:
        raise ValueError("Frequency, timeout and max_joint_step must be positive")
    initial_step = float(cfg.get("max_initial_joint_step", cfg["max_joint_step"]))
    if not math.isfinite(initial_step) or initial_step <= 0:
        raise ValueError("max_initial_joint_step must be finite and positive")
    if cfg.get("websocket_compression") not in (None, "deflate"):
        raise ValueError("websocket_compression must be null or deflate")
    validate_smoothing_config(cfg)
    tolerance = float(cfg.get("gripper_boundary_tolerance", DEFAULT_GRIPPER_BOUNDARY_TOLERANCE))
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("gripper_boundary_tolerance must be finite and nonnegative")
    hardware = Hardware(cfg, execute=execute) if execute or observe_only else None
    width, height = cfg["image_size"]
    cycles = 0
    try:
        # Local RGB observations are large; deflate adds avoidable CPU latency.
        compression = cfg.get("websocket_compression", None if cfg["server_host"] in
                              ("127.0.0.1", "localhost", "::1") else "deflate")
        async with websockets.connect(f"ws://{cfg['server_host']}:{cfg['server_port']}",
                                      max_size=64 * 1024**2, compression=compression) as ws:
            meta = unpack(await asyncio.wait_for(ws.recv(), cfg["request_timeout"]))
            validate_handshake(meta)
            print("Policy metadata:", meta)
            if hardware:
                hardware.open()
            try:
                need_obs = True
                while True:
                    started = time.monotonic()
                    obs = {}
                    requested_observation = need_obs
                    if need_obs:
                        obs = {
                            "images": hardware.images() if hardware else {k: np.zeros((3, height, width), np.uint8) for k in CAMERAS},
                            "state": hardware.state() if hardware else {k: np.zeros(n, np.float32) for k, n in PARTS.items()},
                            "task": cfg["task"], "frequency": cfg["control_frequency"],
                            "embodiment_type": cfg["embodiment_type"],
                        }
                    observation_ms = (time.monotonic() - started) * 1000
                    await asyncio.wait_for(ws.send(pack(obs)), cfg["request_timeout"])
                    response = unpack(await asyncio.wait_for(ws.recv(), cfg["request_timeout"]))
                    if "error" in response:
                        raise RuntimeError(response["error"])
                    action = response["action"]
                    if requested_observation:
                        print(f"policy refresh step {cycles}: observation={observation_ms:.1f}ms, "
                              f"request_and_reply={(time.monotonic() - started) * 1000 - observation_ms:.1f}ms, "
                              f"total={(time.monotonic() - started) * 1000:.1f}ms", flush=True)
                    if execute:
                        previous = hardware.previous
                        first_action = not hardware.action_sent
                        try:
                            hardware.send(action)
                        except ValueError as exc:
                            reference = "initial joint feedback" if first_action else "previous policy target"
                            raise ValueError(f"execute step {cycles} ({reference}): {exc}") from exc
                        deltas = [float(np.max(np.abs(new[:6] - old[:6])))
                                  for new, old in zip(hardware.previous, previous)]
                        print(f"execute step {cycles}: sent, "
                              f"left_delta={deltas[0]:.4f} rad, right_delta={deltas[1]:.4f} rad, "
                              f"grippers=({hardware.previous[0][6]:.4f}, "
                              f"{hardware.previous[1][6]:.4f})", flush=True)
                    else:
                        validate_action(action, gripper_tolerance=cfg.get(
                            "gripper_boundary_tolerance", DEFAULT_GRIPPER_BOUNDARY_TOLERANCE))
                        mode = "observe-only" if observe_only else "dry-run"
                        print(f"{mode} step {cycles}: valid dual-arm action (not executed)")
                    need_obs = bool(response.get("need_obs", True))
                    cycles += 1
                    limit = cfg["max_cycles"] or (0 if execute else 3)
                    if limit and cycles >= limit:
                        break
                    await asyncio.sleep(max(0, 1 / cfg["control_frequency"] - (time.monotonic() - started)))
            finally:
                # Stop hardware before waiting for the WebSocket close handshake.
                if hardware:
                    hardware.close()
                    hardware = None
    finally:
        if hardware:
            hardware.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / "configs/deployment/pc-direct.local.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--observe-only", action="store_true", help="Read real cameras/joints but never send model actions")
    mode.add_argument("--execute", action="store_true", help="Enable real cameras and execute model actions")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    try:
        asyncio.run(run(cfg, args.execute, args.observe_only))
    except KeyboardInterrupt:
        print("Stopped; no automatic return-to-zero motion.")


if __name__ == "__main__":
    main()
