# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Single-left A1Z π0.5 client: head/wrist cameras and one SocketCAN arm.

The default mode sends synthetic observations and never opens hardware.
Use --observe-only to read real hardware without model commands, then
--execute only with an operator present.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

import numpy as np
import websockets
from device_checks import cameras_from_config, check_can
from single_config import normalize_single_config

from dual_arm_client import DEFAULT_GRIPPER_BOUNDARY_TOLERANCE, pack, unpack
from diagnostic_recorder import DiagnosticRecorder
from trajectory_smoother import ArmCommandStreamer


PARTS = {"left_arm": 6, "left_gripper": 1}
CAMERAS = {"head_rgb": "camera_head", "left_wrist_rgb": "camera_left_wrist"}


def validate_config(cfg: dict) -> None:
    if cfg.get("topology") != "pc-direct-single-left":
        raise ValueError("Use topology=pc-direct-single")
    if cfg.get("embodiment_type") != "galaxea_a1z":
        raise ValueError("embodiment_type must be galaxea_a1z")
    if not str(cfg.get("task", "")).strip():
        raise ValueError("task must be set")
    for key in ("control_frequency", "request_timeout", "max_joint_step"):
        value = float(cfg[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    initial_step = float(cfg.get("max_initial_joint_step", cfg["max_joint_step"]))
    if not math.isfinite(initial_step) or initial_step <= 0:
        raise ValueError("max_initial_joint_step must be finite and positive")
    gripper_tolerance = float(cfg.get("gripper_boundary_tolerance", DEFAULT_GRIPPER_BOUNDARY_TOLERANCE))
    if not math.isfinite(gripper_tolerance) or gripper_tolerance < 0:
        raise ValueError("gripper_boundary_tolerance must be finite and nonnegative")
    if not cfg.get("server_host") or not 0 < int(cfg["server_port"]) < 65536:
        raise ValueError("Set server_host and server_port")
    size = cfg["image_size"]
    if len(size) != 2 or any(int(x) <= 0 for x in size):
        raise ValueError("image_size must be [width, height]")
    if int(cfg.get("max_cycles", 0)) < 0:
        raise ValueError("max_cycles must be nonnegative")
    if cfg.get("websocket_compression") not in (None, "deflate"):
        raise ValueError("websocket_compression must be null or deflate")
    smoothing = cfg.get("smoothing", {})
    if not isinstance(smoothing, dict):
        raise ValueError("smoothing must be an object")
    if smoothing.get("enabled", False):
        for key in ("frequency", "max_velocity", "max_acceleration",
                    "response_time", "stale_timeout"):
            try:
                value = float(smoothing[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"smoothing.{key} is required and must be numeric") from exc
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"smoothing.{key} must be finite and positive")
        if float(smoothing["frequency"]) > 250:
            raise ValueError("smoothing.frequency must not exceed SDK control rate 250 Hz")
        if float(smoothing["max_velocity"]) > 7:
            raise ValueError("smoothing.max_velocity must not exceed SDK's 7 rad/s joint cap")
        if float(smoothing["max_acceleration"]) > 20:
            raise ValueError("smoothing.max_acceleration must not exceed SDK's 20 rad/s^2 cap")
        if float(smoothing["stale_timeout"]) <= 0.6:
            raise ValueError("smoothing.stale_timeout must exceed normal 0.6s recompute")


def validate_hardware_config(cfg: dict) -> None:
    if not cfg.get("left_can"):
        raise ValueError("Set left_can")
    paths = [cfg.get(key) for key in CAMERAS.values()]
    if any(not value for value in paths) or len(set(paths)) != 2:
        raise ValueError("Set two distinct camera devices")


def validate_action(action: dict, previous: np.ndarray | None = None,
                    max_step: float = 0.1,
                    gripper_tolerance: float = DEFAULT_GRIPPER_BOUNDARY_TOLERANCE) -> np.ndarray:
    if not isinstance(action, dict):
        raise ValueError("Policy action must be a dictionary")
    values = []
    for key, dim in PARTS.items():
        if key not in action:
            raise ValueError(f"Policy omitted {key}; refusing to execute")
        value = np.asarray(action[key], dtype=np.float64)
        if value.shape != (dim,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid {key}: expected finite ({dim},), got {value}")
        values.append(value)
    gripper = float(values[1][0])
    if not -gripper_tolerance <= gripper <= 1 + gripper_tolerance:
        raise ValueError(
            f"left_gripper={gripper:.6f} exceeds [0, 1] by more than "
            f"{gripper_tolerance}; command refused"
        )
    values[1] = np.clip(values[1], 0.0, 1.0)
    target = np.concatenate(values)
    if previous is not None:
        delta = np.abs(target[:6] - previous[:6])
        joint = int(np.argmax(delta))
        if delta[joint] > max_step:
            raise ValueError(
                f"left joint{joint + 1} step {delta[joint]:.3f} rad exceeds "
                f"{max_step:.3f}; current={previous[joint]:.3f}, "
                f"target={target[joint]:.3f}; command refused"
            )
    return target


def validate_handshake(meta: dict) -> None:
    entries = meta.get("shape_meta", {}).get("action")
    if entries is None:
        by_embodiment = meta.get("shape_meta_by_embodiment", {})
        entries = by_embodiment.get("galaxea_a1z", {}).get("action")
    actual = {str(item["key"]): int(item["raw_shape"]) for item in entries or []}
    if actual != PARTS:
        raise ValueError(f"Checkpoint action layout {actual} does not match single {PARTS}")


class Hardware:
    def __init__(self, cfg: dict, execute: bool = False):
        self.cfg = cfg
        self.execute = execute
        self.arm = None
        self.cameras = None
        self.previous = None
        self.action_sent = False
        self.streamer = None

    def open(self) -> None:
        check_can(self.cfg)
        try:
            self.cameras = cameras_from_config(self.cfg)
            self.cameras.start()
            self.images()  # Verify both streams before enabling motors.
            check_can(self.cfg)
            print("Both camera frames verified; initializing arm", flush=True)
            from a1z.robots.get_robot import get_a1z_robot
            self.arm = get_a1z_robot(can_channel=self.cfg["left_can"],
                                     with_gripper=True, zero_gravity_mode=False)
            self.arm.start()
            self.previous = np.asarray(self.arm.get_joint_pos(), dtype=np.float64)
            if self.previous.shape != (7,) or not np.isfinite(self.previous).all():
                raise RuntimeError("Invalid initial left arm state")
            smoothing = self.cfg.get("smoothing", {})
            if self.execute and smoothing.get("enabled", False):
                self.streamer = ArmCommandStreamer(
                    self.arm, self.previous[:6],
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

    def images(self) -> dict:
        return {name: frame[:, :, ::-1].transpose(2, 0, 1).copy()
                for name, frame in self.cameras.frames().items()}

    def state(self) -> dict:
        value = np.asarray(self.arm.get_joint_pos(), dtype=np.float32)
        if value.shape != (7,) or not np.isfinite(value).all():
            raise RuntimeError("Invalid left arm feedback")
        return {"left_arm": value[:6], "left_gripper": value[6:7]}

    def send(self, action: dict) -> None:
        max_step = (self.cfg.get("max_initial_joint_step", self.cfg["max_joint_step"])
                    if not self.action_sent else self.cfg["max_joint_step"])
        target = validate_action(
            action, self.previous, float(max_step),
            float(self.cfg.get("gripper_boundary_tolerance", DEFAULT_GRIPPER_BOUNDARY_TOLERANCE)),
        )
        if not self.arm.is_running or self.arm.is_estopped:
            raise RuntimeError(f"SDK fault: {self.arm.get_fault_status()}")
        if self.streamer is not None:
            self.streamer.set_target(target[:6])
            self.arm.command_gripper(float(target[6]))
        elif not self.arm.command_joint_pos(target):
            self.hold()
            raise RuntimeError(f"SDK refused action: {self.arm.get_fault_status()}")
        self.previous = target
        self.action_sent = True

    def hold(self) -> None:
        if self.streamer is not None:
            try:
                self.streamer.stop()
            except Exception as exc:
                print(f"Stream stop failed: {exc}", flush=True)
            self.streamer = None
        if self.arm is not None:
            try:
                self.arm.latch_last_command(reason="A1Z single client stopped")
            except Exception as exc:
                print(f"Hold failed: {exc}", flush=True)

    def close(self) -> None:
        errors = []
        had_arm = self.arm is not None
        if self.streamer is not None:
            for attempt in range(2):
                try:
                    self.streamer.stop()
                    print(f"Smooth stream: {self.streamer.ticks} ticks, "
                          f"max_lateness={self.streamer.max_lateness:.4f}s",
                          flush=True)
                    break
                except KeyboardInterrupt:
                    print("Ctrl+C during stream shutdown; finishing device cleanup", flush=True)
                    if attempt == 1:
                        errors.append("stream: shutdown interrupted twice")
                except BaseException as exc:
                    errors.append(f"stream: {type(exc).__name__}: {exc}")
                    break
            self.streamer = None
        if self.arm is not None:
            if self.action_sent:
                try:
                    self.arm.latch_last_command(reason="A1Z single client stopped")
                except BaseException as exc:
                    errors.append(f"final hold: {type(exc).__name__}: {exc}")
            # A repeated Ctrl+C must not skip SDK stop or camera cleanup.
            for attempt in range(2):
                try:
                    print("Stopping single arm...", flush=True)
                    self.arm.stop()
                    break
                except KeyboardInterrupt:
                    print("Ctrl+C during arm shutdown; finishing device cleanup", flush=True)
                    if attempt == 1:
                        errors.append("arm: shutdown interrupted twice")
                except BaseException as exc:
                    errors.append(f"arm: {type(exc).__name__}: {exc}")
                    break
            try:
                status = self.arm.get_fault_status()
                print(f"single arm shutdown status: {json.dumps(status, ensure_ascii=False)}", flush=True)
                if status.get("state") not in ("STOPPED", "HARD_DISABLED"):
                    errors.append(f"arm: shutdown not confirmed, state={status.get('state')}")
            except BaseException as exc:
                errors.append(f"arm: could not read shutdown state: {type(exc).__name__}: {exc}")
            self.arm = None
        if self.cameras is not None:
            for attempt in range(2):
                try:
                    self.cameras.close()
                    break
                except KeyboardInterrupt:
                    print("Ctrl+C during camera shutdown; finishing device cleanup", flush=True)
                    if attempt == 1:
                        errors.append("cameras: shutdown interrupted twice")
                except BaseException as exc:
                    errors.append(f"cameras: {type(exc).__name__}: {exc}")
                    break
            self.cameras = None
        if errors:
            print("Shutdown needs attention: " + "; ".join(errors), flush=True)
        elif had_arm:
            print("SDK stop transaction completed for single arm; physical motor disable is not independently verified.", flush=True)


async def run(cfg: dict, *, execute: bool, observe_only: bool,
              record_dir: Path | None = None, record_every: int = 4,
              record_max_frames: int = 2400) -> None:
    validate_config(cfg)
    if execute or observe_only:
        validate_hardware_config(cfg)
    print(
        "Client tuning: "
        f"control_frequency={cfg['control_frequency']} Hz, "
        f"max_initial_joint_step={cfg.get('max_initial_joint_step', cfg['max_joint_step'])} rad, "
        f"max_joint_step={cfg['max_joint_step']} rad, "
        f"gripper_boundary_tolerance={cfg.get('gripper_boundary_tolerance', DEFAULT_GRIPPER_BOUNDARY_TOLERANCE)}, "
        f"max_cycles={cfg.get('max_cycles', 0)}",
        flush=True,
    )
    smoothing = cfg.get("smoothing", {})
    if execute and smoothing.get("enabled", False):
        print(
            "250Hz smoothing: "
            f"frequency={smoothing['frequency']}Hz, "
            f"max_velocity={smoothing['max_velocity']}rad/s, "
            f"max_acceleration={smoothing['max_acceleration']}rad/s^2, "
            f"response_time={smoothing['response_time']}s, "
            f"stale_timeout={smoothing['stale_timeout']}s",
            flush=True,
        )
    hardware = Hardware(cfg, execute=execute) if execute or observe_only else None
    if record_dir is not None and hardware is None:
        raise ValueError("Recording requires --execute or --observe-only")
    recorder = (DiagnosticRecorder(record_dir, record_every, record_max_frames)
                if record_dir is not None else None)
    if recorder is not None:
        print(f"Diagnostic recording: {recorder.directory}", flush=True)
    width, height = map(int, cfg["image_size"])
    cycles = 0
    try:
        uri = f"ws://{cfg['server_host']}:{cfg['server_port']}"
        compression = cfg.get("websocket_compression", None if cfg["server_host"] in
                              ("127.0.0.1", "localhost", "::1") else "deflate")
        async with websockets.connect(uri, max_size=64 * 1024**2,
                                      compression=compression) as ws:
            try:
                meta = unpack(await asyncio.wait_for(ws.recv(), cfg["request_timeout"]))
                validate_handshake(meta)
                print("Single policy metadata:", meta, flush=True)
                if hardware is not None:
                    hardware.open()
                need_obs = True
                while True:
                    started = time.monotonic()
                    obs = {}
                    images = None
                    observed_state = None
                    frame_name = None
                    if hardware is not None and (
                        need_obs or (recorder is not None
                                     and recorder.frames < recorder.max_frames
                                     and cycles % recorder.every == 0)
                    ):
                        images = hardware.images()
                    if need_obs:
                        observed_state = (hardware.state() if hardware else
                                          {key: np.zeros(dim, np.float32)
                                           for key, dim in PARTS.items()})
                        obs = {
                            "images": images if hardware else
                                {key: np.zeros((3, height, width), np.uint8) for key in CAMERAS},
                            "state": observed_state,
                            "task": cfg["task"],
                            "frequency": cfg["control_frequency"],
                            "embodiment_type": cfg["embodiment_type"],
                        }
                    if recorder is not None and images is not None:
                        frame_name = recorder.frame(
                            cycles, images, model_observation=need_obs
                        )
                    response_started = time.monotonic()
                    await asyncio.wait_for(ws.send(pack(obs)), cfg["request_timeout"])
                    response = unpack(await asyncio.wait_for(ws.recv(), cfg["request_timeout"]))
                    response_seconds = time.monotonic() - response_started
                    if "error" in response:
                        raise RuntimeError(response["error"])
                    action = response["action"]
                    max_step = (cfg.get("max_initial_joint_step", cfg["max_joint_step"])
                                if execute and not hardware.action_sent else cfg["max_joint_step"])
                    gripper_tolerance = float(cfg.get(
                        "gripper_boundary_tolerance", DEFAULT_GRIPPER_BOUNDARY_TOLERANCE
                    ))
                    target = validate_action(
                        action, hardware.previous if execute else None,
                        float(max_step), gripper_tolerance,
                    )
                    raw_gripper = float(np.asarray(action["left_gripper"])[0])
                    if raw_gripper != target[6]:
                        print(
                            f"left_gripper={raw_gripper:.6f} clipped to {target[6]:.6f} "
                            f"(boundary tolerance {gripper_tolerance})",
                            flush=True,
                        )
                    previous_command = hardware.previous.copy() if execute else None
                    if execute:
                        max_delta = float(np.max(np.abs(target[:6] - hardware.previous[:6])))
                        hardware.send(action)
                        print(
                            f"execute step {cycles}: sent, max_joint_delta={max_delta:.3f} rad, "
                            f"gripper={target[6]:.3f}",
                            flush=True,
                        )
                    else:
                        mode = "observe-only" if observe_only else "dry-run"
                        if observe_only:
                            current = hardware.state()
                            delta = target[:6] - current["left_arm"]
                            print(
                                f"{mode} step {cycles}: current={np.round(current['left_arm'], 3).tolist()} "
                                f"target={np.round(target[:6], 3).tolist()} "
                                f"delta={np.round(delta, 3).tolist()} "
                                f"max_abs_delta={np.max(np.abs(delta)):.3f} rad; not executed",
                                flush=True,
                            )
                        else:
                            print(f"{mode} step {cycles}: valid single action; not executed", flush=True)
                    if recorder is not None:
                        recorder.action(
                            cycles, target, previous_command, frame=frame_name,
                            model_observation=need_obs, observed_state=observed_state,
                            response_seconds=response_seconds,
                        )
                    need_obs = bool(response.get("need_obs", True))
                    cycles += 1
                    limit = int(cfg.get("max_cycles", 0)) or (0 if execute else 3)
                    if limit and cycles >= limit:
                        print(f"Completed {cycles} cycles; stopping hardware.", flush=True)
                        break
                    await asyncio.sleep(max(0, 1 / cfg["control_frequency"] -
                                            (time.monotonic() - started)))
            finally:
                # Stop devices before waiting for the WebSocket close handshake.
                if hardware is not None:
                    hardware.close()
                    hardware = None

    finally:
        if hardware is not None:
            hardware.close()
        if recorder is not None:
            try:
                recorder.close()
                print(f"Recording saved: {recorder.directory} "
                      f"({recorder.frames} frames, {recorder.dropped} dropped)",
                      flush=True)
            except Exception as exc:
                print(f"Recording finalization failed: {exc}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="Validate config without opening hardware or server")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--observe-only", action="store_true", help="Read hardware but do not send model commands")
    mode.add_argument("--execute", action="store_true", help="Enable model actions on the single arm")
    parser.add_argument("--record-dir", type=Path,
                        help="Save timestamped dual-camera JPEGs and action trace")
    parser.add_argument("--record-every", type=int, default=4,
                        help="Additional camera sample every N action steps")
    parser.add_argument("--record-max-frames", type=int, default=2400)
    args = parser.parse_args()
    cfg = normalize_single_config(json.loads(args.config.read_text()))
    validate_config(cfg)
    if args.execute or args.observe_only:
        validate_hardware_config(cfg)
    if args.check:
        print("Single configuration OK; hardware and server not opened")
        return
    try:
        asyncio.run(run(cfg, execute=args.execute, observe_only=args.observe_only,
                        record_dir=args.record_dir, record_every=args.record_every,
                        record_max_frames=args.record_max_frames))
    except KeyboardInterrupt:
        print("Stopped; no automatic return-to-zero motion.", flush=True)


if __name__ == "__main__":
    main()
