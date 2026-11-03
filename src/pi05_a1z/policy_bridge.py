# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Serve openpi actions to the existing A1Z single/dual robot clients.

The bridge only translates protocols. Joint limits, command-step checks, motor
shutdown and 250 Hz smoothing remain in the robot client and SDK. Each client
connection owns its own upstream socket and action queue. Reset discards the
queue and reconnects; openpi's standard pi0.5 policy is observation-stateless and
its WebsocketClientPolicy.reset() does not send a server-side reset request.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import logging
import math
import threading
from typing import Any, Protocol

import msgpack
import numpy as np

LOGGER = logging.getLogger(__name__)
MAX_MESSAGE_BYTES = 64 * 1024**2
PARTS = {
    "single": {"left_arm": 6, "left_gripper": 1},
    "dual": {"left_arm": 6, "left_gripper": 1, "right_arm": 6, "right_gripper": 1},
}
CAMERAS = {
    "head_rgb": "cam_high",
    "left_wrist_rgb": "cam_left_wrist",
    "right_wrist_rgb": "cam_right_wrist",
}


def _encode(value: Any) -> dict:
    """The downstream A1Z protocol uses string keys for NumPy envelopes."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf":
            raise ValueError(f"Unsupported array dtype: {value.dtype}")
        return {"__ndarray__": True, "data": value.tobytes(),
                "dtype": value.dtype.str, "shape": value.shape}
    if isinstance(value, np.generic) and value.dtype.kind in "biuf":
        return {"__npgeneric__": True, "data": value.item(), "dtype": value.dtype.str}
    raise TypeError(f"Unsupported message value: {type(value).__name__}")


def _decode(value: dict) -> Any:
    if value.get("__ndarray__"):
        dtype = np.dtype(value["dtype"])
        if dtype.kind not in "biuf":
            raise ValueError(f"Unsupported array dtype: {dtype}")
        return np.frombuffer(value["data"], dtype=dtype).reshape(value["shape"])
    if value.get("__npgeneric__"):
        dtype = np.dtype(value["dtype"])
        if dtype.kind not in "biuf":
            raise ValueError(f"Unsupported scalar dtype: {dtype}")
        return dtype.type(value["data"])
    return value


def pack(value: Any) -> bytes:
    return msgpack.packb(value, default=_encode, use_bin_type=True)


def unpack(value: bytes) -> dict:
    if not isinstance(value, bytes):
        raise ValueError("Expected a binary A1Z protocol message")
    result = msgpack.unpackb(value, object_hook=_decode, raw=False)
    if not isinstance(result, dict):
        raise ValueError("Expected an A1Z protocol object")
    return result


@dataclass(frozen=True)
class BridgeConfig:
    mode: str = "single"
    action_steps: int = 16
    gripper_tolerance: float = 0.02
    upstream_timeout: float = 30.0

    def __post_init__(self) -> None:
        if self.mode not in PARTS:
            raise ValueError("mode must be single or dual")
        if isinstance(self.action_steps, bool) or not isinstance(self.action_steps, int) or self.action_steps < 1:
            raise ValueError("action_steps must be a positive integer")
        if not math.isfinite(self.gripper_tolerance) or self.gripper_tolerance < 0:
            raise ValueError("gripper_tolerance must be finite and nonnegative")
        if not math.isfinite(self.upstream_timeout) or self.upstream_timeout <= 0:
            raise ValueError("upstream_timeout must be finite and positive")

    @property
    def action_dim(self) -> int:
        return sum(PARTS[self.mode].values())

    def metadata(self) -> dict:
        layout = {"action": [{"key": key, "raw_shape": size}
                             for key, size in PARTS[self.mode].items()]}
        return {"model_family": "pi05", "mode": self.mode,
                "action_steps": self.action_steps, "shape_meta": layout,
                "shape_meta_by_embodiment": {"galaxea_a1z": layout}}


class UpstreamPolicy(Protocol):
    def infer(self, observation: dict) -> dict: ...
    def get_server_metadata(self) -> dict: ...
    def close(self) -> None: ...


def observation_for_openpi(observation: dict, config: BridgeConfig) -> dict:
    """Convert A1Z CHW RGB observations without ALOHA calibration or resizing."""
    if observation.get("embodiment_type") != "galaxea_a1z":
        raise ValueError("embodiment_type must be galaxea_a1z")
    prompt = observation.get("task")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Set a nonempty task description")
    frequency = observation.get("frequency")
    if isinstance(frequency, bool) or not isinstance(frequency, (int, float)) or not math.isfinite(frequency) or frequency <= 0:
        raise ValueError("frequency must be finite and positive")
    states = observation.get("state")
    if not isinstance(states, Mapping) or set(states) != set(PARTS[config.mode]):
        raise ValueError(f"State layout must match {config.mode}: {tuple(PARTS[config.mode])}")
    pieces = []
    for key, size in PARTS[config.mode].items():
        value = np.asarray(states[key], dtype=np.float32)
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid state {key}: expected finite ({size},)")
        if "gripper" in key and not 0 <= value[0] <= 1:
            raise ValueError(f"State {key} must be normalized to [0, 1]")
        pieces.append(value)
    expected_cameras = tuple(CAMERAS)[:2 if config.mode == "single" else 3]
    images = observation.get("images")
    if not isinstance(images, Mapping) or set(images) != set(expected_cameras):
        raise ValueError(f"Images must contain {expected_cameras}")
    mapped_images = {}
    for key in expected_cameras:
        image = np.asarray(images[key])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[0] != 3 or min(image.shape[1:]) < 1:
            raise ValueError(f"{key} must be a nonempty CHW RGB uint8 image")
        mapped_images[CAMERAS[key]] = np.ascontiguousarray(image)
    return {"state": np.concatenate(pieces), "images": mapped_images, "prompt": prompt}


def validate_chunk(response: dict, config: BridgeConfig) -> np.ndarray:
    """Validate the complete prediction before exposing even its first action."""
    if not isinstance(response, dict) or "actions" not in response:
        raise ValueError("openpi must return an actions array")
    actions = np.asarray(response["actions"])
    if actions.dtype.kind not in "iuf" or actions.ndim != 2 or actions.shape[1] != config.action_dim:
        raise ValueError(f"Expected actions [horizon, {config.action_dim}], got {actions.shape}")
    if actions.shape[0] < config.action_steps:
        raise ValueError(f"action_steps={config.action_steps} exceeds returned horizon={actions.shape[0]}")
    if not np.isfinite(actions).all():
        raise ValueError("openpi returned nonfinite actions; entire chunk refused")
    grippers = actions[:, [6] if config.mode == "single" else [6, 13]]
    if np.any(grippers < -config.gripper_tolerance) or np.any(grippers > 1 + config.gripper_tolerance):
        raise ValueError(f"openpi gripper actions exceed [0, 1] by more than {config.gripper_tolerance}; entire chunk refused")
    with np.errstate(over="ignore"):
        selected = actions[:config.action_steps].astype(np.float32, copy=True)
    if not np.isfinite(selected).all():
        raise ValueError("openpi actions overflow float32; entire chunk refused")
    return selected


def _validate_upstream_metadata(metadata: dict, config: BridgeConfig) -> int:
    if not isinstance(metadata, dict):
        raise ValueError("openpi server metadata must be an object")
    expected = {"a1z_mode": config.mode, "a1z_action_dim": config.action_dim,
                "action_representation": "absolute"}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"openpi metadata {key}={metadata.get(key)!r} does not match {value!r}; use an A1Z policy server")
    horizon = metadata.get("action_horizon")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < config.action_steps:
        raise ValueError("openpi action_horizon is missing, smaller than action_steps or invalid")
    return horizon


def make_upstream_factory(host: str, port: int, timeout: float) -> Callable[[], UpstreamPolicy]:
    """Use the official client with bounded connection and receive operations."""
    def factory() -> UpstreamPolicy:
        from openpi_client import msgpack_numpy
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        import websockets.sync.client

        class ReceiveDeadline:
            def __init__(self, connection):
                self.connection = connection

            def recv(self):
                return self.connection.recv(timeout=timeout)

            def send(self, value):
                return self.connection.send(value)

            def close(self):
                self.connection.close()

        class BoundedPolicy(WebsocketClientPolicy):
            # The upstream constructor otherwise retries a refused connection
            # forever. The protocol and infer implementation stay upstream-owned.
            def _wait_for_server(self):
                connection = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=MAX_MESSAGE_BYTES,
                    open_timeout=timeout, close_timeout=min(timeout, 2.0),
                )
                try:
                    raw = connection.recv(timeout=timeout)
                    if not isinstance(raw, bytes):
                        raise ValueError("openpi server returned a nonbinary handshake")
                    return ReceiveDeadline(connection), msgpack_numpy.unpackb(raw)
                except BaseException:
                    connection.close()
                    raise

            def close(self):
                self._ws.close()

        return BoundedPolicy(host=host, port=port)

    return factory


class BridgeSession:
    def __init__(self, config: BridgeConfig, policy_factory: Callable[[], UpstreamPolicy]):
        self.config = config
        self.policy_factory = policy_factory
        self.policy: UpstreamPolicy | None = None
        self.action_horizon: int | None = None
        self.actions: deque[np.ndarray] = deque()

    async def open(self) -> None:
        if self.policy is None:
            pending = asyncio.create_task(asyncio.to_thread(self.policy_factory))
            try:
                self.policy = await asyncio.shield(pending)
            except asyncio.CancelledError:
                # A connection finishing after cancellation must not escape its
                # session. Closing it in a daemon thread keeps the loop usable.
                def close_late_connection(done):
                    if not done.cancelled() and done.exception() is None:
                        threading.Thread(target=done.result().close, daemon=True).start()

                pending.add_done_callback(close_late_connection)
                raise
            try:
                metadata = await asyncio.to_thread(self.policy.get_server_metadata)
                self.action_horizon = _validate_upstream_metadata(metadata, self.config)
            except BaseException:
                await self.close()
                raise

    async def close(self) -> None:
        self.actions.clear()
        self.action_horizon = None
        policy, self.policy = self.policy, None
        if policy is not None:
            try:
                await asyncio.to_thread(policy.close)
            except Exception:
                LOGGER.exception("Could not close upstream policy connection")

    async def respond(self, request: dict) -> dict:
        try:
            if not isinstance(request, dict):
                raise ValueError("Expected a request object")
            if "reset" in request:
                if set(request) != {"reset"} or request["reset"] is not True:
                    raise ValueError("Send reset as a separate {'reset': true} request")
                await self.close()
                return {"reset": True, "need_obs": True}
            if request:
                self.actions.clear()
                observation = observation_for_openpi(request, self.config)
                await self.open()
                result = await asyncio.wait_for(
                    asyncio.to_thread(self.policy.infer, observation),
                    timeout=self.config.upstream_timeout,
                )
                chunk = validate_chunk(result, self.config)
                if len(result["actions"]) != self.action_horizon:
                    raise ValueError("openpi returned a different action horizon from its handshake; entire chunk refused")
                self.actions.extend(chunk)
            if not self.actions:
                raise ValueError("A fresh observation is required; no cached action remains")
            row = self.actions.popleft()
            action, offset = {}, 0
            for key, size in PARTS[self.config.mode].items():
                action[key] = row[offset:offset + size].copy()
                offset += size
            return {"action": action, "need_obs": not self.actions}
        except BaseException:
            await self.close()
            raise


class PolicyBridge:
    def __init__(self, config: BridgeConfig, policy_factory: Callable[[], UpstreamPolicy]):
        self.config, self.policy_factory = config, policy_factory

    async def handler(self, websocket) -> None:
        from websockets.exceptions import ConnectionClosed

        session = BridgeSession(self.config, self.policy_factory)
        try:
            # Robot clients open hardware after the handshake. First check that
            # the configured upstream is reachable and has compatible metadata.
            await session.open()
            await websocket.send(pack(self.config.metadata()))
            async for raw in websocket:
                response = await session.respond(unpack(raw))
                await websocket.send(pack(response))
        except ConnectionClosed:
            pass
        except Exception as exc:
            LOGGER.warning("Policy bridge refused a request: %s", exc)
            try:
                await websocket.send(pack({"error": f"{type(exc).__name__}: {exc}", "need_obs": True}))
                await websocket.close(code=1011, reason="Policy request rejected")
            except ConnectionClosed:
                pass
        finally:
            await session.close()

    async def run(self, host: str, port: int) -> None:
        from websockets.asyncio.server import serve

        async with serve(self.handler, host, port, compression=None, max_size=MAX_MESSAGE_BYTES) as server:
            LOGGER.info("A1Z %s bridge listening on ws://%s:%s", self.config.mode, host, port)
            await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(PARTS), default="single")
    parser.add_argument("--upstream-host", default="127.0.0.1")
    parser.add_argument("--upstream-port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--action-steps", type=int, default=16)
    parser.add_argument("--gripper-tolerance", type=float, default=0.02)
    parser.add_argument("--upstream-timeout", type=float, default=30.0)
    args = parser.parse_args()
    for port in (args.port, args.upstream_port):
        if not 0 < port < 65536:
            parser.error("Ports must be between 1 and 65535")
    config = BridgeConfig(args.mode, args.action_steps, args.gripper_tolerance, args.upstream_timeout)
    logging.basicConfig(level=logging.INFO)
    bridge = PolicyBridge(config, make_upstream_factory(
        args.upstream_host, args.upstream_port, args.upstream_timeout))
    try:
        asyncio.run(bridge.run(args.host, args.port))
    except KeyboardInterrupt:
        LOGGER.info("Policy bridge stopped; robot motor control remains with its client")


if __name__ == "__main__":
    main()
