# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""A1Z data transforms without ALOHA joint or gripper calibration."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

Mode = Literal["single", "dual"]


def parts_for_mode(mode: Mode) -> tuple[str, ...]:
    if mode == "single":
        return ("left_arm", "left_gripper")
    if mode == "dual":
        return ("left_arm", "left_gripper", "right_arm", "right_gripper")
    raise ValueError(f"Expected mode single or dual, got {mode!r}")


def action_dimension(mode: Mode) -> int:
    return 7 if len(parts_for_mode(mode)) == 2 else 14


def action_sequence_keys(mode: Mode) -> tuple[str, ...]:
    return tuple(f"action.{part}" for part in parts_for_mode(mode))


def delta_action_mask(mode: Mode) -> tuple[bool, ...]:
    return ((True,) * 6 + (False,)) * (len(parts_for_mode(mode)) // 2)


def _vector(value, width: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if width == 1 and array.ndim == 0:
        array = array.reshape(1)
    if array.shape != (width,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape ({width},), got {array.shape}")
    return array


def _action_part(value, width: int, horizon: int | None, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    # LeRobot v2.1 scalar Arrow columns yield [T], not [T,1], for grippers.
    if width == 1 and array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[-1] != width or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape (T,{width}), got {array.shape}")
    if horizon is not None and array.shape[0] != horizon:
        raise ValueError(f"{name} action horizon differs from the other action fields")
    if array.shape[0] == 0:
        raise ValueError(f"{name} has an empty action horizon")
    return array


def _check_grippers(array: np.ndarray, mode: Mode, name: str) -> None:
    indices = (6,) if mode == "single" else (6, 13)
    grip = array[..., indices]
    if np.any(grip < 0) or np.any(grip > 1):
        raise ValueError(f"{name} grippers must use normalized absolute values in [0,1]")


@dataclass(frozen=True)
class RepackA1ZDataset:
    """Join the existing split v2.1 fields; videos remain in their original files."""

    mode: Mode

    def __post_init__(self):
        parts_for_mode(self.mode)

    def __call__(self, data: dict) -> dict:
        parts = parts_for_mode(self.mode)
        state = np.concatenate([
            _vector(data[f"observation.state.{part}"], 6 if part.endswith("_arm") else 1,
                    f"observation.state.{part}")
            for part in parts
        ])
        _check_grippers(state, self.mode, "state")
        camera_names = {
            "cam_high": "observation.images.head_rgb",
            "cam_left_wrist": "observation.images.left_wrist_rgb",
        }
        if self.mode == "dual":
            camera_names["cam_right_wrist"] = "observation.images.right_wrist_rgb"
        images = {out: data[source] for out, source in camera_names.items() if source in data}
        result = {"state": state, "images": images}
        keys = action_sequence_keys(self.mode)
        if any(key in data for key in keys):
            if not all(key in data for key in keys):
                raise ValueError("Dataset is missing one or more split A1Z action fields")
            arrays = []
            horizon = None
            for part, key in zip(parts, keys, strict=True):
                array = _action_part(data[key], 6 if part.endswith("_arm") else 1, horizon, key)
                horizon = array.shape[0]
                arrays.append(array)
            actions = np.concatenate(arrays, axis=-1)
            _check_grippers(actions, self.mode, "actions")
            result["actions"] = actions
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


def _rgb_image(value, name: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"{name} must be a three-channel RGB image")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] != 3 or min(image.shape[:2]) < 1:
        raise ValueError(f"{name} must have HWC or CHW shape with three channels")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or np.any(image < 0) or np.any(image > 1):
            raise ValueError(f"{name} floating images must be finite and in [0,1]")
        image = np.rint(image * 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        raise ValueError(f"{name} must use uint8 RGB or floating RGB in [0,1]")
    return np.ascontiguousarray(image)


@dataclass(frozen=True)
class A1ZInputs:
    """Canonical A1Z observation to openpi images, state and optional targets."""

    mode: Mode

    def __post_init__(self):
        parts_for_mode(self.mode)

    def __call__(self, data: dict) -> dict:
        state = _vector(data["state"], action_dimension(self.mode), "state").copy()
        _check_grippers(state, self.mode, "state")
        incoming = data["images"]
        allowed = {"cam_high", "cam_left_wrist", "cam_right_wrist"}
        if set(incoming) - allowed:
            raise ValueError(f"Unknown cameras: {sorted(set(incoming) - allowed)}")
        if "cam_high" not in incoming:
            raise ValueError("The head camera cam_high is required")
        required = {"cam_high", "cam_left_wrist"}
        if self.mode == "dual":
            required.add("cam_right_wrist")
        if required - set(incoming):
            raise ValueError(f"Missing required A1Z camera: {sorted(required - set(incoming))}")
        base = _rgb_image(incoming["cam_high"], "cam_high")
        images = {"base_0_rgb": base}
        masks = {"base_0_rgb": np.True_}
        for output, source in (("left_wrist_0_rgb", "cam_left_wrist"),
                               ("right_wrist_0_rgb", "cam_right_wrist")):
            present = source in incoming and not (self.mode == "single" and source == "cam_right_wrist")
            images[output] = _rgb_image(incoming[source], source) if present else np.zeros_like(base)
            masks[output] = np.bool_(present)
        result = {"image": images, "image_mask": masks, "state": state}
        if "actions" in data:
            actions = _action_part(data["actions"], action_dimension(self.mode), None, "actions").copy()
            _check_grippers(actions, self.mode, "actions")
            result["actions"] = actions
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclass(frozen=True)
class A1ZOutputs:
    """Remove model padding after unnormalization and absolute-action restoration."""

    mode: Mode

    def __post_init__(self):
        parts_for_mode(self.mode)

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)
        width = action_dimension(self.mode)
        if actions.ndim != 2 or actions.shape[-1] < width or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite model actions with shape (T,D), D >= {width}")
        # Gripper boundary tolerance and robot joint limits belong to the execution client.
        return {"actions": actions[:, :width].copy()}
