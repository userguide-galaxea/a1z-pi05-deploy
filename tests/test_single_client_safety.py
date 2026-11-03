# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Single-client regression checks; all cameras, SDK calls and policy IO are mocked."""
import asyncio
from contextlib import contextmanager
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       "a1z_runtime" / "client"))
import single_left_client as client


def config():
    return {
        "topology": "pc-direct-single-left", "embodiment_type": "galaxea_a1z",
        "task": "Pick and place an object", "server_host": "127.0.0.1",
        "server_port": 8765, "request_timeout": 5, "control_frequency": 15,
        "image_size": [1, 1], "max_joint_step": .1,
        "max_initial_joint_step": .1, "max_cycles": 1,
        "left_can": "test-can", "camera_head": "test-head",
        "camera_left_wrist": "test-wrist", "gripper_boundary_tolerance": .05,
        "smoothing": {"enabled": True, "frequency": 250, "max_velocity": 4,
                      "max_acceleration": 20, "response_time": .04,
                      "stale_timeout": 1.0},
    }


def arm():
    value = Mock()
    value.is_running = True
    value.is_estopped = False
    value.get_joint_pos.return_value = np.zeros(7)
    value.get_fault_status.return_value = {"state": "STOPPED"}
    value.command_joint_pos.return_value = True
    return value


@contextmanager
def mocked_devices(robot=None, cameras=None):
    robot = robot if robot is not None else arm()
    if cameras is None:
        cameras = Mock()
        cameras.frames.return_value = {
            name: np.zeros((1, 1, 3), np.uint8) for name in client.CAMERAS
        }
    factory = Mock(return_value=robot)
    with patch.dict(sys.modules, {
        "a1z": SimpleNamespace(), "a1z.robots": SimpleNamespace(),
        "a1z.robots.get_robot": SimpleNamespace(get_a1z_robot=factory),
    }), patch.object(client, "check_can") as check, \
         patch.object(client, "cameras_from_config", return_value=cameras):
        cameras.sdk_factory = factory
        cameras.can_check = check
        yield robot, cameras


class PolicyConnection:
    def __init__(self, gripper=1.03, on_exit=None):
        self.on_exit = on_exit
        metadata = {"shape_meta": {"action": [
            {"key": key, "raw_shape": dimension}
            for key, dimension in client.PARTS.items()
        ]}}
        action = {"left_arm": np.full(6, .02), "left_gripper": [gripper]}
        self.ws = SimpleNamespace(
            recv=AsyncMock(side_effect=[client.pack(metadata), client.pack({"action": action})]),
            send=AsyncMock(),
        )

    async def __aenter__(self):
        return self.ws

    async def __aexit__(self, *_args):
        if self.on_exit is not None:
            self.on_exit()
        return False


class SingleClientSafetyTests(unittest.TestCase):
    def test_dry_run_does_not_construct_hardware_or_run_device_checks(self):
        with patch.object(client, "Hardware") as hardware, \
             patch.object(client, "check_can") as check, \
             patch.object(client, "cameras_from_config") as cameras, \
             patch.object(client.websockets, "connect", return_value=PolicyConnection()), \
             patch("sys.stdout", new_callable=io.StringIO):
            asyncio.run(client.run(config(), execute=False, observe_only=False))
        hardware.assert_not_called()
        check.assert_not_called()
        cameras.assert_not_called()

    def test_observe_only_never_starts_stream_or_sends_actions(self):
        with mocked_devices() as (robot, cameras), \
             patch.object(client, "ArmCommandStreamer") as streamer, \
             patch.object(client.websockets, "connect", return_value=PolicyConnection(gripper=1.0)), \
             patch("sys.stdout", new_callable=io.StringIO) as output:
            asyncio.run(client.run(config(), execute=False, observe_only=True))
        streamer.assert_not_called()
        robot.command_joint_pos.assert_not_called()
        robot.command_gripper.assert_not_called()
        robot.latch_last_command.assert_not_called()
        robot.stop.assert_called_once()
        cameras.close.assert_called_once()
        self.assertIn("not executed", output.getvalue())

    def test_execute_uses_custom_tolerance_and_clipped_target(self):
        cfg = config()
        cfg["smoothing"]["enabled"] = False
        with mocked_devices() as (robot, _), \
             patch.object(client.websockets, "connect", return_value=PolicyConnection()), \
             patch("sys.stdout", new_callable=io.StringIO):
            asyncio.run(client.run(cfg, execute=True, observe_only=False))
        robot.command_joint_pos.assert_called_once()
        target = robot.command_joint_pos.call_args.args[0]
        np.testing.assert_allclose(target[:6], .02)
        self.assertEqual(target[6], 1.0)

    def test_observe_only_honors_custom_tolerance_at_both_boundaries(self):
        for tolerance in (.02, .05):
            for value in (-.03, 1.03):
                with self.subTest(tolerance=tolerance, value=value):
                    cfg = config()
                    cfg["gripper_boundary_tolerance"] = tolerance
                    with mocked_devices() as (robot, cameras), \
                         patch.object(client.websockets, "connect", return_value=PolicyConnection(value)), \
                         patch("sys.stdout", new_callable=io.StringIO) as output:
                        if tolerance == .02:
                            with self.assertRaisesRegex(ValueError, "exceeds"):
                                asyncio.run(client.run(cfg, execute=False, observe_only=True))
                        else:
                            asyncio.run(client.run(cfg, execute=False, observe_only=True))
                            self.assertIn("boundary tolerance 0.05", output.getvalue())
                            self.assertIn("not executed", output.getvalue())
                    robot.command_joint_pos.assert_not_called()
                    robot.command_gripper.assert_not_called()
                    robot.stop.assert_called_once()
                    cameras.close.assert_called_once()

    def test_execute_smoothing_still_starts_and_receives_clipped_target(self):
        with mocked_devices() as (robot, _), \
             patch.object(client, "ArmCommandStreamer") as streamer:
            hardware = client.Hardware(config(), execute=True)
            stream = streamer.return_value
            stream.ticks = 0
            stream.max_lateness = 0.0
            hardware.open()
            hardware.send({"left_arm": np.full(6, .02), "left_gripper": [1.03]})
            hardware.close()
        streamer.assert_called_once()
        stream.start.assert_called_once()
        stream.set_target.assert_called_once()
        robot.command_gripper.assert_called_once_with(1.0)

    def test_bad_gripper_values_are_rejected_before_commands(self):
        for value in (1.051, -.051, np.nan, np.inf):
            with self.subTest(value=value):
                hardware = client.Hardware(config(), execute=True)
                hardware.arm = arm()
                hardware.previous = np.zeros(7)
                with self.assertRaises(ValueError):
                    hardware.send({"left_arm": np.zeros(6), "left_gripper": [value]})
                hardware.arm.command_joint_pos.assert_not_called()
                hardware.arm.command_gripper.assert_not_called()

    def test_repeated_ctrl_c_cannot_skip_sdk_status_or_camera_release(self):
        hardware = client.Hardware(config(), execute=True)
        robot = arm()
        hardware.arm = robot
        hardware.action_sent = True
        hardware.streamer = Mock()
        hardware.streamer.stop.side_effect = [KeyboardInterrupt(), None]
        hardware.streamer.ticks = 0
        hardware.streamer.max_lateness = 0.0
        robot.latch_last_command.side_effect = KeyboardInterrupt()
        robot.stop.side_effect = [KeyboardInterrupt(), None]
        cameras = Mock()
        cameras.close.side_effect = [KeyboardInterrupt(), None]
        hardware.cameras = cameras
        with patch("sys.stdout", new_callable=io.StringIO) as output:
            hardware.close()
        self.assertEqual(robot.stop.call_count, 2)
        robot.get_fault_status.assert_called_once()
        self.assertEqual(cameras.close.call_count, 2)
        self.assertIsNone(hardware.arm)
        self.assertIsNone(hardware.streamer)
        self.assertIsNone(hardware.cameras)
        self.assertIn('"state": "STOPPED"', output.getvalue())
        self.assertIn("Shutdown needs attention", output.getvalue())

    def test_partial_sdk_start_failure_stops_arm_and_releases_cameras(self):
        robot = arm()
        robot.start.side_effect = RuntimeError("startup failed")
        with mocked_devices(robot) as (_, cameras), \
             patch("sys.stdout", new_callable=io.StringIO):
            hardware = client.Hardware(config(), execute=True)
            with self.assertRaisesRegex(RuntimeError, "startup failed"):
                hardware.open()
        robot.stop.assert_called_once()
        robot.get_fault_status.assert_called_once()
        cameras.close.assert_called_once()
        self.assertIsNone(hardware.arm)

    def test_exhausted_interrupt_retries_still_attempt_every_cleanup_step(self):
        hardware = client.Hardware(config(), execute=True)
        robot = arm()
        hardware.arm = robot
        hardware.action_sent = True
        stream = Mock()
        stream.stop.side_effect = KeyboardInterrupt()
        hardware.streamer = stream
        robot.latch_last_command.side_effect = KeyboardInterrupt()
        robot.stop.side_effect = KeyboardInterrupt()
        robot.get_fault_status.return_value = {"state": "COMMAND_HOLD"}
        cameras = Mock()
        cameras.close.side_effect = KeyboardInterrupt()
        hardware.cameras = cameras
        with patch("sys.stdout", new_callable=io.StringIO) as output:
            hardware.close()
            hardware.close()
        self.assertEqual(stream.stop.call_count, 2)
        self.assertEqual(robot.stop.call_count, 2)
        self.assertEqual(cameras.close.call_count, 2)
        self.assertIn("shutdown interrupted twice", output.getvalue())
        self.assertIn("shutdown not confirmed", output.getvalue())
        self.assertNotIn("SDK stop transaction completed", output.getvalue())

    def test_sdk_stop_error_still_releases_cameras_and_reports_unconfirmed_stop(self):
        hardware = client.Hardware(config(), execute=True)
        robot = arm()
        robot.stop.side_effect = RuntimeError("stop failed")
        robot.get_fault_status.return_value = {"state": "COMMAND_HOLD"}
        hardware.arm = robot
        cameras = Mock()
        hardware.cameras = cameras
        with patch("sys.stdout", new_callable=io.StringIO) as output:
            hardware.close()
        cameras.close.assert_called_once()
        self.assertIn("stop failed", output.getvalue())
        self.assertIn("shutdown not confirmed", output.getvalue())
        self.assertNotIn("SDK stop transaction completed", output.getvalue())

    def test_failed_camera_open_is_released_before_any_arm_is_created(self):
        cameras = Mock()
        cameras.start.side_effect = RuntimeError("Camera startup failed")
        with mocked_devices(cameras=cameras) as (robot, _), \
             patch("sys.stdout", new_callable=io.StringIO):
            with self.assertRaisesRegex(RuntimeError, "Camera startup failed"):
                client.Hardware(config(), execute=True).open()
        cameras.close.assert_called_once()
        cameras.sdk_factory.assert_not_called()
        robot.start.assert_not_called()

    def test_all_cameras_and_can_are_verified_before_sdk_initialization(self):
        order = []
        with mocked_devices() as (robot, cameras):
            cameras.can_check.side_effect = lambda _cfg: order.append("can")
            cameras.start.side_effect = lambda: order.append("cameras_start")
            cameras.frames.side_effect = lambda: (order.append("frames") or {
                name: np.array([[[1, 2, 3]]], np.uint8) for name in client.CAMERAS
            })
            cameras.sdk_factory.side_effect = lambda **_kwargs: (order.append("sdk") or robot)
            hardware = client.Hardware(config(), execute=False)
            hardware.open()
            images = hardware.images()
            hardware.close()
        self.assertEqual(order[:5], ["can", "cameras_start", "frames", "can", "sdk"])
        for image in images.values():
            self.assertEqual(image.shape, (3, 1, 1))
            np.testing.assert_array_equal(image[:, 0, 0], [3, 2, 1])

    def test_camera_frame_failure_cannot_initialize_sdk(self):
        cameras = Mock()
        cameras.frames.side_effect = TimeoutError("camera stale")
        with mocked_devices(cameras=cameras), patch("sys.stdout", new_callable=io.StringIO):
            with self.assertRaisesRegex(TimeoutError, "camera stale"):
                client.Hardware(config(), execute=True).open()
        cameras.sdk_factory.assert_not_called()
        cameras.close.assert_called_once()

    def test_can_identity_change_after_camera_start_cannot_initialize_sdk(self):
        with mocked_devices() as (_, cameras), patch("sys.stdout", new_callable=io.StringIO):
            cameras.can_check.side_effect = [None, ValueError("CAN identity mismatch")]
            with self.assertRaisesRegex(ValueError, "CAN identity mismatch"):
                client.Hardware(config(), execute=True).open()
        cameras.sdk_factory.assert_not_called()
        cameras.close.assert_called_once()

    def test_initial_can_failure_cannot_start_cameras_or_initialize_sdk(self):
        with mocked_devices() as (_, cameras), patch("sys.stdout", new_callable=io.StringIO):
            cameras.can_check.side_effect = ValueError("CAN identity mismatch")
            with self.assertRaisesRegex(ValueError, "CAN identity mismatch"):
                client.Hardware(config(), execute=True).open()
        cameras.start.assert_not_called()
        cameras.sdk_factory.assert_not_called()

    def test_hardware_closes_before_websocket_exit_on_success_or_policy_error(self):
        for gripper in (1.03, 1.2):
            with self.subTest(gripper=gripper):
                order = []
                connection = PolicyConnection(gripper, on_exit=lambda: order.append("websocket_exit"))
                with mocked_devices() as (robot, cameras), \
                     patch.object(client.websockets, "connect", return_value=connection), \
                     patch("sys.stdout", new_callable=io.StringIO):
                    robot.stop.side_effect = lambda: order.append("arm_stop")
                    cameras.close.side_effect = lambda: order.append("cameras_close")
                    if gripper > 1.05:
                        with self.assertRaises(ValueError):
                            asyncio.run(client.run(config(), execute=False, observe_only=True))
                    else:
                        asyncio.run(client.run(config(), execute=False, observe_only=True))
                self.assertEqual(order, ["arm_stop", "cameras_close", "websocket_exit"])

    def test_interrupted_websocket_exit_does_not_skip_or_repeat_hardware_cleanup(self):
        order = []

        def interrupted_exit():
            order.append("websocket_exit")
            raise KeyboardInterrupt()

        with mocked_devices() as (robot, cameras), \
             patch.object(client.websockets, "connect", return_value=PolicyConnection(on_exit=interrupted_exit)), \
             patch("sys.stdout", new_callable=io.StringIO):
            robot.stop.side_effect = lambda: order.append("arm_stop")
            cameras.close.side_effect = lambda: order.append("cameras_close")
            with self.assertRaises(KeyboardInterrupt):
                asyncio.run(client.run(config(), execute=False, observe_only=True))
        self.assertEqual(order, ["arm_stop", "cameras_close", "websocket_exit"])
        robot.stop.assert_called_once()
        cameras.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
