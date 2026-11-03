# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Safety-relevant checks for optional single-arm and dual-arm target smoothing."""
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       "a1z_runtime" / "client"))
from trajectory_smoother import ArmCommandStreamer, BoundedJointInterpolator
import trajectory_smoother as smoothing_module


class FakeArm:
    is_running = True
    is_estopped = False

    def __init__(self):
        self.commands = []
        self.held = False

    def command_joint_pos(self, position):
        self.commands.append(np.asarray(position).copy())
        return True

    def get_fault_status(self):
        return "none"

    def latch_last_command(self, reason):
        self.held = True


class SmoothingTests(unittest.TestCase):
    def moving_interpolator(self):
        smoother = BoundedJointInterpolator(np.zeros(6), 4, 20, .04)
        smoother.set_target(np.full(6, .05))
        for _ in range(10):
            smoother.advance(.004)
        return smoother

    def assert_bounded_step(self, smoother, dt):
        position, velocity = smoother.position.copy(), smoother.velocity.copy()
        point = smoother.advance(dt)
        used_dt = min(dt, .02)
        actual_velocity = (point - position) / used_dt
        np.testing.assert_allclose(actual_velocity, smoother.velocity, atol=1e-11)
        self.assertLessEqual(np.max(np.abs(actual_velocity)), smoother.max_velocity + 1e-10)
        self.assertLessEqual(np.max(np.abs(actual_velocity - velocity)) / used_dt,
                             smoother.max_acceleration + 1e-8)
        self.assertTrue(np.isfinite(point).all())

    def test_reversal_brakes_without_resetting_reference_velocity(self):
        smoother = self.moving_interpolator()
        position, velocity = smoother.position.copy(), smoother.velocity.copy()
        smoother.set_target(np.zeros(6))
        np.testing.assert_array_equal(smoother.position, position)
        np.testing.assert_array_equal(smoother.velocity, velocity)
        self.assert_bounded_step(smoother, .004)
        # The old positive velocity cannot reverse instantly at finite acceleration.
        self.assertTrue((smoother.position > position).all())
        for _ in range(300):
            self.assert_bounded_step(smoother, .004)
        np.testing.assert_allclose(smoother.position, 0, atol=1e-8)

    def test_retarget_inside_stopping_distance_does_not_snap_to_endpoint(self):
        for offset in (0., .0001):
            with self.subTest(offset=offset):
                smoother = self.moving_interpolator()
                target = smoother.position + offset
                smoother.set_target(target)
                self.assert_bounded_step(smoother, .004)
                self.assertTrue((smoother.position > target).all())
                for _ in range(300):
                    self.assert_bounded_step(smoother, .004)
                np.testing.assert_allclose(smoother.position, target, atol=1e-8)

    def test_repeated_target_preserves_state_and_copies_input(self):
        smoother = self.moving_interpolator()
        position, velocity = smoother.position.copy(), smoother.velocity.copy()
        target = np.full(6, .05)
        smoother.set_target(target)
        target[:] = -1
        np.testing.assert_array_equal(smoother.position, position)
        np.testing.assert_array_equal(smoother.velocity, velocity)
        np.testing.assert_array_equal(smoother.target, np.full(6, .05))

    def test_static_targets_brake_before_crossing_even_with_fast_response(self):
        for response in (.0001, .004, .04):
            for distance in (.0001, .01, .2, -.2):
                with self.subTest(response=response, distance=distance):
                    smoother = BoundedJointInterpolator(np.zeros(6), 4, 20, response)
                    smoother.set_target(np.full(6, distance))
                    for _ in range(500):
                        self.assert_bounded_step(smoother, .004)
                        directed = smoother.position * np.sign(distance)
                        self.assertTrue((directed >= -1e-10).all())
                        self.assertTrue((directed <= abs(distance) + 1e-10).all())
                    np.testing.assert_allclose(smoother.position, distance, atol=1e-8)

    def test_six_axis_retargets_remain_bounded_on_variable_integration_steps(self):
        rng = np.random.default_rng(705)
        smoother = BoundedJointInterpolator(np.zeros(6), 4, 20, .04)
        for step in range(2000):
            if step % 9 == 0:
                smoother.set_target(rng.uniform(-.2, .2, 6))
            self.assert_bounded_step(smoother, float(rng.choice([.002, .004, .008, .016])))

    def test_scheduler_stall_keeps_existing_capped_clock_without_replaying_time(self):
        stalled, reference = self.moving_interpolator(), self.moving_interpolator()
        self.assert_bounded_step(stalled, .08)
        reference.advance(.02)
        np.testing.assert_array_equal(stalled.position, reference.position)
        np.testing.assert_array_equal(stalled.velocity, reference.velocity)
        self.assert_bounded_step(stalled, .004)

    def test_targets_are_continuous_and_do_not_overshoot(self):
        smoother = BoundedJointInterpolator(np.zeros(6), 4, 20, .04)
        smoother.set_target(np.full(6, .2))
        points = np.array([smoother.advance(.004) for _ in range(100)])
        self.assertTrue(np.isfinite(points).all())
        self.assertTrue((points >= 0).all())
        self.assertTrue((points <= .2 + 1e-9).all())
        self.assertLess(np.max(np.abs(np.diff(points, axis=0))), .0161)
        self.assertTrue(np.allclose(points[-1], .2, atol=1e-3))
        smoother.set_target(np.zeros(6))
        reverse = np.array([smoother.advance(.004) for _ in range(100)])
        self.assertTrue((reverse >= -1e-9).all())
        self.assertTrue(np.allclose(reverse[-1], 0, atol=1e-3))

    def test_stale_policy_holds_and_stops_stream(self):
        arm = FakeArm()
        stream = ArmCommandStreamer(arm, np.zeros(6), stale_timeout=.08)
        try:
            stream.start()
            stream.set_target(np.full(6, .1))
            time.sleep(.13)
            with self.assertRaisesRegex(RuntimeError, "stale"):
                stream.check()
            self.assertTrue(arm.held)
            self.assertGreater(len(arm.commands), 10)
        finally:
            stream.stop()


class StreamOutputTests(unittest.TestCase):
    def assert_stream_output_bounded(self, dual, near_target):
        class Clock:
            now = 0.0

            def monotonic(self):
                return self.now

        clock = Clock()

        class Event:
            stopped = False

            def is_set(self):
                return self.stopped

            def set(self):
                self.stopped = True

            def wait(self, delay):
                clock.now += max(0.0, delay)

        class RetargetArm(FakeArm):
            def command_joint_pos(self, position):
                super().command_joint_pos(position)
                if self is arms[-1]:
                    if len(self.commands) == 11:
                        targets = [arm.commands[-1] + sign * .0001
                                   if near_target else np.zeros(6)
                                   for arm, sign in zip(arms, signs)]
                        if dual:
                            stream.set_targets(targets)
                        else:
                            stream.set_target(targets[0])
                    if len(self.commands) == 120:
                        stream.stop_event.set()
                return True

        arms = [RetargetArm() for _ in range(2 if dual else 1)]
        signs = (1, -1) if dual else (1,)
        options = dict(frequency=250, max_velocity=4, max_acceleration=20,
                       response_time=.04, stale_timeout=1.0)
        if dual:
            stream = smoothing_module.DualArmCommandStreamer(
                arms, [np.zeros(6), np.zeros(6)], **options)
            stream.set_targets([np.full(6, sign * .05) for sign in signs])
        else:
            stream = ArmCommandStreamer(arms[0], np.zeros(6), **options)
            stream.set_target(np.full(6, .05))
        stream.stop_event = Event()
        stream.last_target_time = 0.0
        with patch.object(smoothing_module.time, 'monotonic', clock.monotonic):
            stream._run()
        self.assertIsNone(stream.error)
        for arm in arms:
            self.assertEqual(len(arm.commands), 120)
            velocity = np.diff(np.asarray(arm.commands), axis=0) / .004
            acceleration = np.diff(velocity, axis=0) / .004
            self.assertLessEqual(np.abs(velocity).max(), 4 + 1e-10)
            self.assertLessEqual(np.abs(acceleration).max(), 20 + 1e-8)
            self.assertFalse(arm.held)

    def test_single_stream_sends_bounded_positions_across_retargets(self):
        for near in (False, True):
            with self.subTest(near_target=near):
                self.assert_stream_output_bounded(False, near)

    def test_dual_stream_sends_bounded_positions_across_retargets(self):
        for near in (False, True):
            with self.subTest(near_target=near):
                self.assert_stream_output_bounded(True, near)


class DualSmoothingTests(unittest.TestCase):
    def test_dual_client_checks_both_targets_before_sending(self):
        from dual_arm_client import Hardware

        class GripperArm(FakeArm):
            def __init__(self):
                super().__init__()
                self.grippers = []

            def command_gripper(self, value):
                self.grippers.append(value)

        class TargetStream:
            def __init__(self):
                self.targets = []

            def check(self):
                pass

            def set_targets(self, targets):
                self.targets.append([v.copy() for v in targets])

        left, right = GripperArm(), GripperArm()
        hardware = Hardware({"max_joint_step": .3}, execute=True)
        hardware.arms = [left, right]
        hardware.previous = (np.zeros(7), np.zeros(7))
        hardware.streamer = TargetStream()
        action = {"left_arm": np.full(6, .1), "left_gripper": [.4],
                  "right_arm": np.full(6, -.1), "right_gripper": [.6]}
        hardware.send(action)
        self.assertEqual(left.grippers, [.4])
        self.assertEqual(right.grippers, [.6])
        self.assertEqual(len(hardware.streamer.targets), 1)
        self.assertTrue(np.allclose(hardware.streamer.targets[0][0], .1))
        self.assertTrue(np.allclose(hardware.streamer.targets[0][1], -.1))
        self.assertEqual(left.commands + right.commands, [])
        action["right_arm"] = np.full(6, 1.0)
        with self.assertRaisesRegex(ValueError, r"right joint\d+ step"):
            hardware.send(action)
        self.assertEqual(left.grippers, [.4])
        self.assertEqual(right.grippers, [.6])
        self.assertEqual(len(hardware.streamer.targets), 1)

    def test_both_arms_hold_when_one_side_rejects_command(self):
        from trajectory_smoother import DualArmCommandStreamer

        left, right = FakeArm(), FakeArm()
        right.command_joint_pos = lambda _position: False
        stream = DualArmCommandStreamer(
            [left, right], [np.zeros(6), np.zeros(6)], stale_timeout=1.0
        )
        try:
            stream.start()
            stream.thread.join(timeout=1.0)
            with self.assertRaisesRegex(RuntimeError, "SDK refused target"):
                stream.check()
            self.assertTrue(left.held)
            self.assertTrue(right.held)
        finally:
            stream.stop()

    def test_both_arms_receive_bounded_targets_and_hold_when_stale(self):
        from trajectory_smoother import DualArmCommandStreamer

        left, right = FakeArm(), FakeArm()
        stream = DualArmCommandStreamer(
            [left, right], [np.zeros(6), np.zeros(6)],
            frequency=250, max_velocity=4, max_acceleration=20,
            stale_timeout=.08,
        )
        try:
            stream.start()
            stream.set_targets([np.full(6, .2), np.full(6, -.2)])
            time.sleep(.13)
            with self.assertRaisesRegex(RuntimeError, "stale"):
                stream.check()
            self.assertTrue(left.held and right.held)
            self.assertGreater(len(left.commands), 10)
            self.assertEqual(len(left.commands), len(right.commands))
            self.assertTrue(all(cmd.shape == (6,) for cmd in left.commands + right.commands))
            self.assertTrue(all((cmd <= .2 + 1e-9).all() for cmd in left.commands))
            self.assertTrue(all((cmd >= -.2 - 1e-9).all() for cmd in right.commands))
        finally:
            stream.stop()


if __name__ == "__main__":
    unittest.main()
