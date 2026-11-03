# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Collection regression checks using mocks only; never open devices."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
COLLECT = ROOT / "a1z-data-collect"
sys.path.insert(0, str(COLLECT))
from collectors.arm_collector import ArmCollector
spec = importlib.util.spec_from_file_location("a1z_record_data", COLLECT / "scripts/record_data.py")
record = importlib.util.module_from_spec(spec)
spec.loader.exec_module(record)


class CollectionIntegrationTests(unittest.TestCase):
    def test_read_failure_holds_and_propagates(self):
        leader, follower = Mock(), Mock()
        leader.read_as_vector.side_effect = OSError("serial disconnected")
        collector = ArmCollector(leader, follower, None)
        collector.start()
        collector._thread.join(timeout=1.0)
        collector.stop()
        with self.assertRaisesRegex(RuntimeError, "Arm collection stopped"):
            collector.raise_if_failed()
        follower.hold.assert_called_once_with("serial disconnected")

    def test_aliases_cannot_open_same_dual_serial_port(self):
        with tempfile.TemporaryDirectory() as tmp:
            port = Path(tmp)/"serial"
            port.touch()
            alias = Path(tmp)/"alias"
            alias.symlink_to(port)
            cfg = SimpleNamespace(leader_port_left=str(port), leader_port_right=str(alias))
            with patch.object(record, "A1ZTLeaderArm") as constructor:
                with self.assertRaisesRegex(ValueError, "separate"):
                    record._make_dual_leader(cfg)
                constructor.assert_not_called()

    def test_stop_failure_does_not_skip_other_cleanup(self):
        leader, follower, camera = Mock(), Mock(), Mock()
        follower.stop.side_effect = RuntimeError("disable failed")
        events = {"start_recording": False, "stop": True}
        cfg = SimpleNamespace(num_episodes=1, collector_cfg=SimpleNamespace(
            arm_freq=30, camera_freq=15, cpu_affinity=None, realtime_priority=False),
            arm_cfg=SimpleNamespace(use_velocity=False))
        with patch.object(record,"make_arm_readers",return_value=(leader,follower,None)), \
             patch.object(record,"make_camera_readers",return_value={"cam_high":camera}), \
             patch.object(record,"H5Dataset"), patch.object(record,"ArmCollector"), \
             patch.object(record,"CameraCollector"), \
             patch.object(record,"init_keyboard_listener",return_value=(Mock(),events)):
            with self.assertRaisesRegex(RuntimeError,"disable failed"):
                record.run(cfg)
        leader.close.assert_called_once()
        camera.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
