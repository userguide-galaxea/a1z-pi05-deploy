# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""CAN mapping and blocked-camera regressions; no hardware or SDK is opened."""
import copy
import json
from pathlib import Path
import sys
import subprocess
import os
import importlib.util
import tempfile
import time
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'a1z_runtime/client'))
import camera_capture
from camera_capture import CameraGroup, CameraSpec
from device_checks import camera_specs, can_roles, check_can, usb_identity


def deployment(mode='dual'):
    cfg = {'topology': 'pc-direct', 'left_can': 'can_b', 'right_can': 'can_a',
           'camera_head': '/head', 'camera_left_wrist': '/left', 'camera_right_wrist': '/right',
           'image_size': [12, 8], 'control_frequency': 15,
           'hardware_checks': {'can_adapter_serials': {'left': 'LEFT_ADAPTER', 'right': 'RIGHT_ADAPTER'}}}
    if mode == 'single':
        cfg.update(topology='pc-direct-single', can_channel='can_b', camera_wrist='/wrist')
        cfg['hardware_checks']['can_adapter_serials'] = {'single': 'LEFT_ADAPTER'}
    return cfg


def ready(_):
    return {'flags': ['UP'], 'linkinfo': {'info_kind': 'can', 'info_data': {
        'state': 'ERROR-ACTIVE', 'bittiming': {'bitrate': 1000000}}}}


def identity(channel):
    return {'adapter_serial': {'can_b': 'LEFT_ADAPTER', 'can_a': 'RIGHT_ADAPTER'}[channel]}


class CanChecks(unittest.TestCase):
    def test_collector_removes_launcher_options_but_keeps_arm_roles(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/normalize_collect_config.py'
        spec = importlib.util.spec_from_file_location('normalize_for_test', path)
        entry = importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
        cfg = {'arm_cfg': {'leader_port': '/leader', 'follower_can': 'can_b'},
               'camera_cfg': {'camera_names': ['head', 'wrist'], 'topics': {'head': '/head', 'wrist': '/wrist'}},
               'hardware_checks': {'can_adapter_serials': {'single': 'ADAPTER'}}}
        result = entry.normalize(cfg)
        self.assertNotIn('hardware_checks', result)
        self.assertIn('hardware_checks', cfg)
        self.assertEqual(result['arm_cfg']['follower_can_left'], 'can_b')
        self.assertEqual(result['camera_cfg']['camera_names'], ['cam_high', 'cam_left_wrist'])

    def test_collection_refuses_hardware_overrides_before_opening_devices(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp)/'config.yaml';cfg.write_text('task_description: example\n')
            for override in ('arm_cfg.follower_can_left=wrong', 'arm_cfg={follower_can_left:wrong}', 'config_file=other.yaml'):
                result = subprocess.run(['bash',str(root/'scripts/collect.sh'),'dual','--check',override],
                    env=dict(os.environ,A1Z_COLLECT_CONFIG=str(cfg),A1Z_PYTHON=sys.executable),capture_output=True,text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn('bypass identity checks', result.stderr)

    def test_single_dual_roles_follow_serials_not_interface_numbers(self):
        for mode in ('single', 'dual'):
            cfg = deployment(mode)
            rows = check_can(cfg, identity_reader=identity, status_reader=ready)
            self.assertEqual(rows['single' if mode == 'single' else 'left']['interface'], 'can_b')
        cfg = deployment();cfg['left_can'], cfg['right_can'] = cfg['right_can'], cfg['left_can']
        status = Mock()
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            check_can(cfg, identity_reader=identity, status_reader=status)
        status.assert_not_called()

    def test_incomplete_duplicate_and_wrong_serials_are_refused(self):
        for serials in ({'left': '', 'right': ''}, {'left': 'SAME', 'right': 'SAME'}, {'left': 'LEFT_ADAPTER'}):
            cfg = deployment();cfg['hardware_checks']['can_adapter_serials'] = serials
            with self.assertRaises(ValueError):check_can(cfg, identity_reader=identity, status_reader=ready)

    def test_down_wrong_bitrate_and_bus_off_are_refused(self):
        for mutate in ('down', 'bitrate', 'bus-off'):
            status = ready(None)
            if mutate == 'down':status['flags'] = []
            elif mutate == 'bitrate':status['linkinfo']['info_data']['bittiming']['bitrate'] = 500000
            else:status['linkinfo']['info_data']['state'] = 'BUS-OFF'
            with self.assertRaises(ValueError):
                check_can(deployment(), identity_reader=identity, status_reader=lambda _: status)

    def test_explicit_manual_binding_warns_and_still_checks_can(self):
        cfg = deployment();cfg['hardware_checks']['can_adapter_serials'] = None
        with self.assertWarnsRegex(UserWarning, 'serial binding disabled'):
            self.assertEqual(len(check_can(cfg, identity_reader=identity, status_reader=ready)), 2)

    def test_sysfs_identity_and_missing_interface(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);net = root/'net';net.mkdir()
            usb = root/'usb';iface = usb/'interface';iface.mkdir(parents=True)
            (usb/'idVendor').write_text('test');(usb/'serial').write_text('LEFT_ADAPTER')
            (net/'can_b').mkdir();(net/'can_b/device').symlink_to(iface)
            self.assertEqual(usb_identity('can_b', net)['adapter_serial'], 'LEFT_ADAPTER')
            with self.assertRaisesRegex(ValueError, 'missing'):usb_identity('can_a', net)

    def test_collection_names_and_camera_roles(self):
        for mode in ('single', 'dual'):
            cfg = deployment(mode)
            arm = ({'collection_type': 'one_master_slave', 'follower_can': 'can_b'} if mode == 'single' else
                   {'collection_type': 'two_master_slave', 'follower_can_left': 'can_b', 'follower_can_right': 'can_a'})
            names = ['head', 'wrist'] if mode == 'single' else ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
            collection = {'arm_cfg': arm, 'camera_cfg': {'camera_names': names,
                'topics': {n: '/'+n for n in names}, 'width': 12, 'height': 8},
                'collector_cfg': {'camera_freq': 15}}
            self.assertEqual(can_roles(collection), can_roles(cfg))
            self.assertEqual([s.name for s in camera_specs(collection)], [s.name for s in camera_specs(cfg)])


FRAME_WORKER = '''
import sys,time
from multiprocessing.connection import Connection
import numpy as np
c=Connection(int(sys.argv[1]),readable=False)
for _ in range(3):
 c.send(('frame',time.monotonic(),np.zeros((8,12,3),np.uint8)))
 time.sleep(.01)
time.sleep(30)
'''


class BoundedCameraChecks(unittest.TestCase):
    def specs(self, root, count=2):
        result = []
        for i in range(count):
            device = root/str(i);device.touch()
            result.append(CameraSpec(str(i), str(device), 12, 8, 15))
        return result

    def test_two_and_three_streams_start_and_shutdown_without_hardware(self):
        for count in (2, 3):
            with tempfile.TemporaryDirectory() as tmp:
                group = CameraGroup(self.specs(Path(tmp), count), startup_timeout=5)
                with patch.object(camera_capture, '_worker_command', side_effect=lambda fd,s: [sys.executable,'-c',FRAME_WORKER,str(fd)]):
                    try:
                        group.start()
                        self.assertEqual(len(group.frames()), count)
                        self.assertTrue(all(f.shape == (8,12,3) for f in group.frames().values()))
                    finally:group.close()
                self.assertTrue(all(s.process.poll() is not None for s in group.streams))

    def test_blocked_open_is_bounded_even_when_sigterm_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            group = CameraGroup(self.specs(Path(tmp)), startup_timeout=.2)
            code = 'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)'
            with patch.object(camera_capture, '_worker_command', return_value=[sys.executable,'-c',code]):
                start = time.monotonic()
                with self.assertRaisesRegex(TimeoutError, 'startup timed out'):group.start()
            self.assertLess(time.monotonic()-start, 2.5)
            self.assertTrue(all(s.process.poll() is not None for s in group.streams))

    def test_frozen_capture_rejects_stale_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            group = CameraGroup(self.specs(Path(tmp)), startup_timeout=5, frame_timeout=.15)
            with patch.object(camera_capture, '_worker_command', side_effect=lambda fd,s: [sys.executable,'-c',FRAME_WORKER,str(fd)]):
                try:
                    group.start();time.sleep(.25)
                    with self.assertRaisesRegex(TimeoutError, 'stale'):group.frames()
                finally:group.close()

    def test_symlink_aliases_cannot_open_same_camera_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            specs = self.specs(Path(tmp));Path(specs[1].device).unlink()
            Path(specs[1].device).symlink_to(specs[0].device)
            with self.assertRaisesRegex(ValueError, 'distinct'):CameraGroup(specs)

    def test_camera_failure_prevents_sdk_import_for_both_modes(self):
        import dual_arm_client
        import single_left_client
        for module in (dual_arm_client, single_left_client):
            group = Mock();group.start.side_effect = TimeoutError('camera frozen')
            sdk = types.ModuleType('a1z.robots.get_robot');sdk.get_a1z_robot = Mock()
            cfg = deployment();cfg['left_can']='can_b'
            with patch.object(module, 'check_can'), patch.object(module, 'cameras_from_config', return_value=group), \
                 patch.dict(sys.modules, {'a1z': types.ModuleType('a1z'),
                     'a1z.robots': types.ModuleType('a1z.robots'), 'a1z.robots.get_robot': sdk}):
                hardware = module.Hardware(cfg)
                with self.assertRaisesRegex(TimeoutError, 'frozen'):hardware.open()
                sdk.get_a1z_robot.assert_not_called()
                group.close.assert_called()


if __name__ == '__main__':
    unittest.main()
