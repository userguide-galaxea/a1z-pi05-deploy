# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Launch the local PC SDK client. Default never sends model actions."""
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_TOPOLOGY = "pc-direct"
sys.path.insert(0, str(ROOT / 'a1z_runtime' / 'client'))
from single_config import normalize_single_config
from device_checks import hardware_options


def command(cfg, path, execute=False, observe=False, record_dir=None,
            record_every=4, record_max_frames=2400):
    cfg = normalize_single_config(cfg)
    hardware_options(cfg)
    topology = cfg.get('topology')
    if record_dir is not None and not (execute or observe):
        raise ValueError('--record-dir requires --execute or --observe-only')
    if record_every < 1 or record_max_frames < 1:
        raise ValueError('Recording interval and frame limit must be positive')
    if record_dir is not None and topology != 'pc-direct-single-left':
        raise ValueError('Diagnostic recording currently supports single mode only')
    if not str(cfg.get('task', '')).strip():
        raise ValueError('task must be set')
    if cfg.get('embodiment_type') != 'galaxea_a1z':
        raise ValueError('Set embodiment_type to galaxea_a1z')
    if 'websocket_compression' in cfg and cfg['websocket_compression'] not in (None, 'deflate'):
        raise ValueError('websocket_compression must be null or deflate')
    if topology == 'pc-direct' and 'max_initial_joint_step' in cfg:
        value = float(cfg['max_initial_joint_step'])
        if not math.isfinite(value) or value <= 0:
            raise ValueError('max_initial_joint_step must be finite and positive')
    if topology == 'pc-direct' and 'gripper_boundary_tolerance' in cfg:
        value = float(cfg['gripper_boundary_tolerance'])
        if not math.isfinite(value) or value < 0:
            raise ValueError('gripper_boundary_tolerance must be finite and nonnegative')
    if topology == 'pc-direct':
        for name in ('control_frequency', 'request_timeout', 'max_joint_step'):
            if not math.isfinite(float(cfg[name])) or float(cfg[name]) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if int(cfg.get('max_cycles', 0)) < 0:
            raise ValueError('max_cycles must be nonnegative')
        smoothing = cfg.get('smoothing', {})
        if not isinstance(smoothing, dict):
            raise ValueError('smoothing must be an object')
        if smoothing.get('enabled', False):
            limits = {'frequency': 250, 'max_velocity': 7,
                      'max_acceleration': 20, 'response_time': None,
                      'stale_timeout': None}
            for name, maximum in limits.items():
                try:
                    value = float(smoothing[name])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f'smoothing.{name} must be numeric') from exc
                if not math.isfinite(value) or value <= 0 or (
                        maximum is not None and value > maximum):
                    raise ValueError(f'invalid smoothing.{name}: {value}')
            if float(smoothing['stale_timeout']) <= 0.6:
                raise ValueError('smoothing.stale_timeout must exceed 0.6s')
        if not cfg.get('server_host') or not 0 < int(cfg['server_port']) < 65536:
            raise ValueError('Set server_host/server_port')
        if execute or observe:
            devices = [cfg.get(k) for k in ('camera_head', 'camera_left_wrist', 'camera_right_wrist')]
            if any(d is None or d == '' for d in devices) or len(set(devices)) != 3:
                raise ValueError('Set three distinct camera devices before opening hardware')
            if not cfg.get('left_can') or not cfg.get('right_can') or cfg['left_can'] == cfg['right_can']:
                raise ValueError('Set distinct left_can/right_can')
        return [os.environ.get('A1Z_CLIENT_PYTHON', sys.executable),
                str(ROOT/'a1z_runtime/client/dual_arm_client.py'), '--config', str(path),
                *(['--execute'] if execute else ['--observe-only'] if observe else [])]
    if topology == 'pc-direct-single-left':
        for name in ('control_frequency', 'request_timeout', 'max_joint_step'):
            if not math.isfinite(float(cfg[name])) or float(cfg[name]) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if 'max_initial_joint_step' in cfg:
            value = float(cfg['max_initial_joint_step'])
            if not math.isfinite(value) or value <= 0:
                raise ValueError('max_initial_joint_step must be finite and positive')
        if 'gripper_boundary_tolerance' in cfg:
            value = float(cfg['gripper_boundary_tolerance'])
            if not math.isfinite(value) or value < 0:
                raise ValueError('gripper_boundary_tolerance must be finite and nonnegative')
        if int(cfg.get('max_cycles', 0)) < 0:
            raise ValueError('max_cycles must be nonnegative')
        smoothing = cfg.get('smoothing', {})
        if not isinstance(smoothing, dict):
            raise ValueError('smoothing must be an object')
        if smoothing.get('enabled', False):
            limits = {'frequency': 250, 'max_velocity': 7,
                      'max_acceleration': 20, 'response_time': None,
                      'stale_timeout': None}
            for name, maximum in limits.items():
                try:
                    value = float(smoothing[name])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f'smoothing.{name} must be numeric') from exc
                if not math.isfinite(value) or value <= 0 or (
                    maximum is not None and value > maximum
                ):
                    raise ValueError(f'invalid smoothing.{name}: {value}')
            if float(smoothing['stale_timeout']) <= 0.6:
                raise ValueError('smoothing.stale_timeout must exceed 0.6s')
        if not cfg.get('server_host') or not 0 < int(cfg['server_port']) < 65536:
            raise ValueError('Set server_host/server_port')
        if execute or observe:
            devices = [cfg.get(k) for k in ('camera_head', 'camera_left_wrist')]
            if any(d is None or d == '' for d in devices) or len(set(devices)) != 2:
                raise ValueError('Set two distinct camera devices before opening hardware')
            if not cfg.get('left_can'):
                raise ValueError('Set left_can')
        return [os.environ.get('A1Z_CLIENT_PYTHON', sys.executable),
                str(ROOT/'a1z_runtime/client/single_client.py'), '--config', str(path),
                *(['--execute'] if execute else ['--observe-only'] if observe else []),
                *(['--record-dir', str(record_dir),
                   '--record-every', str(record_every),
                   '--record-max-frames', str(record_max_frames)]
                  if record_dir is not None else [])]
    raise ValueError('topology must be pc-direct or pc-direct-single')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/deployment' / f'{DEPLOYMENT_TOPOLOGY}.local.json')
    p.add_argument('--check', action='store_true', help='Validate and print command only; no hardware or network')
    p.add_argument('--device-check', action='store_true', help='Read-only CAN identity and camera checks; no SDK or policy server')
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--execute', action='store_true', help='Enable model action execution')
    modes.add_argument('--observe-only', action='store_true', help='PC SDK: starts hardware and reads observations, no model actions')
    p.add_argument('--record-dir', type=Path, help='Timestamped dual-camera JPEGs and action trace')
    p.add_argument('--record-every', type=int, default=4)
    p.add_argument('--record-max-frames', type=int, default=2400)
    a = p.parse_args()
    try:
        path = a.config.resolve()
        cfg = normalize_single_config(json.loads(path.read_text()))
        if cfg.get('topology') not in (DEPLOYMENT_TOPOLOGY, 'pc-direct-single-left'):
            raise ValueError('Use topology=pc-direct or pc-direct-single')
        cmd = command(cfg, path, a.execute, a.observe_only,
                      a.record_dir, a.record_every, a.record_max_frames)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        p.error(str(exc))
    if a.check:
        print('Configuration OK (not a hardware, dependency or inference test).')
        print(shlex.join(cmd))
        return 0
    if a.device_check:
        if a.execute or a.observe_only:
            p.error('--device-check cannot be combined with --execute or --observe-only')
        cmd = [os.environ.get('A1Z_CLIENT_PYTHON', sys.executable),
               str(ROOT / 'scripts/check_devices.py'), '--config', str(path)]
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT/'a1z-data-collect/a1z-t/GALAXEA-A1Z') + os.pathsep + env.get('PYTHONPATH', '')
    # Replace the launcher so SIGINT cannot kill a child during SDK cleanup.
    os.execvpe(cmd[0], cmd, env)


if __name__ == '__main__':
    sys.exit(main())
