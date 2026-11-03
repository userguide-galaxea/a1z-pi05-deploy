# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Read-only CAN identity checks and camera configuration for single/dual mode."""
import json
import math
from pathlib import Path
import subprocess
import warnings

from camera_capture import CameraGroup, CameraSpec


def hardware_options(cfg):
    options = cfg.get('hardware_checks', {})
    if not isinstance(options, dict):
        raise ValueError('hardware_checks must be an object')
    for key, default in (('camera_startup_timeout', 12), ('camera_frame_timeout', 1)):
        value = options.get(key, default)
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f'hardware_checks.{key} must be finite and positive')
    serials = options.get('can_adapter_serials')
    if serials is not None and not isinstance(serials, dict):
        raise ValueError('hardware_checks.can_adapter_serials must be an object or null')
    return options


def can_roles(cfg):
    if 'arm_cfg' in cfg:
        arm = cfg['arm_cfg']
        if arm['collection_type'] == 'one_master_slave':
            return {'single': arm.get('follower_can', arm.get('follower_can_left'))}
        if arm['collection_type'] != 'two_master_slave':
            raise ValueError('Expected one_master_slave or two_master_slave')
        return {side: arm.get('follower_can_' + side) for side in ('left', 'right')}
    if cfg.get('topology') in ('pc-direct-single', 'pc-direct-single-left'):
        return {'single': cfg.get('can_channel', cfg.get('left_can'))}
    if cfg.get('topology') != 'pc-direct':
        raise ValueError('Expected single or dual PC configuration')
    return {side: cfg.get(side + '_can') for side in ('left', 'right')}


def usb_identity(channel, net_root=Path('/sys/class/net')):
    if not isinstance(channel, str) or not channel or Path(channel).name != channel:
        raise ValueError(f'Invalid CAN interface: {channel!r}')
    node = net_root / channel
    if not node.exists():
        raise ValueError(f'CAN interface missing: {channel}')
    device = (node / 'device').resolve()
    serial = None
    for parent in (device, *device.parents):
        if (parent / 'idVendor').is_file():
            if (parent / 'serial').is_file():
                serial = (parent / 'serial').read_text().strip()
            break
    return {'adapter_serial': serial, 'device_path': str(device)}


def link_status(channel):
    return json.loads(subprocess.check_output(
        ['ip', '-details', '-json', 'link', 'show', channel], text=True))[0]


def check_can(cfg, *, identity_reader=usb_identity, status_reader=link_status):
    roles = can_roles(cfg)
    channels = list(roles.values())
    if any(not isinstance(x, str) or not x for x in channels) or len(set(channels)) != len(channels):
        raise ValueError('Set distinct CAN interfaces for every arm')
    options = hardware_options(cfg)
    serials = options.get('can_adapter_serials')
    if serials is not None:
        if set(serials) != set(roles) or any(not isinstance(x, str) or not x.strip() for x in serials.values()):
            raise ValueError(f'Fill can_adapter_serials for {list(roles)} from physically identified adapters; '
                             'use null only for manually verified stable interfaces')
        if len(set(serials.values())) != len(serials):
            raise ValueError('CAN adapter serials must differ')
    else:
        warnings.warn('CAN serial binding disabled: interface names do not establish physical arm roles. '
                      'Verify cabling and stable interface names before enabling motors.', stacklevel=2)
    result = {}
    for role, channel in roles.items():
        identity = identity_reader(channel)
        actual = identity['adapter_serial']
        if serials is not None and actual != serials[role]:
            raise ValueError(f'CAN identity mismatch for {role}: {channel} serial={actual!r}, '
                             f'expected={serials[role]!r}; robot startup refused')
        result[role] = {'interface': channel, **identity}
    for role, channel in roles.items():
        status = status_reader(channel)
        info = status.get('linkinfo', {})
        data = info.get('info_data', {})
        if info.get('info_kind') != 'can' or 'UP' not in status.get('flags', []) or data.get('bittiming', {}).get('bitrate') != 1000000:
            raise ValueError(f'{role}: {channel} must be an UP SocketCAN interface at 1000000 bit/s')
        if data.get('state') in ('BUS-OFF', 'STOPPED', 'SLEEPING'):
            raise ValueError(f'{role}: {channel} is not ready: {data.get("state")}')
        result[role]['state'] = data.get('state')
        print(f'CAN verified: {role} -> {channel}, adapter serial={result[role]["adapter_serial"]}', flush=True)
    return result


def camera_specs(cfg):
    if 'camera_cfg' in cfg:
        camera = cfg['camera_cfg']
        fps = float(cfg['collector_cfg']['camera_freq'])
        names = camera['camera_names']
        aliases = {'head': 'head_rgb', 'cam_high': 'head_rgb',
                   'wrist': 'left_wrist_rgb', 'cam_left_wrist': 'left_wrist_rgb',
                   'cam_right_wrist': 'right_wrist_rgb'}
        return [CameraSpec(aliases[n], camera['topics'][n],
                           *map(int, camera.get('resolutions', {}).get(n, [camera['width'], camera['height']])), fps)
                for n in names]
    single = cfg.get('topology') in ('pc-direct-single', 'pc-direct-single-left')
    if not single and cfg.get('topology') != 'pc-direct':
        raise ValueError('Expected single or dual PC configuration')
    devices = {'head_rgb': cfg.get('camera_head'),
               'left_wrist_rgb': cfg.get('camera_wrist', cfg.get('camera_left_wrist'))}
    if not single:
        devices['right_wrist_rgb'] = cfg.get('camera_right_wrist')
    width, height = map(int, cfg['image_size'])
    return [CameraSpec(name, device, width, height, float(cfg['control_frequency']))
            for name, device in devices.items()]


def cameras_from_config(cfg):
    options = hardware_options(cfg)
    return CameraGroup(camera_specs(cfg), options.get('camera_startup_timeout', 12),
                       options.get('camera_frame_timeout', 1))
