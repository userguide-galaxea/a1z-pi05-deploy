# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Discover CAN adapter identities or check CAN/cameras; never import the SDK."""
import argparse
import json
from pathlib import Path
import sys
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'a1z_runtime/client'))
from device_checks import cameras_from_config, check_can, link_status, usb_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--list-can', action='store_true')
    mode.add_argument('--config', type=Path, help='Collection YAML or deployment JSON')
    args = parser.parse_args()
    cameras = None
    try:
        if args.list_can:
            rows = []
            for node in sorted(Path('/sys/class/net').iterdir()):
                status = link_status(node.name)
                if status.get('linkinfo', {}).get('info_kind') == 'can':
                    rows.append({'interface': node.name, **usb_identity(node.name),
                                 'state': status['linkinfo']['info_data'].get('state'),
                                 'bitrate': status['linkinfo']['info_data'].get('bittiming', {}).get('bitrate')})
            print(json.dumps(rows, indent=2))
            return 0
        import yaml
        cfg = yaml.safe_load(args.config.read_text())
        check_can(cfg)
        cameras = cameras_from_config(cfg)
        cameras.start()
        for spec in cameras.specs:
            print(f'Camera verified: {spec.name} RGB {spec.width}x{spec.height}', flush=True)
        cameras.frames()
        print('Device checks passed; no CAN frames sent, SDK imported or motors enabled.', flush=True)
        return 0
    except KeyboardInterrupt:
        print('Device check cancelled; no motors enabled.', flush=True)
        return 130
    except (ValueError, RuntimeError, TimeoutError, OSError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(f'Device check failed: {exc}', file=sys.stderr)
        return 2
    finally:
        if cameras is not None:
            cameras.close()


if __name__ == '__main__':
    raise SystemExit(main())
