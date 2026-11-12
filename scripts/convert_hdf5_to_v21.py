# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Convert pinned a1z_data_collection dual-arm HDF5 episodes to LeRobot v2.1."""
import argparse
from functools import partial
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import shutil

import cv2
import h5py
import numpy as np

PARTS = {'left_arm': slice(0, 6), 'left_gripper': slice(6, 7),
         'right_arm': slice(7, 13), 'right_gripper': slice(13, 14)}
CAMERAS = ('head_rgb', 'left_wrist_rgb', 'right_wrist_rgb')
SINGLE_LEFT_PARTS = {'left_arm': slice(0, 6), 'left_gripper': slice(6, 7)}


def timestamps(values, name):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
        raise ValueError(f'{name}: need at least two finite, strictly increasing timestamps')
    return values


def align_past(source, target, max_age, name):
    # Zero-order hold: never select a sample completed after the target time.
    indices = np.searchsorted(source, target + 1e-9, side='right') - 1
    if np.any(indices < 0):
        raise ValueError(f'{name}: target precedes source')
    age = np.maximum(0.0, target - source[indices])
    if np.any(age > max_age + 1e-9):
        raise ValueError(f'{name}: alignment age {age.max():.6f}s exceeds {max_age}s; check dropped frames')
    return indices, {'max_age_s': float(age.max()), 'mean_age_s': float(age.mean()),
                     'reused_frames': int(np.sum(np.diff(indices) == 0)),
                     'source_frames': len(source), 'used_frames': len(np.unique(indices))}


def decode_rgb(dataset, index):
    encoded = np.asarray(dataset[int(index)])
    if encoded.dtype != np.uint8 or encoded.ndim != 1 or not encoded.size:
        raise ValueError(f'{dataset.name}[{index}]: expected JPEG uint8 bytes')
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f'{dataset.name}[{index}]: image decode failed')
    h, w = bgr.shape[:2]
    if h % 2 or w % 2:
        raise ValueError(f'{dataset.name}: H.264 requires even width and height, got {w}x{h}')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def inspect_episode(handle, cameras, fps, max_age, max_arm_skew, min_frames, task_override=None, parts=PARTS):
    task = task_override if task_override is not None else handle.attrs.get('task', '')
    if isinstance(task, bytes):
        task = task.decode('utf-8')
    if not isinstance(task, str) or not task.strip():
        raise ValueError('Missing task attribute; supply --task explicitly')
    arrays = {}
    for key in ('state', 'action'):
        array = np.asarray(handle[f'arm/{key}'], dtype=np.float32)
        width = max(span.stop for span in parts.values())
        if array.ndim != 2 or array.shape[1] != width or not np.isfinite(array).all():
            raise ValueError(f'arm/{key}: expected finite [N,{width}] data')
        grippers = [span.start for name, span in parts.items() if name.endswith('_gripper')]
        if np.any(array[:, grippers] < 0) or np.any(array[:, grippers] > 1):
            raise ValueError(f'arm/{key}: SDK grippers must be in [0,1]; no automatic clipping/unit conversion')
        arrays[key] = array
    follower = timestamps(handle['arm/follower_timestamps'][:], 'follower')
    leader = timestamps(handle['arm/leader_timestamps'][:], 'leader')
    if not (len(follower) == len(leader) == len(arrays['state']) == len(arrays['action'])):
        raise ValueError('Arm row counts differ')
    skew = follower - leader
    if np.any(skew < 0) or np.any(skew > max_arm_skew):
        raise ValueError(f'Leader/follower timing differs from collector contract; max skew={skew.max():.6f}s')
    streams = {'arm': follower}
    for out, name in cameras.items():
        group = handle[f'cameras/{name}']
        streams[out] = timestamps(group['timestamps'][:], name)
        if len(group['frames']) != len(streams[out]):
            raise ValueError(f'{name}: image/timestamp counts differ')
    start = max(t[0] for t in streams.values())
    end = min(t[-1] for t in streams.values())
    if end < start:
        raise ValueError('Streams have no overlapping time range')
    grid = start + np.arange(int(np.floor((end - start) * fps)) + 1) / fps
    if len(grid) < min_frames:
        raise ValueError(f'Only {len(grid)} aligned frames; require at least {min_frames}')
    indices, alignment = {}, {}
    for key, stream in streams.items():
        indices[key], alignment[key] = align_past(stream, grid, max_age, key)
        alignment[key].update(trim_start_s=float(start-stream[0]), trim_end_s=float(stream[-1]-grid[-1]))
    # Decode every selected source frame during preflight, not merely the first image.
    shapes = {}
    for out, name in cameras.items():
        frames = handle[f'cameras/{name}/frames']
        for i in np.unique(indices[out]):
            shape = decode_rgb(frames, i).shape
            if out in shapes and shapes[out] != shape:
                raise ValueError(f'{name}: resolution changes inside episode')
            shapes[out] = shape
    return {'task': task.strip(), 'frames': len(grid), 'indices': indices, 'arrays': arrays,
            'shapes': shapes, 'report': {'frames': len(grid), 'alignment': alignment,
                                      'max_arm_skew_s': float(skew.max())}}


def feature_schema(shapes, parts=PARTS):
    features = {}
    for group in ('observation.state', 'action'):
        for name, span in parts.items():
            size = span.stop - span.start
            features[f'{group}.{name}'] = {'dtype': 'float32', 'shape': (size,),
                                           'names': [f'{name}_{i}' for i in range(size)]}
    for name, shape in shapes.items():
        features[f'observation.images.{name}'] = {'dtype': 'video', 'shape': tuple(shape),
                                                 'names': ['height', 'width', 'channels']}
    return features


def writer_class():
    if version('lerobot') != '0.3.3':
        raise ValueError('Conversion requires lerobot==0.3.3; use requirements-convert.txt in a separate environment')
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, CODEBASE_VERSION
    from lerobot.datasets.video_utils import encode_video_frames
    from lerobot.datasets.utils import write_info
    if CODEBASE_VERSION != 'v2.1':
        raise ValueError('Writer does not produce v2.1')

    class H264Dataset(LeRobotDataset):
        # Adapt the v0.3.3 encoding hook to select H.264; metadata and statistics
        # remain produced by the official writer (Apache-2.0).
        def _save_episode_table(self, episode_buffer, episode_index):
            # v2.1's (1,) features are scalar Arrow columns. Keep [N,1] arrays
            # for official statistics, but explicitly flatten the table copy.
            table_buffer = dict(episode_buffer)
            for group in ('observation.state', 'action'):
                for key in tuple(table_buffer):
                    if key.startswith(f'{group}.') and key.endswith('_gripper'):
                        table_buffer[key] = table_buffer[key].reshape(-1)
            super()._save_episode_table(table_buffer, episode_index)

        def encode_episode_videos(self, episode_index):
            for key in self.meta.video_keys:
                path = self.root / self.meta.get_video_file_path(episode_index, key)
                images = self._get_image_file_path(episode_index, key, 0).parent
                encode_video_frames(images, path, self.fps, vcodec='h264', crf=18, overwrite=True)
                shutil.rmtree(images)
            if episode_index == 0:
                self.meta.update_video_info()
                write_info(self.meta.info, self.meta.root)
    return H264Dataset


def source_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(partial(stream.read, 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def convert(args):
    if not str(args.input).strip():
        raise ValueError('--input must not be empty')
    source = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else None
    if not source.is_dir():
        raise ValueError('--input must be a directory containing completed episode_*.hdf5 files')
    files = sorted((p for p in source.rglob('episode_*.hdf5')
                    if re.fullmatch(r'episode_\d+\.hdf5', p.name)),
                   key=lambda p: (str(p.parent.relative_to(source)), int(p.stem.split('_')[-1])))
    if not files:
        raise ValueError('No completed episode_N.hdf5 files found')
    if not args.check_only and output is None:
        raise ValueError('--output is required for conversion')
    stage = output.with_name(output.name + '.incomplete') if output else None
    if output and (output.exists() or output.is_symlink() or stage.exists() or stage.is_symlink()):
        raise ValueError('Output or .incomplete directory already exists; choose a new output path')
    if args.max_age <= 0 or args.max_arm_skew <= 0 or not np.isfinite([args.max_age, args.max_arm_skew]).all():
        raise ValueError('Timing limits must be finite and positive')
    if args.min_frames < 2:
        raise ValueError('--min-frames must be at least 2')
    single_left = getattr(args, 'single_left', False)
    parts = SINGLE_LEFT_PARTS if single_left else PARTS
    names = CAMERAS[:2] if single_left else CAMERAS
    cameras = dict(zip(names, (args.head, args.left_wrist, args.right_wrist)))
    if len(set(cameras.values())) != len(cameras):
        raise ValueError('Choose distinct source cameras')
    inspect = partial(inspect_episode, cameras=cameras, fps=args.fps, max_age=args.max_age,
                      max_arm_skew=args.max_arm_skew, min_frames=args.min_frames,
                      task_override=args.task, parts=parts)
    reports, shapes = [], None
    # Inspect all episodes before creating output. Only one episode's arrays are held at a time.
    for path in files:
        with h5py.File(path, 'r') as handle:
            item = inspect(handle)
        if shapes is not None and shapes != item['shapes']:
            raise ValueError(f'{path.name}: camera dimensions differ between episodes')
        shapes = item['shapes']
        reports.append(dict(item['report'], source=str(path.relative_to(source)), sha256=source_hash(path)))
        print(f'Checked {path.relative_to(source)}: {item["frames"]} frames; ages=' +
              ', '.join(f'{k}:{v["max_age_s"]:.4f}s' for k, v in item['report']['alignment'].items()), flush=True)
    report = {'format': 'v2.1', 'fps': args.fps, 'embodiment': 'single_left' if single_left else 'dual',
              'camera_mapping': cameras, 'writer': 'lerobot==0.3.3',
              'alignment': 'past-sample zero-order hold; paired state/action at follower timestamp',
              'max_age_s': args.max_age, 'max_arm_skew_s': args.max_arm_skew,
              'joint_unit': 'radians', 'gripper_unit': 'SDK [0,1]', 'action_source': 'arm/action',
              'episodes': reports}
    if args.check_only:
        print('HDF5 check passed; no dataset written. No training or hardware validation performed.')
        return report
    Writer = writer_class()
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = Writer.create(repo_id='local/a1z', root=stage, fps=args.fps, robot_type='a1z',
                            features=feature_schema(shapes, parts), video_backend='pyav')
    (stage/'INCOMPLETE').write_text('Conversion/checks are not complete. Do not train on this directory.\n')
    try:
        for path, entry in zip(files, reports):
            if source_hash(path) != entry['sha256']:
                raise ValueError(f'{path.name}: input changed after preflight; stop recording before conversion')
            with h5py.File(path, 'r') as handle:
                item = inspect(handle)
                for i in range(item['frames']):
                    arm_i = item['indices']['arm'][i]
                    frame = {f'{prefix}.{part}': item['arrays'][group][arm_i, span].copy()
                             for group, prefix in [('state', 'observation.state'), ('action', 'action')]
                             for part, span in parts.items()}
                    for name, camera in cameras.items():
                        frame[f'observation.images.{name}'] = decode_rgb(handle[f'cameras/{camera}/frames'], item['indices'][name][i])
                    dataset.add_frame(frame, task=item['task'])
                dataset.save_episode()
            if source_hash(path) != entry['sha256']:
                raise ValueError(f'{path.name}: input changed while converting')
            print(f'Written episode {dataset.num_episodes}/{len(files)}', flush=True)
        (stage/'conversion_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
        from check_v21_dataset import check_dataset
        validation = check_dataset(stage, allow_incomplete=True)
        (stage/'validation_report.json').write_text(json.dumps(validation, indent=2)+'\n')
        # Rename only fully checked output; failed staging stays marked incomplete.
        if output.exists():
            raise ValueError('Output appeared during conversion; refusing to replace it')
        stage.rename(output)
        (output/'INCOMPLETE').unlink()
    finally:
        dataset.stop_image_writer()
    print(f'Converted and checked: {output}\nSet A1Z_DATASET to this directory.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Directory of completed episode_N.hdf5 files (recursive)')
    parser.add_argument('--output', help='New output dataset directory; never overwrites')
    parser.add_argument('--fps', type=int, choices=(15, 30), default=30)
    parser.add_argument('--head', default='cam_high')
    parser.add_argument('--left-wrist', default='cam_left_wrist')
    parser.add_argument('--right-wrist', default='cam_right_wrist')
    parser.add_argument('--single', '--single-left', dest='single_left', action='store_true', help='7D single arm and head/wrist cameras')
    parser.add_argument('--max-age', type=float, default=0.05, help='Maximum sample age in seconds')
    parser.add_argument('--max-arm-skew', type=float, default=0.02, help='Maximum leader-to-follower read delay')
    parser.add_argument('--min-frames', type=int, default=32)
    parser.add_argument('--task', help='Explicitly override every episode task (otherwise read HDF5 task)')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    try:
        convert(args)
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        parser.exit(1, f'Conversion failed: {exc}\n')


if __name__ == '__main__':
    main()
