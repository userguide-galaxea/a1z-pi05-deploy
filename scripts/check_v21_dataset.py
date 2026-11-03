# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Check A1Z v2.1 tables, metadata and every video frame without loading a model."""
import argparse
import json
from pathlib import Path
import sys

import av
import numpy as np
import pyarrow.parquet as pq

PART_DIMS = {'left_arm': 6, 'left_gripper': 1, 'right_arm': 6, 'right_gripper': 1}
IMAGE_KEYS = tuple(f'observation.images.{name}' for name in ('head_rgb', 'left_wrist_rgb', 'right_wrist_rgb'))
SINGLE_LEFT_PART_DIMS = {'left_arm': 6, 'left_gripper': 1}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def jsonlines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def local_path(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), f'Dataset path escapes root: {relative}')
    require(path.is_file(), f'Missing file: {relative}')
    return path


def check_dataset(root, allow_incomplete=False):
    root = Path(root).expanduser().resolve()
    require(allow_incomplete or not (root/'INCOMPLETE').exists(), 'Dataset is INCOMPLETE')
    info = json.loads((root/'meta/info.json').read_text())
    require(info.get('codebase_version') == 'v2.1', 'Expected real LeRobot v2.1, not v3')
    fps = info['fps']
    require(isinstance(fps, (int, float)) and np.isfinite(fps) and fps > 0, 'Invalid fps')
    episodes = jsonlines(root/'meta/episodes.jsonl')
    tasks = jsonlines(root/'meta/tasks.jsonl')
    stats = jsonlines(root/'meta/episodes_stats.jsonl')
    require(len(episodes) > 0 and len(episodes) == info['total_episodes'] == len(stats), 'Episode/stat counts differ')
    task_ids = {int(t['task_index']) for t in tasks}
    require(task_ids == set(range(len(tasks))) and len(tasks) == info['total_tasks'] and bool(tasks), 'Invalid task index table')
    require(all(isinstance(t['task'], str) and t['task'].strip() for t in tasks), 'Empty task text')
    features = info['features']
    single_left = 'action.right_arm' not in features and 'action.right_gripper' not in features
    part_dims = SINGLE_LEFT_PART_DIMS if single_left else PART_DIMS
    image_keys = IMAGE_KEYS[:2] if single_left else IMAGE_KEYS
    vectors = {f'{group}.{part}': n for group in ('observation.state', 'action') for part, n in part_dims.items()}
    arm_features = {key for key in features if key.startswith(('action.', 'observation.state.'))}
    require(arm_features == set(vectors), 'Inconsistent arm features')
    for key, n in vectors.items():
        require(features.get(key, {}).get('dtype') == 'float32' and features[key]['shape'] == [n], f'Invalid feature: {key}')
    for key in image_keys:
        shape = features.get(key, {}).get('shape', [])
        require(features.get(key, {}).get('dtype') == 'video' and len(shape) == 3 and shape[2] == 3, f'Invalid RGB video feature: {key}')
    stats_by_episode = {row['episode_index']: row['stats'] for row in stats}
    require(len(stats_by_episode) == len(stats), 'Duplicate episode statistics')
    total = 0
    for episode_index, ep in enumerate(episodes):
        require(ep['episode_index'] == episode_index, 'Episode indices must be contiguous')
        length = ep['length']
        require(isinstance(length, int) and length > 0, 'Empty/invalid episode')
        fmt = {'episode_index': episode_index, 'episode_chunk': episode_index // info['chunks_size']}
        table = pq.read_table(local_path(root, info['data_path'].format(**fmt))).to_pydict()
        require(len(table['index']) == length, f'Episode {episode_index}: parquet length mismatch')
        require(np.array_equal(table['index'], np.arange(total, total+length)), 'Invalid global frame index')
        require(np.array_equal(table['frame_index'], np.arange(length)), 'Invalid frame index')
        require(np.all(np.asarray(table['episode_index']) == episode_index), 'Wrong episode index')
        require(np.allclose(table['timestamp'], np.arange(length)/fps, atol=1e-4, rtol=0), 'Nonuniform frame timestamps')
        require(set(table['task_index']).issubset(task_ids), 'Unknown task index')
        task_map = {int(t['task_index']): t['task'] for t in tasks}
        require(set(ep['tasks']) == {task_map[i] for i in table['task_index']}, 'Episode tasks disagree with parquet')
        for key, n in vectors.items():
            array = np.asarray(table[key])
            if n == 1 and array.ndim == 1:
                array = array[:, None]  # Official v2.1 stores one-dimensional features as scalars.
            require(array.shape == (length, n) and np.isfinite(array).all(), f'Invalid vector values: {key}')
            if key.endswith('_gripper'):
                require(np.all((array >= 0) & (array <= 1)), f'{key}: expected SDK [0,1] gripper units')
        ep_stats = stats_by_episode[episode_index]
        for key in (*vectors, *image_keys):
            require(key in ep_stats, f'Missing normalization statistics: {key}')
            for stat in ('min', 'max', 'mean', 'std', 'count'):
                require(stat in ep_stats[key], f'Missing {stat} statistics: {key}')
                values = np.asarray(ep_stats[key][stat], dtype=float)
                require(values.size > 0 and np.isfinite(values).all(), f'Invalid statistics: {key}/{stat}')
                if key in vectors and stat != 'count':
                    require(values.shape == (vectors[key],), f'Invalid statistics shape: {key}/{stat}')
                if stat in ('std', 'count'):
                    require(np.all(values >= 0), f'Negative statistics: {key}/{stat}')
            if key in vectors:
                array = np.asarray(table[key], dtype=float).reshape(length, vectors[key])
                for name, expected in [('min', array.min(0)), ('max', array.max(0)), ('mean', array.mean(0)), ('std', array.std(0))]:
                    require(np.allclose(ep_stats[key][name], expected, atol=1e-5, rtol=1e-4), f'Statistics disagree with data: {key}/{name}')
                require(np.all(np.asarray(ep_stats[key]['count']) == length), f'Wrong stats count: {key}')
        for key in image_keys:
            path = local_path(root, info['video_path'].format(**fmt, video_key=key))
            count = 0
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                require(abs(float(stream.average_rate) - fps) < 1e-6, f'Wrong video FPS: {key}')
                for frame in container.decode(stream):
                    require(frame.time is not None and abs(frame.time-count/fps) <= 1e-4, f'Video timing mismatch: {key}')
                    require(list(frame.to_ndarray(format='rgb24').shape) == features[key]['shape'], f'Video shape mismatch: {key}')
                    count += 1
            require(count == length, f'Video frame count differs: {key}: {count} != {length}')
        total += length
    require(total == info['total_frames'], 'Total frame count mismatch')
    require(info['total_videos'] == len(image_keys)*len(episodes), 'Total video count mismatch')
    return {'format': 'v2.1', 'episodes': len(episodes), 'frames': total, 'fps': fps,
            'all_video_frames_decoded': True, 'vector_statistics_checked': True,
            'gpu_training_tested': False, 'physical_alignment_reviewed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    args = parser.parse_args()
    try:
        root = args.dataset.expanduser().resolve()
        report = check_dataset(root)
        print(json.dumps(report, indent=2))
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        parser.exit(1, f'Dataset check failed: {exc}\n')


if __name__ == '__main__':
    main()
