# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Configuration and local-binding guards without model libraries or hardware."""

import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pi05_a1z.runtime import bind_dataset, dataset_fingerprint

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("a1z_model_entry_test", ROOT / "scripts/model.py")
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


@dataclasses.dataclass(frozen=True)
class Model:
    pi05: bool = True
    action_dim: int = 32
    action_horizon: int = 16
    paligemma_variant: str = "gemma_2b_lora"
    action_expert_variant: str = "gemma_300m_lora"

    def get_freeze_filter(self):
        return (self.paligemma_variant, self.action_expert_variant)


@dataclasses.dataclass(frozen=True)
class Assets:
    asset_id: str = "local/a1z_single"
    assets_dir: str | None = None


@dataclasses.dataclass(frozen=True)
class Data:
    mode: str = "single"
    repo_id: str = "local/a1z_single"
    use_delta_joint_actions: bool = True
    assets: Assets = dataclasses.field(default_factory=Assets)


@dataclasses.dataclass(frozen=True)
class Schedule:
    peak_lr: float = 1e-4


@dataclasses.dataclass(frozen=True)
class Config:
    name: str = "a1z_single_lora"
    model: Model = dataclasses.field(default_factory=Model)
    data: Data = dataclasses.field(default_factory=Data)
    freeze_filter: object = "stale"
    policy_metadata: dict = dataclasses.field(default_factory=dict)
    batch_size: int = 1
    seed: int = 42
    optimizer: Schedule = dataclasses.field(default_factory=Schedule)
    lr_schedule: Schedule = dataclasses.field(default_factory=Schedule)
    ema_decay: float | None = None
    fsdp_devices: int = 1
    num_train_steps: int = 20_000
    assets_dirs: Path = Path("assets")


def write_dataset(root, frames=4, episodes=1):
    meta = root / "meta"
    meta.mkdir(parents=True)
    features = {}
    for group in ("observation.state", "action"):
        for part, shape in (("left_arm", [6]), ("left_gripper", [1])):
            features[f"{group}.{part}"] = {"shape": shape}
    for camera in ("head_rgb", "left_wrist_rgb"):
        features[f"observation.images.{camera}"] = {"shape": [3, 8, 12]}
    (meta / "info.json").write_text(json.dumps({
        "codebase_version": "v2.1", "total_frames": frames,
        "total_episodes": episodes, "features": features,
    }))
    (meta / "tasks.jsonl").write_text('{"task_index":0,"task":"pick and place the cube"}\n')
    for name in ("episodes.jsonl", "episodes_stats.jsonl"):
        (meta / name).write_text('{"episode_index":0}\n')


class ModelRuntimeTests(unittest.TestCase):
    def test_binding_sets_cache_before_import_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "dataset"
            write_dataset(dataset)
            with patch.dict(os.environ, {"A1Z_DATASET": str(dataset),
                                         "HF_LEROBOT_HOME": str(root / "cache"),
                                         "A1Z_REPO_ID": "local/a1z_single"}):
                actual, _ = bind_dataset("single")
                self.assertEqual(actual, dataset.resolve())
                self.assertEqual((root / "cache/local/a1z_single").resolve(), dataset.resolve())
                other = root / "other"
                write_dataset(other)
                os.environ["A1Z_DATASET"] = str(other)
                with self.assertRaisesRegex(ValueError, "another dataset"):
                    bind_dataset("single")
                self.assertEqual((root / "cache/local/a1z_single").resolve(), dataset.resolve())

    def test_empty_datasets_are_rejected(self):
        for frames, episodes in ((0, 1), (4, 0)):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                write_dataset(root, frames, episodes)
                with patch.dict(os.environ, {"A1Z_DATASET": str(root)}):
                    with self.assertRaisesRegex(ValueError, "at least one"):
                        bind_dataset("single")

    def test_cli_guards_and_recomputed_freeze_filter(self):
        base = Config()
        result = entry.validate_config(base, "single", "lora", "local/a1z_single")
        self.assertEqual(result.freeze_filter, ("gemma_2b_lora", "gemma_300m_lora"))
        invalid = (
            dataclasses.replace(base, data=dataclasses.replace(base.data, mode="dual")),
            dataclasses.replace(base, data=dataclasses.replace(base.data, repo_id="other/data")),
            dataclasses.replace(base, data=dataclasses.replace(base.data, assets=Assets("other/data"))),
            dataclasses.replace(base, model=dataclasses.replace(base.model, pi05=False)),
            dataclasses.replace(base, model=dataclasses.replace(base.model, action_dim=7)),
            dataclasses.replace(base, model=dataclasses.replace(base.model, action_horizon=0)),
            dataclasses.replace(base, model=dataclasses.replace(base.model, paligemma_variant="gemma_2b")),
        )
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                entry.validate_config(config, "single", "lora", "local/a1z_single")

    def test_statistics_bind_dataset_delta_horizon_and_content(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "dataset"
            write_dataset(dataset)
            config = dataclasses.replace(Config(), assets_dirs=root / "assets")
            stats = entry.statistics_path(config)
            stats.parent.mkdir(parents=True)
            stats.write_text('{"norm_stats":{}}\n')
            bound = {"signature": entry.statistics_signature(config, dataset),
                     "norm_stats_sha256": entry.file_sha256(stats)}
            stats.with_name("pi05_a1z_norm.json").write_text(json.dumps(bound))
            self.assertEqual(entry.check_statistics(config, dataset), bound["norm_stats_sha256"])
            for changed in (
                dataclasses.replace(config, data=dataclasses.replace(config.data, use_delta_joint_actions=False)),
                dataclasses.replace(config, model=dataclasses.replace(config.model, action_horizon=8)),
            ):
                with self.assertRaisesRegex(ValueError, "Statistics belong"):
                    entry.check_statistics(changed, dataset)
            stats.write_text('{"norm_stats":{"changed":true}}')
            with self.assertRaisesRegex(ValueError, "changed"):
                entry.check_statistics(config, dataset)
            old = dataset_fingerprint(dataset)
            (dataset / "meta/episodes_stats.jsonl").write_text('{"episode_index":1}\n')
            self.assertNotEqual(old, dataset_fingerprint(dataset))

    def test_resume_requires_manifest_and_allows_only_target_extension(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "dataset"
            write_dataset(dataset)
            config = Config()
            manifest = entry.save_manifest(config, dataset, "stats-hash")
            path = root / "pi05_a1z_run.json"
            with self.assertRaisesRegex(ValueError, "without"):
                entry.check_resume_manifest(path, manifest)
            path.write_text(json.dumps(manifest))
            entry.check_resume_manifest(path, manifest)
            extended = entry.save_manifest(dataclasses.replace(config, num_train_steps=30_000), dataset, "stats-hash")
            entry.check_resume_manifest(path, extended)
            for changed in (
                dataclasses.replace(config, batch_size=2),
                dataclasses.replace(config, lr_schedule=Schedule(2e-4)),
                dataclasses.replace(config, num_train_steps=10_000),
            ):
                with self.assertRaises(ValueError):
                    entry.check_resume_manifest(path, entry.save_manifest(changed, dataset, "stats-hash"))


if __name__ == "__main__":
    unittest.main()
