#!/usr/bin/env python3
# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""A1Z entry points for the pinned official JAX trainer, statistics and server."""
import argparse
import dataclasses
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "openpi/src"),
                str(ROOT / "openpi/packages/openpi-client/src")]
from pi05_a1z.runtime import bind_dataset, dataset_fingerprint


def upstream_script(name):
    path = ROOT / "openpi/scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location("pi05_upstream_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolved_metadata(config):
    mode = config.data.mode
    metadata = dict(config.policy_metadata or {})
    metadata.update(a1z_mode=mode, a1z_action_dim=7 if mode == "single" else 14,
                    action_horizon=config.model.action_horizon, action_representation="absolute",
                    training_delta_joint_actions=config.data.use_delta_joint_actions)
    return dataclasses.replace(config, policy_metadata=metadata)


def validate_config(config, mode, method, bound_repo_id):
    """Keep CLI overrides within the bound A1Z dataset and selected model family."""
    if config.data.mode != mode or config.data.repo_id != bound_repo_id:
        raise ValueError("Do not override data.mode or data.repo-id; select the mode and A1Z_REPO_ID before binding")
    if config.data.assets.asset_id != bound_repo_id or config.data.assets.assets_dir is not None:
        raise ValueError("Keep the bound asset ID and default assets directory; recompute this config's statistics")
    model = config.model
    if not model.pi05 or model.action_dim != 32:
        raise ValueError("A1Z pi0.5 requires model.pi05=True and model.action-dim=32")
    if not isinstance(model.action_horizon, int) or isinstance(model.action_horizon, bool) or model.action_horizon < 1:
        raise ValueError("model.action-horizon must be a positive integer")
    expected = ("gemma_2b_lora", "gemma_300m_lora") if method == "lora" else ("gemma_2b", "gemma_300m")
    if (model.paligemma_variant, model.action_expert_variant) != expected:
        raise ValueError(f"The {method} method requires model variants {expected}")
    return dataclasses.replace(config, freeze_filter=model.get_freeze_filter())


def statistics_signature(config, dataset):
    return {"format": 1, "dataset_metadata_sha256": dataset_fingerprint(dataset),
            "mode": config.data.mode, "repo_id": config.data.repo_id,
            "asset_id": config.data.assets.asset_id,
            "use_delta_joint_actions": config.data.use_delta_joint_actions,
            "action_horizon": config.model.action_horizon}


def statistics_path(config):
    return config.assets_dirs / config.data.assets.asset_id / "norm_stats.json"


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_statistics(config, dataset):
    path = statistics_path(config)
    sidecar = path.with_name("pi05_a1z_norm.json")
    if not path.is_file() or not sidecar.is_file():
        raise ValueError("Compute normalization statistics for this config before training")
    saved = json.loads(sidecar.read_text())
    if saved.get("signature") != statistics_signature(config, dataset):
        raise ValueError("Statistics belong to another dataset, mode or delta/action-horizon setting; rerun norm")
    if saved.get("norm_stats_sha256") != file_sha256(path):
        raise ValueError("Normalization statistics changed after their binding was saved; rerun norm")
    return saved["norm_stats_sha256"]


def named_dataclass(value):
    return {"type": type(value).__name__, "values": dataclasses.asdict(value)}


def save_manifest(config, dataset, stats_sha256):
    return {"format": 2, "config_name": config.name,
            "model": dataclasses.asdict(config.model), "repo_id": config.data.repo_id,
            "use_delta_joint_actions": config.data.use_delta_joint_actions,
            "dataset_metadata_sha256": dataset_fingerprint(dataset),
            "norm_stats_sha256": stats_sha256,
            "policy_metadata": config.policy_metadata,
            "training": {"batch_size": config.batch_size, "seed": config.seed,
                         "optimizer": named_dataclass(config.optimizer),
                         "lr_schedule": named_dataclass(config.lr_schedule),
                         "ema_decay": config.ema_decay, "fsdp_devices": config.fsdp_devices},
            "num_train_steps": config.num_train_steps}


def check_resume_manifest(path, manifest):
    if not path.is_file():
        raise ValueError("Cannot resume without the run's pi05_a1z_run.json manifest")
    saved = json.loads(path.read_text())
    # Extending a run is allowed; model, data, normalization and optimizer semantics must stay fixed.
    old_steps, new_steps = saved.get("num_train_steps"), manifest["num_train_steps"]
    if not isinstance(old_steps, int) or new_steps < old_steps:
        raise ValueError("Resume may extend num-train-steps; it cannot reduce the saved training target")
    saved.pop("num_train_steps")
    current = {k: v for k, v in manifest.items() if k != "num_train_steps"}
    if saved != current:
        raise ValueError("Resume model, dataset, statistics or training parameters differ from the saved run")


def load_checkpoint_config(config_name, checkpoint):
    path = Path(checkpoint).expanduser().resolve(strict=True)
    if not (path / "params").is_dir() or not (path / "assets").is_dir():
        raise ValueError("Select a numeric checkpoint directory containing params/ and assets/")
    manifest = json.loads((path.parent / "pi05_a1z_run.json").read_text())
    if manifest.get("format") != 2:
        raise ValueError("Checkpoint is missing a supported Pi05_A1Z run manifest")
    if manifest["config_name"] != config_name:
        raise ValueError("Requested mode/method differs from the saved training configuration")
    stats = path / "assets" / manifest["repo_id"] / "norm_stats.json"
    if not stats.is_file() or file_sha256(stats) != manifest["norm_stats_sha256"]:
        raise ValueError("Checkpoint normalization statistics differ from the saved run manifest")
    os.environ["A1Z_REPO_ID"] = manifest["repo_id"]
    os.environ["A1Z_DELTA_ACTIONS"] = "1" if manifest["use_delta_joint_actions"] else "0"
    from pi05_a1z.openpi_configs import get_config
    config = get_config(config_name)
    model = dataclasses.replace(config.model, **manifest["model"])
    data = dataclasses.replace(config.data, use_delta_joint_actions=manifest["use_delta_joint_actions"])
    config = dataclasses.replace(config, model=model, data=data)
    mode, method = config_name.removeprefix("a1z_").split("_")
    config = validate_config(config, mode, method, manifest["repo_id"])
    return resolved_metadata(config), path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "norm", "data-check", "serve"))
    parser.add_argument("mode", choices=("single", "dual"))
    parser.add_argument("method", choices=("lora", "full"))
    args, rest = parser.parse_known_args()
    config_name = f"a1z_{args.mode}_{args.method}"
    if args.command == "serve":
        p = argparse.ArgumentParser(description="Serve a saved A1Z pi0.5 checkpoint")
        p.add_argument("checkpoint")
        p.add_argument("--host", default="127.0.0.1")
        p.add_argument("--port", type=int, default=8000)
        a = p.parse_args(rest)
        config, checkpoint = load_checkpoint_config(config_name, a.checkpoint)
        from openpi.policies import policy_config
        from openpi.serving.websocket_policy_server import WebsocketPolicyServer
        policy = policy_config.create_trained_policy(config, checkpoint)
        print(f"Policy server listening on {a.host}:{a.port}", flush=True)
        WebsocketPolicyServer(policy, host=a.host, port=a.port, metadata=config.policy_metadata).serve_forever()
        return
    # Import LeRobot only after its local cache location is defined.
    dataset, info = bind_dataset(args.mode)
    from pi05_a1z.openpi_configs import get_config
    base_config = dataclasses.replace(get_config(config_name), assets_base_dir=str(ROOT / "assets"),
                                      checkpoint_base_dir=str(ROOT / "checkpoints"))
    if args.command == "train":
        import tyro
        dry = "--dry-run" in rest
        rest = [a for a in rest if a != "--dry-run"]
        config = tyro.extras.overridable_config_cli(
            {config_name: (config_name, base_config)}, args=[config_name, *rest])
        config = validate_config(config, args.mode, args.method, base_config.data.repo_id)
        config = resolved_metadata(config)
        if dry:
            print(config)
            print("Configuration parsed; no weights loaded or training performed.")
            return
        from openpi.training import checkpoints
        stats_sha256 = check_statistics(config, dataset)
        manifest = save_manifest(config, dataset, stats_sha256)
        old_initializer = checkpoints.initialize_checkpoint_dir

        def initialize(*values, **kwargs):
            output = config.checkpoint_dir / "pi05_a1z_run.json"
            if config.resume:
                check_resume_manifest(output, manifest)
            result = old_initializer(*values, **kwargs)
            output.write_text(json.dumps(manifest, indent=2) + "\n")
            return result

        checkpoints.initialize_checkpoint_dir = initialize
        try:
            logging.basicConfig(level=logging.INFO)
            upstream_script("train").main(config)
        finally:
            checkpoints.initialize_checkpoint_dir = old_initializer
        return
    if args.command == "norm":
        p = argparse.ArgumentParser(description="Compute A1Z statistics using the official data transforms")
        p.add_argument("--max-frames", type=int)
        p.add_argument("--action-horizon", type=int)
        a = p.parse_args(rest)
        if a.max_frames is not None and a.max_frames < 1:
            p.error("--max-frames must be positive")
        if a.action_horizon is not None:
            base_config = dataclasses.replace(base_config, model=dataclasses.replace(
                base_config.model, action_horizon=a.action_horizon))
        base_config = validate_config(base_config, args.mode, args.method, base_config.data.repo_id)
        from openpi.training import config as official
        official._CONFIGS_DICT[config_name] = dataclasses.replace(base_config, num_workers=0)
        upstream_script("compute_norm_stats").main(config_name, max_frames=a.max_frames)
        path = statistics_path(base_config)
        path.with_name("pi05_a1z_norm.json").write_text(json.dumps({
            "signature": statistics_signature(base_config, dataset),
            "norm_stats_sha256": file_sha256(path), "max_frames": a.max_frames,
        }, indent=2) + "\n")
        return
    p = argparse.ArgumentParser(description="Read actual LeRobot samples and transformed action windows")
    p.add_argument("--samples", type=int, default=5)
    a = p.parse_args(rest)
    if a.samples < 1:
        p.error("--samples must be positive")
    from openpi.training import data_loader
    from pi05_a1z.robot_adapter import RepackA1ZDataset, A1ZInputs
    # The raw reader check does not construct a tokenizer, model or normalizer.
    from openpi.training.config import DataConfig
    from pi05_a1z.robot_adapter import action_sequence_keys
    raw_config = DataConfig(repo_id=base_config.data.repo_id, prompt_from_task=True,
                            action_sequence_keys=action_sequence_keys(args.mode))
    ds = data_loader.create_torch_dataset(raw_config, base_config.model.action_horizon, base_config.model)
    if len(ds) < 1:
        raise ValueError("LeRobot reader returned an empty dataset")
    import numpy as np
    for index in np.linspace(0, len(ds)-1, min(a.samples, len(ds)), dtype=int):
        sample = A1ZInputs(args.mode)(RepackA1ZDataset(args.mode)(ds[int(index)]))
        assert sample["state"].shape == (7 if args.mode == "single" else 14,)
        assert sample["actions"].shape == (base_config.model.action_horizon, sample["state"].size)
        print(f"Sample {index}: state={sample['state'].shape}, actions={sample['actions'].shape}, prompt={sample['prompt']!r}")
    print(f"LeRobot reader + A1Z mapping passed: {info['total_episodes']} episodes; no model or hardware opened.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc
