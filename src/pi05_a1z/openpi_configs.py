# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Project-owned JAX pi0.5 configurations; no upstream source patch is required."""

import os
import re

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import optimizer, weight_loaders

from pi05_a1z.data_config import LeRobotA1ZDataConfig
from pi05_a1z.robot_adapter import action_dimension


def get_configs() -> dict[str, _config.TrainConfig]:
    configs = {}
    for mode in ("single", "dual"):
        for method in ("lora", "full"):
            name = f"a1z_{mode}_{method}"
            lora = method == "lora"
            model = pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora" if lora else "gemma_2b",
                action_expert_variant="gemma_300m_lora" if lora else "gemma_300m",
            )
            repo_id = os.environ.get("A1Z_REPO_ID", f"local/a1z_{mode}")
            if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo_id) or any(
                part in (".", "..") for part in repo_id.split("/")
            ):
                raise ValueError("A1Z_REPO_ID must be a relative namespace/dataset identifier")
            delta = os.environ.get("A1Z_DELTA_ACTIONS", "1")
            if delta not in ("0", "1"):
                raise ValueError("A1Z_DELTA_ACTIONS must be 0 or 1")
            configs[name] = _config.TrainConfig(
                name=name,
                project_name="pi05_a1z",
                model=model,
                data=LeRobotA1ZDataConfig(
                    repo_id=repo_id,
                    mode=mode,
                    use_delta_joint_actions=delta == "1",
                    assets=_config.AssetsConfig(asset_id=repo_id),
                ),
                weight_loader=weight_loaders.CheckpointWeightLoader(
                    os.environ.get("A1Z_BASE_PARAMS", "gs://openpi-assets/checkpoints/pi05_base/params")
                ),
                freeze_filter=model.get_freeze_filter(),
                ema_decay=None if lora else 0.99,
                batch_size=1,
                num_workers=2,
                num_train_steps=20_000 if lora else 30_000,
                save_interval=5_000,
                keep_period=5_000,
                log_interval=10,
                lr_schedule=optimizer.CosineDecaySchedule(
                    warmup_steps=1_000,
                    peak_lr=1e-4 if lora else 2.5e-5,
                    decay_steps=20_000 if lora else 30_000,
                    decay_lr=1e-5 if lora else 2.5e-6,
                ),
                optimizer=optimizer.AdamW(clip_gradient_norm=1.0),
                wandb_enabled=False,
                policy_metadata={
                    "a1z_mode": mode,
                    "a1z_action_dim": action_dimension(mode),
                    "action_horizon": model.action_horizon,
                    "action_representation": "absolute",
                    "training_delta_joint_actions": delta == "1",
                },
            )
    return configs


def get_config(name: str) -> _config.TrainConfig:
    try:
        return get_configs()[name]
    except KeyError as exc:
        raise ValueError(f"Unknown A1Z config {name!r}; choose one of {', '.join(get_configs())}") from exc
