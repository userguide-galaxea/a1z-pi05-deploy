# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""LeRobot v2.1 configuration for A1Z single and dual absolute-joint datasets."""

import dataclasses
import pathlib

from openpi.training import config as _config
from openpi import transforms

from pi05_a1z.robot_adapter import A1ZInputs, A1ZOutputs, Mode, RepackA1ZDataset
from pi05_a1z.robot_adapter import action_sequence_keys, delta_action_mask


@dataclasses.dataclass(frozen=True)
class LeRobotA1ZDataConfig(_config.DataConfigFactory):
    mode: Mode = "single"
    use_delta_joint_actions: bool = True
    default_prompt: str | None = None

    def create(self, assets_dirs: pathlib.Path, model_config) -> _config.DataConfig:
        # Statistics are specific to this dataset and to its delta/absolute setting.
        # Do not reuse G0.5, ALOHA or the pi05 base model's normalization assets.
        data_transforms = transforms.Group(
            inputs=[A1ZInputs(self.mode)],
            outputs=[A1ZOutputs(self.mode)],
        )
        if self.use_delta_joint_actions:
            mask = delta_action_mask(self.mode)
            data_transforms = data_transforms.push(
                inputs=[transforms.DeltaActions(mask)],
                outputs=[transforms.AbsoluteActions(mask)],
            )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=transforms.Group(inputs=[RepackA1ZDataset(self.mode)]),
            data_transforms=data_transforms,
            model_transforms=_config.ModelTransformFactory(default_prompt=self.default_prompt)(model_config),
            action_sequence_keys=action_sequence_keys(self.mode),
            prompt_from_task=True,
        )
