# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DataConfigFactory for the pi05_droid checkpoint.

The DROID observation schema (per `openpi/policies/droid_policy.py`):
- observation/exterior_image_1_left (RGB, uint8)   third-person view
- observation/wrist_image_left      (RGB, uint8)   EE-mounted
- observation/joint_position        (7,)            arm joint angles
- observation/gripper_position      (1,)            gripper width scalar

DroidInputs concatenates joint_position + gripper_position into an 8-dim state,
maps exterior -> base_0_rgb, wrist -> left_wrist_0_rgb, masks the third
right_wrist_0_rgb slot. Actions are 8-dim (7 joint velocity + 1 gripper).

We reuse openpi's `droid_policy.DroidInputs / DroidOutputs` directly so any
upstream change to DROID conventions flows through without code edits here.
"""
import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.policies import droid_policy
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """Wraps openpi's DROID transforms for use by RLinf's openpi integration."""

    default_prompt: str | None = None

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        # Repack from RLinf env's obs_processor output keys to the
        # namespaced keys DroidInputs expects. obs_processor (patched for
        # "pi05_droid") emits observation/image, observation/wrist_image,
        # observation/joint_position, observation/gripper_position, prompt.
        # DroidInputs reads observation/exterior_image_1_left,
        # observation/wrist_image_left, observation/joint_position,
        # observation/gripper_position.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
