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

"""Joint-space GELLO teleop: closed-loop P-control velocity.

Unlike GelloIntervention (which runs FrankaFK to an EE pose and emits a
cartesian delta), this wrapper drives the robot joints directly toward the
GELLO leader joints and emits an 8-D joint-velocity action
``[v(7), gripper]`` -- pi05_droid's native action. The same vector is sent to
the robot (via the joint-velocity env step) and recorded as the dataset action.
"""

import time

import gymnasium as gym
import numpy as np


def compute_joint_velocity_action(
    q_gello: np.ndarray,
    q_robot: np.ndarray,
    gripper: float,
    kp: float,
    vmax: float,
) -> np.ndarray:
    """8-D joint-velocity action from a P-controller on the joint error.

    v = clip(kp * (q_gello - q_robot), -vmax, vmax); action[7] = clip(gripper, 0, 1).
    Returns shape (8,). Pure function -- no hardware, unit-tested."""
    q_gello = np.asarray(q_gello, dtype=np.float64).reshape(-1)
    q_robot = np.asarray(q_robot, dtype=np.float64).reshape(-1)
    assert q_gello.shape == (7,) and q_robot.shape == (7,)
    v = np.clip(kp * (q_gello - q_robot), -vmax, vmax)
    grip = float(np.clip(gripper, 0.0, 1.0))
    return np.concatenate([v, [grip]]).astype(np.float64)


class GelloJointIntervention(gym.ActionWrapper):
    """Override the policy action with a GELLO joint-velocity teleop command.

    Args:
        env: wrapped joint-velocity env (must expose ``get_arm_joint_position()``).
        port: GELLO serial port (from env config ``gello_port``).
        kp: P gain mapping joint error (rad) to normalized joint velocity.
        vmax: per-joint velocity clip (normalized units, DROID scales 1.0->max).
        gripper_enabled: whether action[7] carries a gripper command.
    """

    def __init__(self, env, port, kp: float = 4.0, vmax: float = 1.0,
                 gripper_enabled: bool = True):
        super().__init__(env)
        from gello_teleop.gello_teleop_agent import GelloTeleopAgent

        # Inject the FR3 leader calibration into gello's PORT_CONFIG_MAP before
        # opening the port — gello_software ships only stock configs and would
        # otherwise assert "Port /dev/gello not in config map". Keeping the
        # calibration in-repo (not in the ephemeral gello_software install) is
        # what makes it survive a container rebuild.
        from rlinf.envs.realworld.common.gello.fr3_gello_config import (
            register_fr3_gello_config,
        )

        register_fr3_gello_config(port)
        self.agent = GelloTeleopAgent(port=port)
        self.kp = float(kp)
        self.vmax = float(vmax)
        self.gripper_enabled = gripper_enabled
        self.last_intervene = 0.0

    def action(self, action: np.ndarray):
        from rlinf.envs.realworld.common.step_timing import timed as _timed

        with _timed("gello_read"):
            gello_joints, gello_gripper = self.agent.get_action()
        q_gello = np.asarray(gello_joints, dtype=np.float64).reshape(-1)[:7]
        q_robot = np.asarray(
            self.get_wrapper_attr("get_arm_joint_position")(), dtype=np.float64
        ).reshape(-1)[:7]
        gripper = float(gello_gripper) if self.gripper_enabled else 0.0
        expert_a = compute_joint_velocity_action(
            q_gello, q_robot, gripper, self.kp, self.vmax
        )
        if np.linalg.norm(expert_a[:7]) > 0.001 or (
            self.gripper_enabled and abs(expert_a[7]) > 0.5
        ):
            self.last_intervene = time.time()
        if time.time() - self.last_intervene < 0.5:
            return expert_a, True
        return action, False

    def step(self, action):
        new_action, replaced = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        return obs, rew, done, truncated, info
