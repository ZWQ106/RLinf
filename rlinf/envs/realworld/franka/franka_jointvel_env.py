# Copyright 2025 The RLinf Authors.
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

"""Joint-velocity Franka env (pi05_droid-native action space).

Reuses FrankaEnv's hardware/camera/reset/reward machinery but replaces the
cartesian EE-delta action with an 8-D joint-velocity action [dq(7), gripper]
and exposes a joint-space observation (joint_position + gripper_position).
Selected via gym id "FrankaJointVelEnv-v1"."""

import copy
import time

import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.franka.franka_env import FrankaEnv


class FrankaJointVelEnv(FrankaEnv):
    def _init_action_obs_spaces(self):
        # Build the parent's spaces first so that the camera "frames" space
        # (derived from self._camera_infos) and the safety boxes are set up,
        # then overwrite the action/observation spaces with the joint-space
        # variant. Runs in __init__ regardless of is_dummy, so the dummy
        # _get_observation() samples from the overridden joint-space space.
        super()._init_action_obs_spaces()

        self.action_space = gym.spaces.Box(
            low=-np.ones(8, dtype=np.float32),
            high=np.ones(8, dtype=np.float32),
        )
        frames_space = self.observation_space["frames"]
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "joint_position": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,)
                        ),
                        "gripper_position": gym.spaces.Box(0.0, 1.0, shape=(1,)),
                    }
                ),
                "frames": frames_space,
            }
        )
        self._base_observation_space = copy.deepcopy(self.observation_space)

    def get_arm_joint_position(self) -> np.ndarray:
        """Current 7-D arm joint positions (rad), served from the state cached by
        step()/reset(). Reusing the cache (instead of a fresh get_state) drops one
        controller RPC per step; staleness is at most one control period (~66 ms),
        negligible for the P-control velocity in GelloJointIntervention."""
        return np.asarray(self._franka_state.arm_joint_position, dtype=np.float64)

    def step(self, action: np.ndarray):
        """Joint-velocity step. action = [dq0..dq6 (joint velocity, [-1,1]),
        gripper (0=open, 1=close)] — pi05_droid-native, sent via
        controller.step_joint_velocity. No cartesian/IK path."""
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # Joint-vel path carries the gripper inside the 8-D action (no discrete
        # _end_effector_action), so there's no gripper-change signal; pass False
        # so an enabled gripper_penalty doesn't fire on every step.
        is_ee_action_effective = False
        if not self.config.is_dummy:
            # One actor RPC: command the velocity and read the resulting state
            # together (replaces separate move + get_state; the teleop wrapper's
            # joint read is now served from this cached state) -> 3 controller
            # RPCs/step down to 1, steadying the command cadence. The no-op
            # _clear_error() RPC is dropped (polymetis impedance self-recovers).
            self._franka_state = self._controller.step_joint_velocity(
                action.astype(np.float32)
            ).wait()[0]
        self._num_steps += 1
        step_time = time.time() - start_time
        time.sleep(max(0, (1.0 / self.config.step_frequency) - step_time))
        observation = self._get_observation()
        reward = self._calc_step_reward(observation, is_ee_action_effective)
        terminated = (reward == 1.0) and (
            self._success_hold_counter >= self.config.success_hold_steps
        )
        truncated = self._num_steps >= self.config.max_num_steps
        reward *= self.config.reward_scale
        return observation, reward, terminated, truncated, {}

    def _get_observation(self) -> dict:
        if self.config.is_dummy:
            return self._base_observation_space.sample()
        frames = self._get_camera_frames()
        grip01 = float(self._franka_state.gripper_position) / 255.0
        state = {
            "joint_position": np.asarray(
                self._franka_state.arm_joint_position, dtype=np.float64
            ),
            "gripper_position": np.array([grip01], dtype=np.float64),
        }
        return copy.deepcopy({"state": state, "frames": frames})
