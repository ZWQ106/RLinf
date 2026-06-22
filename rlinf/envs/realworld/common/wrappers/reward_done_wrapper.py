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

import time
from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import ActType, ObsType

from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener


class BaseKeyboardRewardDoneWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, reward_mode: str = "always_replace"):
        super().__init__(env)
        self.reward_modifier = 0
        self.listener = KeyboardListener()
        self.reward_mode = reward_mode
        assert self.reward_mode in ["always_replace"]

    def reset(self, *, seed=None, options=None):
        """Reset the inner env (homes the arm), then BLOCK until the operator
        presses 's' to start the next episode — gives time to reposition the
        scene and a clear go signal. `options.skip_wait_for_start` bypasses
        the wait (collect_real_data sets it on the final reset). Mirrors
        dosw1's free-teleop start gate.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        skip = bool((options or {}).get("skip_wait_for_start", False))
        if not skip:
            # Sentinel FILE (not just stdout): Ray buffers actor stdout, so a
            # print may not reach the driver log promptly — the dashboard
            # reads this file via `docker exec cat` to drive the 开始下一条
            # button reliably. stdout line kept for human debugging.
            self._write_gate("WAIT")
            print("WAIT_FOR_START: reset done — press 's' to start episode",
                  flush=True)
            while True:
                if self.listener.get_key() == "s":
                    print("EPISODE_START: 's' received", flush=True)
                    break
                time.sleep(0.05)
            self._write_gate("RUN")
        return obs, info

    @staticmethod
    def _write_gate(state: str) -> None:
        try:
            with open("/tmp/collect_gate", "w") as f:
                f.write(state)
        except Exception:
            pass

    def _check_keypress(self) -> tuple[bool, bool, float]:
        raise NotImplementedError

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        last_intervened, updated_reward, updated_terminated = self.reward_terminated()
        if last_intervened or self.reward_mode == "always_replace":
            reward = updated_reward
        return observation, reward, updated_terminated, truncated, info

    def reward_terminated(
        self,
    ) -> tuple[float, bool]:
        last_intervened, terminated, keyboard_reward = self._check_keypress()
        return last_intervened, keyboard_reward, terminated


class KeyboardRewardDoneWrapper(BaseKeyboardRewardDoneWrapper):
    def _check_keypress(self) -> tuple[bool, bool, float]:
        last_intervened = False
        done = False
        reward = 0
        key = self.listener.get_key()
        if key is not None:
            print(f"Key pressed: {key}")
        if key not in ["a", "b", "c"]:
            return last_intervened, done, reward

        last_intervened = True
        if key == "a":
            reward = -1
            done = True
            last_intervened = True
        elif key == "b":
            reward = 0
            last_intervened = True
        elif key == "c":
            reward = 1
            done = True
            last_intervened = True
        return last_intervened, done, reward


class KeyboardRewardDoneMultiStageWrapper(BaseKeyboardRewardDoneWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.stage_rewards = [0, 0.1, 1]

    def reset(self, *, seed=None, options=None):
        self.reward_stage = 0
        return super().reset(seed=seed, options=options)

    def _check_keypress(self) -> tuple[bool, bool, float]:
        last_intervened = False
        done = False
        reward = 0
        key = self.listener.get_key()
        if key is not None:
            print(f"Key pressed: {key}")
        if key == "a":
            self.reward_stage = 0
        elif key == "b":
            self.reward_stage = 1
        elif key == "c":
            self.reward_stage = 2

        if self.reward_stage == 2:
            done = True

        reward = self.stage_rewards[self.reward_stage]
        if key == "q":
            reward = -1
            done = False
        return last_intervened, done, reward
