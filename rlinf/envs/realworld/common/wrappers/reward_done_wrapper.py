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
    """Keyboard start gate + reward/done labelling for teleop collection.

    Start gate, one-stage (default, unchanged): reset homes the arm, then
    blocks until 's' — after which every step is recorded.

    Start gate, two-stage (``release_before_record=True``): 's' *releases* the
    robot (teleop goes live) but recording does NOT begin; the operator drives
    the arm into its true starting pose, then presses 'r' to begin recording.
    The steps in between are stepped through the env — the arm must actually
    move — but the collector drops them, because getting into position is not
    part of the demonstration. Gate states: WAIT -> LIVE -> RUN.
    """

    def __init__(self, env: gym.Env, reward_mode: str = "always_replace",
                 release_before_record: bool = False):
        super().__init__(env)
        self.reward_modifier = 0
        self.listener = KeyboardListener()
        self.reward_mode = reward_mode
        assert self.reward_mode in ["always_replace"]
        self.release_before_record = bool(release_before_record)
        # One-stage runs are always recording once reset returns.
        self._recording = True
        self._warned_no_hook = False

    @property
    def is_recording(self) -> bool:
        """False only during the two-stage release phase (teleop live, frames
        discarded). Published on every step as ``info['recording']``."""
        return self._recording

    def reset(self, *, seed=None, options=None):
        """Reset the inner env (homes the arm), then BLOCK until the operator
        presses 's'. `options.skip_wait_for_start` bypasses the wait
        (collect_real_data sets it on the final reset). Mirrors dosw1's
        free-teleop start gate.

        With ``release_before_record`` the block ends at the *release*, not at
        the start of recording — see the class docstring.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        skip = bool((options or {}).get("skip_wait_for_start", False))
        self._recording = True
        if not skip:
            # Sentinel FILE (not just stdout): Ray buffers actor stdout, so a
            # print may not reach the driver log promptly — the dashboard
            # reads this file via `docker exec cat` to drive the 开始下一条
            # button reliably. stdout line kept for human debugging.
            self._write_gate("WAIT")
            print("WAIT_FOR_START: reset done — press 's' to "
                  + ("release the robot" if self.release_before_record
                     else "start episode"), flush=True)
            self._await_key("s")
            if self.release_before_record:
                self._recording = False
                self._write_gate("LIVE")
                print("ROBOT_RELEASED: teleop live, NOT recording — "
                      "position the arm, then press 'r' to start recording",
                      flush=True)
            else:
                print("EPISODE_START: 's' received", flush=True)
                self._write_gate("RUN")
        return obs, info

    def _await_key(self, want: str) -> None:
        while True:
            if self.listener.get_key() == want:
                return
            time.sleep(0.05)

    def _zero_episode_counters(self) -> None:
        """Hold the env's per-episode counters at zero during the release phase.

        Called on EVERY release-phase step, not just at the transition: the
        operator may spend a while positioning, and `elapsed_steps` would
        otherwise reach `max_episode_steps`, truncate, and trip the env's
        auto-reset — re-homing the arm in the middle of being placed.
        """
        try:
            self.env.unwrapped.begin_recorded_episode()
        except AttributeError:
            # Env predates the hook. Recording still works; the positioning
            # steps just count toward the episode length. Warn once, don't
            # fail the run.
            if not self._warned_no_hook:
                self._warned_no_hook = True
                print("WARN: env has no begin_recorded_episode(); release-phase "
                      "steps will count toward max_episode_steps", flush=True)

    def _begin_recording(self) -> None:
        """Release phase -> recording, with clean per-episode counters."""
        self._recording = True
        self._zero_episode_counters()
        self._write_gate("RUN")
        print("EPISODE_START: 'r' received — recording", flush=True)

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
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`.

        Publishes ``info['recording']`` so the collector knows whether the
        transition belongs in the demonstration.
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        if not self._recording:
            # Release phase: the arm follows the leader so the operator can get
            # into position, but none of this is part of the demonstration —
            # no keyboard labelling, no episode end. Only 'r' matters here.
            self._zero_episode_counters()
            if self.listener.get_key() == "r":
                self._begin_recording()
            # Still False even on the transitioning step: the action was taken
            # while positioning. Recording starts from the NEXT step, with this
            # observation as its starting state.
            info = dict(info)
            info["recording"] = False
            return observation, reward, terminated, truncated, info
        last_intervened, updated_reward, updated_terminated = self.reward_terminated()
        if last_intervened or self.reward_mode == "always_replace":
            reward = updated_reward
        info = dict(info)
        info["recording"] = True
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
    def __init__(self, env, release_before_record: bool = False):
        super().__init__(env, release_before_record=release_before_record)
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
