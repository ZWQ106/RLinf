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
"""GELLO joint teleop with an emulated forward (operator -> robot) delay.

Remote-teleop study: the leader arm is sampled continuously in a background
thread and every sample is stamped with a *send* time and an emulated
*arrival* time (send + tau_f + jitter). The robot-side P-controller only ever
sees what has "arrived", exactly as a follower would on the far end of a
delayed link. Two robot-side policies for consuming the delayed stream:

- ``direct``  use the most recent sample that has arrived (out-of-order
              samples are dropped). This is the naive remote setup.
- ``queued``  MATER-style command queue: play the stream back at a fixed
              playout offset ``playout_s`` behind the newest arrival and
              linearly interpolate between stamped samples, trading a little
              extra delay for a smooth, jitter-free target.

Recorded actions keep the baseline semantics (8-D joint velocity actually
sent), so datasets remain comparable with the zero-delay GELLO baseline.
"""
import threading
import time
from collections import deque
from typing import Optional

import numpy as np

from rlinf.envs.realworld.common.wrappers.gello_joint_intervention import (
    GelloJointIntervention,
    compute_joint_velocity_action,
)


class LeaderSample:
    __slots__ = ("t_send", "t_arrive", "q", "gripper")

    def __init__(self, t_send: float, t_arrive: float, q: np.ndarray, gripper: float):
        self.t_send = t_send
        self.t_arrive = t_arrive
        self.q = q
        self.gripper = gripper


class DelayedLeaderStream:
    """Timestamped leader samples with emulated arrival times.

    ``push`` is called by the sampler; ``latest_arrived`` / ``interpolated``
    are the two consumption policies. Pure logic, no hardware, unit-tested.
    """

    def __init__(
        self,
        tau_f: float,
        jitter_s: float = 0.0,
        jitter_dist: str = "uniform",
        seed: Optional[int] = None,
        horizon_s: float = 10.0,
    ):
        assert tau_f >= 0.0 and jitter_s >= 0.0
        assert jitter_dist in ("uniform", "gaussian")
        self.tau_f = float(tau_f)
        self.jitter_s = float(jitter_s)
        self.jitter_dist = jitter_dist
        self.horizon_s = float(horizon_s)
        self._rng = np.random.default_rng(seed)
        self._lock = threading.Lock()
        self._samples: deque[LeaderSample] = deque()

    def _draw_jitter(self) -> float:
        if self.jitter_s == 0.0:
            return 0.0
        if self.jitter_dist == "uniform":
            return float(self._rng.uniform(0.0, self.jitter_s))
        return float(abs(self._rng.normal(0.0, self.jitter_s)))

    def push(self, q: np.ndarray, gripper: float, t_send: Optional[float] = None) -> LeaderSample:
        t_send = time.time() if t_send is None else float(t_send)
        s = LeaderSample(
            t_send, t_send + self.tau_f + self._draw_jitter(), np.asarray(q, dtype=np.float64), float(gripper)
        )
        with self._lock:
            self._samples.append(s)
            cutoff = t_send - self.horizon_s
            while self._samples and self._samples[0].t_send < cutoff:
                self._samples.popleft()
        return s

    def _arrived(self, now: float) -> list[LeaderSample]:
        with self._lock:
            return [s for s in self._samples if s.t_arrive <= now]

    def latest_arrived(self, now: Optional[float] = None) -> Optional[LeaderSample]:
        """``direct``: newest-by-send-time among arrived samples (late/out-of-order dropped)."""
        now = time.time() if now is None else now
        arrived = self._arrived(now)
        if not arrived:
            return None
        return max(arrived, key=lambda s: s.t_send)

    def interpolated(self, now: Optional[float] = None, playout_s: float = 0.05) -> Optional[LeaderSample]:
        """``queued``: sample the arrived stream at send-time ``t_play`` = newest
        arrived send-time − playout_s, linearly interpolating between the two
        bracketing samples (hold the last one if the buffer ran dry)."""
        now = time.time() if now is None else now
        arrived = sorted(self._arrived(now), key=lambda s: s.t_send)
        if not arrived:
            return None
        t_play = arrived[-1].t_send - playout_s
        prev = None
        for s in arrived:
            if s.t_send >= t_play:
                if prev is None or s.t_send == t_play:
                    return LeaderSample(t_play, now, s.q, s.gripper)
                w = (t_play - prev.t_send) / max(s.t_send - prev.t_send, 1e-9)
                q = prev.q + w * (s.q - prev.q)
                g = prev.gripper + w * (s.gripper - prev.gripper)
                return LeaderSample(t_play, now, q, g)
            prev = s
        return LeaderSample(t_play, now, prev.q, prev.gripper)


class DelayedGelloJointIntervention(GelloJointIntervention):
    """GelloJointIntervention whose leader input is delayed by ``tau_f`` (+ jitter).

    Args:
        tau_f: forward delay in seconds (0 reproduces the baseline wrapper,
            except that the leader is sampled in a thread at ``sample_hz``).
        jitter_s: extra per-sample delay, drawn uniform [0, jitter_s] or
            |N(0, jitter_s)| depending on ``jitter_dist``.
        mode: ``"direct"`` or ``"queued"`` (see module docstring).
        playout_s: playout buffer for ``queued``.
        sample_hz: leader sampling rate of the background thread.
        seed: RNG seed for the jitter draw (recorded in ``info["leader_delay"]``).
        match_tolerance_rad: pose-match gate — no motion until every leader
            joint is within this of the robot (wrapped error) once; the
            per-joint mismatch is logged every second meanwhile. Prevents the
            lunge when the leader rests far from the robot's home pose.
    """

    def __init__(
        self,
        env,
        port,
        kp: float = 4.0,
        vmax: float = 1.0,
        gripper_enabled: bool = True,
        tau_f: float = 0.0,
        jitter_s: float = 0.0,
        jitter_dist: str = "uniform",
        mode: str = "direct",
        playout_s: float = 0.05,
        sample_hz: float = 50.0,
        seed: Optional[int] = 0,
        match_tolerance_rad: float = 0.35,
    ):
        super().__init__(env, port, kp=kp, vmax=vmax, gripper_enabled=gripper_enabled)
        assert mode in ("direct", "queued")
        self.mode = mode
        self.playout_s = float(playout_s)
        self.sample_hz = float(sample_hz)
        self.stream = DelayedLeaderStream(tau_f, jitter_s, jitter_dist, seed)
        self._cfg = {
            "tau_f": float(tau_f),
            "jitter_s": float(jitter_s),
            "jitter_dist": jitter_dist,
            "mode": mode,
            "playout_s": float(playout_s),
            "sample_hz": float(sample_hz),
            "seed": seed,
        }
        self.match_tolerance_rad = float(match_tolerance_rad)
        self._matched = False
        self._last_match_log = 0.0
        self._sampler_stop = threading.Event()
        self._sampler = threading.Thread(target=self._sample_loop, daemon=True)
        self._sampler.start()

    def _sample_loop(self):
        period = 1.0 / self.sample_hz
        while not self._sampler_stop.is_set():
            t0 = time.time()
            try:
                joints, gripper = self.agent.get_action()
            except Exception:
                time.sleep(period)
                continue
            self.stream.push(np.asarray(joints, dtype=np.float64).reshape(-1)[:7], float(gripper), t0)
            time.sleep(max(0.0, period - (time.time() - t0)))

    def close(self):
        self._sampler_stop.set()
        return super().close()

    def reset(self, **kwargs):
        # Robot re-homes on reset; the operator must match the pose again.
        self._matched = False
        return self.env.reset(**kwargs)

    def _leader(self):
        now = time.time()
        if self.mode == "queued":
            s = self.stream.interpolated(now, self.playout_s)
        else:
            s = self.stream.latest_arrived(now)
        return s, now

    def action(self, action: np.ndarray):
        s, now = self._leader()
        if s is None:
            # Nothing has "arrived" yet (first tau_f seconds): hold still.
            self._last_delay_info = dict(self._cfg, applied_s=None, arrived=False)
            return action, False
        q_robot = np.asarray(
            self.get_wrapper_attr("get_arm_joint_position")(), dtype=np.float64
        ).reshape(-1)[:7]
        gripper = s.gripper if self.gripper_enabled else 0.0
        if not self._matched:
            err = (s.q - q_robot + np.pi) % (2.0 * np.pi) - np.pi
            if np.all(np.abs(err) <= self.match_tolerance_rad):
                self._matched = True
                print("[leader gate] pose matched — teleop live", flush=True)
            else:
                if now - self._last_match_log > 1.0:
                    self._last_match_log = now
                    print("[leader gate] move GELLO to the robot pose; joint error (rad): "
                          + np.array2string(err, precision=2, suppress_small=True), flush=True)
                self._last_delay_info = dict(self._cfg, applied_s=now - s.t_send, arrived=True, matched=False)
                return action, False
        expert_a = compute_joint_velocity_action(s.q, q_robot, gripper, self.kp, self.vmax)
        self._last_delay_info = dict(self._cfg, applied_s=now - s.t_send, arrived=True)
        if np.linalg.norm(expert_a[:7]) > 0.001 or (
            self.gripper_enabled and abs(expert_a[7]) > 0.5
        ):
            self.last_intervene = time.time()
        if time.time() - self.last_intervene < 0.5:
            return expert_a, True
        return action, False

    def step(self, action):
        obs, rew, done, truncated, info = super().step(action)
        info["leader_delay"] = getattr(self, "_last_delay_info", dict(self._cfg))
        return obs, rew, done, truncated, info
