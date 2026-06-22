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

"""Thin zerorpc client wrapper around DROID's FrankaRobot.

Connects via zerorpc to the FrankaRobot exposed at NUC1:4242 by DROID's
run_server.py. Key design facts preserved:

1.  Server expects POSITIONAL args for `update_command` — kwargs over zerorpc
    are silently dropped on this version, falling back to the default
    `action_space="cartesian_velocity"`. Always use positional.

2.  `get_robot_state()` returns a 2-list `[state_dict, timestamp_dict]`,
    not a single dict. We unwrap and merge for caller convenience.

3.  Dual connections: `_client` (timeout=30s, for bootstrap/blocking ops) and
    `_fast` (timeout=5s, for state reads + streaming) — bounds stall time.

4.  The container's FrankaRobot is "lazy" — polymetis driver isn't spawned at
    container start. Call `bootstrap()` once before sending any motion or state
    read; subsequent calls are no-ops.
"""

import logging
import threading
import time
from typing import Sequence

import numpy as np
import zerorpc

_log = logging.getLogger("droid_zerorpc_client")


class DroidZerorpcClient:
    """Thin zerorpc wrapper around DROID's FrankaRobot.

    One instance per process. Re-entrant — bootstrap() is idempotent.
    """

    def __init__(
        self,
        address: str = "tcp://100.75.6.62:4242",
        heartbeat: int = 20,
        timeout: int = 30,
        fast_timeout: int = 5,
    ):
        # Long-timeout connection: bootstrap (launch_controller ~10s) and
        # blocking moves (one-shot positions can take several seconds).
        self._client = zerorpc.Client(heartbeat=heartbeat, timeout=timeout)
        self._client.connect(address)
        # Short-timeout connection: 15 Hz streaming + state reads. Bounds
        # how long the eval loop (and /status) can stall on a wedged call —
        # with the shared 60s timeout a single stuck get_robot_state held
        # the stop flag hostage for up to a minute.
        self._fast = zerorpc.Client(heartbeat=heartbeat, timeout=fast_timeout)
        self._fast.connect(address)
        self._bootstrapped = False
        self._owner_thread = threading.get_ident()
        _log.info("droid client connected to %s", address)

    @property
    def bootstrapped(self) -> bool:
        return self._bootstrapped

    def _check_thread(self) -> None:
        # gevent hubs are thread-local: zerorpc calls from a thread other
        # than the creating one die with an opaque LoopExit. Fail loudly
        # instead. If another thread needs a client, it must create its own.
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError(
                "DroidZerorpcClient used from a different thread than the one "
                "that created it; create a separate client in that thread"
            )

    # ---- lifecycle ----

    def bootstrap(self, settle_seconds: float = 8.0) -> None:
        """Ensure the polymetis driver + RobotInterface are up.

        Idempotent ACROSS PROCESSES: if a controller is already running and
        healthy (get_robot_state succeeds — e.g. the dashboard left one up
        from a Recover/Home), REUSE it and skip the launch. The server-side
        FrankaRobot is shared across zerorpc clients, so re-launching would
        force launch_controller to KILL the live driver and relaunch — that
        kill+relaunch hangs (observed 2026-06-15: collection startup wedged
        for minutes when a dashboard controller was already running). Only
        do the full launch when no healthy controller exists.
        """
        self._check_thread()
        if self._bootstrapped:
            return
        # Already-healthy controller? Reuse it — no kill+relaunch.
        try:
            self._fast.get_robot_state()
            self._bootstrapped = True
            _log.info("bootstrap: controller already healthy, reusing (no relaunch)")
            return
        except Exception:
            pass
        _log.info("bootstrap step 1/3: launch_controller (spawn polymetis driver)")
        self._client.launch_controller()
        _log.info(
            "bootstrap step 2/3: sleep %.1fs to let driver come up", settle_seconds
        )
        time.sleep(settle_seconds)
        _log.info("bootstrap step 3/3: launch_robot (connect RobotInterface)")
        self._client.launch_robot()
        self._bootstrapped = True
        _log.info("bootstrap complete")

    def close(self) -> None:
        for c in (self._client, self._fast):
            try:
                c.close()
            except Exception:
                pass

    # ---- state ----

    def get_robot_state(self) -> dict:
        """Flat state dict with joint positions, velocities, torques, EE pose,
        gripper, and timestamp. Unwraps DROID's [state, ts] 2-list."""
        self._check_thread()
        st_pair = self._fast.get_robot_state()
        state, ts = st_pair[0], st_pair[1]
        # Normalize to numpy for caller convenience.
        out = {
            "joint_positions": np.asarray(state["joint_positions"]),
            "joint_velocities": np.asarray(state["joint_velocities"]),
            "joint_torques_computed": np.asarray(state["joint_torques_computed"]),
            "motor_torques_measured": np.asarray(state["motor_torques_measured"]),
            "cartesian_position": np.asarray(state["cartesian_position"]),
            "gripper_position": float(state["gripper_position"]),
            "prev_command_successful": bool(
                state.get("prev_command_successful", False)
            ),
            "prev_controller_latency_ms": float(
                state.get("prev_controller_latency_ms", 0.0)
            ),
            "timestamp_seconds": int(ts["robot_timestamp_seconds"]),
            "timestamp_nanos": int(ts["robot_timestamp_nanos"]),
        }
        return out

    def get_joint_positions(self) -> np.ndarray:
        return np.asarray(self._fast.get_joint_positions())

    def get_gripper_position(self) -> float:
        return float(self._fast.get_gripper_position())

    # ---- motion (positional args only — kwargs are dropped by this zerorpc) ----

    def update_joint_velocity(
        self,
        action_8d: Sequence[float],
        blocking: bool = False,
    ) -> None:
        """Send one tick of an 8-D joint-velocity action.

        Args:
            action_8d: shape (8,) in [-1, 1]. action[:7] = joint velocities,
                action[7] = gripper command (0=open, 1=close).
            blocking: True for one-shot (e.g. reset). False for streaming
                eval at 15 Hz (production path).

        Note: kwargs over zerorpc are silently dropped on this build and we
        get the default action_space="cartesian_velocity", which IK-solves
        into completely different joint motion. Always use positional args.
        """
        self._check_thread()
        action_list = [float(x) for x in np.asarray(action_8d, dtype=np.float64)]
        assert len(action_list) == 8, f"expected 8-D action, got {len(action_list)}"
        # POSITIONAL args. Arg 3 = gripper_action_space = "position". DROID's
        # default for action_space="joint_velocity" is gripper="velocity" →
        # treats action[7] as a DELTA and accumulates → gripper saturates
        # closed within a few iters and stays stuck. pi05_droid emits action[7]
        # as ABSOLUTE position in [0, 1] (0=open, 1=close), matching DROID
        # training data; force "position" so the gripper actually tracks.
        c = self._client if blocking else self._fast
        c.update_command(action_list, "joint_velocity", "position", blocking)

    def update_joint_position(
        self,
        target_q_7d: Sequence[float],
        gripper_cmd: float = 0.0,
        blocking: bool = True,
    ) -> None:
        """One-shot move to an absolute 7-D joint target. Blocking by default
        (used for resets / Go Home; not the streaming eval path).

        Args:
            target_q_7d: shape (7,) joint positions in radians.
            gripper_cmd: gripper command (0=open, 1=close).
            blocking: True for one-shot moves (robot waits). False for streaming.
        """
        self._check_thread()
        q = [float(x) for x in np.asarray(target_q_7d, dtype=np.float64)]
        assert len(q) == 7, f"expected 7 joint targets, got {len(q)}"
        action = q + [float(gripper_cmd)]
        self._client.update_command(action, "joint_position", None, blocking)

    def update_cartesian_position(
        self, pose_6d: Sequence[float], gripper_cmd: float = 0.0
    ) -> None:
        """One non-blocking absolute EE target. 6D [xyz + euler-xyz] + gripper.

        Args:
            pose_6d: shape (6,) [x, y, z, euler_x, euler_y, euler_z] in radians.
            gripper_cmd: gripper command (0=open, 1=close).

        Note: kwargs over zerorpc are silently dropped by this server version —
        always positional args to update_command.
        """
        self._check_thread()
        a = [float(x) for x in pose_6d] + [float(gripper_cmd)]
        assert len(a) == 7
        self._fast.update_command(a, "cartesian_position", "position", False)

    def stream_joint_position(
        self,
        q7: Sequence[float],
        duration_s: float = 2.0,
        hz: int = 15,
        gripper_cmd: float = 0.0,
    ) -> None:
        """Streamed joint reset — a blocking one-shot is a no-op for small
        displacement (DROID adaptive_time_to_go=0 gotcha).

        Args:
            q7: shape (7,) target joint positions in radians.
            duration_s: duration of streaming in seconds.
            hz: streaming rate in Hz.
            gripper_cmd: gripper command (0=open, 1=close).

        Streams on the fast (5s) connection — a wedged server stalls a tick by
        at most 5s. Per-tick pacing ignores RPC RTT, so effective rate is
        slightly below hz; fine for resets.

        Why streaming instead of blocking one-shot: DROID's adaptive_time_to_go()
        returns 0 for small displacements (default t_min=0), and polymetis
        `move_to_joint_positions(q, t=0)` becomes a no-op. So going to a pose
        close to current position silently does nothing. Streaming the same
        absolute target at 15 Hz with cartesian impedance running converges
        reliably regardless of starting distance.
        """
        self._check_thread()
        a = [float(x) for x in q7] + [float(gripper_cmd)]
        assert len(a) == 8
        for _ in range(int(duration_s * hz)):
            self._fast.update_command(a, "joint_position", "position", False)
            time.sleep(1.0 / hz)

    def leashed_move_to_joint(
        self,
        q7: Sequence[float],
        gripper_cmd: float = 0.0,
        max_step: float = 0.06,
        hz: int = 15,
        timeout_s: float = 25.0,
        tol: float = 0.03,
    ) -> float:
        """Speed-bounded joint move; returns max joint error (rad).

        Each tick reads the ACTUAL joint position and commands an impedance
        setpoint at most `max_step` rad ahead toward the target, so the
        impedance error (hence torque, hence speed ~= max_step*hz) is bounded
        — no full-speed lunge, no Franka velocity/accel reflex. `stream_joint_
        position` (constant final target) lunges from a far pose; blocking
        move_to_joint_positions WEDGES the polymetis gRPC server. max_step=0.06
        because a gravity-loaded wrist joint won't track a smaller setpoint
        error (verified on hardware 2026-06-15).
        """
        self._check_thread()
        target = [float(x) for x in q7]
        deadline = time.monotonic() + timeout_s
        maxerr = float("inf")
        while True:
            cur = [float(x) for x in self._fast.get_robot_state()[0]["joint_positions"]]
            err = [t - c for t, c in zip(target, cur)]
            maxerr = max(abs(e) for e in err)
            if maxerr < tol or time.monotonic() > deadline:
                return maxerr
            sp = [c + max(-max_step, min(max_step, e)) for c, e in zip(cur, err)]
            self._fast.update_command(sp + [float(gripper_cmd)],
                                      "joint_position", "position", False)
            time.sleep(1.0 / hz)

    # ---- gripper standalone ----

    def update_gripper(self, command: float, blocking: bool = False) -> None:
        """0=open, 1=closed. Independent of arm motion.

        Args:
            command: gripper position command (0=open, 1=close), clipped to [0,1].
            blocking: True for one-shot. False for streaming.
        """
        self._check_thread()
        cmd = float(np.clip(command, 0.0, 1.0))
        # POSITIONAL args: command, velocity, blocking
        self._client.update_gripper(cmd, False, blocking)

    # ---- recovery ----

    def kill_controller(self) -> None:
        """Tear down polymetis driver. Future commands will need bootstrap()."""
        self._check_thread()
        try:
            self._client.kill_controller()
        finally:
            self._bootstrapped = False
