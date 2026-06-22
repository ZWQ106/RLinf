"""DroidLikeClient — Desktop-side thin wrapper around the droid_nuc container.

Connects via zerorpc to the FrankaRobot exposed at NUC1:4242 by DROID's
run_server.py. Mirrors the public surface of penn-pal-lab's
server_interface.py while accommodating two DROID-specific quirks:

1.  Server expects POSITIONAL args for `update_command` — kwargs over zerorpc
    are silently dropped on this version, falling back to the default
    `action_space="cartesian_velocity"`. Always use positional.

2.  `get_robot_state()` returns a 2-list `[state_dict, timestamp_dict]`,
    not a single dict. We unwrap and merge for caller convenience.

The container's FrankaRobot is "lazy" — polymetis driver isn't spawned at
container start. Call `bootstrap()` once before sending any motion or state
read; subsequent calls are no-ops.
"""
import math
import time
import logging
from typing import Optional, Sequence

import numpy as np
import zerorpc

_log = logging.getLogger("droid_client")

# DROID home pose (matches our existing dashboard constant)
DROID_HOME_Q = [
    0.0,
    -math.pi / 5,
    0.0,
    -4 * math.pi / 5,
    0.0,
    3 * math.pi / 5,
    0.0,
]


class DroidLikeClient:
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
        self._addr = address
        # Long-timeout connection: bootstrap (launch_controller ~10s) and
        # blocking moves (Go Home can take several seconds).
        self._client = zerorpc.Client(heartbeat=heartbeat, timeout=timeout)
        self._client.connect(address)
        # Short-timeout connection: 15 Hz streaming + state reads. Bounds
        # how long the eval loop (and /status) can stall on a wedged call —
        # with the shared 60s timeout a single stuck get_robot_state held
        # the stop flag hostage for up to a minute.
        self._fast = zerorpc.Client(heartbeat=heartbeat, timeout=fast_timeout)
        self._fast.connect(address)
        self._bootstrapped = False
        _log.info("droid client connected to %s", address)

    @property
    def bootstrapped(self) -> bool:
        return self._bootstrapped

    # ---- lifecycle ----

    def bootstrap(self, settle_seconds: float = 8.0) -> None:
        """Spawn polymetis driver + Robotiq driver inside the container,
        then connect FrankaRobot's RobotInterface to them.

        Idempotent. First call takes ~10s. Subsequent calls are no-ops.
        Must be called before any motion or state read.
        """
        if self._bootstrapped:
            return
        _log.info("bootstrap step 1/3: launch_controller (spawn polymetis driver)")
        self._client.launch_controller()
        _log.info("bootstrap step 2/3: sleep %.1fs to let driver come up", settle_seconds)
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
            "prev_command_successful": bool(state.get("prev_command_successful", False)),
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
        """
        action_list = [float(x) for x in np.asarray(action_8d, dtype=np.float64)]
        assert len(action_list) == 8, f"expected 8-D action, got {len(action_list)}"
        # POSITIONAL args. kwargs over zerorpc are silently dropped on this
        # build and we get the default action_space="cartesian_velocity",
        # which IK-solves into completely different joint motion.
        #
        # Arg 3 = gripper_action_space = "position". DROID's default for
        # action_space="joint_velocity" is gripper="velocity" → treats
        # action[7] as a DELTA and accumulates → gripper saturates closed
        # within a few iters and stays stuck. pi05_droid emits action[7]
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
        (used for resets / Go Home; not the streaming eval path)."""
        q = [float(x) for x in np.asarray(target_q_7d, dtype=np.float64)]
        assert len(q) == 7, f"expected 7 joint targets, got {len(q)}"
        action = q + [float(gripper_cmd)]
        self._client.update_command(action, "joint_position", None, blocking)

    def go_home(self, gripper_cmd: float = 0.0,
                duration_s: float = 1.5, hz: int = 15) -> None:
        """Smooth move to DROID home pose via NON-blocking streaming.

        Why streaming instead of `update_joint_position(blocking=True)`:
        DROID's adaptive_time_to_go() returns 0 for small displacements
        (default t_min=0), and polymetis `move_to_joint_positions(q, t=0)`
        becomes a no-op. So Go Home from a pose close to home silently
        does nothing. Streaming the same absolute target at 15 Hz with
        cartesian impedance running converges reliably regardless of
        starting distance.
        """
        _log.info(f"go_home: stream {duration_s:.1f}s @ {hz}Hz to DROID reset pose")
        action_list = [float(x) for x in DROID_HOME_Q] + [float(gripper_cmd)]
        dt = 1.0 / hz
        n = int(duration_s * hz)
        for _ in range(n):
            self._client.update_command(action_list, "joint_position", "position", False)
            time.sleep(dt)

    # ---- gripper standalone ----

    def update_gripper(self, command: float, blocking: bool = False) -> None:
        """0=open, 1=closed. Independent of arm motion."""
        cmd = float(np.clip(command, 0.0, 1.0))
        # update_gripper(self, command, velocity=True, blocking=False)
        self._client.update_gripper(cmd, False, blocking)

    # ---- recovery ----

    def kill_controller(self) -> None:
        """Tear down polymetis driver. Future commands will need bootstrap()."""
        try:
            self._client.kill_controller()
        finally:
            self._bootstrapped = False
