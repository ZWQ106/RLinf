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
import threading
import time
import logging
from typing import Optional
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


class ControllerNotResponding(RuntimeError):
    """The NUC accepted our commands but the arm did not follow them —
    polymetis's impedance controller is not running (typically after a
    libfranka reflex) and DROID's robot.py drops ticks silently. Remedy:
    tasl/launch/nuc-restart.sh (or the portal's 🔧 Reset NUC button)."""


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
        # Serializes bootstrap: two concurrent bootstrap() calls would both
        # run launch_controller and tear down each other's polymetis driver.
        self._bootstrap_lock = threading.Lock()
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

        launch_robot races the polymetis driver startup (gRPC "Socket
        closed" when the driver is still coming up) — retry with backoff
        so the first button press after a container restart self-heals.
        """
        with self._bootstrap_lock:
            if self._bootstrapped:
                return
            _log.info("bootstrap step 1/3: launch_controller (spawn polymetis driver)")
            self._client.launch_controller()
            _log.info("bootstrap step 2/3: sleep %.1fs to let driver come up", settle_seconds)
            time.sleep(settle_seconds)
            _log.info("bootstrap step 3/3: launch_robot (connect RobotInterface)")
            last_exc: Optional[Exception] = None
            for attempt in range(4):
                try:
                    self._client.launch_robot()
                    self._bootstrapped = True
                    _log.info("bootstrap complete")
                    return
                except Exception as exc:
                    last_exc = exc
                    _log.warning("launch_robot attempt %d/4 failed: %s; retrying in 6s",
                                 attempt + 1, str(exc)[:120])
                    time.sleep(6.0)
            raise last_exc

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

    def stream_to_joint_position(
        self,
        target_q_7d: Sequence[float],
        gripper_cmd: float = 0.0,
        max_joint_vel: float = 0.3,
        hz: int = 15,
        min_duration_s: float = 1.5,
        max_duration_s: float = 10.0,
        settle_ticks: int = 8,
        verify: bool = True,
    ) -> dict:
        """Move to an absolute 7-D joint target by STREAMING interpolated
        setpoints at `hz`, non-blocking — the same wire path the 15 Hz eval
        loop uses, so the NUC's single-threaded zerorpc server never enters a
        blocking `move_to_joint_positions` / `gripper.goto(blocking=True)`.

        Why: the blocking variant (`update_joint_position(blocking=True)`)
        wedged the NUC server twice on 2026-08-25 — a blocking home issued
        right after an RTC episode raced the controller's non-blocking tick
        threads inside polymetis and never returned; every later call
        (state, gripper, home) then timed out until the container was
        restarted. Streaming keeps the server responsive throughout.

        Duration = max(min_duration_s, max joint displacement / max_joint_vel),
        capped at max_duration_s; a cosine ramp keeps velocity continuous at
        both ends. Returns {"duration_s", "disp", "residual"}; with
        verify=True raises ControllerNotResponding when the arm plainly did
        not follow (moved < half of a non-trivial displacement) — the
        "frozen controller" failure that otherwise shows up only as a policy
        that "doesn't move".
        """
        target = np.asarray(target_q_7d, dtype=np.float64)
        assert target.shape == (7,), f"expected 7 joint targets, got {target.shape}"
        q0 = np.asarray(self.get_joint_positions(), dtype=np.float64)[:7]
        disp = float(np.max(np.abs(target - q0)))
        dur = min(max_duration_s, max(min_duration_s, disp / max(max_joint_vel, 1e-3)))
        n = max(1, int(round(dur * hz)))
        dt = 1.0 / hz
        grip = float(np.clip(gripper_cmd, 0.0, 1.0))
        _log.info("stream_to_joint_position: disp=%.3f rad -> %.1fs @ %dHz", disp, dur, hz)
        for i in range(1, n + 1 + settle_ticks):
            s_ = min(1.0, i / n)
            alpha = 0.5 - 0.5 * float(np.cos(np.pi * s_))   # cosine ease-in/out
            q = q0 + alpha * (target - q0)
            action_list = [float(x) for x in q] + [grip]
            self._fast.update_command(action_list, "joint_position", "position", False)
            time.sleep(dt)
        q1 = np.asarray(self.get_joint_positions(), dtype=np.float64)[:7]
        residual = float(np.max(np.abs(target - q1)))
        moved = float(np.max(np.abs(q1 - q0)))
        _log.info("stream_to_joint_position: done, moved=%.3f residual=%.3f rad", moved, residual)
        if verify and disp > 0.03 and moved < 0.5 * disp:
            raise ControllerNotResponding(
                f"arm did not follow streamed setpoints (moved {moved:.3f} of {disp:.3f} rad) — "
                "NUC polymetis controller is not executing commands; run "
                "tasl/launch/nuc-restart.sh (portal: 🔧 Reset NUC)")
        return {"duration_s": dur, "disp": disp, "residual": residual}

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

    # ---- cartesian EE motion (zerorpc; the legacy HTTP robot_server is
    #      stopped, so RS.* HTTP endpoints are dead) ----

    def get_ee_pose(self) -> list:
        """EE pose as [x, y, z, roll, pitch, yaw] (DROID robot.py convention:
        pos + quat_to_euler)."""
        return list(self._client.get_ee_pose())

    def update_pose(self, pose: list, blocking: bool = False) -> None:
        """Absolute cartesian pose target, orientation as Euler angles."""
        self._client.update_pose(list(pose), False, blocking)

    def lift_ee(self, dz: float = 0.05, blocking: bool = True) -> None:
        """Cartesian +Z lift by dz meters, keeping orientation."""
        pose = self.get_ee_pose()
        pose[2] = float(pose[2]) + dz
        self._client.update_pose(pose, False, blocking)

    # ---- recovery ----

    def kill_controller(self) -> None:
        """Tear down polymetis driver. Future commands will need bootstrap()."""
        try:
            self._client.kill_controller()
        finally:
            self._bootstrapped = False
