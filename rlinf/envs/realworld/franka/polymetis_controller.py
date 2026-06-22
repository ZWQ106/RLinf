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

"""FrankaController-compatible Worker backed by DROID's polymetis container
over zerorpc — no ROS, works on FR3 fw >= 5.9 (libfranka 0.18.1 inside the
container). Selected via FrankaRobotConfig.controller_type == "polymetis".

Must stay a plain sync Ray actor (no max_concurrency / async def / background
threads): the zerorpc client is bound to the actor's task-execution thread
via gevent hub thread-locality. All public methods are sync; the client is
created in __init__ and only ever called from the same actor task thread.

Interface parity note: ``launch_controller`` and ``__init__`` accept the same
full argument set as ``FrankaController`` (including ``ros_pkg``,
``end_effector_type``, ``end_effector_config``, ``gripper_type``) so that
``FrankaEnv._setup_hardware`` can dispatch to either backend with an identical
call.  ROS-specific / end-effector-type args are accepted and ignored; this
backend always drives the Robotiq 2F-85 gripper via DROID's zerorpc server.
"""

import time
from typing import Optional

import numpy as np

from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker
from rlinf.utils.logging import get_logger

from .droid_zerorpc_client import DroidZerorpcClient
from .franka_robot_state import FrankaRobotState
from .polymetis_conversions import (
    droid_state_to_franka_fields,
    rlinf_ee_action_to_droid,
    rlinf_gripper_to_droid,
)


class PolymetisController(Worker):
    """Franka robot arm controller backed by DROID's polymetis zerorpc server."""

    @staticmethod
    def launch_controller(
        robot_ip: str,
        env_idx: int = 0,
        node_rank: int = 0,
        worker_rank: int = 0,
        ros_pkg: str = "serl_franka_controllers",
        end_effector_type: str = "franka_gripper",
        end_effector_config: Optional[dict] = None,
        gripper_type: Optional[str] = None,
        gripper_connection: Optional[str] = None,
    ):
        """Launch a PolymetisController on the specified worker's node.

        Signature mirrors FrankaController.launch_controller exactly so that
        FrankaEnv._setup_hardware can call either backend without branching on
        arguments.  ``ros_pkg``, ``end_effector_type``, ``end_effector_config``,
        and ``gripper_type`` are forwarded to __init__ for interface parity
        (they are accepted and ignored by this backend).
        """
        cluster = Cluster()
        placement = NodePlacementStrategy(node_ranks=[node_rank])
        return PolymetisController.create_group(
            robot_ip,
            ros_pkg,
            end_effector_type,
            end_effector_config or {},
            gripper_type,
            gripper_connection,
        ).launch(
            cluster=cluster,
            placement_strategy=placement,
            name=f"PolymetisController-{worker_rank}-{env_idx}",
        )

    def __init__(
        self,
        robot_ip: str,
        ros_pkg: str = "serl_franka_controllers",
        end_effector_type: str = "franka_gripper",
        end_effector_config: Optional[dict] = None,
        gripper_type: Optional[str] = None,
        gripper_connection: Optional[str] = None,
    ):
        super().__init__()
        # Mirror FrankaController.__init__: also obtain a named logger.
        self._logger = get_logger()
        self._robot_ip = robot_ip
        # ros_pkg / end_effector_type / end_effector_config / gripper_type /
        # gripper_connection are accepted for interface parity with FrankaController
        # but unused here — this backend always uses the DROID zerorpc Robotiq path.

        self._client = DroidZerorpcClient(address=f"tcp://{robot_ip}:4242")
        self._client.bootstrap()  # idempotent; spawns polymetis + robotiq drivers

        # Sync cached gripper command with physical state so the binary
        # open/close logic in command_end_effector starts from truth.
        self._last_grip: float = float(
            self._client.get_robot_state()["gripper_position"]
        )

    # ---- liveness --------------------------------------------------------

    def is_robot_up(self) -> bool:
        """Check whether the arm and gripper are reachable."""
        try:
            self._client.get_robot_state()
            return True
        except Exception:
            return False

    # ---- state -----------------------------------------------------------

    def get_state(self) -> FrankaRobotState:
        """Get the current state of the Franka robot."""
        fields = droid_state_to_franka_fields(self._client.get_robot_state())
        return FrankaRobotState(**fields)

    # ---- arm motion ------------------------------------------------------

    def move_arm(self, position: np.ndarray):
        """ABSOLUTE 7D pose target [xyz + quat xyzw] — one non-blocking
        impedance-equilibrium update per env step (serl semantics).

        A wedged server raises zerorpc.TimeoutExpired after 5s (fast-client timeout) and the env step crashes — correct behavior; recover via recover()."""
        assert len(position) == 7, (
            f"Invalid position, expected 7 dimensions but got {len(position)}"
        )
        pose6 = rlinf_ee_action_to_droid(np.asarray(position, dtype=np.float64))
        self._client.update_cartesian_position(pose6, gripper_cmd=self._last_grip)
        self.log_debug(f"Move arm to position: {position}")

    def move_joint_velocity(self, action_8d: np.ndarray):
        """Stream one tick of an 8-D joint-velocity action.

        action[:7] = joint velocities in [-1, 1]; action[7] = gripper position
        (0=open, 1=close). Matches pi05_droid's native action; sent via DROID's
        action_space="joint_velocity". Non-blocking (15 Hz streaming path)."""
        a = np.asarray(action_8d, dtype=np.float64).reshape(-1)
        assert a.shape == (8,), f"expected 8-D joint-velocity action, got {a.shape}"
        self._client.update_joint_velocity(a, blocking=False)
        self.log_debug(f"Move joint velocity: {a}")

    def step_joint_velocity(self, action_8d: np.ndarray) -> FrankaRobotState:
        """Command an 8-D joint-velocity action AND return the resulting state
        in ONE actor RPC.

        Folding the command + state read into a single env<->controller round
        trip (instead of separate move_joint_velocity + get_state calls, with the
        teleop wrapper's own joint read now served from the env's cached state)
        cuts per-step controller RPCs from 3 to 1. Fewer round trips means a
        steadier command cadence — the DROID joint-velocity controller ramps the
        arm down when a fresh command arrives late, which is felt as a stutter."""
        a = np.asarray(action_8d, dtype=np.float64).reshape(-1)
        assert a.shape == (8,), f"expected 8-D joint-velocity action, got {a.shape}"
        self._client.update_joint_velocity(a, blocking=False)
        return self.get_state()

    # ---- end-effector actions --------------------------------------------

    def command_end_effector(self, action: np.ndarray) -> bool:
        """GRIPPER path (binary): <= -0.5 closes if open, >= 0.5 opens if
        closed, else no-op. Mirrors serl FrankaController semantics."""
        value = float(np.asarray(action).reshape(-1)[0])
        currently_open = self._last_grip < 0.5
        if value <= -0.5 and currently_open:
            self.close_gripper()
            return True
        if value >= 0.5 and not currently_open:
            self.open_gripper()
            return True
        return False

    def reset_end_effector(self, target_state: np.ndarray | None = None) -> None:
        """Reset the end-effector to a target or default state.

        Gripper EE type: route through binary gripper logic (serl behavior).
        This backend does not support dexterous hands.
        """
        if target_state is not None:
            self.command_end_effector(np.asarray(target_state))

    # ---- joint reset -----------------------------------------------------

    def reset_joint(self, reset_pos, timeout_s: float = 25.0):
        """Speed-bounded leashed move to the reset joint pose; when this
        returns the arm is AT the target (within tol) or we warned loudly.

        Uses the closed-loop leash (setpoint kept just ahead of the measured
        pose) so the inter-episode joint reset does NOT lunge at full speed
        and trip the Franka velocity/accel reflex — the old stream_joint_
        position (constant final target) did exactly that from a far pose.
        """
        err = self._client.leashed_move_to_joint(
            reset_pos, gripper_cmd=self._last_grip, timeout_s=timeout_s
        )
        if err >= 0.03:
            self.log_warning(f"reset_joint not converged: max err {err:.4f} rad")

    # ---- compliance / error handling ------------------------------------

    def reconfigure_compliance_params(self, params: dict) -> None:
        """Compliance reconfiguration is a ROS/serl concept — log and ignore."""
        self.log_warning(f"polymetis backend ignores compliance params {params}")

    def clear_errors(self) -> None:
        """Called by FrankaEnv before EVERY move (_move_action → _clear_error).

        Must be cheap. Real recovery is done by recover() (relaunches the DROID
        controller) which is never on the hot path.
        """
        pass  # polymetis impedance runs continuously; no per-step reset needed

    def recover(self) -> None:
        """Full DROID controller relaunch — for reflex/error recovery only."""
        self._client.kill_controller()
        self._client.bootstrap()

    # ---- impedance lifecycle (no-op: always running in DROID) -----------

    def start_impedance(self) -> None:
        pass  # impedance runs whenever the DROID controller is launched

    def stop_impedance(self) -> None:
        pass

    # ---- gripper helpers -------------------------------------------------

    def open_gripper(self) -> None:
        # Deliberately update the cache BEFORE the blocking RPC: move_arm
        # re-asserts _last_grip at ~15 Hz, so on a response-lost timeout
        # where the gripper DID move, post-update would stream the stale
        # command and reopen mid-grasp. Pre-update converges to intent.
        self._last_grip = 0.0
        self._client.update_gripper(0.0, blocking=True)
        self.log_debug("Open gripper")

    def close_gripper(self) -> None:
        # Deliberately update the cache BEFORE the blocking RPC: move_arm
        # re-asserts _last_grip at ~15 Hz, so on a response-lost timeout
        # where the gripper DID move, post-update would stream the stale
        # command and reopen mid-grasp. Pre-update converges to intent.
        self._last_grip = 1.0
        self._client.update_gripper(1.0, blocking=True)
        self.log_debug("Close gripper")

    def move_gripper(self, position: int, speed: float = 0.3) -> None:
        """Move gripper to absolute position (0-255 scale, 0=open, 255=closed).

        ``speed`` is accepted for interface parity with FrankaController but
        ignored by this backend (DROID's update_gripper does not expose speed).
        """
        assert 0 <= position <= 255, (
            f"Invalid gripper position {position}, must be between 0 and 255"
        )
        cmd = rlinf_gripper_to_droid(position)
        self._last_grip = cmd
        self._client.update_gripper(cmd, blocking=False)
        self.log_debug(f"Move gripper to position: {position}")

    # ---- hand / end-effector info ---------------------------------------

    def get_hand_type(self) -> str:
        return "gripper"

    def get_hand_state(self) -> np.ndarray | None:
        return None

    def get_hand_detailed_state(self) -> dict:
        grip = self._client.get_robot_state()["gripper_position"]
        return {
            "gripper_position": int(round(grip * 255)),
            "gripper_open": grip < 0.5,
        }

    def get_hand_finger_names(self) -> list[str]:
        return ["gripper"]
