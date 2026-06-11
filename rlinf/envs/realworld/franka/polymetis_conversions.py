"""Pure conversions between DROID polymetis zerorpc state/commands and the
field/action conventions FrankaEnv expects. No rlinf imports — unit-testable
without ray/torch.

CONTRACT (verified 2026-06-11 against franka_controller.py:293-335,
franka_env.py:297-360+823-875, and droid robot.py + transformations.py in the
droid-nuc-fr3 container):
- DROID cartesian_position: 6D [x, y, z, euler-xyz(rad)] — scipy
  R.as_euler("xyz"), quat is scipy xyzw. Same conventions as RLinf. VERIFIED.
- DROID update_command action spaces: cartesian_position / joint_position /
  cartesian_velocity / joint_velocity. VERIFIED.
- DROID gripper_position: float [0,1], 1 = closed.
- RLinf ARM path: `move_arm(position: 7D [x,y,z,qx,qy,qz,qw])` — ABSOLUTE
  impedance equilibrium target, one non-blocking call per env step.
  (`command_end_effector` is NOT the arm path — it is the gripper path.)
- RLinf GRIPPER path: `command_end_effector(scalar)` — binary semantics:
  value <= -0.5 AND currently open → close; value >= 0.5 AND currently
  closed → open; else no-op. Returns bool (action effective).
- `clear_errors()` is called by the env BEFORE EVERY move (`_move_action` →
  `_clear_error()`), so it MUST be cheap — a no-op for polymetis (impedance
  recovery is handled by relaunching the DROID controller, exposed
  separately, never per-step).
- RLinf FrankaRobotState.gripper_position: int (0..255), gripper_open: bool
"""
import numpy as np
from scipy.spatial.transform import Rotation as R


def droid_state_to_franka_fields(state: dict) -> dict:
    pos_euler = np.asarray(state["cartesian_position"], dtype=np.float64)
    quat = R.from_euler("xyz", pos_euler[3:6]).as_quat()
    grip_open, grip_pos = droid_gripper_to_rlinf(float(state["gripper_position"]))
    return {
        "tcp_pose": np.concatenate([pos_euler[:3], quat]),
        "tcp_vel": np.zeros(6),
        "arm_joint_position": np.asarray(state["joint_positions"], dtype=np.float64),
        "arm_joint_velocity": np.asarray(state["joint_velocities"], dtype=np.float64),
        "tcp_force": np.zeros(3),
        "tcp_torque": np.zeros(3),
        "arm_jacobian": np.zeros((6, 7)),  # FrankaEnv never reads it (verified)
        "gripper_position": grip_pos,
        "gripper_open": grip_open,
    }


def rlinf_ee_action_to_droid(target_pose_7d: np.ndarray) -> np.ndarray:
    t = np.asarray(target_pose_7d, dtype=np.float64)
    assert t.shape == (7,), f"expected 7D pos+quat, got {t.shape}"
    euler = R.from_quat(t[3:7]).as_euler("xyz")
    return np.concatenate([t[:3], euler])


def droid_gripper_to_rlinf(droid_pos: float) -> tuple[bool, int]:
    droid_pos = min(1.0, max(0.0, droid_pos))
    return droid_pos < 0.5, int(round(droid_pos * 255))


def rlinf_gripper_to_droid(position_255: int) -> float:
    return min(1.0, max(0.0, position_255 / 255.0))
