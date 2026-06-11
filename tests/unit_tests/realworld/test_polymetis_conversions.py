import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R
import sys
import importlib.util

# Load polymetis_conversions directly without triggering rlinf/__init__
spec = importlib.util.spec_from_file_location(
    "polymetis_conversions",
    "rlinf/envs/realworld/franka/polymetis_conversions.py"
)
poly_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(poly_module)

droid_state_to_franka_fields = poly_module.droid_state_to_franka_fields
rlinf_ee_action_to_droid = poly_module.rlinf_ee_action_to_droid
droid_gripper_to_rlinf = poly_module.droid_gripper_to_rlinf
rlinf_gripper_to_droid = poly_module.rlinf_gripper_to_droid


def _droid_state(euler=(0.1, -0.2, 0.3)):
    return {
        "joint_positions": np.arange(7, dtype=np.float64),
        "joint_velocities": np.full(7, 0.5),
        "cartesian_position": np.array([0.4, 0.0, 0.3, *euler]),
        "gripper_position": 0.8,  # DROID: 1=closed
    }


def test_state_pose_is_pos_plus_quat():
    f = droid_state_to_franka_fields(_droid_state())
    assert f["tcp_pose"].shape == (7,)
    np.testing.assert_allclose(f["tcp_pose"][:3], [0.4, 0.0, 0.3])
    expect_q = R.from_euler("xyz", [0.1, -0.2, 0.3]).as_quat()
    np.testing.assert_allclose(f["tcp_pose"][3:], expect_q, atol=1e-9)


def test_state_joints_and_jacobian_passthrough():
    f = droid_state_to_franka_fields(_droid_state())
    np.testing.assert_allclose(f["arm_joint_position"], np.arange(7))
    assert f["arm_jacobian"].shape == (6, 7)  # zeros: env never reads it
    assert not f["arm_jacobian"].any()


def test_ee_action_roundtrip():
    # RLinf side: 7D absolute target pos+quat -> DROID 6D pos+euler
    quat = R.from_euler("xyz", [0.0, 0.1, -0.1]).as_quat()
    target = np.array([0.5, 0.1, 0.25, *quat])
    d = rlinf_ee_action_to_droid(target)
    assert d.shape == (6,)
    np.testing.assert_allclose(d[:3], target[:3])
    np.testing.assert_allclose(d[3:], [0.0, 0.1, -0.1], atol=1e-9)


def test_gripper_conventions():
    # DROID 1=closed; RLinf gripper_open bool + 0-255 int position
    open_, pos = droid_gripper_to_rlinf(0.0)
    assert open_ is True and pos == 0
    open_, pos = droid_gripper_to_rlinf(1.0)
    assert open_ is False and pos == 255
    # RLinf move_gripper(position: int 0-255) -> DROID [0,1] cmd
    assert rlinf_gripper_to_droid(0) == pytest.approx(0.0)
    assert rlinf_gripper_to_droid(255) == pytest.approx(1.0)
