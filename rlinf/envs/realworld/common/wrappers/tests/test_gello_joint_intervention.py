import numpy as np
from rlinf.envs.realworld.common.wrappers.gello_joint_intervention import (
    compute_joint_velocity_action,
)


def test_zero_error_gives_zero_velocity():
    q = np.array([0.1, -0.6, 0.0, -2.5, 0.0, 1.8, 0.0])
    a = compute_joint_velocity_action(q_gello=q, q_robot=q, gripper=0.0, kp=4.0, vmax=1.0)
    assert a.shape == (8,)
    np.testing.assert_allclose(a[:7], 0.0)
    assert a[7] == 0.0


def test_positive_error_positive_velocity_and_clip():
    q_robot = np.zeros(7)
    q_gello = np.full(7, 1.0)  # large error -> saturates
    a = compute_joint_velocity_action(q_gello, q_robot, gripper=1.0, kp=4.0, vmax=1.0)
    np.testing.assert_allclose(a[:7], 1.0)  # clipped to +vmax
    assert a[7] == 1.0  # gripper passthrough


def test_small_error_scales_by_kp_within_clip():
    q_robot = np.zeros(7)
    q_gello = np.full(7, 0.1)
    a = compute_joint_velocity_action(q_gello, q_robot, gripper=0.3, kp=4.0, vmax=1.0)
    np.testing.assert_allclose(a[:7], 0.4)  # 4.0 * 0.1, below clip
    assert a[7] == 0.3


def test_gripper_clipped_to_unit_interval():
    q = np.zeros(7)
    a = compute_joint_velocity_action(q, q, gripper=2.0, kp=1.0, vmax=1.0)
    assert a[7] == 1.0
