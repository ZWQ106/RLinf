import numpy as np
import pytest

from rlinf.envs.realworld.franka.franka_jointvel_env import FrankaJointVelEnv


@pytest.fixture
def dummy_env():
    env = FrankaJointVelEnv(
        override_cfg={
            "is_dummy": True,
            "task_description": "pick up the cube",
            "camera_serials": ["dummy0"],
        },
        worker_info=None,
        hardware_info=None,
        env_idx=0,
    )
    return env


def test_action_space_is_8d(dummy_env):
    assert dummy_env.action_space.shape == (8,)


def test_observation_state_has_joint_position(dummy_env):
    obs = dummy_env._get_observation()
    assert "joint_position" in obs["state"]
    assert np.asarray(obs["state"]["joint_position"]).shape == (7,)
    assert "gripper_position" in obs["state"]
    assert "tcp_pose" not in obs["state"]


def test_get_arm_joint_position_returns_7d(dummy_env):
    q = dummy_env.get_arm_joint_position()
    assert np.asarray(q).shape == (7,)


def test_dummy_step_returns_wellformed_5tuple(dummy_env):
    action = dummy_env.action_space.sample()
    obs, reward, terminated, truncated, info = dummy_env.step(action)
    assert "joint_position" in obs["state"]
    assert isinstance(bool(terminated), bool)
    assert isinstance(bool(truncated), bool)
    assert isinstance(info, dict)
