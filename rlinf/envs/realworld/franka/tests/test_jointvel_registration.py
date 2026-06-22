import gymnasium as gym
import rlinf.envs.realworld.franka.tasks  # noqa: F401  (registers ids)


def test_jointvel_env_is_registered():
    assert "FrankaJointVelEnv-v1" in gym.registry
