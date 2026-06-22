from rlinf.envs.realworld.franka.franka_env import FrankaEnv, FrankaRobotConfig


def _bare_env(resolution):
    e = FrankaEnv.__new__(FrankaEnv)
    e.config = FrankaRobotConfig(
        camera_serials=["36443134", "17150101"],
        camera_type="zed",
        camera_resolution=resolution,
    )
    return e


def test_build_camera_infos_sets_hd1080_resolution():
    e = _bare_env([1920, 1080])
    infos = e._build_camera_infos()
    assert len(infos) == 2
    assert all(tuple(i.resolution) == (1920, 1080) for i in infos)


def test_build_camera_infos_defaults_resolution_when_unset():
    e = _bare_env(None)
    infos = e._build_camera_infos()
    assert all(tuple(i.resolution) == (640, 480) for i in infos)
