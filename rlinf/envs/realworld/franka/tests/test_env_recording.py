import os

from rlinf.envs.realworld.franka.franka_env import FrankaEnv, FrankaRobotConfig


class _FakeCam:
    def __init__(self, name):
        self._camera_info = type("CI", (), {"name": name})()
        self.started = None
        self.stopped = False

    def start_recording(self, path):
        self.started = path

    def stop_recording(self):
        self.stopped = True


def _bare_env(cams, dummy=False):
    e = FrankaEnv.__new__(FrankaEnv)
    e.config = FrankaRobotConfig(is_dummy=dummy)
    e._cameras = cams
    return e


def test_start_recording_builds_paths_and_returns_map(tmp_path):
    svo_dir = str(tmp_path / "svo")
    cams = [_FakeCam("wrist_1"), _FakeCam("wrist_2")]
    e = _bare_env(cams)
    paths = e.start_recording(svo_dir, "_rec_5")
    assert paths == {
        "wrist_1": os.path.join(svo_dir, "_rec_5_wrist_1.svo2"),
        "wrist_2": os.path.join(svo_dir, "_rec_5_wrist_2.svo2"),
    }
    assert cams[0].started == os.path.join(svo_dir, "_rec_5_wrist_1.svo2")
    assert cams[1].started == os.path.join(svo_dir, "_rec_5_wrist_2.svo2")


def test_stop_recording_stops_all():
    cams = [_FakeCam("wrist_1"), _FakeCam("wrist_2")]
    e = _bare_env(cams)
    e.stop_recording()
    assert all(c.stopped for c in cams)


def test_start_recording_noop_when_dummy():
    e = _bare_env([_FakeCam("wrist_1")], dummy=True)
    assert e.start_recording("/data/svo", "_rec_0") == {}
