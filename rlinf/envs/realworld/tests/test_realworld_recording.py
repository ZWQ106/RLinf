from rlinf.envs.realworld.realworld_env import RealWorldEnv


class _FakeVec:
    """Stand-in for the NoAutoResetSyncVectorEnv: .call invokes a method on
    each sub-env and returns a per-env tuple."""

    def __init__(self):
        self.calls = []

    def call(self, name, *args):
        self.calls.append((name, args))
        if name == "start_recording":
            return ({"wrist_1": "/svo/_rec_0_wrist_1.svo2"},)  # 1-env tuple
        return (None,)


def test_realworld_start_recording_proxies_to_subenv():
    e = RealWorldEnv.__new__(RealWorldEnv)
    e.env = _FakeVec()
    out = e.start_recording("/svo", "_rec_0")
    assert out == {"wrist_1": "/svo/_rec_0_wrist_1.svo2"}
    assert ("start_recording", ("/svo", "_rec_0")) in e.env.calls


def test_realworld_stop_recording_proxies():
    e = RealWorldEnv.__new__(RealWorldEnv)
    e.env = _FakeVec()
    e.stop_recording()
    assert any(c[0] == "stop_recording" for c in e.env.calls)
