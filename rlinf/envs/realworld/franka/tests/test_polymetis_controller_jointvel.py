import numpy as np
from rlinf.envs.realworld.franka.polymetis_controller import PolymetisController


def _bare_controller():
    """A PolymetisController instance with __init__ bypassed and a fake client."""
    c = PolymetisController.__new__(PolymetisController)
    sent = {}

    class FakeClient:
        def update_joint_velocity(self, action_8d, blocking=False):
            sent["action"] = np.asarray(action_8d, dtype=np.float64)
            sent["blocking"] = blocking

    c._client = FakeClient()
    c._sent = sent
    c.log_debug = lambda *a, **k: None
    return c


def test_move_joint_velocity_forwards_8d_nonblocking():
    c = _bare_controller()
    a = np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 1.0])
    c.move_joint_velocity(a)
    assert c._sent["action"].shape == (8,)
    np.testing.assert_allclose(c._sent["action"], a)
    assert c._sent["blocking"] is False


def test_move_joint_velocity_rejects_wrong_dim():
    c = _bare_controller()
    try:
        c.move_joint_velocity(np.zeros(7))
        assert False, "expected AssertionError"
    except AssertionError:
        pass


def test_step_joint_velocity_commands_then_returns_state():
    c = _bare_controller()
    sentinel = object()
    c.get_state = lambda: sentinel  # stand in for the single state read
    a = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 1.0])
    out = c.step_joint_velocity(a)
    assert c._sent["action"].shape == (8,)  # velocity command sent
    np.testing.assert_allclose(c._sent["action"], a)
    assert c._sent["blocking"] is False
    assert out is sentinel  # returns the state read in the same call


def test_step_joint_velocity_rejects_wrong_dim():
    c = _bare_controller()
    c.get_state = lambda: None
    try:
        c.step_joint_velocity(np.zeros(7))
        assert False, "expected AssertionError"
    except AssertionError:
        pass
