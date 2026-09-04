import numpy as np

from rlinf.envs.realworld.common.wrappers.delayed_gello_joint_intervention import (
    DelayedLeaderStream,
)


def _ramp(stream, n=10, dt=0.02, t0=100.0):
    # q ramps linearly with send time so interpolation is checkable.
    for i in range(n):
        stream.push(np.full(7, i * 1.0), gripper=0.0, t_send=t0 + i * dt)
    return t0, dt


def test_nothing_arrives_before_tau_f():
    s = DelayedLeaderStream(tau_f=0.3)
    t0, _ = _ramp(s)
    assert s.latest_arrived(now=t0 + 0.29) is None
    assert s.interpolated(now=t0 + 0.29) is None


def test_direct_returns_newest_arrived_by_send_time():
    s = DelayedLeaderStream(tau_f=0.3)
    t0, dt = _ramp(s)
    got = s.latest_arrived(now=t0 + 0.3 + 4 * dt)
    assert got is not None and got.q[0] == 4.0
    assert abs((t0 + 0.3 + 4 * dt) - got.t_send - 0.3) < 1e-9  # applied delay == tau_f


def test_direct_drops_out_of_order_arrivals():
    # Large jitter: later sends may arrive earlier; the consumer must never go
    # backwards in send time.
    s = DelayedLeaderStream(tau_f=0.1, jitter_s=0.2, seed=1)
    t0, dt = _ramp(s, n=30)
    last = -1.0
    for k in range(60):
        got = s.latest_arrived(now=t0 + 0.1 + k * 0.01)
        if got is not None:
            assert got.q[0] >= last
            last = got.q[0]


def test_queued_interpolates_between_samples():
    s = DelayedLeaderStream(tau_f=0.1)
    t0, dt = _ramp(s)
    # newest arrived at now: sample 5 (send t0+5dt); playout half a sample back
    now = t0 + 0.1 + 5 * dt
    got = s.interpolated(now=now, playout_s=dt / 2)
    assert got is not None
    assert abs(got.q[0] - 4.5) < 1e-9
    assert abs(got.gripper) < 1e-9


def test_queued_holds_last_when_buffer_dry():
    s = DelayedLeaderStream(tau_f=0.0)
    s.push(np.ones(7), gripper=0.3, t_send=10.0)
    got = s.interpolated(now=11.0, playout_s=5.0)
    assert got is not None and got.q[0] == 1.0 and got.gripper == 0.3


def test_zero_delay_zero_jitter_is_passthrough():
    s = DelayedLeaderStream(tau_f=0.0)
    t0, dt = _ramp(s)
    got = s.latest_arrived(now=t0 + 9 * dt)
    assert got.q[0] == 9.0


def test_jitter_is_nonnegative_and_seeded():
    a = DelayedLeaderStream(tau_f=0.2, jitter_s=0.05, seed=7)
    b = DelayedLeaderStream(tau_f=0.2, jitter_s=0.05, seed=7)
    for i in range(20):
        sa = a.push(np.zeros(7), 0.0, t_send=float(i))
        sb = b.push(np.zeros(7), 0.0, t_send=float(i))
        assert sa.t_arrive >= sa.t_send + 0.2
        assert sa.t_arrive == sb.t_arrive
