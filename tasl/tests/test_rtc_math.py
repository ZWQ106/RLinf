"""CPU tests for rtc/rtc_math.py (JAX; no checkpoint needed).

Run:  cd ~/RLinf/tasl && JAX_PLATFORMS=cpu ~/work/openpi/.venv/bin/python -m pytest tests/test_rtc_math.py -q
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rtc import rtc_math as rtc  # noqa: E402


def test_prefix_weights_linear_matches_reference_docstring():
    # Reference (kinetix src/model.py): start=2, end=6, total=10 ->
    # 1 1 4/5 3/5 2/5 1/5 0 0 0 0
    w = rtc.get_prefix_weights(2, 6, 10, rtc.schedule_code("linear"))
    np.testing.assert_allclose(np.asarray(w), [1, 1, 0.8, 0.6, 0.4, 0.2, 0, 0, 0, 0], atol=1e-6)


def test_prefix_weights_schedules():
    d, end, h = 3, 11, 15
    zeros = np.asarray(rtc.get_prefix_weights(d, end, h, rtc.schedule_code("zeros")))
    ones = np.asarray(rtc.get_prefix_weights(d, end, h, rtc.schedule_code("ones")))
    lin = np.asarray(rtc.get_prefix_weights(d, end, h, rtc.schedule_code("linear")))
    exp = np.asarray(rtc.get_prefix_weights(d, end, h, rtc.schedule_code("exp")))
    for w in (zeros, ones, lin, exp):   # frozen prefix 1, beyond `end` 0
        assert (w[:d] == 1).all() and (w[end:] == 0).all()
    assert (zeros[d:end] == 0).all()
    assert (ones[d:end] == 1).all()
    assert (exp[d:end] <= lin[d:end] + 1e-6).all()      # exp decays faster
    assert (np.diff(exp[d:end]) <= 1e-6).all()          # and is monotone
    jitted = jax.jit(lambda s, e, c: rtc.get_prefix_weights(s, e, h, c))  # traced ints
    np.testing.assert_allclose(np.asarray(jitted(d, end, 2)), exp, atol=1e-6)


def test_prefix_weights_end_before_start():
    w = np.asarray(rtc.get_prefix_weights(8, 4, 10, rtc.schedule_code("exp")))
    assert (w[:4] == 1).all() and (w[4:] == 0).all()


def test_guidance_weight_clipped_and_finite():
    beta = 5.0
    w = np.asarray(jax.vmap(lambda t: rtc.guidance_weight(t, beta))(jnp.linspace(1.0, 0.0, 11)))
    assert np.isfinite(w).all() and (w <= beta + 1e-6).all() and (w > 0).all()
    assert w[0] == beta and w[-1] == beta   # both ends clipped
    assert w[5] < beta                      # tau = 0.5 -> 2.0


def _gaussian_flow_velocity(mu, sigma):
    """Exact openpi-convention flow velocity for x0 ~ N(mu, sigma^2 I): the
    denoiser x_t - t v(x_t) recovers E[x0 | x_t], which has a non-trivial
    Jacobian for the pseudoinverse guidance to act on."""

    def velocity(x_t, t):
        k = (1 - t) * sigma**2 / ((1 - t) ** 2 * sigma**2 + t**2)
        m = mu + k * (x_t - (1 - t) * mu)
        return (x_t - (1 - t) * m) / t - m

    return velocity


def _sample(velocity, noise, num_steps, prev=None, weights=None, beta=5.0):
    dt = -1.0 / num_steps

    def step(carry):
        x_t, time = carry
        if prev is None:
            v = velocity(x_t, time)
        else:
            v = rtc.guided_velocity(lambda x: velocity(x, time), x_t, time, prev, weights, beta)
        return x_t + dt * v, time + dt

    x0, _ = jax.lax.while_loop(lambda c: c[1] >= -dt / 2, step, (noise, 1.0))
    return x0


def test_guided_sampling_inpaints_prefix_and_leaves_tail_free():
    h, a, batch = 8, 2, 256
    d, s = 2, 3
    velocity = _gaussian_flow_velocity(0.0, 1.0)
    noise = jax.random.normal(jax.random.key(0), (batch, h, a))
    target = 3.0
    prev = jnp.full((batch, h, a), target)
    weights = rtc.get_prefix_weights(d, h - s, h, rtc.schedule_code("exp"))
    plain = np.asarray(_sample(velocity, noise, 10))
    guided = np.asarray(_sample(velocity, noise, 10, prev, weights))
    assert abs(plain.mean()) < 0.2 and abs(plain.std() - 1.0) < 0.2
    assert np.abs(guided[:, :d] - target).mean() < 0.35           # frozen prefix inpainted
    soft = np.abs(guided[:, d: h - s] - target).mean(axis=(0, 2))
    assert (np.diff(soft) > -1e-3).all() and soft[0] < soft[-1]   # soft region: decaying pull
    np.testing.assert_allclose(guided[:, h - s:], plain[:, h - s:], atol=1e-4)  # tail untouched


def test_hard_mask_only_affects_prefix():
    h, a, batch = 8, 1, 64
    velocity = _gaussian_flow_velocity(0.0, 1.0)
    noise = jax.random.normal(jax.random.key(1), (batch, h, a))
    prev = jnp.full((batch, h, a), -2.0)
    weights = rtc.get_prefix_weights(3, 8, h, rtc.schedule_code("zeros"))
    plain = np.asarray(_sample(velocity, noise, 10))
    guided = np.asarray(_sample(velocity, noise, 10, prev, weights))
    assert np.abs(guided[:, :3] + 2.0).mean() < 0.35
    np.testing.assert_allclose(guided[:, 3:], plain[:, 3:], atol=1e-4)
