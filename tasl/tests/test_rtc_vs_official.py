"""Numerical equivalence of tasl/rtc against the official RTC code.

The reference functions below are copied VERBATIM from
Physical-Intelligence/real-time-chunking-kinetix, src/model.py (commit 9296f31):
`get_prefix_weights` and the ΠGDM step inside `FlowPolicy.realtime_action`.
They use the paper's time convention (tau: 0 = noise -> 1 = data, dt = +1/n);
ours uses openpi's (t: 1 = noise -> 0 = data, dt = -1/n). Both integrators are
driven by the SAME toy velocity field and the SAME noise; the sampled chunks
must agree element-wise.

Run:  cd ~/RLinf/tasl && JAX_PLATFORMS=cpu ~/work/openpi/.venv/bin/python -m pytest tests/test_rtc_vs_official.py -q
"""

import functools
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rtc import rtc_math  # noqa: E402

# ─── official (verbatim) ───────────────────────────────────────────────────


def official_get_prefix_weights(start: int, end: int, total: int, schedule: str) -> jax.Array:
    start = jnp.minimum(start, end)
    if schedule == "ones":
        w = jnp.ones(total)
    elif schedule == "zeros":
        w = (jnp.arange(total) < start).astype(jnp.float32)
    elif schedule == "linear" or schedule == "exp":
        w = jnp.clip((start - 1 - jnp.arange(total)) / (end - start + 1) + 1, 0, 1)
        if schedule == "exp":
            w = w * jnp.expm1(w) / (jnp.e - 1)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")
    return jnp.where(jnp.arange(total) >= end, 0, w)


def official_realtime_action(model_paper, noise, num_steps, prev_action_chunk, inference_delay,
                             prefix_attention_horizon, prefix_attention_schedule, max_guidance_weight,
                             action_chunk_size):
    """FlowPolicy.realtime_action with the network call `self(obs, x, t)` replaced by
    `model_paper(x, t)` (toy field, paper time convention) and the noise passed in."""
    dt = 1 / num_steps

    def step(carry, _):
        x_t, time = carry

        @functools.partial(jax.vmap, in_axes=(0, 0, None))  # over batch
        def pinv_corrected_velocity(x_t, y, t):
            def denoiser(x_t):
                v_t = model_paper(x_t[None], t)[0]
                return x_t + v_t * (1 - t), v_t

            x_1, vjp_fun, v_t = jax.vjp(denoiser, x_t, has_aux=True)
            weights = official_get_prefix_weights(
                inference_delay, prefix_attention_horizon, action_chunk_size, prefix_attention_schedule
            )
            error = (y - x_1) * weights[:, None]
            pinv_correction = vjp_fun(error)[0]
            # constants from paper
            inv_r2 = (t**2 + (1 - t) ** 2) / ((1 - t) ** 2)
            c = jnp.nan_to_num((1 - t) / t, posinf=max_guidance_weight)
            guidance_weight = jnp.minimum(c * inv_r2, max_guidance_weight)
            return v_t + guidance_weight * pinv_correction

        v_t = pinv_corrected_velocity(x_t, prev_action_chunk, time)
        return (x_t + dt * v_t, time + dt), None

    (x_1, _), _ = jax.lax.scan(step, (noise, 0.0), length=num_steps)
    return x_1


# ─── ours, driven the way rtc_policy.sample_actions_rtc drives it ──────────


def ours_sample(model_openpi, noise, num_steps, prev, d, ph, schedule, beta, H):
    dt = -1.0 / num_steps
    weights = rtc_math.get_prefix_weights(d, ph, H, rtc_math.schedule_code(schedule))

    def step(carry):
        x_t, time = carry
        v_t = rtc_math.guided_velocity(lambda x: model_openpi(x, time), x_t, time, prev, weights, beta)
        return x_t + dt * v_t, time + dt

    x_0, _ = jax.lax.while_loop(lambda c: c[1] >= -dt / 2, step, (noise, 1.0))
    return x_0


# ─── toy flow field: x0 ~ N(mu, sigma^2), exact posterior-mean velocity ─────


def make_fields(mu=0.3, sigma=0.8):
    def v_openpi(x_t, t):  # openpi: x_t = t*eps + (1-t)*x0, v = eps - x0
        k = (1 - t) * sigma**2 / ((1 - t) ** 2 * sigma**2 + t**2)
        m = mu + k * (x_t - (1 - t) * mu)
        return (x_t - (1 - t) * m) / t - m

    def v_paper(x_tau, tau):  # paper: A^tau = (1-tau)*eps + tau*x0, v = x0 - eps
        return -v_openpi(x_tau, 1.0 - tau)

    return v_openpi, v_paper


def test_prefix_weights_identical_to_official():
    for schedule in ("zeros", "linear", "exp", "ones"):
        for start, end, total in [(2, 6, 10), (3, 12, 16), (0, 16, 16), (8, 4, 10), (5, 5, 8), (0, 0, 8), (16, 16, 16)]:
            ours = np.asarray(rtc_math.get_prefix_weights(start, end, total, rtc_math.schedule_code(schedule)))
            ref = np.asarray(official_get_prefix_weights(start, end, total, schedule))
            np.testing.assert_allclose(ours, ref, atol=2e-7, err_msg=f"{schedule} {start} {end} {total}")


def test_guided_sampling_identical_to_official():
    H, A, batch, n = 16, 3, 32, 10
    v_openpi, v_paper = make_fields()
    noise = jax.random.normal(jax.random.key(3), (batch, H, A))
    prev = jax.random.normal(jax.random.key(4), (batch, H, A)) * 0.5 + 1.0
    for schedule in ("exp", "zeros", "linear", "ones"):
        for d, s in [(3, 4), (0, 4), (2, 8), (5, 5)]:
            ref = np.asarray(official_realtime_action(v_paper, noise, n, prev, d, H - s, schedule, 5.0, H))
            ours = np.asarray(ours_sample(v_openpi, noise, n, prev, d, H - s, schedule, 5.0, H))
            np.testing.assert_allclose(ours, ref, rtol=1e-5, atol=1e-5, err_msg=f"{schedule} d={d} s={s}")


def test_guidance_weight_identical_to_official_on_the_grid():
    beta = 5.0
    for n in (5, 10):
        for k in range(n):
            tau = jnp.float32(k / n)   # jnp so 1/0 -> inf (as in the official code), not ZeroDivisionError
            inv_r2 = (tau**2 + (1 - tau) ** 2) / ((1 - tau) ** 2)
            c = jnp.nan_to_num((1 - tau) / tau, posinf=beta)
            ref = float(jnp.minimum(c * inv_r2, beta))
            ours = float(rtc_math.guidance_weight(1.0 - tau, beta))
            assert abs(ours - ref) < 1e-6, (n, k, ours, ref)
