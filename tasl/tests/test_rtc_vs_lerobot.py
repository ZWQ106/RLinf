"""Numerical comparison of tasl/rtc against LeRobot's RTCProcessor (the code
behind Intel's pi05-rtc-ov pipeline) and against Intel's OpenVINO-exported
variant.

Vendored verbatim (math only):
  * LeRobot src/lerobot/policies/rtc/modeling_rtc.py @ bf31dd7 — `denoise_step`
    guidance block + `get_prefix_weights` (linspace construction).
  * Intel edge-ai-suites robotics-ai-suite/pipelines/pi05-rtc-ov, patch
    0002/0004 `convert_ov_rtc.py::rtc_denoise_step` — identical except
    `correction = err` (the autograd VJP is dropped so the graph can be exported).

All three integrators use openpi's time convention (t: 1 -> 0), the same toy
velocity field and the same noise.

Run:  cd ~/RLinf/tasl && JAX_PLATFORMS=cpu ~/work/openpi/.venv/bin/python -m pytest tests/test_rtc_vs_lerobot.py -q -s
"""

import math
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rtc import rtc_math  # noqa: E402

# ─── toy field (openpi convention), numpy-agnostic ──────────────────────────
MU, SIGMA = 0.3, 0.8


def v_openpi(x_t, t):
    k = (1 - t) * SIGMA**2 / ((1 - t) ** 2 * SIGMA**2 + t**2)
    m = MU + k * (x_t - (1 - t) * MU)
    return (x_t - (1 - t) * m) / t - m


# ─── LeRobot (verbatim math) ────────────────────────────────────────────────


def lerobot_get_prefix_weights(start, end, total, schedule):
    start = min(start, end)
    if schedule == "zeros":
        weights = torch.zeros(total)
        weights[:start] = 1.0
        return weights
    if schedule == "ones":
        weights = torch.ones(total)
        weights[end:] = 0.0
        return weights
    skip_steps_at_end = max(total - end, 0)
    linspace_steps = total - skip_steps_at_end - start
    lin = torch.tensor([]) if (end <= start or linspace_steps <= 0) else torch.linspace(1, 0, linspace_steps + 2)[1:-1]
    if schedule == "exp":
        lin = lin * torch.expm1(lin).div(math.e - 1)
    zeros_len = total - end
    w = torch.cat([lin, torch.zeros(zeros_len)]) if zeros_len > 0 else lin
    ones_len = min(start, total)
    if ones_len > 0:
        w = torch.cat([torch.ones(ones_len), w])
    return w


def lerobot_guided_step(x_t, prev, inference_delay, execution_horizon, time, beta, schedule, vjp=True):
    """RTCProcessor.denoise_step core; vjp=False reproduces Intel's exported variant."""
    tau = 1 - time
    T = x_t.shape[1]
    weights = lerobot_get_prefix_weights(inference_delay, execution_horizon, T, schedule).unsqueeze(0).unsqueeze(-1)
    x_t = x_t.clone().detach()
    with torch.enable_grad():
        x_t.requires_grad_(True)
        v_t = v_openpi(x_t, time)
        x1_t = x_t - time * v_t
        err = (prev - x1_t) * weights
        if vjp:
            correction = torch.autograd.grad(x1_t, x_t, err.clone().detach(), retain_graph=False)[0]
        else:
            correction = err                       # Intel convert_ov_rtc.py
    beta_t = torch.as_tensor(float(beta))
    tau_t = torch.as_tensor(tau)
    sq1 = (1 - tau_t) ** 2
    inv_r2 = (sq1 + tau_t**2) / sq1
    c = torch.nan_to_num((1 - tau_t) / tau_t, posinf=beta_t)
    gw = torch.nan_to_num(c * inv_r2, posinf=beta_t)
    gw = torch.minimum(gw, beta_t)
    return (v_t - gw * correction).detach()


def lerobot_integrate(noise, prev, d, end, num_steps, beta, schedule, vjp=True):
    dt = -1.0 / num_steps
    x_t = noise.clone()
    for step in range(num_steps):
        time = 1.0 + step * dt
        v_t = lerobot_guided_step(x_t, prev, d, end, time, beta, schedule, vjp=vjp)
        x_t = x_t + dt * v_t
    return x_t


# ─── ours ──────────────────────────────────────────────────────────────────


def ours_integrate(noise, prev, d, end, num_steps, beta, schedule):
    dt = -1.0 / num_steps
    H = noise.shape[1]
    weights = rtc_math.get_prefix_weights(d, end, H, rtc_math.schedule_code(schedule))

    def step(carry):
        x_t, time = carry
        v_t = rtc_math.guided_velocity(lambda x: v_openpi(x, time), x_t, time, prev, weights, beta)
        return x_t + dt * v_t, time + dt

    x_0, _ = jax.lax.while_loop(lambda c: c[1] >= -dt / 2, step, (noise, 1.0))
    return x_0


def test_prefix_weights_identical_to_lerobot():
    for schedule in ("zeros", "ones", "linear", "exp"):
        for d, end, T in [(3, 12, 16), (2, 6, 10), (0, 16, 16), (8, 4, 10), (5, 45, 50), (0, 0, 8)]:
            ours = np.asarray(rtc_math.get_prefix_weights(d, end, T, rtc_math.schedule_code(schedule)))
            ref = lerobot_get_prefix_weights(d, end, T, schedule).numpy()
            np.testing.assert_allclose(ours, ref, atol=1e-6, err_msg=f"{schedule} {d} {end} {T}")


def test_guided_sampling_identical_to_lerobot():
    H, A, batch, n = 16, 3, 16, 10
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((batch, H, A)).astype(np.float32)
    prev = (rng.standard_normal((batch, H, A)) * 0.5 + 1.0).astype(np.float32)
    for schedule in ("exp", "linear"):
        for d, end, beta in [(3, 12, 5.0), (2, 8, 10.0), (0, 12, 5.0)]:
            ref = lerobot_integrate(torch.tensor(noise), torch.tensor(prev), d, end, n, beta, schedule).numpy()
            ours = np.asarray(ours_integrate(jnp.asarray(noise), jnp.asarray(prev), d, end, n, beta, schedule))
            np.testing.assert_allclose(ours, ref, rtol=1e-4, atol=1e-4, err_msg=f"{schedule} d={d} end={end} beta={beta}")


def test_intel_openvino_variant_is_a_different_algorithm():
    """Intel's exported graph drops the VJP (correction = err). Document that this
    is NOT equivalent, and how far its frozen prefix lands from the target."""
    H, A, batch, n = 16, 3, 256, 10
    d, end = 3, 12
    rng = np.random.default_rng(1)
    noise = torch.tensor(rng.standard_normal((batch, H, A)).astype(np.float32))
    prev = torch.full((batch, H, A), 3.0)
    paper = lerobot_integrate(noise, prev, d, end, n, 5.0, "exp", vjp=True).numpy()
    intel = lerobot_integrate(noise, prev, d, end, n, 5.0, "exp", vjp=False).numpy()
    err_paper = np.abs(paper[:, :d] - 3.0).mean()
    err_intel = np.abs(intel[:, :d] - 3.0).mean()
    print(f"\n[toy field] frozen-prefix |x - Y|: paper/VJP={err_paper:.3f}  intel/identity-jacobian={err_intel:.3f}")
    assert not np.allclose(paper, intel, atol=1e-3)
    # tail (weight 0) is untouched in both
    np.testing.assert_allclose(paper[:, end:], intel[:, end:], atol=1e-5)
