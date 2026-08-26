"""Real-Time Chunking (RTC): inference-time inpainting guidance for flow policies.

Reference
    Black, Galliker, Levine. "Real-Time Execution of Action Chunking Flow
    Policies" (arXiv:2506.07339) - Sec. 3, Eq. 2-5, Algorithm 1.
    Simulation code: github.com/Physical-Intelligence/real-time-chunking-kinetix
    (src/model.py: get_prefix_weights / realtime_action).

Time convention
    openpi integrates from t=1 (noise) down to t=0 (data) with dt < 0. The paper
    uses tau in [0, 1) with tau=0 noise and tau=1 data, so tau = 1 - t and the
    learned velocity flips sign (v_openpi = noise - data = -v_paper). Every
    helper below takes the *openpi* time t and returns an openpi-convention
    velocity, so `x_t + dt * v` stays the integration step in both branches.

Nothing in this module is model specific: it only needs a callable that maps
a noisy chunk x_t to the model's velocity at a fixed time.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp

# Integer codes so the schedule can be passed through jax.jit as a plain
# (traced) scalar instead of a static string argument.
SCHEDULES: tuple[str, ...] = ("zeros", "linear", "exp", "ones")


def schedule_code(name: str) -> int:
    """Map a schedule name to its integer code (see SCHEDULES)."""
    try:
        return SCHEDULES.index(name)
    except ValueError as exc:
        raise ValueError(f"unknown prefix attention schedule {name!r}; choose from {SCHEDULES}") from exc


def get_prefix_weights(start, end, total: int, schedule) -> jax.Array:
    """Soft mask W over the action chunk (paper Eq. 5).

    Args:
        start: inference delay d. Actions [0, start) are frozen (weight 1).
        end: prefix attention horizon H - s. Actions [end, total) get weight 0
            (they lie beyond the end of the previous chunk). `end` takes
            precedence: if end < start, start is pushed down to end.
        total: chunk length H.
        schedule: int code into SCHEDULES (traced scalar is fine) -
            "zeros"  = hard mask (only the frozen prefix),
            "linear" = linear decay from 1 at `start` to 0 at `end`,
            "exp"    = the paper's default, c * (e^c - 1) / (e - 1),
            "ones"   = attend to the whole previous chunk with weight 1.

    With start=2, end=6, total=10 and schedule="linear":
        1 1 4/5 3/5 2/5 1/5 0 0 0 0
    """
    start = jnp.minimum(start, end)
    idx = jnp.arange(total)
    # (end - i) / (end - start + 1), clipped: 1 for i < start, 0 for i >= end.
    lin = jnp.clip((start - 1 - idx) / (end - start + 1) + 1, 0.0, 1.0)
    w = jnp.select(
        [schedule == 0, schedule == 1, schedule == 2, schedule == 3],
        [
            (idx < start).astype(jnp.float32),
            lin,
            lin * jnp.expm1(lin) / (jnp.e - 1),
            jnp.ones(total, dtype=jnp.float32),
        ],
        default=lin,
    )
    # Frozen prefix is exactly 1 (the exp schedule lands at 1 - 1 ulp otherwise).
    w = jnp.where(idx < start, 1.0, w)
    return jnp.where(idx >= end, 0.0, w)


def guidance_weight(t, max_guidance_weight) -> jax.Array:
    """min(beta, (1 - tau) / (tau * r_tau^2)) from Eq. 2/4, with tau = 1 - t.

    r_tau^2 = (1 - tau)^2 / (tau^2 + (1 - tau)^2), so the unclipped weight is
    (tau^2 + (1 - tau)^2) / (tau * (1 - tau)); it diverges at both ends of the
    trajectory and the clip at beta keeps the few-step integration stable.
    """
    tau = 1.0 - t
    num = tau**2 + (1.0 - tau) ** 2
    den = jnp.maximum(tau * (1.0 - tau), 1e-6)
    return jnp.minimum(num / den, max_guidance_weight)


def guided_velocity(
    velocity_fn: Callable[[jax.Array], jax.Array],
    x_t: jax.Array,
    t,
    prev_action_chunk: jax.Array,
    weights: jax.Array,
    max_guidance_weight,
) -> jax.Array:
    """One guided denoising step (paper Eq. 2, Algorithm 1 lines 26-29).

    Args:
        velocity_fn: x_t -> v_t at fixed openpi time t (shape [b, H, A]).
        x_t: current noisy chunk [b, H, A].
        t: openpi time of x_t (1 = noise, 0 = data).
        prev_action_chunk: Y, the previous chunk shifted to the new frame and
            right-padded to H, in *model* (normalized) action space [b, H, A].
        weights: W from get_prefix_weights, shape [H].
        max_guidance_weight: beta.

    Returns:
        The corrected velocity in openpi convention; integrate as x_t + dt * v.
    """

    def denoiser(x):
        v = velocity_fn(x)
        # Eq. 3 in openpi time: x0_hat = x_t + (0 - t) * v.
        return x - t * v, v

    x0_hat, vjp_fn, v_t = jax.vjp(denoiser, x_t, has_aux=True)
    error = (prev_action_chunk - x0_hat) * weights[None, :, None]
    correction = vjp_fn(error)[0]
    # Paper: A += (1/n) (v_paper + w g). With v_openpi = -v_paper and dt = -1/n
    # this is x += dt (v_openpi - w g).
    return v_t - guidance_weight(t, max_guidance_weight) * correction
