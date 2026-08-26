"""RTC-capable wrapper around an openpi JAX `Policy` (Pi0 / Pi05).

`RTCPolicy.infer(obs)` is byte-for-byte the wrapped policy's `infer` unless the
request dict carries an `rtc` key:

    obs["rtc"] = {
        "prev_actions": None | float32 [H, action_dim]   # model-space chunk of
                        # the previous response (`actions_model`), already
                        # shifted by the executed steps + zero right-padded
        "inference_delay": d,              # frozen prefix length (ticks)
        "prefix_attention_horizon": H - s, # soft mask reaches 0 here
        "schedule": "exp" | "linear" | "zeros" | "ones",
        "max_guidance_weight": 5.0,        # beta
    }

With `rtc` present the response additionally carries
    "actions_model": float32 [H, action_dim]  (normalized, full width — feed
                                               it back as prev_actions next time)
    "rtc": {"guided": bool, "inference_delay": d, "prefix_attention_horizon": H-s}

`prev_actions=None` (the first chunk of an episode) samples the plain way but
still returns `actions_model`.

The guided sampler below mirrors `Pi0.sample_actions` from openpi (commit
c23745b, src/openpi/models/pi0.py) and swaps the velocity in the integration
step for `rtc_math.guided_velocity`. Pinned to that method's structure: prefix
KV cache once, then a suffix pass per denoising step.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy

from openpi.models import model as _model
from openpi.models.pi0 import Pi0, make_attn_mask
from openpi.policies.policy import Policy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rtc import rtc_math  # noqa: E402

_log = logging.getLogger("rtc.policy")


def sample_actions_rtc(
    model: Pi0,
    rng: jax.Array,
    observation: _model.Observation,
    *,
    prev_action_chunk: jax.Array,
    inference_delay,
    prefix_attention_horizon,
    prefix_attention_schedule,
    max_guidance_weight,
    num_steps: int = 10,
    noise: jax.Array | None = None,
) -> jax.Array:
    """Guided flow integration (paper Algorithm 1, GuidedInference)."""
    observation = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    if noise is None:
        noise = jax.random.normal(rng, (batch_size, model.action_horizon, model.action_dim))

    # Prefix (images + language) is encoded once and cached.
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    def velocity(x_t, time):
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size)
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = model.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        return model.action_out_proj(suffix_out[:, -model.action_horizon :])

    weights = rtc_math.get_prefix_weights(
        inference_delay, prefix_attention_horizon, model.action_horizon, prefix_attention_schedule
    )

    def step(carry):
        x_t, time = carry
        v_t = rtc_math.guided_velocity(
            lambda x: velocity(x, time), x_t, time, prev_action_chunk, weights, max_guidance_weight
        )
        return x_t + dt * v_t, time + dt

    def cond(carry):
        _, time = carry
        return time >= -dt / 2  # robust to floating-point error

    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    return x_0


def _jit_with_frozen_module(model: nnx.Module, fn):
    """jax.jit `fn(model, *args, **kw)` with the module state frozen — same idea
    as openpi.shared.nnx_utils.module_jit but for a free function."""
    graphdef, state = nnx.split(model)

    def fun(state, *args, **kwargs):
        return fn(nnx.merge(graphdef, state), *args, **kwargs)

    jitted = jax.jit(fun)

    def wrapper(*args, **kwargs):
        return jitted(state, *args, **kwargs)

    return wrapper


class RTCPolicy(_base_policy.BasePolicy):
    """Wraps a JAX openpi Policy; adds real-time-chunking guided sampling."""

    def __init__(self, policy: Policy, model_config: _model.BaseModelConfig | None = None):
        """`model_config` (the TrainConfig's `.model`) is only needed by
        `warmup()`, which builds a fake observation from its input spec."""
        if getattr(policy, "_is_pytorch_model", False):
            raise NotImplementedError("RTCPolicy supports the JAX model only (guidance uses jax.vjp).")
        if not isinstance(policy._model, Pi0):  # noqa: SLF001
            raise NotImplementedError(f"RTCPolicy expects a Pi0/Pi05 model, got {type(policy._model)}")  # noqa: SLF001
        self._policy = policy
        self._model: Pi0 = policy._model  # noqa: SLF001
        self._model_config = model_config
        self._sample_rtc = _jit_with_frozen_module(self._model, sample_actions_rtc)

    @property
    def metadata(self) -> dict:
        return self._policy.metadata

    @property
    def action_horizon(self) -> int:
        return int(self._model.action_horizon)

    @property
    def action_dim(self) -> int:
        return int(self._model.action_dim)

    def reset(self) -> None:
        self._policy.reset()

    def warmup(self) -> float:
        """Compile the guided trace on a fake observation (same shapes as real
        requests). Returns seconds spent. Call at server start so the first RTC
        episode does not stall on compilation."""
        if self._model_config is None:
            _log.warning("RTCPolicy.warmup skipped: no model_config given (first guided request will compile)")
            return 0.0
        t0 = time.monotonic()
        obs = self._model_config.fake_obs(1)
        prev = jnp.zeros((1, self.action_horizon, self.action_dim), jnp.float32)
        kwargs = dict(self._policy._sample_kwargs)  # noqa: SLF001
        kwargs.pop("noise", None)
        # Plain trace first (the first vanilla request otherwise pays ~4 s), then the guided one.
        jax.block_until_ready(self._policy._sample_actions(jax.random.key(0), obs, **kwargs))  # noqa: SLF001
        out = self._sample_rtc(
            jax.random.key(0), obs,
            prev_action_chunk=prev, inference_delay=2,
            prefix_attention_horizon=self.action_horizon - 4,
            prefix_attention_schedule=rtc_math.schedule_code("exp"),
            max_guidance_weight=5.0, **kwargs,
        )
        jax.block_until_ready(out)
        dt = time.monotonic() - t0
        _log.info(f"plain + RTC guided samplers compiled and warmed up in {dt:.1f}s")
        return dt

    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[override]
        if not isinstance(obs, dict) or "rtc" not in obs:
            return self._policy.infer(obs) if noise is None else self._policy.infer(obs, noise=noise)

        obs = dict(obs)
        rtc = obs.pop("rtc") or {}
        p = self._policy
        # --- mirrors Policy.infer (JAX branch) -------------------------------
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = p._input_transform(inputs)  # noqa: SLF001
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        p._rng, sample_rng = jax.random.split(p._rng)  # noqa: SLF001
        observation = _model.Observation.from_dict(inputs)
        sample_kwargs: dict[str, Any] = dict(p._sample_kwargs)  # noqa: SLF001
        if noise is not None:
            n = jnp.asarray(noise)
            sample_kwargs["noise"] = n if n.ndim == 3 else n[None, ...]
        # --- RTC request -----------------------------------------------------
        prev = rtc.get("prev_actions")
        guided = prev is not None
        d = int(rtc.get("inference_delay", 0))
        ph = int(rtc.get("prefix_attention_horizon", self.action_horizon))
        start = time.monotonic()
        if not guided:
            actions = p._sample_actions(sample_rng, observation, **sample_kwargs)  # noqa: SLF001
        else:
            prev = jnp.asarray(prev, dtype=jnp.float32)
            if prev.ndim == 2:
                prev = prev[None, ...]
            expected = (1, self.action_horizon, self.action_dim)
            if prev.shape != expected:
                raise ValueError(f"rtc/prev_actions must have shape {expected[1:]} (model space), got {tuple(prev.shape[1:])}")
            schedule = rtc.get("schedule", "exp")
            if isinstance(schedule, bytes):
                schedule = schedule.decode("utf-8")
            actions = self._sample_rtc(
                sample_rng, observation,
                prev_action_chunk=prev,
                inference_delay=d,
                prefix_attention_horizon=ph,
                prefix_attention_schedule=rtc_math.schedule_code(str(schedule)),
                max_guidance_weight=float(rtc.get("max_guidance_weight", 5.0)),
                **sample_kwargs,
            )
        outputs = {"state": inputs["state"], "actions": actions}
        model_time = time.monotonic() - start
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        actions_model = outputs["actions"]
        outputs = p._output_transform(outputs)  # noqa: SLF001
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        outputs["actions_model"] = np.asarray(actions_model, dtype=np.float32)
        outputs["rtc"] = {"guided": guided, "inference_delay": d, "prefix_attention_horizon": ph}
        return outputs
