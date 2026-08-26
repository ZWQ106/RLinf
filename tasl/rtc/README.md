# Real-Time Chunking (RTC) for the FR3 openpi eval portal

Inference-time RTC from *Real-Time Execution of Action Chunking Flow Policies*
(Black, Galliker, Levine — [arXiv:2506.07339](https://arxiv.org/abs/2506.07339);
reference sim code: [real-time-chunking-kinetix](https://github.com/Physical-Intelligence/real-time-chunking-kinetix)),
packaged as an **opt-in module**: nothing in `~/work/openpi` is modified, and
`dashboards/openpi.py` only carries a few one-line hooks (`grep -n _rtc_hook`)
plus the two loader buttons. Loaded w/o RTC → the portal is byte-for-byte the
old synchronous loop. Principle write-up (中文): [`PRINCIPLE.md`](PRINCIPLE.md).

## What it does

Synchronous (today): `obs → infer (~100 ms) → stream 4 actions @15 Hz → repeat`;
the arm pauses at every chunk boundary and each chunk is generated blind to
the one it continues from.

RTC: a **controller thread** consumes one action per 15 Hz tick from the
current chunk and never waits. Once `s ≥ s_min` actions were consumed, the
**inference loop** captures a fresh obs and asks the policy for the next
chunk, passing the remaining actions of the current chunk (shifted into the
new frame) and the expected inference delay `d`. The policy **inpaints**:
during flow denoising the first `d` actions are frozen to what will already
have been executed (hard mask), the next `H-s-d` are softly pulled toward the
old chunk with exponentially decaying weight, and the last `s` are free
(paper Eq. 2–5, ΠGDM guidance with clip β). When the new chunk arrives the
executor swaps it in at index `elapsed` (ticks that passed meanwhile) — no
pause, no jump. `d` is re-estimated as the max over the last `delay_buffer`
observed delays (Algorithm 1).

Constraints (paper): `d ≤ s ≤ H − d`. `pi05_droid_franka_lora` serves `H = 16`
(base `pi05_droid`: 15) at 15 Hz. Measured on the desktop GPU (smoke, 2026-08-25):
plain 95 ms, guided 138 ms ≈ 2.1 ticks → `delay_init = 3` with `s_min = 4`.

## Layout

| file | side | role |
|---|---|---|
| `rtc_math.py` | server (JAX) | soft mask `get_prefix_weights` (Eq. 5), `guidance_weight` (Eq. 4 + β clip), `guided_velocity` (Eq. 2 via `jax.vjp`); model-agnostic, openpi time convention (t=1 noise) |
| `rtc_policy.py` | server | `RTCPolicy` wraps an openpi `Policy`; `sample_actions_rtc` mirrors `Pi0.sample_actions` with the guided step; `warmup()` compiles the guided trace at start |
| `scripts/serve_policy.py` | server | drop-in for `openpi/scripts/serve_policy.py` (same CLI, run with openpi's venv). Named so the dashboard's `pgrep/pkill serve_policy.py` keep working |
| `executor.py` | client | `RTCConfig` (knobs) + `RTCExecutor` (Algorithm 1: controller thread + inference loop), callback-wired, hardware-free |
| `dashboard_hook.py` | client | glue for `dashboards/openpi.py`: `run_episode`, `/rtc` routes, `status`, UI card, obs capture (mirror of the sync loop's obs block) |
| `smoke.py` | tool | offline check against a running `:8000` — inpainting really holds the prefix, schedule behaviour, latency → suggested `delay_init` |
| `../tests/test_rtc_{math,executor,hook}.py` | tests | see run lines in each file |

## Wire protocol (client ↔ policy server)

Request = normal DROID obs dict + `obs["rtc"] = {prev_actions, inference_delay,
prefix_attention_horizon, schedule, max_guidance_weight}`. `prev_actions` is
the previous response's `actions_model` (**model space**: normalized, 32-dim)
shifted by the executed steps and zero-padded — so the server stays stateless
and no un/normalize round trip is needed. `None` for the first chunk.
Response adds `actions_model` and `rtc: {guided, inference_delay, prefix_attention_horizon}`.
Requests without `rtc` are served exactly as before (vanilla clients unaffected).

## Using it

The switch is the checkpoint loader in the portal:

* **Load w/o RTC** → openpi's own `scripts/serve_policy.py`; the eval loop is
  the old synchronous one, byte for byte.
* **Load with RTC** → `tasl/rtc/scripts/serve_policy.py` (same CLI); the eval
  loop takes the RTC path. The ckpt state line shows `· RTC` / `· sync`, and
  episodes are recorded under ckpt label `<label> +rtc` so the "by ckpt"
  grouping in the episode list separates the two modes for the same step.
* The **Real-time chunking (RTC)** card only holds knobs (`s_min`, `delay_init`,
  β, schedule, warm-up) → **Apply** (refused while an eval runs) and shows the
  live `s / d / elapsed / infer ms / starved ticks` during an RTC episode.
* `start_openpi.sh`'s cold-start serve is the vanilla one (= w/o RTC). If the
  dashboard restarts while a serve is running it adopts it and detects the
  mode from the process command line.
* Every RTC episode writes `rtc.json` (config + stats) next to `meta.json` in
  `eval_episodes/<task>/<ep>/`; `traj.jsonl` rows carry an `rtc` dict.

Offline check (no robot):
```bash
~/work/openpi/.venv/bin/python ~/RLinf/tasl/rtc/scripts/serve_policy.py --port 8000 \
    policy:checkpoint --policy.config=pi05_droid_franka_lora \
    --policy.dir=$HOME/ckpts/pi05_droid_franka_lora_10task/16000 &
PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$HOME/work/openpi/packages/openpi-client/src \
    /usr/bin/python3 ~/RLinf/tasl/rtc/smoke.py --s 4 --d 3
```
Expected: guided prefix error well below the independent-sample baseline
(2026-08-25: 0.040 vs 0.095; hard mask 0.026), tail identical, `✓ RTC smoke OK`.

Tests:
```bash
cd ~/RLinf/tasl
JAX_PLATFORMS=cpu ~/work/openpi/.venv/bin/python -m pytest tests/test_rtc_math.py tests/test_rtc_executor.py -q
PYTHONPATH=$HOME/.local/lib/python3.10/site-packages /usr/bin/python3 -m pytest tests/test_rtc_hook.py -q
```

## Not covered (yet)

* The hooked PyTorch server (`VLA-PatchLen-cp/serve_policy_patched.py`) —
  guidance there would need `torch.autograd` through the hooked forward.
* Legacy binary gripper modes: the RTC path streams the absolute gripper
  command per tick (= the `proportional` mode).
* Training-time RTC (arXiv:2512.05964) — different recipe, not inference-only.
