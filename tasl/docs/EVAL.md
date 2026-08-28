# TASL FR3 — pi05_droid Policy Eval Manual (RLinf runtime, polymetis controller)

How to run a **pi05_droid policy eval** on the FR3 bench through the **RLinf
runtime** (`eval_embodied_agent.py`), driving the arm with the **same polymetis
controller as teleop/collect**. We use RLinf only for its efficient pi05_droid
inference + infra; the robot is driven by the proven polymetis joint-velocity
streaming path. For GELLO teleop/collection see [TELEOP.md](TELEOP.md); for the
component map see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 0. The two pi05 deployment paths (and what they share)

The same pi05_droid VLA can drive the arm through **two runtimes**, but — per the
design decision on 2026-06-22 — **both use the polymetis zerorpc controller**
(backend B). The franky `robot_server` (backend A) is *not* used; it was too
limited in practice.

| | **openpi standalone** (`openpi.py` :8003) | **RLinf eval** (`rlinf.py` :8003) — *this doc* |
|---|---|---|
| Policy | openpi `serve_policy` WS :8000 + in-process loop | model loaded inside RLinf, under **Ray** |
| Runtime | thin client on the Desktop host | `eval_embodied_agent.py` in `rlinf-eval` container |
| Checkpoint | openpi-format `~/.cache/openpi/.../pi05_droid` | RLinf-format `/ckpts/pi05_droid_pt` |
| Rollout owner | the dashboard loop | the RLinf env (`FrankaJointVelEnv` + `RealWorldEnv.chunk_step`) |
| **NUC controller** | polymetis **zerorpc** (backend B) | polymetis **zerorpc** (backend B) ✅ same |
| Data written | none (pure inference) | LeRobot dataset (`/tmp/rlinf_pi05_droid_eval`) |
| Use it to… | quickly sanity-check the checkpoint | eval the policy in the RLinf env at scale |

> **Real-time chunking (RTC):** the openpi standalone portal has an opt-in RTC mode
> (async controller + guided inpainting, arXiv:2506.07339) — see [`../rtc/README.md`](../rtc/README.md).

**Why polymetis works for chunked VLA execution:** RLinf's pi05 rollout emits an
action *chunk* (e.g. 8 waypoints). `RealWorldEnv.chunk_step` executes it by
streaming the waypoints action-by-action over the polymetis `update_joint_velocity`
(non-blocking, ~15 Hz) — the exact path GELLO teleop already streams smoothly. So
we do **not** need franky's blocking HTTP `/move/joint_velocity_chunk`. Same
controller, same `:4242`, same FCI bring-up as teleop.

---

## 1. One-time prerequisite — unified eval checkout (pi05 inference on the polymetis branch)

> **Consolidated 2026-07-03:** the `rlinf-eval` container now mounts **`~/RLinf`**
> at `/workspace/rlinf` (not `~/work/rlinf-clone`, which is retired), and the eval
> code below is merged into it. Data is on the separate `~/rlinf_data` mount. See
> [SETUP.md](SETUP.md) and the tasl README for the container recipe + env vars.

The eval reuses the **polymetis-controller** work (which already drives the arm
for collect) and adds **only the pi05_droid inference** on top. The mounted
checkout `~/RLinf` already has the controller (`polymetis_controller.py`,
`droid_zerorpc_client.py`, the polymetis `FrankaJointVelEnv`,
`RealWorldEnv.chunk_step`) and the openpi action model, plus the integration
pieces below:

- `rlinf/models/embodiment/openpi/dataconfig/droid_dataconfig.py` — register the
  **`pi05_droid`** TrainConfig (mirror the existing `pi05_droid_polaris`, point
  `pytorch_weight_path` at `/ckpts/pi05_droid_pt`, set our DROID cam/action map).
- `examples/embodiment/config/realworld_eval_pi05_droid_polymetis.yaml` — the eval
  config: `model/pi0_5` + openpi `pi05_droid` + `env/realworld_franka_jointvel_polymetis`
  (polymetis controller) + `only_eval: True` + chunk via `RealWorldEnv.chunk_step`,
  with `robot_ip: 172.16.0.2` (robot net) and our camera serials.
- `rlinf.py` dashboard: robot panel swapped from HTTP `RS` → zerorpc `DroidClient`
  (port the proven plumbing from `collect.py`); eval spawn → the polymetis config.

`eval.sh` **preflights** the config + `polymetis_controller.py` + `droid_dataconfig.py`
and refuses to launch (with the missing-file list) if `/workspace/rlinf` lacks them.

The checkpoint `/ckpts/pi05_droid_pt` is already mounted (verified present). Because
this is all on the polymetis branch, **one checkout/container serves both collect
and eval** — no backend A/B split.

---

## 2. Start an eval session

```bash
sudo ~/RLinf/tasl/launch/eval.sh
```

Four stages (Stages 1–2 are identical to `teleop.sh` — same polymetis backend):

1. **NUC1 backend (B)** — ssh `tasl@172.16.0.2`: stop `franka-robot-server` (FCI
   exclusivity), `docker compose up -d` the polymetis container, wait for zerorpc
   `:4242`.
2. **FCI gate (manual)** — open **Franka Desk** `https://172.16.0.1`, unlock joints,
   **Activate FCI** + execution mode, then press Enter.
3. **Eval dashboard `:8003`** — frees the ZEDs, launches `rlinf.py --mode host`.
   The dashboard **owns the cameras permanently** and serves frames over HTTP
   (`/cam/<name>.jpg`) to the in-container env — no camera handoff.
4. **Bootstrap + auto-home** — bootstraps the polymetis controller against the live
   FCI (zerorpc) and homes to the DROID anchor pose, via the dashboard.

---

## 3. Run eval episodes

1. Open `http://<printed IP>:8003` (or `http://localhost:8003` on the Desktop).
   Confirm live ZED previews + the **policy view** tiles (224×224, exactly what
   the policy sees) + green robot status.
2. **Type the task prompt** (e.g. "pick up the cup") and press **Start**.
   The **pbc** checkbox next to Start picks the image preprocessing and must
   match the checkpoint's training data: unchecked = DROID `resize_with_pad`
   (224×126 + black bars; `pi05_droid`, `…-10task`, `…-10task-v2` ckpts),
   checked = center-crop 720×720 → 224² (`…-pbc` / `…-pbc-v2` ckpts, built by
   `tasl/tools/make_centercrop_dataset.py`). The policy-view tiles follow the
   checkbox live; the choice is stored per-episode as `image_mode` in
   `meta.json`.
   The dashboard `docker exec`s `eval_embodied_agent.py --config-name
   realworld_eval_pi05_droid_polymetis` with Hydra overrides for the prompt + home
   pose; the env runs one episode: read frames → policy → action chunk →
   `RealWorldEnv.chunk_step` streams it over polymetis `update_joint_velocity`.
3. **SAFETY — the first rollout is autonomous** (a neural net drives the arm). Hand
   on the E-stop, arm at home, watch the first few action chunks for sane direction
   and magnitude before trusting it.
4. **Stop** ends the run. Data lands in `/tmp/rlinf_pi05_droid_eval` (LeRobot)
   inside the container.

**Gripper / action convention** matches collect: action `[7] ∈ [0,1]`, binary at
`0.5`; `action_scale 0.3`, `joint_vel_max 0.4 rad/s`, `dynamics_factor 0.3`.

---

## 4. Stop

- **Stop just the current episode** → dashboard **Stop** button (FCI + dashboard stay up).
- **Done / crash recovery (release FCI, free everything)**:
  ```bash
  sudo ~/RLinf/tasl/launch/eval-stop.sh
  ```
  Stops the eval dashboard, reaps in-container eval/PolymetisController/ray procs
  (frees ZEDs), brings the polymetis container **down**, verifies `:4242` closed →
  FCI released. Then re-lock the joints in Desk.

> To switch back to teleop/collect afterwards: just run `teleop.sh` — same
> controller, so it's effectively the same bring-up with the collect dashboard.

---

## 5. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `eval … MISSING eval code` preflight die | container lacks the pi05 inference port / polymetis eval config | Complete §7 integration and mount the unified checkout. |
| policy "doesn't move": actions non-zero but joints frozen; robot row shows `loop=STALE`; Go home reports `moved 0.000` / `ControllerNotResponding` | polymetis control loop out after a libfranka reflex (it runs `automaticErrorRecovery` retries by itself); meanwhile `GetRobotState` returns the last buffered state and DROID `robot.py` drops every tick silently — so the portal sees identical joints + a frozen state timestamp while zerorpc calls "succeed" | Live verdict = the **motion** chip (🟡 policy 静止 / 🟢 执行中 / 🟠 有指令未动 Ns / 🔴 NUC 未执行 Ns — net commanded vs actual joint displacement over 1.5 s + polymetis timestamp; hover the chip for numbers). The watchdog stops the episode when the polymetis timestamp stays frozen ≥ 3 s (the 30-step frozen-joints rule only warns while the timestamp is alive — an arm blocked at contact looks the same from the desktop) and `meta.json` gets `abort`; Mark then records `aborted`, not success/fail. Wait ~10 s: if `loop` turns `alive`, Go home + Start; if it stays STALE, **🔧 Reset NUC** (= `nuc-restart.sh`, brings the container up via compose if it is missing) → Go home. A policy that merely hovers keeps the timestamp advancing and joints jittering ≥ 3.5e-4 rad, so it never trips either rule. |
| ~1 s freezes every 20–60 s during a policy, or repeated "NUC not executing" aborts | libfranka `communication_constraints_violation` reflex: the NUC must answer each 1 kHz state packet within ≈0.62 ms and the polymetis RT thread was unpinned (often on a 1.7 GHz E-core) → ~10 % late commands → reflex. Root cause + data: `saved_demo/bug/BUGLOG.md` §1.6 | Fixed 2026-08-26: `tasl/launch/nuc-pin-rt.sh` pins the driver to P-cores and is run automatically after every bootstrap / Reset NUC (log line `NUC RT threads pinned to P-cores`). If freezes return, run it by hand, then escalate per BUGLOG §1.7 (irqbalance off + NIC IRQs → CPUs 4–5, `isolcpus`). |
| portal shows robot + gripper DOWN, Go home does nothing, state calls time out | NUC `run_server.py` (single-threaded zerorpc) stuck in a blocking call | `~/RLinf/tasl/launch/nuc-restart.sh` (restarts `droid-nuc-fr3` + re-bootstraps; FCI must be active). Since 2026-08-25 Mark/Go home stream setpoints instead of blocking moves, which is what wedged it. |
| `recover` / state calls time out | polymetis controller wedged (FCI wasn't active at bootstrap) | `eval-stop.sh` → confirm FCI active + brakes off in Desk → `eval.sh` again. |
| Jerky chunk execution | streaming rate / `action_scale` / `joint_vel_max` mismatch | tune `step_duration_s` (0.0667=15 Hz), `action_scale`, `joint_vel_max_rad_s` in the config. |
| Policy view tiles black | cameras not acquired / wrong serials | check `36443134` (exterior) + `17150101` (wrist) enumerated; reseat USB. |
| `:4242` open but eval can't bootstrap | franky robot_server somehow up (HTTP, not zerorpc) | `ssh tasl@172.16.0.2 'sudo systemctl stop franka-robot-server'`; `eval.sh` does this in Stage 1. |

---

## 5b. Getting rollout videos onto TASL2

Rollouts land on TASL1 in two places (both gitignored — never commit):

- `~/RLinf/tasl/eval_episodes/<task>/ep_<YYYYMMDD_HHMMSS>/` — every rollout, automatic:
  `video.mp4` (exterior | wrist tiled, 2560x720 @15 fps), `meta.json` (task / ckpt /
  steps / mark / note, `t0` epoch), `traj.jsonl` (one line per policy step: `t`, `iter`,
  `q[7]`, `grip`, `infer_ms`, executed `actions[n][8]`, `q_target[7]`),
  `frame_times.json` (wall-clock per video frame — aligns curves to the video)
- `~/RLinf/saved_demo/<task>/<ep_id>.{mp4,json,traj.jsonl,frames.json}` — what the
  portal's **💾 Save demo** button exports (newest rollout, after Stop / Mark);
  `saved_demo/<task>/{layouts,init_layouts}/` hold the task's layouts and every
  episode's initial frame (see `saved_demo/README_layouts.md`)

TASL1 cannot reach TASL2, so **TASL2 pulls** with rsync over ssh (TASL1's sshd
is already up on Tailscale `100.79.65.37` and campus LAN `10.12.159.30`):

```bash
# on TASL2, one-time
ssh-copy-id franka_desktop@100.79.65.37          # or paste TASL2's pubkey into TASL1 ~/.ssh/authorized_keys
scp franka_desktop@100.79.65.37:RLinf/tasl/launch/pull_demos_from_tasl1.sh ~/
# every time / cron (*/10 * * * *)
~/pull_demos_from_tasl1.sh                       # → ~/Franka_RealRobot/saved_demo/{eval_episodes,saved_demo}/
```

Incremental and resumable; never deletes on the receiving side. `--dry-run`
to preview, `DEST=…` / `TASL1_HOST=…` to override. TASL2 needs to be on the
same tailnet (or on campus) — `tailscale ping 100.79.65.37` must answer.

## 6. Quick reference

- **Start / stop:** `sudo ~/RLinf/tasl/launch/eval.sh` · `sudo ~/RLinf/tasl/launch/eval-stop.sh`
- **Dashboard:** `:8003` (rlinf.py, host mode) — *same port as the openpi dashboard; one at a time*
- **Backend:** polymetis `droid-nuc-fr3` container, **zerorpc** `tcp://172.16.0.2:4242` (backend **B**, same as teleop)
- **Config:** `realworld_eval_pi05_droid_polymetis` (env `realworld_franka_jointvel_polymetis`, model `pi0_5`/openpi `pi05_droid`)
- **Checkpoint:** `/ckpts/pi05_droid_pt` (RLinf-format, already mounted)
- **Data:** `/tmp/rlinf_pi05_droid_eval` (LeRobot) inside `rlinf-eval`
- **Logs:** `~/RLinf/tasl/logs/eval.log` (dashboard) · `_dashboard_eval.log` in-container (eval)
- **Home pose (rad):** `[0, -0.6283, 0, -2.5133, 0, 1.8850, 0]`

---

## 7. Integration status

**✅ DONE — framework inference port (validated in `rlinf-eval`).** On branch
`franka-fr3/eval-pi05-polymetis` in `~/work/rlinf-clone` (commit `feat(eval):
pi05_droid inference over the polymetis controller`):
- `droid_dataconfig.LeRobotDROIDDataConfig` + `pi05_droid` TrainConfig registered
  (verified: `"pi05_droid" in _CONFIGS_DICT`).
- `openpi_action_model.obs_processor` `pi05_droid` branch (emits DroidInputs keys).
- `realworld_eval_pi05_droid_polymetis.yaml` — composes + resolves
  (`controller_type: polymetis`, `robot_ip: 172.16.0.2`, `config_name: pi05_droid`,
  `num_action_chunks: 8`, `use_gello: False`).
- `rlinf.py` eval spawn → `config_name = realworld_eval_pi05_droid_polymetis`.

> This inference code is now merged into `~/RLinf`, which the `rlinf-eval`
> container mounts (consolidated 2026-07-03). No separate checkout/branch step:
> the launchers build the container against `~/RLinf` via `ensure_rlinf_container`.

**☐ TODO — dashboard robot panel zerorpc swap (best done + validated at the bench).**
`rlinf.py` still drives its robot panel over HTTP `RS` (franky). The eval *run*
doesn't need it (the in-container env bootstraps + homes on Start), but the panel
buttons + `eval.sh` Stage 4 do. Port the proven per-request `DroidClient` from
`collect.py` (gevent/zerorpc thread-affinity → fresh client per request, positional
args only) and rewire:
- `RS.state()` / status poll → `DroidClient().get_robot_state()`.
- endpoints → `/api/robot/{state,home,recover,gripper/<a>}` (match `collect.py` +
  `eval.sh`); update the dashboard JS that calls `/home` `/recover`.
- `recover` → `DroidClient(timeout=90)` kill+bootstrap+state-recheck (copy
  `collect.py:api_robot_recover`); `home` → leashed `move_to_joint_target`.
- `EvalRunner.stop()` HTTP `/move/joint_velocity_stop` + `/stop` → kill the eval
  proc (already does) + optional zerorpc zero-velocity; drop the HTTP calls.

**☐ TODO — bench validation:** mount the eval branch → `eval.sh` → Start → watch
the first chunks (E-stop in hand). Then `eval-stop.sh`.

**☐ Optional later — `rtc_guidance.py`** for real-time chunking (deferred; baseline first).
