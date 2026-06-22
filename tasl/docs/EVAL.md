# TASL FR3 — pi05_droid Policy Eval Manual (RLinf runtime)

How to run a **pi05_droid policy eval** on the FR3 bench through the **RLinf
runtime** (`eval_embodied_agent.py`), end to end. For GELLO teleop/collection see
[TELEOP.md](TELEOP.md); for the component map see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 0. The two pi05 deployment paths (and why this one is different)

The same pi05_droid VLA can drive the arm through **two different runtimes**, and
they even use **different NUC controllers**:

| | **openpi standalone** (`openpi.py` :8003) | **RLinf eval** (`rlinf.py` :8003) — *this doc* |
|---|---|---|
| Policy | openpi `serve_policy` WS :8000 + in-process loop | model loaded inside RLinf, under **Ray** |
| Runtime | thin client on the Desktop host | `eval_embodied_agent.py` in `rlinf-eval` container |
| Checkpoint | openpi-format `~/.cache/openpi/.../pi05_droid` | RLinf-format `/ckpts/pi05_droid_pt` |
| Rollout owner | the dashboard loop | the RLinf **`FrankaJointVelEnv`** |
| **NUC controller** | DroidLikeClient **zerorpc** (backend B) | franky **`robot_server` HTTP** (backend A) |
| Data written | none (pure inference) | LeRobot dataset (`/tmp/rlinf_pi05_droid_eval`) |
| Use it to… | quickly sanity-check the checkpoint | eval the policy in the exact env training/RL use |

**Key consequence:** RLinf eval is the *mirror-opposite* of teleop/collect.
Teleop drives backend **B** (polymetis zerorpc); eval drives backend **A** (franky
`robot_server` HTTP). Both bind NUC1 `:4242` and are mutually exclusive, so
`eval.sh` brings backend A up and takes backend B down. `teleop.sh` does the reverse.

---

## 1. One-time prerequisite — mount the eval checkout

The RLinf eval needs code + config that live **only on the eval branch**
(`franka-fr3/rlinf-pi05-droid-eval`): `realworld_eval_pi05_droid.yaml`,
`env/realworld_franka_jointvel.yaml`, `rlinf/envs/realworld/realworld_chunk_env.py`
(the `realworld_chunk_native` wrapper), and the openpi `droid_dataconfig.py` that
registers the `pi05_droid` TrainConfig. The default mounted checkout
(`~/work/rlinf-clone`, on the **polymetis collect** branch) does **not** have these
— and that branch *deletes* `polymetis_controller.py`, so one checkout can't serve
both teleop and eval.

A full eval-branch checkout already exists at **`~/work/rlinf-clone-rtc-old`**
(branch `franka-fr3/rtc-pi05-droid-deploy`). Point the `rlinf-eval` container's
`/workspace/rlinf` at the eval checkout before running eval. Either:

- **Recreate the container** with the eval checkout mounted:
  `… -v ~/work/rlinf-clone-rtc-old:/workspace/rlinf …`, **or**
- **Check out the eval branch** inside `~/work/rlinf-clone`
  (`git checkout franka-fr3/rlinf-pi05-droid-eval`) — but that takes teleop/collect
  offline until you switch back.

`eval.sh` **preflights this** and refuses to launch (with the missing-file list) if
`/workspace/rlinf` doesn't have the eval code — so a wrong mount fails loudly, not
mid-rollout.

Also fix the controller IP in the staged config: set
`env.{train,eval}.override_cfg.robot_server_url: http://172.16.0.2:4242` (the branch
ships the stale Tailscale `100.75.6.62`). `eval.sh` passes `--robot-server` to the
dashboard, but the env reads `robot_server_url` from the YAML, so both must point at
the robot net.

The checkpoint `/ckpts/pi05_droid_pt` is already mounted (verified present).

---

## 2. Start an eval session

```bash
sudo ~/RLinf/tasl/launch/eval.sh
```

Four stages (mirror of teleop.sh, opposite backend):

1. **NUC1 backend (A)** — ssh `tasl@172.16.0.2`: `docker compose down` the
   polymetis container (release `:4242`), then `systemctl start franka-robot-server`
   (the unit we keep *disabled at boot*). Waits until `:4242/state` answers HTTP.
   *⚠ The franky `robot_server` was set up but never verified end-to-end — if Stage 1
   times out, debug it on NUC1 (`journalctl -u franka-robot-server`).*
2. **FCI gate (manual)** — open **Franka Desk** `https://172.16.0.1`, unlock joints,
   **Activate FCI** + execution mode, then press Enter.
3. **Eval dashboard `:8003`** — frees the ZEDs, launches `rlinf.py --mode host`.
   The dashboard **owns the cameras permanently** and serves frames over HTTP
   (`/cam/<name>.jpg`) to the in-container env — no camera handoff.
4. **Recover + auto-home** — recovers the franky controller and homes to the DROID
   anchor pose via the dashboard (proxied to `robot_server`).

---

## 3. Run eval episodes

1. Open `http://<printed IP>:8003` (or `http://localhost:8003` on the Desktop).
   Confirm live ZED previews + the **policy view** tiles (224×224 center-crop — the
   exact frames the model sees) + green robot status.
2. **Type the task prompt** (e.g. "pick up the cup") and press **Start**.
   The dashboard `docker exec`s `eval_embodied_agent.py --config-name
   realworld_eval_pi05_droid` with Hydra overrides for the prompt + home pose; the
   `FrankaJointVelEnv` then runs one episode: read frames → policy → action chunk →
   `robot_server /move/joint_velocity_chunk`.
3. **SAFETY — the first rollout is autonomous** (a neural net drives the arm). Hand
   on the E-stop, arm at home, watch the first few action chunks for sane direction
   and magnitude before trusting it. `eval_rollout_epoch: 1` → one episode per Start.
4. **Stop** ends the run (halts ruckig motion first, then kills the eval).
   Data lands in `/tmp/rlinf_pi05_droid_eval` (LeRobot) inside the container.

**Gripper / action convention** matches collect: action `[7] ∈ [0,1]`, binary at
`0.5`; `action_scale 0.3`, `joint_vel_max 0.4 rad/s`, `dynamics_factor 0.3`.

---

## 4. Stop

- **Stop just the current episode** → dashboard **Stop** button (FCI + dashboard stay up).
- **Done / crash recovery (release FCI, free everything)**:
  ```bash
  sudo ~/RLinf/tasl/launch/eval-stop.sh
  ```
  Halts motion, stops the eval dashboard, reaps in-container eval/ray procs (frees
  ZEDs), **stops `franka-robot-server`** on NUC1, and verifies `:4242` closed → FCI
  released. Then re-lock the joints in Desk.

> To switch back to teleop/collect afterwards: just run `teleop.sh` — it stops the
> franky server and brings the polymetis container back up.

---

## 5. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Stage 1: `robot_server never answered :4242/state` | franky `robot_server` not up / not working on NUC1 | `ssh tasl@172.16.0.2 'systemctl status franka-robot-server; journalctl -u franka-robot-server -n 50'`. This backend is unverified — may need fixing. |
| `eval … MISSING eval code` preflight die | container mounts the polymetis checkout, not the eval branch | Point `/workspace/rlinf` at `~/work/rlinf-clone-rtc-old` (§1). |
| Eval starts but env can't reach the robot | `robot_server_url` still Tailscale `100.75.6.62` | Set it to `http://172.16.0.2:4242` in the staged config. |
| `:4242` open but it's zerorpc, not HTTP | polymetis container still up (backend B) | `eval.sh` brings it down; if it lingers: `ssh tasl@172.16.0.2 'cd ~/polymetis_fr3 && docker compose down'`. |
| Policy view tiles black | cameras not acquired / wrong serials | check `36443134` (exterior) + `17150101` (wrist) enumerated; reseat USB. |

---

## 6. Quick reference

- **Start / stop:** `sudo ~/RLinf/tasl/launch/eval.sh` · `sudo ~/RLinf/tasl/launch/eval-stop.sh`
- **Dashboard:** `:8003` (rlinf.py, host mode) — *same port as the openpi dashboard; one at a time*
- **Backend:** franky `robot_server` HTTP `http://172.16.0.2:4242` (backend **A**)
- **Config:** `realworld_eval_pi05_droid` (env `realworld_franka_jointvel`, model `pi0_5`/openpi `pi05_droid`)
- **Checkpoint:** `/ckpts/pi05_droid_pt` (RLinf-format, already mounted)
- **Eval checkout:** `~/work/rlinf-clone-rtc-old` (branch `franka-fr3/rtc-pi05-droid-deploy`)
- **Data:** `/tmp/rlinf_pi05_droid_eval` (LeRobot) inside `rlinf-eval`
- **Logs:** `~/RLinf/tasl/logs/eval.log` (dashboard) · `_dashboard_eval.log` in-container (eval)
- **Home pose (rad):** `[0, -0.6283, 0, -2.5133, 0, 1.8850, 0]`
