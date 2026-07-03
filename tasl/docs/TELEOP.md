# TASL FR3 — Teleoperation Manual (GELLO data collection)

How to run a GELLO teleop data-collection session on the FR3 bench, end to end,
plus the failure modes we've actually hit and how to clear them. For the
component map see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 0. TL;DR

```bash
# START everything (Desktop):
sudo ~/RLinf/tasl/launch/teleop.sh        # → activate FCI when prompted → auto-homes

# then in the dashboard (laptop or Desktop browser):
#   http://<address shown in concole>:8004   (Tailscale)   — move GELLO, press Start, mark Success, Stop

# DONE for the session (release FCI, free everything):
sudo ~/RLinf/tasl/launch/teleop-stop.sh
```

| | command |
|---|---|
| Start session | `sudo ~/RLinf/tasl/launch/teleop.sh` |
| Full teardown / crash recovery | `sudo ~/RLinf/tasl/launch/teleop-stop.sh` |
| Stop just the current run | dashboard **Stop** button (stays idle, FCI still held) |

---

## 1. The setup (who's who)

| Machine | IP (robot net) | Role |
|---|---|---|
| Desktop **TASL-1** | `172.16.0.3` | runs `teleop.sh`, the dashboard, the ZEDs + GELLO |
| **NUC1** | `172.16.0.2` | polymetis controller container (`droid-nuc-fr3`), zerorpc `:4242`, holds FCI |
| **FR3 Control** | `172.16.0.1` | Franka Desk / FCI |

- Controller link is the **robot network**, not Tailscale (Tailscale is the legacy
  path and is often offline). Override with `NUC1_HOST=…` if the NUC IP changes.
- GELLO leader is the udev symlink **`/dev/gello`** (→ `ttyACM0`).
- Cameras: 2× ZED 2i + ZED Mini wrist on the Desktop's USB3.

---

## 2. Before you start (prerequisites)

1. **FR3 powered on**, e-stop within reach.
2. **GELLO plugged in** — confirm: `ls -l /dev/gello` (should resolve to `ttyACM*`).
3. **Desktop on the robot net** — `ping -c1 172.16.0.1` and `172.16.0.2` both reply.
4. ZED cameras connected (the launcher checks they're free).

You do **not** pre-start the NUC controller or `rlinf-eval` — `teleop.sh` does that.

---

## 3. Start a session

```bash
sudo ~/RLinf/tasl/launch/teleop.sh
```

It runs four stages:

1. **NUC1 backend** — ssh's to `tasl@172.16.0.2`, stops the franky `robot_server`
   (FCI exclusivity), and `docker compose up`s the polymetis container.
   *You may be prompted for the NUC ssh/sudo password — enter it.*
2. **FCI gate (manual)** — it prints instructions and **waits**. Open **Franka Desk**:
   - Browse to **`https://172.16.0.1`** (reachable from the Desktop; accept the cert).
   - **Unlock the joints** (brakes click open).
   - **Activate FCI** + put the robot in **execution mode**.
   - Make sure there's **no error/reflex light**.
   - Back in the terminal, **press Enter**.
3. **Dashboard** — frees the ZEDs and launches the collect dashboard on `:8004`.
   It prints the URL:
   `✓ dashboard up — http://100.79.65.37:8004`
4. **Bootstrap + auto-home** — launches the controller against the live FCI and
   homes the arm to the anchor pose. Ends with `✓ READY …`.

> A cold bootstrap can take 30–90 s; the launcher waits. If it still prints
> `recover failed`, see Troubleshooting — the controller is often actually up.

---

## 4. Open the dashboard

- **From the laptop (Tailscale):** `http://100.79.65.37:8004`
  (laptop must be on the same tailnet). The exact IP is printed by the launcher
  and derived live, so trust the printed one if it differs.
- **From the Desktop:** `http://localhost:8004`.

Idle state shows live ZED previews + green robot status.

---

## 5. Record episodes

1. **Test tracking first:** gently move the GELLO leader and confirm the FR3
   follows (correct direction, smooth). GELLO is read synchronously, so expect
   slight loop-rate lag — that's normal, not a fault.
2. Press **Start** → the dashboard releases the cameras and launches the teleop
   collection run (`collect_real_data.py`, joint-velocity GELLO, over the robot net).
3. Teleoperate the task with the GELLO leader.
4. **Mark success** with the dashboard's success button (injects `c` to the
   in-container key listener). Only successful episodes are saved.
5. Repeat for more episodes, or press **Stop**.

**Gripper convention:** state `gripper_position ∈ [0,1]`, **1 = closed**;
action `[7] ∈ [0,1]`, **0 = open / 1 = close**.

**Data lands in:** `~/rlinf_data/outputs/lerobot/` (LeRobot v2.1), plus a
per-run `~/rlinf_data/outputs/collect_<timestamp>/`, and datasets under
`~/rlinf_data/datasets/`. Verify: `ls -lt ~/rlinf_data/outputs/lerobot/`.
(Data lives outside the code checkout; the container sees it at
`/workspace/rlinf/{datasets,outputs}` — see [SETUP.md](SETUP.md).)

---

## 6. Stop

Two different things:

- **Stop the current run, keep collecting later** → dashboard **Stop** button.
  Kills the run, cleans up Ray, reclaims the cameras, returns to **idle**. The
  dashboard, NUC controller, and **FCI stay up**. Saved episodes are preserved.
- **Done for the session (release FCI, free everything)** →
  ```bash
  sudo ~/RLinf/tasl/launch/teleop-stop.sh
  ```
  Stops the dashboard, reaps in-container procs (frees ZEDs + GELLO), brings the
  NUC controller down, and **verifies `:4242` closed → FCI released**. Then
  **re-lock the joints in Desk**.

---

## 7. Troubleshooting (things we've actually hit)

| Symptom | Cause | Fix |
|---|---|---|
| `recover failed — controller could not bootstrap` but the arm is actually live | Cold `launch_controller` took longer than the client timeout — false negative | Check state is good (arm homed / status green). Just **Home** from the UI and continue; no need to re-run. (Fixed in newer dashboard: 90 s timeout + state re-check.) |
| `Address already in use — Port 8004` then recover times out | A **stale older dashboard** still holds `:8004` (and points at the dead Tailscale link) | `sudo pkill -9 -f '[/_]collect[.]py'`; confirm `ss -ltn \| grep :8004` is free; re-run `teleop.sh`. Hardened launcher now reaps both old/new names. |
| `recover` / state calls time out at 30 s repeatedly | Polymetis controller **wedged** on the NUC (FCI wasn't active when bootstrap ran) | `teleop-stop.sh` (compose-down clears it) → confirm FCI **active + brakes off** in Desk → `teleop.sh` again. |
| `camera open timeout` / "refused" on Start | A crashed run left `collect_real_data`/`PolymetisController` holding the ZEDs+GELLO | Re-run `teleop.sh` (auto-reaps), or `teleop-stop.sh` then start again. |
| `teleop-stop` prints `controller :4242 STILL OPEN — FCI NOT released` | NUC `docker compose down` didn't complete (ssh/password) | `ssh tasl@172.16.0.2 'cd ~/polymetis_fr3 && docker compose down'`; verify `docker ps \| grep droid` is empty. |
| Huge `ray::… "" "" …` zombie dump on Stop | Ray reporting already-dead (zombie) procs it can't re-kill | Harmless — ignore. They clear on the next `rlinf-eval` restart. |
| NUC1 unreachable in Stage 1 | Robot net down / NUC off | `ping 172.16.0.2`; check the cable/switch and that NUC1 is powered. |
| `/dev/gello` missing | GELLO unplugged or udev rule not applied | Replug; check `ls /dev/serial/by-id/`. |
| `AssertionError: Port /dev/gello not in config map` | GELLO leader calibration not registered (e.g. after a container rebuild that dropped the deps) | The calibration is in-repo (`fr3_gello_config.py`) and injected by `gello_joint_intervention`; ensure the GELLO deps are installed (`ensure_container_deps` / re-run `teleop.sh`). See [GELLO calibration](#gello-calibration). |
| One joint lurches ~90° on the first teleop tick | That joint's GELLO offset is wrong by π/2 | Flip that joint's entry in `FR3_GELLO_OFFSETS_PI_2` (`fr3_gello_config.py`) and re-verify. See [GELLO calibration](#gello-calibration). |

**FCI exclusivity (root cause of most controller hangs):** only one FCI client at
a time. The franky `robot_server` (systemd `franka-robot-server`) must be
**stopped** while the polymetis container holds FCI — `teleop.sh` stops it in
Stage 1, but if that ssh step fails, bootstrap will hang. Verify on NUC1:
`systemctl is-active franka-robot-server` → should be `inactive`.

---

## 8. GELLO calibration

The GELLO leader (OpenRB-150 on `/dev/gello`) needs a per-leader
`DynamixelRobotConfig` — joint IDs, **offsets** (multiples of π/2), **signs**, and
**gripper** open/close. `gello_software` ships only stock FTDI configs, so ours is
kept **in the repo** and injected into gello's `PORT_CONFIG_MAP` at runtime:

- **Config:** `rlinf/envs/realworld/common/gello/fr3_gello_config.py`
  (`FR3_GELLO_OFFSETS_PI_2`, `FR3_GELLO_JOINT_SIGNS`, `FR3_GELLO_GRIPPER_CONFIG`).
- **Injected by:** `GelloJointIntervention.__init__` → `register_fr3_gello_config()`
  before the agent opens the port. Keeping it in-repo (not hand-edited into the
  ephemeral `gello_software` install) is why it survives a container rebuild.

**Re-calibrate** when the leader is re-assembled, or if the config is ever lost.
Offsets snap to π/2, so you only need the leader within ±45° of the start pose.

1. **Raw offsets** — hold the leader at `START = [0,0,0,-1.571,0,1.571,0]`, gripper
   fully **open**, and run:
   ```bash
   docker exec rlinf-eval /opt/venv/openpi/bin/python \
     /opt/gello_software/scripts/gello_get_offset.py \
     --port /dev/gello --start-joints 0 0 0 -1.571 0 1.571 0 \
     --joint-signs 1 -1 1 -1 1 1 1 --gripper
   ```
   Read `best offsets function of pi` (the `N*np.pi/2` list) and the gripper
   open/close degrees. Base joints are hard to hold steady — expect a couple of
   joints to be ambiguous across runs.
2. **Validate per joint** — put the candidate offsets into `fr3_gello_config.py`,
   then confirm each joint independently against the homed **robot** (a rigid
   reference, far better than holding in air): with the arm at the home pose, hold
   the leader to match and check that each `q_gello − q_robot` is small. Any joint
   stuck near ±π/2 → its offset is wrong by one step; adjust that entry.
3. Current values were re-derived **2026-07-03**: offsets `[2,0,0,2,2,1,2]`, signs
   `[1,-1,1,-1,1,1,1]`, gripper `(8,264,222)`.

> Note: the calibration is **not** in the base image or in `gello_software` — only
> in `fr3_gello_config.py`. Do not hand-edit `PORT_CONFIG_MAP` in the container; it
> lives in the ephemeral layer and is lost on rebuild.

---

## 9. Quick reference

- **Dashboard:** `http://100.79.65.37:8004` (Tailscale) / `http://localhost:8004`
- **DROID home pose (rad):** `[0, -0.6283, 0, -2.5133, 0, 1.8850, 0]`
- **Controller:** zerorpc `tcp://172.16.0.2:4242` (robot net)
- **Config:** `realworld_collect_data_polymetis_jointvel` + `env.eval.gello_port=/dev/gello`
- **Logs:** `~/RLinf/tasl/logs/collect.log`
- **Data:** `~/rlinf_data/outputs/lerobot/` (datasets: `~/rlinf_data/datasets/`)
- **GELLO calibration:** `rlinf/envs/realworld/common/gello/fr3_gello_config.py`
- **Ports:** collect `:8004`, openpi `:8003`, serve_policy `:8000`, zed_viewer `:8002`
