# TASL FR3 — System Setup Guide (NUC1 + Desktop)

How the TASL Franka FR3 bench is wired and configured end to end, so the whole
system can be rebuilt or onboarded. Two machines + the robot:

- **Desktop (TASL-1)** — GPU workstation: cameras, GELLO, dashboards, policy.
- **NUC1** — real-time controller: the only libfranka/FCI client.
- **FR3** — the arm + Franka Desk/FCI.

For day-to-day operation see [TELEOP.md](TELEOP.md); for the component model see
[ARCHITECTURE.md](ARCHITECTURE.md). The canonical install plan (with the full
DROID image build) lives in the Notion "Franka Install" doc — this guide is the
repo-local summary plus the parts verified on the live machines.

> **Scope note.** Both sections are verified against the live machines
> (Desktop and NUC1 inspected 2026-06-22).

---

## 1. Network & wiring

Everything robot-side is on a private **robot network `172.16.0.0/24`**:

| Host | Robot-net IP | Interface / link |
|---|---|---|
| FR3 Control | `172.16.0.1` | Cat6 from the FR3 **C2 (Shop-Floor) port** |
| NUC1 | `172.16.0.2` | direct/switched to the FR3 + Desktop |
| Desktop TASL-1 | `172.16.0.3` | NIC `enp3s0`, static `172.16.0.3/24` |

- The controller path (zerorpc `:4242`) and Desk all ride this network. **Tailscale
  is legacy/often-offline** (Desktop `100.79.65.37`, NUC1 `100.75.6.62`) — don't
  rely on it. The launchers default to the robot net; override with `NUC1_HOST=…`.
- FCI must be on the **C2 port** (port 1337 is filtered on the X5 port).
- Desktop `enp3s0` static config (NetworkManager or netplan): address
  `172.16.0.3/24`, no gateway needed (isolated subnet).

Verify from the Desktop:
```bash
ip -4 addr show enp3s0 | grep 172.16.0.3
ping -c1 172.16.0.1 && ping -c1 172.16.0.2     # FR3 + NUC1 reachable
```

---

## 2. FR3 / Franka Desk (one-time)

1. Power the FR3; connect its **C2 port** to the robot network.
2. Set the Control's IP to `172.16.0.1` (or match your subnet) in Desk → Settings → Network.
3. Browse to **`https://172.16.0.1`**, create the Desk admin account, accept the cert.
4. Register the **end effector** (Franka Hand) — payload/inertia, homing test.
5. Confirm firmware **5.9.2** (libfranka 0.18.x / FCI protocol 10).
6. FCI is enabled per-session in Desk (sidebar **Activate FCI**) — see TELEOP.md.

---

## 3. NUC1 setup (controller)  ✅ verified 2026-06-22

> **Full controller build + config guide:** [CONTROLLER.md](CONTROLLER.md) — how
> `droid-nuc-fr3:0.18.1` is built (libfranka 0.13.5 → 0.18.1), what the compose
> mounts, the control contract, and which files in `~/polymetis_fr3/` are live vs
> deprecated. This section is the bench-level summary.

**Base:** Ubuntu **22.04.5** + **RT kernel `5.15.0-1105-realtime`**, user `tasl`,
Docker installed, `tasl` in the `docker` group. Cat6 to the FR3.

**The controller lives in `~/polymetis_fr3/`:**
```
~/polymetis_fr3/
├── Dockerfile.droid-0181        # builds the image (libfranka 0.18.1 patch)
├── docker-compose.yaml          # starts the droid-nuc-fr3 container
├── parameters.py                # mounted in — overrides DROID's default IPs
└── conf/
    ├── franka_hardware.yaml      # FR3 robot_ip
    └── franka_panda.yaml         # joint limits
```

**Docker image `droid-nuc-fr3:0.18.1`** (~18 GB): based on
`ghcr.io/droid-dataset/droid_nuc:fr3`, with **libfranka 0.13.5 → 0.18.1** (FR3 fw
5.9.2 needs 0.18.x) and the build patches (Pinocchio 3→2 API, C++17,
`BUILD_FRANKA=ON`, conda libstdc++). Full build steps are in the Notion doc — the
image is prebuilt; you only rebuild if libfranka/firmware changes.

**Container `droid-nuc-fr3`:** host-network, privileged, rtprio 99. Runs
`run_server.py` → `zerorpc.Server(FrankaRobot()).bind(:4242)`. Processes live in
the **`polymetis-local` conda env** (`source /root/miniconda3/etc/profile.d/conda.sh
&& conda activate polymetis-local` if you exec in).

**Critical config points (gotchas):**
- **IPs are reversed from DROID defaults.** Ours: `nuc_ip=172.16.0.2,
  robot_ip=172.16.0.1` (DROID ships the opposite). Set in `parameters.py` +
  `conf/franka_hardware.yaml`; re-check after any image rebuild.
- **Lazy init:** the container starts but the polymetis driver does **not** spawn
  until a client calls `launch_controller()` + `launch_robot()` (the dashboard's
  recover/bootstrap does this; ~10 s cold).
- **zerorpc drops kwargs** — always pass `update_command` args positionally
  (`_droid_client.py` / the env client already wrap this).
- The **gripper** is driven through this container over zerorpc. **Verified: it
  runs a Robotiq 2F** — `launch_gripper.py gripper=robotiq_2f
  gripper.comport=/dev/ttyUSB0` (FTDI USB-RS485 bridge `0403:6015` on `ttyUSB0`,
  driver actively running). ⚠ This contradicts the "Franka Hand" assumption
  elsewhere — confirm which end-effector is physically mounted; the data's
  gripper channel comes from this Robotiq driver.

**FCI exclusivity — the #1 footgun:** only one libfranka client at a time. The
franky **`franka-robot-server` systemd unit must stay STOPPED** whenever the
droid container holds FCI. Bring-up:
```bash
sudo systemctl stop franka-robot-server
cd ~/polymetis_fr3 && docker compose up -d
ss -tlnp | grep ':4242'          # zerorpc listening
```
(`teleop.sh` on the Desktop does this for you over ssh.)

**NUC1 config notes (2026-06-22):**
- ✅ **FIXED** — `franka-robot-server` was `enabled` (boot FCI-conflict risk);
  now `disabled` (`systemctl disable`). Service stays `inactive`; re-enable only
  if you switch the bench back to the franky stack.
- ✅ **FIXED** — `parameters.py laptop_ip` + compose `LAPTOP_IP` corrected from the
  stale `100.66.31.78` to `100.79.65.37` (`.bak-ipfix` backups kept on NUC1).
  Note: compose env changes apply on the next `docker compose up -d` recreate.
- ⚠ **Remaining** — compose env `LIBFRANKA_VERSION: "0.13.5"` is misleading; the
  image actually ships **0.18.1** (the var is unused — version baked into the
  image). Left as-is to avoid implying a rebuild.

---

## 4. Desktop (TASL-1) setup  ✅ verified

**Base (current):** Ubuntu **22.04.5**, kernel **6.8** (no RT needed here), Docker
**29.2.1**, NIC `enp3s0` on the robot net.

**GPU:** RTX 4090, NVIDIA driver **570.211.01**, CUDA 12.x.

**ZED cameras:**
- ZED SDK **5.3.0** at `/usr/local/zed`, `pyzed` 5.3.0 in system `python3`.
- udev `/etc/udev/rules.d/99-slabs.rules` — grants non-root access (vendor `2b03`,
  `MODE=0666` for usb + hidraw). **Without it the SDK reports "No Camera detected."**
- 2× ZED 2i (SN `36443134`, `34825630`) + ZED Mini wrist (`17150101`) on USB3.

**GELLO leader:**
- udev `/etc/udev/rules.d/99-gello.rules` maps the OpenRB-150 (vendor `2f5d`,
  product `2202`, fixed serial) → stable **`/dev/gello`** (→ `ttyACM*`).
- Verify: `ls -l /dev/gello`.

**uinput** (for the dashboard's virtual keyboard / success key): `/dev/uinput`
present; the dashboard runs under sudo to access it.

**Apply udev changes:** `sudo udevadm control --reload && sudo udevadm trigger`.

**Python deps (system `python3` + user site):** `zerorpc`, `flask`, `numpy`,
`pyzed` (used by the dashboards). `SITE_PKGS=~/.local/lib/python3.10/site-packages`.
`uv` at `~/.local/bin/uv`.

**`rlinf-eval` container** (hosts the RL env / collection):
- Image `rlinf/rlinf:agentic-rlinf0.2-pi05droid-zed`, **host network, privileged**.
- Mounts (**consolidated 2026-07-03**): **`~/RLinf → /workspace/rlinf`** (code),
  **`~/rlinf_data/datasets` + `~/rlinf_data/outputs`** (data — kept *outside* the
  code checkout so swapping the mounted code never touches collected data),
  `/usr/local/cuda`, `/usr/local/zed`, `ckpts/pi05_droid_pt → /ckpts/pi05_droid_pt`,
  and **`/dev → /dev`** (so it sees `/dev/gello`, `/dev/video*`, `/dev/uinput`).
- **Created/started by `ensure_rlinf_container` in `launch/lib.sh`** — the canonical
  `docker run` recipe (mounts are version-controlled, not a hand-run command).
  Mounts are fixed at creation; to repoint them: `docker rm -f rlinf-eval`, then
  re-run any launcher. Override paths with `RLINF_REPO_HOST` / `RLINF_DATA_DIR` /
  `RLINF_CONTAINER` / `RLINF_IMAGE` (defaults in `lib.sh`).
- **In-container GELLO deps** (`zerorpc`, `dynamixel-sdk`, `gello`, `gello_teleop`)
  are NOT in the base image — `ensure_container_deps` (also `lib.sh`) reinstalls
  them from source (PyPI + `wuphilipp/gello_software` + `RLinf/gello-teleop`) when
  missing, so a container rebuild self-heals. The GELLO leader **calibration** lives
  in the repo at `rlinf/envs/realworld/common/gello/fr3_gello_config.py` (injected
  into gello's `PORT_CONFIG_MAP` at runtime) — see [TELEOP.md](TELEOP.md#gello-calibration).
- In-container Python: `/opt/venv/openpi/bin/python`. The collection run is
  `examples/embodiment/collect_real_data.py` from the mounted `~/RLinf` checkout.

---

## 5. Repos & layout (Desktop)

```
~/RLinf/                      # THE fork checkout (github.com/tasl-lab/RLinf);
│   │                         #   ALSO mounted into rlinf-eval at /workspace/rlinf
│   ├── tasl/                 # ← OUR bench tooling (dashboards, launch, docs)
│   └── rlinf/ examples/ …    #   the RLinf framework the container runs
~/rlinf_data/                 # collected data, OUTSIDE the repo (survives code swaps)
│   ├── datasets/             #   → /workspace/rlinf/datasets  (LeRobot + raw SVO)
│   └── outputs/              #   → /workspace/rlinf/outputs   (lerobot, live_cam, logs)
~/ckpts/pi05_droid_pt/        # policy checkpoint (mounted into rlinf-eval)
~/work/                       # legacy support checkouts:
    ├── openpi/               #   VLA policy (serve_policy + openpi-client)
    ├── lerobot/              #   dataset format
    └── rlinf-clone/          #   DEPRECATED — old container mount, no longer used
```

Since **2026-07-03 the container mounts `~/RLinf` directly** (one checkout for both
host tooling and the in-container framework); `~/work/rlinf-clone` is retired. Clone:
```bash
git clone https://github.com/tasl-lab/RLinf ~/RLinf
cd ~/RLinf   # bench tooling lives under tasl/ ; launchers build the container
```

---

## 6. Verify the full stack

1. **Network:** `ping 172.16.0.1 && ping 172.16.0.2` from the Desktop.
2. **Devices:** `ls /dev/gello`; `docker exec rlinf-eval /opt/venv/openpi/bin/python -c
   "import pyzed.sl as sl; print([d.serial_number for d in sl.Camera.get_device_list()])"`
   shows the expected ZED serials.
3. **NUC controller:** start it (§3), then from the Desktop
   `bash -c 'cat </dev/null >/dev/tcp/172.16.0.2/4242' && echo ":4242 open"`.
4. **Robot link smoke (moves the arm):**
   `docker exec rlinf-eval bash -lc "PYTHONPATH=/workspace/rlinf /opt/venv/openpi/bin/python
   /workspace/rlinf/examples/embodiment/scripts/p1_polymetis_smoke.py"`.
5. **End to end:** run `sudo ~/RLinf/tasl/launch/teleop.sh` and follow
   [TELEOP.md](TELEOP.md).

---

## 7. Reference — versions & key facts

| Item | Value |
|---|---|
| FR3 firmware | 5.9.2 (FCI protocol 10) |
| libfranka (NUC container) | 0.18.1 |
| NUC image | `droid-nuc-fr3:0.18.1` (base `ghcr.io/droid-dataset/droid_nuc:fr3`) |
| NUC OS / kernel | Ubuntu 22.04.5 / `5.15.0-1105-realtime` |
| Gripper | **Robotiq 2F** via `/dev/ttyUSB0` (FTDI `0403:6015`), `gripper=robotiq_2f` |
| Desktop OS / kernel | Ubuntu 22.04.5 / 6.8 |
| NVIDIA driver | 570.211.01 (RTX 4090) |
| ZED SDK / pyzed | 5.3.0 |
| Docker | 29.2.1 |
| rlinf-eval image | `rlinf/rlinf:agentic-rlinf0.2-pi05droid-zed` |
| Robot net | FR3 `.1`, NUC1 `.2`, Desktop `.3` on `172.16.0.0/24` |
| Controller | zerorpc `tcp://172.16.0.2:4242` |
| GELLO | `/dev/gello` (vendor 2f5d / product 2202) |
| ZED serials | 2i `36443134`, `34825630`; Mini wrist `17150101` |
| Home pose (rad) | `[0, -0.6283, 0, -2.5133, 0, 1.8850, 0]` |
| Data | `~/rlinf_data/{datasets,outputs}/` (outputs/lerobot for LeRobot) |
| GELLO calibration | `rlinf/envs/realworld/common/gello/fr3_gello_config.py` (in-repo) |
