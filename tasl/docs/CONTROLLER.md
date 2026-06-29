# FR3 polymetis controller — setup guide (NUC)

How to **stand up the `droid-nuc-fr3` polymetis controller on a NUC** — the
libfranka FCI client that owns the FR3 and exposes the arm + gripper over zerorpc
`:4242`. This is the "backend B / current" controller in
[ARCHITECTURE.md](ARCHITECTURE.md); the franky `robot_server` (backend A) is the
mutually-exclusive alternative.

§1 is the from-scratch install. §2+ is reference: how the image is built, what the
container/config do, and the control contract. Repo-managed config lives in
[`../controller/`](../controller/); the running tree is `~/polymetis_fr3/` on the
NUC. For the bench as a whole see [SETUP.md](SETUP.md); for day-to-day bring-up see
[TELEOP.md](TELEOP.md).

> **Verified** against the live `~/polymetis_fr3/` tree, the running NUC1 image,
> and the `HANDOFF_2026-06-01` notes. The libfranka-0.18.1 stack is confirmed
> end-to-end (zerorpc client → FR3 motion, J1 ±11° at 15 Hz streaming).

---

## 1. Installing on a NUC (from scratch)

Provisioning a replacement/second NUC. Steps confirmed against NUC1 (Ubuntu
22.04.5, RT kernel `5.15.0-1105-realtime`, image present locally).

### 1.0 Hardware
- Intel NUC, ≥30 GB free disk (the image alone is ~18 GB).
- **Cat6** from the NUC NIC to the FR3 **C2 (Shop-Floor) port** (FCI is filtered
  on the X5 port).
- USB cable to the **Robotiq 2F-85** (its FTDI USB-RS485 bridge → `/dev/ttyUSB0`).

### 1.1 OS + real-time kernel
The libfranka 1 kHz client **requires a PREEMPT_RT kernel** (the container runs at
`rtprio 99`). On a non-RT kernel the controller will run but miss deadlines /
trigger reflexes.

```bash
# Ubuntu 22.04.5 LTS, user `tasl`
sudo pro attach <UBUNTU_PRO_TOKEN>      # realtime-kernel needs Ubuntu Pro
sudo pro enable realtime-kernel
sudo reboot
uname -r            # must show ...-realtime  (e.g. 5.15.0-1105-realtime)
```
> NUC1 uses the Ubuntu Pro `realtime-kernel` (PREEMPT_RT). If you have no Pro
> token, any PREEMPT_RT kernel works, but match the running NUC if you can.

### 1.2 Docker
```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER && newgrp docker      # log out/in to take effect
docker compose version                               # plugin must be present
```

### 1.3 Network (robot net)
Static address on the robot subnet `172.16.0.0/24`, NUC side `172.16.0.2`, no
gateway (isolated link). Via NetworkManager:
```bash
nmcli con add type ethernet ifname <iface> con-name robotnet \
  ipv4.method manual ipv4.addresses 172.16.0.2/24
nmcli con up robotnet
ping -c1 172.16.0.1        # FR3 reachable
```
(Use the FR3's actual control IP if your bench differs — keep `parameters.py` in
sync. **Remember the IPs are reversed from DROID defaults**, §5.)

### 1.4 Get the controller config — from the repo, no peer-copy
The config is **version-managed in this repo** at
[`tasl/controller/`](../controller/), so you don't copy it off another NUC:

```bash
git clone https://github.com/tasl-lab/RLinf ~/RLinf      # or pull
cp -r ~/RLinf/tasl/controller ~/polymetis_fr3
cd ~/polymetis_fr3
# edit parameters.py / conf/ if this bench's IPs differ, then commit back
```

That tree (`docker-compose.yaml`, `parameters.py`, `Dockerfile.droid-0181`,
`conf/`) is the entire working set — the deprecated `wrapper/`/`scripts/`/`fairo/`
are not needed (§2). See "Reproducing the config without copying" below for why a
clone is enough.

### 1.5 Get the image — two options
The ~18 GB image is **not** in the repo. Either transfer it or rebuild.

**Option A — transfer the prebuilt image (recommended, fastest, no ghcr needed):**
```bash
# on an existing NUC (~18 GB → multi-GB gz, a few minutes):
docker save droid-nuc-fr3:0.18.1 | gzip > /tmp/droid-nuc-fr3-0.18.1.tar.gz
scp /tmp/droid-nuc-fr3-0.18.1.tar.gz newnuc:/tmp/

# on the new NUC:
gunzip -c /tmp/droid-nuc-fr3-0.18.1.tar.gz | docker load
docker images | grep droid-nuc-fr3            # confirm 0.18.1 present
```

**Option B — rebuild from the Dockerfile (needs the ghcr base image):**
```bash
docker login ghcr.io                          # if the base image is gated
docker pull ghcr.io/droid-dataset/droid_nuc:fr3
cd ~/polymetis_fr3
docker build -f Dockerfile.droid-0181 -t droid-nuc-fr3:0.18.1 .
```
See §3 for what the build does. Prefer **Option A** unless you're changing
libfranka/firmware — the build compiles libfranka + relinks polymetis and is slow.

### 1.6 Bring up + verify
```bash
cd ~/polymetis_fr3
docker compose up -d
docker compose logs --tail 20 droid-nuc-fr3
ss -tlnp | grep ':4242'                       # zerorpc listening
```
Then enable FCI in Franka Desk and run the §7 smoke test. If you also installed
the legacy franky stack on this NUC, **stop/disable `franka-robot-server`** first
(§6) — only one libfranka client may hold FCI.

> **No franky unit on a clean NUC.** A fresh install won't have
> `franka-robot-server` at all, so the FCI-exclusivity step is a no-op — it only
> matters if you later add the franky backend.

### Reproducing the config without copying
None of the controller files actually require copying from a peer NUC:

- **`conf/franka_hardware.yaml` + `conf/franka_panda.yaml`** are **byte-identical
  to the copies baked into the image** (verified). The mounts override them with
  the same bytes. You could drop the mounts entirely, or extract them straight
  from the image:
  ```bash
  docker run --rm --entrypoint cat droid-nuc-fr3:0.18.1 \
    /app/droid/fairo/polymetis/polymetis/conf/robot_client/franka_hardware.yaml
  ```
- **`parameters.py`** is the **only real override** (the image's built-in has
  DROID's reversed IPs). It's tiny and fully reproduced in §5 — hand-write it.
- **`docker-compose.yaml` + `Dockerfile.droid-0181`** are short and reproduced in
  §4 / §3.

So: clone the repo (`tasl/controller/`), or paste these few files from this doc —
either way no machine-to-machine copy is needed. Only the image needs `save`/`load`.

---

## 2. What actually runs (and what doesn't)

`~/polymetis_fr3/` has accumulated two generations of files. **Only the DROID
image path is live.** Know which is which before you touch anything:

| Path | Status | Role |
|---|---|---|
| `Dockerfile.droid-0181` | **LIVE** | Builds the controller image `droid-nuc-fr3:0.18.1` |
| `docker-compose.yaml` | **LIVE** | Starts the `droid-nuc-fr3` container |
| `parameters.py` | **LIVE** | Mounted in — overrides DROID's default robot IPs |
| `conf/franka_hardware.yaml` | **LIVE** | FR3 robot client config (robot_ip, limits, safety) |
| `conf/franka_panda.yaml` | **LIVE** | FR3 robot model (joints, rest pose, limits) |
| `Dockerfile` (main) | deprecated | Early from-scratch Ubuntu 20.04 build (no conda). Superseded by `Dockerfile.droid-0181`; kept for reference |
| `wrapper/` | deprecated | Early hand-written zerorpc shim (`server.py` + `FrankaController` + `ik_solver.py`). **Not mounted, not run** — the live image uses DROID's own `run_server.py` → `FrankaRobot` |
| `scripts/` | deprecated | Early `launch_*.sh` (the live image has its own under `/app/scripts/server/`) |
| `tests/` | reference | Pure unit tests for the old `wrapper/ik_solver.py` |
| `fairo/` | reference | polymetis source used during a build; baked into the image now |

The deprecated `wrapper/`/`scripts/` are still useful as a **readable spec** of the
control contract (DROID-style 8-D action, `max_joint_delta=0.2` at 15 Hz, gripper
convention) — but the running controller is DROID's `FrankaRobot`, not this code.
The **repo-managed** copy at [`tasl/controller/`](../controller/) carries only the
five LIVE files.

---

## 3. The controller image — `droid-nuc-fr3:0.18.1`

Built by **`Dockerfile.droid-0181`** as a thin derivative of
`ghcr.io/droid-dataset/droid_nuc:fr3`. The base image already has polymetis + conda
env `polymetis-local`, but ships **libfranka 0.13.5**, which can't speak the FR3 fw
**5.9.2** server protocol (v10). The whole point of the Dockerfile is to swap
libfranka up to **0.18.1** and relink `franka_panda_client` against it.

What the Dockerfile does, in order:

1. **Install 0.18 build deps** the 0.13.5 base lacked — Pinocchio/urdfdom/tinyxml2
   kinematics deps (`liburdfdom-dev`, `libconsole-bridge-dev`, `libtinyxml2-dev`, …).
2. **tinyxml2 CONFIG shim** — writes a `tinyxml2-config.cmake` because the base
   (Ubuntu 20.04) ships the lib but no cmake config, and Pinocchio's
   `find_package(... CONFIG)` needs one.
3. **Clone libfranka `0.18.1`** (recursive) into polymetis'
   `franka_panda_client/third_party/`, replacing the bundled 0.13.5.
4. **Patch + build libfranka:**
   - `cmake_minimum_required → 3.5` across its CMakeLists (modern cmake rejects the
     old pins) plus `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`.
   - **Pinocchio 3→2 API rename:** `sed 's/.parentJoint/.parent/g'` in
     `src/robot_model.cpp` — the base image has Pinocchio 2.x, but libfranka 0.18
     was written against 3.x (`Frame::parentJoint`). Same field, old name.
   - Build with **C++17** (`-DCMAKE_CXX_STANDARD=17`), tests/examples off.
5. **Relink `franka_panda_client`** in polymetis' build dir:
   - Wipe `CMakeCache.txt` + the old binary, re-cmake with `-DBUILD_FRANKA=ON`,
     **C++17** (0.18 headers use `std::optional`), inside the `polymetis-local`
     conda env.
   - Point `LDFLAGS`/rpath at the **conda `libstdc++`**
     (`/root/miniconda3/envs/polymetis-local/lib`) so the binary loads the conda
     C++ runtime, not the system one.
   - `make franka_panda_client`, then `ldd | grep franka` to confirm it links the
     new libfranka.
6. **`LABEL libfranka.version=0.18.1`** so you can tell at a glance.

Build it (only needed if libfranka/firmware changes — the image is normally
prebuilt, ~18 GB):

```bash
cd ~/polymetis_fr3
docker build -f Dockerfile.droid-0181 -t droid-nuc-fr3:0.18.1 .
docker inspect droid-nuc-fr3:0.18.1 --format '{{ index .Config.Labels "libfranka.version" }}'
```

---

## 4. The container — `docker-compose.yaml`

`docker compose up -d` starts container **`droid-nuc-fr3`** from that image:

- **`network_mode: host`** — zerorpc `:4242` and the internal polymetis gRPC
  `:50051` are on the host net.
- **`privileged` + `cap_add: SYS_NICE` + `ulimits rtprio: 99`, `memlock`** — the
  libfranka client needs RT priority (NUC1 runs the `-realtime` kernel).
- **`devices: /dev:/dev`** — so the container sees the Robotiq gripper on
  `/dev/ttyUSB0`.
- **`restart: unless-stopped`.**
- Three **bind mounts** override files baked into the DROID image:

  ```yaml
  volumes:
    - ./parameters.py:/app/droid/misc/parameters.py
    - ./conf/franka_hardware.yaml:/app/droid/fairo/polymetis/polymetis/conf/robot_client/franka_hardware.yaml
    - ./conf/franka_panda.yaml:/app/droid/fairo/polymetis/polymetis/conf/robot_model/franka_panda.yaml
  ```
  (The two `conf/*.yaml` mounts are byte-identical to the image's own copies — §1
  "Reproducing the config"; only `parameters.py` actually changes anything.)

- `environment:` sets `ROBOT_TYPE=fr3`, `NUC_IP=172.16.0.2`, `ROBOT_IP=172.16.0.1`,
  `LAPTOP_IP=100.79.65.37`.
  - ⚠ **`LIBFRANKA_VERSION: "0.13.5"` here is stale/unused** — the image actually
    ships 0.18.1 (baked at build time). Don't trust this env var.

The container's entrypoint is DROID's `launch_server.sh` → `run_server.py` →
`zerorpc.Server(FrankaRobot()).bind(:4242)`. All polymetis processes live in the
**`polymetis-local` conda env** (`source /root/miniconda3/etc/profile.d/conda.sh &&
conda activate polymetis-local` if you `docker exec` in).

---

## 5. Config — the mounted files

### `parameters.py` (robot IPs)
Overrides DROID's `misc/parameters.py`. **This is the only mount that changes
behaviour.** The lines that matter:

```python
nuc_ip   = "172.16.0.2"   # NUC1's direct-eth side
robot_ip = "172.16.0.1"   # our FR3  (DROID default is the OPPOSITE)
laptop_ip = "100.79.65.37" # Desktop (Tailscale; legacy/often offline)
robot_type = "fr3"
```

> **#1 footgun: the IPs are reversed from DROID defaults.** DROID ships
> `nuc_ip=172.16.0.1, robot_ip=172.16.0.2`; ours is the reverse. **Re-check this in
> both `parameters.py` and `conf/franka_hardware.yaml` after any rebuild.**
>
> The repo copy ([`tasl/controller/parameters.py`](../controller/parameters.py))
> has the `sudo_password` scrubbed (read from `$SUDO_PASSWORD`; unused on the
> root-container path) — don't commit the real secret back.

### `conf/franka_hardware.yaml` (robot_client)
1 kHz, real-time, `exec: franka_panda_client`. Sets `robot_ip: "172.16.0.1"`, the
default impedance gains (`default_Kq/Kqd/Kx/Kxd`), and the **safety envelope**:
workspace cartesian box, per-joint position/velocity/torque limits (Franka limits
minus a margin), `collision_behavior` torque/force thresholds, and the
`safety_controller` margins/stiffness. Tune limits here, not in code. **Identical
to the image's built-in copy** — kept in the repo so it's visible/editable.

### `conf/franka_panda.yaml` (robot_model)
FR3 kinematic model: `panda_arm.urdf`, 7 DOF, `ee_link=panda_link8`, the `rest_pose`,
and `joint_limits_low/high`, `joint_damping`, `torque_limits` used by the IK / model.
Also identical to the image's built-in copy.

---

## 6. Bring-up (operational)

Only **one** libfranka client may hold FCI at a time, so the old franky
`franka-robot-server` systemd unit must be **stopped** (it's now `disable`d on NUC1,
so it stays inactive across reboots — see SETUP §3).

```bash
ssh tasl-nuc1
sudo systemctl stop franka-robot-server          # release FCI (no-op if disabled)
cd ~/polymetis_fr3 && docker compose up -d
sleep 5
docker compose logs --tail 20 droid-nuc-fr3
ss -tlnp | grep ':4242'                          # zerorpc listening
```

Then enable FCI in **Franka Desk** (`https://172.16.0.1` → unlock joints →
"Activate FCI"). See TELEOP.md for the operator flow.

### Lazy init — the driver does NOT auto-start
The container binding `:4242` does **not** mean the robot is live. DROID's
`FrankaRobot` is lazy: the polymetis driver (`launch_robot.py` →
`franka_panda_client`) only spawns when a zerorpc client calls `launch_controller()`
then `launch_robot()` (~5–10 s cold). The Desktop's `DroidLikeClient.bootstrap()`
chains those two; the dashboard's Go-Home / recover triggers it.

### Gripper
Driven through this same container over zerorpc: `launch_gripper.py
gripper=robotiq_2f gripper.comport=/dev/ttyUSB0` — a **Robotiq 2F-85** on an FTDI
USB-RS485 bridge (`0403:6015`), `MAX_GRIPPER_WIDTH = 0.085 m`. (This contradicts a
"Franka Hand" assumption elsewhere — the live driver is Robotiq.)

---

## 7. Smoke test

After bring-up, confirm the 5 expected processes inside the container:

```bash
docker exec droid-nuc-fr3 bash -lc \
  "ps auxf | grep -E 'launch_robot|launch_gripper|franka_panda_client|run_server' | grep -v grep"
# expect: run_server.py + launch_robot.py + launch_gripper.py + run_server (cpp) + franka_panda_client (cpp)
```

From the Desktop, the end-to-end link/motion smoke is the `DroidLikeClient` snippet
in [HANDOFF_2026-06-01.md](HANDOFF_2026-06-01.md) ("Smoke test") — it bootstraps,
reads state, and nudges J1 ±0.1 rad. **Pass `update_command` args positionally** —
this zerorpc build silently drops kwargs (the client wrapper already handles this).

---

## 8. Control contract (for reference)

The model/action conventions the controller implements (matching DROID, mirrored in
the deprecated `wrapper/` as a readable spec):

- **Action: 8-D in `[-1, 1]`.** `action[:7]` = joint velocities, `action[7]` =
  gripper command (`0`=open, `1`=close).
- **Joint velocity → delta:** `delta_i = clip(v_i, -1, 1) * 0.2 rad` per step, at the
  **15 Hz** DROID inference rate (no Jacobian/SE3 — DROID was trained against this
  exact scaling).
- **Gripper:** `width = MAX_GRIPPER_WIDTH * (1 - command)`; reported
  `gripper_position = 1 - width/MAX` (1 = closed).
- **Home pose (rad):** `[0, -π/5, 0, -4π/5, 0, 3π/5, 0]`.

---

## 9. Quick reference

| Item | Value |
|---|---|
| Repo-managed config | [`tasl/controller/`](../controller/) (compose, parameters.py, Dockerfile, conf/) |
| Image | `droid-nuc-fr3:0.18.1` (base `ghcr.io/droid-dataset/droid_nuc:fr3`) |
| Built by | `Dockerfile.droid-0181` |
| libfranka | 0.18.1 (FR3 fw 5.9.2, FCI protocol v10) |
| Container | `droid-nuc-fr3` — host net, privileged, rtprio 99 |
| Conda env | `polymetis-local` |
| Controller iface | zerorpc `tcp://172.16.0.2:4242` (DROID `FrankaRobot`) |
| Robot IPs | FR3 `172.16.0.1`, NUC1 `172.16.0.2` (**reversed from DROID default**) |
| Gripper | Robotiq 2F-85, `/dev/ttyUSB0` (FTDI `0403:6015`), width `0.085 m` |
| Control rate / step | 15 Hz, `max_joint_delta = 0.2 rad` |
| FCI exclusivity | stop/disable `franka-robot-server` before `compose up` |
| Real override | only `parameters.py`; conf/*.yaml are identical to the image's |
