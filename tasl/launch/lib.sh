#!/usr/bin/env bash
# Shared helpers + robot preflight for the FR3 dashboard launchers.
#
# DESIGN: these scripts RUN ON THE DESKTOP (TASL-1), self-contained, for ANY
# user with no prior context. They use only LOCAL commands (docker, pkill,
# nohup, a local zerorpc probe) — no ssh, no laptop config, no stored secrets.
# The wrappers self-elevate with sudo (cameras + uinput need root).
#
# LAUNCH_DRY_RUN=1 makes every side-effecting call echo instead of run, so the
# command sequence can be unit-tested anywhere (laptop, no robot).
set -euo pipefail

: "${LAUNCH_DRY_RUN:=}"
ZED_EXTERIOR=36443134
ZED_WRIST=17150101
# NUC1 over the ROBOT network (Desktop is on 172.16.0.x). Tailscale
# (100.75.6.62) is the legacy path. Override either with an env var.
NUC1_HOST="${NUC1_HOST:-172.16.0.2}"
NUC1_SSH="${NUC1_SSH:-tasl@$NUC1_HOST}"
ZERORPC_ADDR="tcp://$NUC1_HOST:4242"
# Desktop dashboard host for the laptop. Prefer the Tailscale IP (derived LIVE
# so it can never go stale); if tailscale is offline, fall back to the
# robot-network IP (172.16.0.x). Override with TS_IP=… if needed.
TS_IP="${TS_IP:-$(tailscale ip -4 2>/dev/null | head -1)}"
if [[ -z "$TS_IP" ]]; then
  TS_IP="$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^172\.16\.0\.' | head -1)"
  TS_IP="${TS_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}"   # last resort: first IP
fi
# Repo root derived from THIS file's location (<repo>/tasl/launch/lib.sh), so it
# is correct wherever the checkout lives and survives the sudo re-exec — unlike
# $HOME, BASH_SOURCE does not change to /root under sudo. DESKTOP_HOME is the dir
# the repo + data sit under. Override the repo with RLINF_REPO_HOST.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # consolidated checkout (~/RLinf)
DESKTOP_HOME="$(dirname "$REPO")"   # home dir (repo + rlinf_data live here)
TASL="$REPO/tasl"                   # bench operator tooling lives here
SITE_PKGS="$DESKTOP_HOME/.local/lib/python3.10/site-packages"

# --- consolidated code + data locations (SINGLE SOURCE OF TRUTH) -------------
# Code lives in ONE checkout ($REPO = ~/RLinf), mounted into the container at
# /workspace/rlinf. DATA (datasets + outputs) lives OUTSIDE the repo so swapping
# the code checkout never touches collected data. The dashboards read these
# same env vars (with matching defaults), so paths are defined here, not
# hard-coded in Python. Override any with an env var before launch.
RLINF_REPO_HOST="${RLINF_REPO_HOST:-$REPO}"                    # host code -> /workspace/rlinf
RLINF_DATA_DIR="${RLINF_DATA_DIR:-$DESKTOP_HOME/rlinf_data}"   # host datasets/ + outputs/
RLINF_CONTAINER="${RLINF_CONTAINER:-rlinf-eval}"
RLINF_IMAGE="${RLINF_IMAGE:-rlinf/rlinf:agentic-rlinf0.2-pi05droid-zed}"
CKPT_HOST="${CKPT_HOST:-$DESKTOP_HOME/ckpts/pi05_droid_pt}"
export RLINF_REPO_HOST RLINF_DATA_DIR RLINF_CONTAINER RLINF_IMAGE

# --- logging ---
step() { printf '\033[36m▶ %s\033[0m\n' "$*"; }
ok()   { printf '\033[32m✓ %s\033[0m\n' "$*"; }
warn() { printf '\033[33m! %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# --- self-elevate: re-exec under sudo if not root (cameras + uinput need it) ---
# Call as:  ensure_root "$0" "$@"   (no-op in dry-run and when already root)
ensure_root() {
  if [[ -n "$LAUNCH_DRY_RUN" ]]; then echo "DRY ensure_root (would re-exec under sudo if not root)"; return 0; fi
  local self="$1"; shift
  if [[ "$(id -u)" -ne 0 ]]; then
    step "Re-running under sudo (cameras + uinput need root) — enter the Desktop password"
    exec sudo bash "$self" "$@"
  fi
}

# --- local command runner (dry-run aware) ---
# Runs a shell command line LOCALLY on the Desktop. In dry-run, just echoes it.
desk() { if [[ -n "$LAUNCH_DRY_RUN" ]]; then echo "DRY: $*"; return 0; fi; eval "$*"; }

# --- NUC1 ssh (key-based, works under sudo and from the dashboard) -----------
# The bench key lives in the OPERATOR's ~/.ssh, but every launcher re-execs
# under sudo, where $HOME is /root — so the operator's `Host FrankaNUC` config
# block never applies and ssh silently falls back to an interactive password
# prompt (and would HANG outright when called from the dashboard, which has no
# tty). Point at the key by absolute path instead. Override with NUC1_SSH_KEY.
NUC1_SSH_KEY="${NUC1_SSH_KEY:-$DESKTOP_HOME/.ssh/id_ed25519_frankanuc}"

# nuc_ssh: run a command on NUC1. Allocates a tty only when we have one;
# without a tty it goes BatchMode so a missing key fails fast and loudly
# instead of blocking forever on a password prompt nobody can answer.
nuc_ssh() {
  local opts=(-o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new)
  [[ -r "$NUC1_SSH_KEY" ]] && opts+=(-i "$NUC1_SSH_KEY" -o IdentitiesOnly=yes)
  if [[ -t 0 && -t 1 ]]; then opts+=(-t); else opts+=(-o BatchMode=yes); fi
  ssh "${opts[@]}" "$NUC1_SSH" "$@"
}

# --- rlinf-eval container: create with the canonical mounts, else (re)start ---
# The bind mounts are the REAL source of truth for where in-container code +
# data live, so they belong here in version control, not in a hand-run
# `docker run`. Idempotent: reuses an existing container. Mounts are fixed at
# creation time, so to REPOINT them (e.g. new code checkout / data dir):
#     docker rm -f "$RLINF_CONTAINER"    # then re-run any launcher
ensure_rlinf_container() {
  if [[ -n "$LAUNCH_DRY_RUN" ]]; then
    echo "DRY: ensure_rlinf_container $RLINF_CONTAINER (code=$RLINF_REPO_HOST data=$RLINF_DATA_DIR)"
    return 0
  fi
  mkdir -p "$RLINF_DATA_DIR/datasets" "$RLINF_DATA_DIR/outputs"
  if docker inspect "$RLINF_CONTAINER" >/dev/null 2>&1; then
    docker start "$RLINF_CONTAINER" >/dev/null
  else
    step "creating $RLINF_CONTAINER (code=$RLINF_REPO_HOST, data=$RLINF_DATA_DIR)"
    docker run -d --name "$RLINF_CONTAINER" \
      --network host --privileged --gpus all --shm-size 64m \
      -w /workspace/rlinf \
      -v /dev:/dev \
      -v "$RLINF_REPO_HOST:/workspace/rlinf" \
      -v "$RLINF_DATA_DIR/datasets:/workspace/rlinf/datasets" \
      -v "$RLINF_DATA_DIR/outputs:/workspace/rlinf/outputs" \
      -v "$CKPT_HOST:/ckpts/pi05_droid_pt:ro" \
      -v /usr/local/zed:/usr/local/zed:ro \
      -v /usr/local/cuda:/usr/local/cuda:ro \
      "$RLINF_IMAGE" -c 'sleep infinity' >/dev/null
  fi
  # Container runs as root over a franka_desktop-owned checkout; git refuses
  # ("dubious ownership") which breaks Hydra's git-SHA logging + our checks.
  # Idempotent allowlist of the mount path.
  docker exec "$RLINF_CONTAINER" bash -lc \
    'git config --global --get-all safe.directory 2>/dev/null | grep -qx /workspace/rlinf \
       || git config --global --add safe.directory /workspace/rlinf' 2>/dev/null || true
  ensure_container_deps
}

# --- in-container runtime deps (GELLO teleop stack) --------------------------
# These are NOT in the base image and were previously pip-installed by hand into
# the container's ephemeral layer (lost on 2026-07-03 when the container was
# recreated). Reinstall them here — from version-controlled sources — whenever
# they are missing, so a container rebuild restores the whole stack. The leader
# CALIBRATION lives in the repo (rlinf/.../gello/fr3_gello_config.py), not here.
ensure_container_deps() {
  if [[ -n "$LAUNCH_DRY_RUN" ]]; then echo "DRY: ensure_container_deps"; return 0; fi
  if docker exec "$RLINF_CONTAINER" /opt/venv/openpi/bin/python -c \
       'import zerorpc, dynamixel_sdk, gello.agents.gello_agent, gello_teleop' 2>/dev/null; then
    return 0   # fast path: already present
  fi
  step "installing in-container GELLO deps (zerorpc, dynamixel-sdk, gello, gello_teleop)"
  docker exec "$RLINF_CONTAINER" bash -lc '
    set -e
    PIP=/opt/venv/openpi/bin/pip
    "$PIP" install -q zerorpc dynamixel-sdk
    if [ ! -e /opt/gello_software/pyproject.toml ] && [ ! -e /opt/gello_software/setup.py ]; then
      git clone --recurse-submodules -q https://github.com/wuphilipp/gello_software.git /opt/gello_software
    fi
    "$PIP" install -q -e /opt/gello_software
    "$PIP" install -q "gello-teleop @ git+https://github.com/RLinf/gello-teleop.git"
  ' && ok "GELLO deps present" \
    || warn "GELLO dep install had issues (see above) — teleop/collect may fail until fixed"
}

# --- health checks (dry-run aware via DRY_* stubs) ---

# robot_ok: does the NUC zerorpc controller answer get_robot_state within ~5s?
robot_ok() {
  if [[ -n "$LAUNCH_DRY_RUN" ]]; then echo "DRY probe: get_robot_state @ $ZERORPC_ADDR"; [[ "${DRY_ROBOT_OK:-1}" == 1 ]]; return; fi
  PYTHONPATH="$SITE_PKGS" timeout 6 /usr/bin/python3 - <<PY 2>/dev/null | grep -q OK
import zerorpc, sys
c = zerorpc.Client(timeout=5); c.connect("$ZERORPC_ADDR")
try:
    c.get_robot_state(); print("OK")
except Exception:
    sys.exit(1)
PY
}

# zeds_free: are both expected ZED serials present + enumerable (not held)?
zeds_free() {
  if [[ -n "$LAUNCH_DRY_RUN" ]]; then echo "DRY zed probe: $ZED_EXTERIOR $ZED_WRIST"; [[ "${DRY_ZEDS_FREE:-1}" == 1 ]]; return; fi
  local serials
  serials=$(docker exec rlinf-eval /opt/venv/openpi/bin/python -c \
    "import pyzed.sl as sl; print([d.serial_number for d in sl.Camera.get_device_list()])" 2>/dev/null | tail -1)
  [[ "$serials" == *"$ZED_EXTERIOR"* && "$serials" == *"$ZED_WRIST"* ]]
}

# STEREOLABS USB ids for the two CAMERA interfaces (not the f6xx HID controls):
#   f880 = ZED 2i (exterior), f682 = ZED-M (wrist). Kept in sync with zed-check.sh.
ZED_USB_IDS=(2b03:f880 2b03:f682)

# zeds_reset: USB-reset both ZEDs to clear the post-kill / hot-plug SDK wedge
# (enumerated on USB but get_device_list() == []). Needs root + a free camera
# (no process holding it), so call this AFTER reaping stale eval/collect procs.
zeds_reset() {
  if [[ -n "$LAUNCH_DRY_RUN" ]]; then echo "DRY zed usbreset: ${ZED_USB_IDS[*]}"; return 0; fi
  if [[ "$(id -u)" != 0 ]]; then warn "zeds_reset needs root (skipping usbreset)"; return 1; fi
  local id
  for id in "${ZED_USB_IDS[@]}"; do
    if usbreset "$id" >/dev/null 2>&1; then ok "usbreset $id"; else warn "usbreset $id failed (camera absent / still held?)"; fi
  done
  sleep 2   # let the SDK re-enumerate before the next probe
}

# zeds_free_or_reset: ensure both ZEDs are SDK-visible, auto USB-resetting on a
# wedge instead of failing. Probe first (no reset if already fine); otherwise
# reset + re-probe up to N times. Returns non-zero only if still wedged after
# all attempts (a real reseat/reboot case). Reap ZED holders before calling.
zeds_free_or_reset() {
  local tries="${1:-2}" i
  zeds_free && return 0
  for ((i = 1; i <= tries; i++)); do
    warn "ZEDs not SDK-visible — auto USB-reset (attempt $i/$tries)"
    zeds_reset
    zeds_free && { ok "ZEDs recovered after USB reset"; return 0; }
  done
  return 1
}

# collection_procs_running: pids inside the container still holding the ZEDs +
# GELLO — a live OR wedged collection. Empty output (rc 1) means nothing holds
# them, which is what distinguishes a real USB wedge from a leftover run.
collection_procs_running() {
  if [[ -n "$LAUNCH_DRY_RUN" ]]; then
    echo "DRY collection_procs probe"; [[ "${DRY_COLLECT_PROCS:-0}" == 1 ]]; return
  fi
  docker exec "$RLINF_CONTAINER" \
    pgrep -f "[c]ollect_real_data|[r]ay::DataCollector|[P]olymetisController" \
    2>/dev/null
}

# reap_collection_procs: THE definition of "free the ZEDs + GELLO". Shared by
# teleop.sh / start_collect.sh / teleop-stop.sh so the kill set is fixed in one
# place — it used to be copy-pasted into all three.
reap_collection_procs() {
  desk "docker exec $RLINF_CONTAINER bash -lc 'pkill -9 -f \"[r]ay::DataCollector\"; pkill -9 -f \"[c]ollect_real_data\"; pkill -9 -f \"[P]olymetisController\"; ${CONTAINER_RAY:-/opt/venv/openpi/bin/ray} stop --force >/dev/null 2>&1; rm -rf /tmp/ray; true' || true"
  [[ -z "$LAUNCH_DRY_RUN" ]] && sleep 2
  return 0
}

# ensure_zeds_free: get both ZEDs SDK-visible, telling apart the two causes that
# look IDENTICAL from get_device_list() == []:
#   (a) a live/wedged collection in the container still owns them  -> reap it
#   (b) a post-hot-plug USB wedge, nothing holding them            -> usbreset
# Diagnosing (a) as (b) is exactly what used to happen: a stale run would fail
# the preflight with a "post-hot-plug USB wedge" message and send the operator
# off to usbreset cameras that were simply still in use.
ensure_zeds_free() {
  zeds_free && { ok "ZEDs free"; return 0; }
  local held
  held="$(collection_procs_running || true)"
  if [[ -n "$held" ]]; then
    warn "ZEDs are held by an in-container collection (pids: $(tr '\n' ' ' <<<"$held")) — reaping it"
    reap_collection_procs
    zeds_free && { ok "ZEDs freed by reaping the stale collection"; return 0; }
    warn "still not visible after the reap — trying a USB reset too"
  fi
  zeds_free_or_reset 2
}

# gpu_free: no compute process on the Desktop GPU (serve + training froze the box).
gpu_free() {
  if [[ -n "$LAUNCH_DRY_RUN" ]]; then echo "DRY nvidia-smi compute-apps"; [[ "${DRY_GPU_FREE:-1}" == 1 ]]; return; fi
  local out rc
  out=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); rc=$?
  [[ "$rc" -ne 0 ]] && die "nvidia-smi failed on the Desktop (driver wedged / GPU unreachable) — cannot verify the GPU is free"
  local n; n=$(grep -c '[0-9]' <<<"$out" || true)
  [[ "${n:-0}" -eq 0 ]]
}

# port_open: is a local TCP port already serving?
port_open() {
  local port="$1"
  if [[ -n "$LAUNCH_DRY_RUN" ]]; then echo "DRY port_open: $port"; [[ "${DRY_PORT_OPEN:-1}" == 1 ]]; return; fi
  curl -s -o /dev/null --max-time 3 "http://127.0.0.1:$port/" >/dev/null 2>&1
}

# wait_http: poll a local HTTP port until it answers (any HTTP status, incl.
# 426 for a WebSocket server) or times out. arg2 = max tries x2s (default 20=40s).
# dry-run: stub.
wait_http() {
  local port="$1" tries="${2:-20}"
  if [[ -n "$LAUNCH_DRY_RUN" ]]; then echo "DRY wait_http: $port (${tries}x2s)"; [[ "${DRY_HTTP_OK:-1}" == 1 ]]; return; fi
  local i code
  for i in $(seq 1 "$tries"); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:$port/" 2>/dev/null)
    [[ "$code" =~ ^[0-9]+$ && "$code" -ne 000 ]] && return 0
    sleep 2
  done
  return 1
}

# kill_other_dashboard: SIGTERM the OTHER dashboard (mutual exclusion on cameras
# + zerorpc). The [_] bracket avoids pkill matching its own argv (self-kill).
# arg1 = which dashboard we are starting (collect|openpi).
kill_other_dashboard() {
  local me="$1" pat
  if [[ "$me" == collect ]]; then pat="[/_]openpi[.]py"; else pat="[/_]collect[.]py"; fi
  desk "pkill -TERM -f '$pat' || true"
}

# preflight_robot: shared checks for both dashboards. arg1 = collect|openpi.
# Robot is CHECK + INSTRUCT (no auto NUC bring-up: a no-context Desktop user
# has no ssh to the NUC, and the droid container is normally already up).
preflight_robot() {
  local me="$1"
  step "Robot: zerorpc get_robot_state @ $ZERORPC_ADDR"
  robot_ok || die "robot controller unreachable at $ZERORPC_ADDR.
    Bring up the controller on NUC1 first (or just run teleop.sh, which does
    this for you):
      ssh $NUC1_SSH            (sudo password: tasl123456)
      sudo systemctl stop franka-robot-server
      cd ~/polymetis_fr3 && docker compose up -d
    Then make sure the FR3 is powered and FCI is active in Desk (brakes off,
    execution mode). Re-run this script."
  ok "robot reachable"
  step "Mutual exclusion: stop the other dashboard"
  kill_other_dashboard "$me"
  ok "other dashboard stopped"
  step "Cameras: both ZEDs free ($ZED_EXTERIOR exterior, $ZED_WRIST wrist)"
  ensure_zeds_free || die "ZED cameras still not visible to the SDK after reaping
    in-container holders AND a USB reset. Escalate:
      1. $TASL/launch/zed-check.sh          # confirm which serial is missing
      2. physically unplug/replug the exterior ZED 2i, then re-run
      3. reboot the Desktop (last resort)"
}
