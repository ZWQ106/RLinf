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
ZERORPC_ADDR="tcp://100.75.6.62:4242"
DESKTOP_HOME=/home/franka_desktop
REPO="$DESKTOP_HOME/RLinf"          # consolidated fork checkout (tasl-lab/RLinf)
TASL="$REPO/tasl"                   # bench operator tooling lives here
SITE_PKGS="$DESKTOP_HOME/.local/lib/python3.10/site-packages"

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
  if [[ "$me" == collect ]]; then pat="dashboards/openpi[.]py"; else pat="dashboards/collect[.]py"; fi
  desk "pkill -TERM -f '$pat' || true"
}

# preflight_robot: shared checks for both dashboards. arg1 = collect|openpi.
# Robot is CHECK + INSTRUCT (no auto NUC bring-up: a no-context Desktop user
# has no ssh to the NUC, and the droid container is normally already up).
preflight_robot() {
  local me="$1"
  step "Robot: zerorpc get_robot_state @ $ZERORPC_ADDR"
  robot_ok || die "robot controller unreachable at $ZERORPC_ADDR.
    Bring up the controller on NUC1 first:
      ssh tasl@100.75.6.62            (sudo password: tasl123456)
      sudo systemctl stop franka-robot-server
      cd ~/polymetis_fr3 && docker compose up -d
    Then make sure the FR3 is powered and FCI is active in Desk (brakes off,
    execution mode). Re-run this script."
  ok "robot reachable"
  step "Mutual exclusion: stop the other dashboard"
  kill_other_dashboard "$me"
  ok "other dashboard stopped"
  step "Cameras: both ZEDs free ($ZED_EXTERIOR exterior, $ZED_WRIST wrist)"
  zeds_free || die "ZED cameras not both free/present. Reseat the exterior 2i USB; make sure no dashboard or collection still holds them."
  ok "ZEDs free"
}
