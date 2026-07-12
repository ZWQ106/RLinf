#!/usr/bin/env bash
# Deploy the version-controlled NUC controller config to NUC1 and restart the
# droid-nuc-fr3 polymetis controller so it re-reads the parameters (impedance
# gains, safety limits, IPs). RUN ON THE DESKTOP.
#
#   ./deploy.sh                       # push conf -> NUC1, restart controller, wait :4242
#   ./deploy.sh --preset stiff        # named joint-stiffness preset (see PRESETS below)
#   ./deploy.sh --kq "20 15 25 12 18 12 5"   # custom 7-joint stiffness, then deploy
#   ./deploy.sh --no-restart          # copy files only (apply later)
#   NUC1_HOST=… ./deploy.sh           # override the NUC host
#   LAUNCH_DRY_RUN=1 ./deploy.sh      # print actions, do nothing
#
# PRESETS (default_Kq, J1..J7) — pick with --preset NAME:
#   stiff   [40 30 50 25 35 25 10]  original — accurate tracking (data collection + policy eval)
#   soft    [20 15 25 12 18 12  5]  halved  — compliant, still decent tracking
#   softer  [15 12 20 10 12  8  4]  current — most compliant (GELLO calibration / teleop testing)
#
# The config files are mounted into the container (see docker-compose.yaml), so a
# restart re-reads them. A restart drops FCI: afterwards re-Activate FCI in Desk
# and Recover/Home in the dashboard — or just run `sudo ../launch/teleop.sh`,
# which does the FCI gate + home for you.
set -euo pipefail
_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$_DIR/../launch/lib.sh"   # NUC1_SSH/HOST, ZERORPC_ADDR, step/ok/warn/die

RESTART=1
KQ=""
PRESET=""
NUC_DIR="${NUC_CONTROLLER_DIR:-polymetis_fr3}"   # ~/polymetis_fr3 on NUC1
HW_YAML="$_DIR/conf/franka_hardware.yaml"

# Named default_Kq presets (J1..J7). Source of truth for --preset; keep the
# --help header table above in sync. Promote a good hand-tuned vector here.
declare -A KQ_PRESETS=(
  [stiff]="40 30 50 25 35 25 10"    # original — accurate tracking (collection + eval)
  [soft]="20 15 25 12 18 12 5"      # halved  — compliant, still decent tracking
  [softer]="15 12 20 10 12 8 4"     # current — most compliant (GELLO calib / teleop)
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-restart) RESTART=0; shift ;;
    --kq) KQ="${2:-}"; shift 2 ;;
    --preset) PRESET="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
    *) die "unknown arg: $1 (see --help)" ;;
  esac
done

# Resolve a named preset into KQ (mutually exclusive with --kq).
if [[ -n "$PRESET" ]]; then
  [[ -n "$KQ" ]] && die "use either --preset or --kq, not both"
  KQ="${KQ_PRESETS[$PRESET]:-}"
  [[ -n "$KQ" ]] || die "unknown --preset '$PRESET' (choices: ${!KQ_PRESETS[*]})"
  ok "preset '$PRESET' -> default_Kq [$KQ]"
fi

# --- optional: set joint stiffness in the repo config before deploying --------
if [[ -n "$KQ" ]]; then
  read -r -a _vals <<<"$KQ"
  [[ ${#_vals[@]} -eq 7 ]] || die "--kq needs 7 space-separated values, got ${#_vals[@]}"
  arr="[$(printf '%s, ' "${_vals[@]}" | sed 's/, $//')]"
  if [[ -z "$LAUNCH_DRY_RUN" ]]; then
    sed -i -E "s/^( *default_Kq:).*/\1 $arr/" "$HW_YAML" \
      || die "failed to set default_Kq in $HW_YAML"
    ok "set default_Kq = $arr in $HW_YAML (commit or revert as you like)"
  else
    echo "DRY: sed default_Kq -> $arr in $HW_YAML"
  fi
fi

# --- push config to the NUC ---------------------------------------------------
step "Deploy controller config -> $NUC1_SSH:~/$NUC_DIR/"
if [[ -z "$LAUNCH_DRY_RUN" ]]; then
  scp "$_DIR/conf/franka_hardware.yaml" "$_DIR/conf/franka_panda.yaml" \
      "$NUC1_SSH:~/$NUC_DIR/conf/" \
    || die "scp conf failed — is $NUC1_SSH reachable on the robot net? pw?"
  scp "$_DIR/docker-compose.yaml" "$_DIR/parameters.py" \
      "$NUC1_SSH:~/$NUC_DIR/" \
    || die "scp compose/parameters failed"
  ok "config copied"
else
  echo "DRY: scp conf/*.yaml docker-compose.yaml parameters.py -> $NUC1_SSH:~/$NUC_DIR/"
fi

# --- restart the controller so it re-reads the config -------------------------
if [[ "$RESTART" == 0 ]]; then
  ok "files copied, no restart requested. Apply later: restart droid-nuc-fr3 on NUC1."
  exit 0
fi

step "Restart droid-nuc-fr3 (re-reads the mounted config)"
if [[ -z "$LAUNCH_DRY_RUN" ]]; then
  ssh -t "$NUC1_SSH" "cd ~/$NUC_DIR && docker compose down && docker compose up -d" \
    || die "controller restart failed on $NUC1_SSH"
else
  echo "DRY: ssh $NUC1_SSH 'cd ~/$NUC_DIR && docker compose down && docker compose up -d'"
fi

step "  wait for zerorpc server @ $ZERORPC_ADDR (server up; controller lazy until FCI)"
if [[ -z "$LAUNCH_DRY_RUN" ]]; then
  for i in $(seq 1 20); do
    if timeout 3 bash -c "cat </dev/null >/dev/tcp/$NUC1_HOST/4242" 2>/dev/null; then
      ok "zerorpc server reachable"; break
    fi
    [[ $i -eq 20 ]] && die "zerorpc never opened $NUC1_HOST:4242 (container up?)"
    sleep 2
  done
fi

ok "controller restarted with the new parameters."
warn "FCI dropped on restart — re-Activate FCI in Desk (https://172.16.0.1), then
     Recover + Home in the dashboard, or run:  sudo $TASL/launch/teleop.sh"
