#!/usr/bin/env bash
# ONE-SHOT cold start for the pi05_droid inference portal.
#
#   ~/RLinf/tasl/launch/start_openpi_full.sh [steer-config]   # normal terminal
#
#   steer-config (default: capture_fig3b) — a YAML name from
#   ../VLA-PatchLen-cp/configs/. The HOOKED PyTorch server is started with it,
#   so every rollout stores per-episode activations + io data under
#   ../VLA-PatchLen-cp/rollouts/ (one ep* subdir per dashboard Start press).
#   Pass `vanilla-jax` to get the old unhooked JAX serve_policy instead.
#
# Chains the stages that were previously separate:
#   Stage 1  NUC1 controller  — stop franky robot_server, docker-compose up the
#            droid-nuc-fr3 polymetis container (zerorpc :4242). Skipped if
#            :4242 already answers. Prompts for the NUC sudo password.
#   Stage 2  MANUAL FCI GATE  — you confirm joints unlocked + FCI active in
#            Desk (https://172.16.0.1), then press Enter.
#   Stage 3  hooked serve_patched.sh on :8000 (capture per steer-config), then
#            start_openpi.sh — robot preflight + dashboard :8003 (sees :8000
#            up, skips its own serve). Re-execs under sudo (Desktop password).
#
# Then open the printed URL, type the task prompt in the portal, press Start.
# SAFETY: first rollout is autonomous — hand on the E-stop.
set -euo pipefail
_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="$_DIR/../VLA-PatchLen-cp"
STEER_CFG="${STEER_CONFIG:-${1:-capture_fig3b}}"

# Real user's home even when invoked under sudo (else $HOME=/root and the
# NUC ssh key + ~/.local zerorpc are not found).
DESKTOP_HOME="$(getent passwd "${SUDO_USER:-$USER}" | cut -d: -f6)"
DESKTOP_HOME="${DESKTOP_HOME:-$HOME}"
NUC1_HOST="${NUC1_HOST:-172.16.0.2}"
NUC1_SSH="${NUC1_SSH:-tasl@$NUC1_HOST}"
NUC1_SSH_KEY="${NUC1_SSH_KEY:-$DESKTOP_HOME/.ssh/id_ed25519_frankanuc}"

step() { printf '\033[36m▶ %s\033[0m\n' "$*"; }
ok()   { printf '\033[32m✓ %s\033[0m\n' "$*"; }
die()  { printf '\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

port4242_open() {
  timeout 3 bash -c "cat < /dev/null > /dev/tcp/$NUC1_HOST/4242" 2>/dev/null
}

# ── Stage 1/3 — NUC1 polymetis controller ───────────────────────────────────
if port4242_open; then
  ok "Stage 1/3 — zerorpc :4242 already up on NUC1, skipping bring-up"
else
  step "Stage 1/3 — NUC1 controller bring-up ($NUC1_SSH; NUC sudo password: tasl123456)"
  ssh -t -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new \
      -i "$NUC1_SSH_KEY" "$NUC1_SSH" \
      'sudo systemctl stop franka-robot-server 2>/dev/null; cd ~/polymetis_fr3 && docker compose up -d' \
    || die "NUC1 bring-up failed — is the NUC reachable at $NUC1_HOST?"
  step "  waiting for zerorpc :4242 (server binds lazily, container start takes a few s)"
  for i in $(seq 1 20); do
    port4242_open && { ok "zerorpc server reachable"; break; }
    [[ $i -eq 20 ]] && die ":4242 never opened — check 'docker ps' on the NUC"
    sleep 2
  done
fi

# ── Stage 2/3 — manual FCI gate ─────────────────────────────────────────────
step "Stage 2/3 — confirm FCI in Desk"
cat <<'EOF'
    ┌────────────────────────────────────────────────────────────┐
    │  Franka Desk:  https://172.16.0.1                          │
    │  1. joints UNLOCKED (brakes clicked open)                  │
    │  2. FCI ACTIVE + execution mode                            │
    └────────────────────────────────────────────────────────────┘
EOF
read -r -p "  Press Enter once FCI is ACTIVE… " _

# ── Stage 2.5 — bootstrap the polymetis driver (cold container) ─────────────
# A freshly started droid-nuc-fr3 container binds zerorpc but has no robot
# driver yet — get_robot_state raises until launch_controller/launch_robot run.
# Needs FCI active, hence after the gate. Idempotent (no-op when bootstrapped).
step "Stage 2.5 — bootstrap polymetis driver (idempotent, ~15s on cold start)"
PYTHONPATH="$_DIR/..:$DESKTOP_HOME/.local/lib/python3.10/site-packages" \
timeout 90 /usr/bin/python3 - <<PYEOF || die "bootstrap failed — is FCI really active in Desk?"
from clients.droid_client import DroidLikeClient
c = DroidLikeClient(address="tcp://$NUC1_HOST:4242", timeout=60)
c.bootstrap()
s = c.get_robot_state()
print("robot state OK, joints:", [round(float(x), 3) for x in s["joint_positions"]])
PYEOF
ok "controller bootstrapped"

# ── Stage 3/3 — policy server :8000 + dashboard :8003 ───────────────────────
if [[ "$STEER_CFG" == "vanilla-jax" ]]; then
  step "Stage 3/3 — start_openpi.sh (vanilla JAX serve, no capture)"
  exec bash "$_DIR/start_openpi.sh"
fi

port8000_open() {
  timeout 2 bash -c "cat < /dev/null > /dev/tcp/127.0.0.1/8000" 2>/dev/null
}

step "Stage 3/3 — hooked serve ($STEER_CFG) + start_openpi.sh"
if port8000_open; then
  die ":8000 is already serving (unknown config — could be the vanilla JAX server, no capture!). Stop it first: $_DIR/openpi-stop.sh, then re-run."
fi
[[ -f "$PATCH_DIR/configs/$STEER_CFG.yaml" ]] \
  || die "no such steer config: $PATCH_DIR/configs/$STEER_CFG.yaml"
# Extra args (past the config name) go to the serve, e.g.:
#   start_openpi_full.sh setting2_rr_inject --watch-reload
# (only shift the config name off if it actually came from $1 — with
# STEER_CONFIG set via env, $1 may already be an extra arg)
if [[ $# -gt 0 && "$1" == "$STEER_CFG" ]]; then shift; fi
"$PATCH_DIR/serve_patched.sh" "$STEER_CFG" "$@" &
step "  waiting for hooked serve on :8000 (ckpt load takes ~1-2 min)"
for i in $(seq 1 90); do
  curl -s -m 2 -o /dev/null "http://127.0.0.1:8000/healthz" && break
  [[ $i -eq 90 ]] && die ":8000 never came up — check $PATCH_DIR/logs/serve_patched.log"
  sleep 2
done
ok "hooked serve up on :8000 (captures → $PATCH_DIR/rollouts/, config $STEER_CFG)"

step "  start_openpi.sh (re-execs under sudo — Desktop password; sees :8000 up → dashboard only)"
exec bash "$_DIR/start_openpi.sh"
