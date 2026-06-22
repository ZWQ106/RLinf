#!/usr/bin/env bash
# Crash-safe RLinf-eval teardown — reverses eval.sh. RUN ON THE DESKTOP (TASL-1):
#
#     ./eval-stop.sh
#
# Mirror of teleop-stop.sh but for backend A (franky robot_server). Idempotent —
# safe to run anytime, even half-up:
#   1. Stop the Desktop eval dashboard (rlinf.py)  — SIGTERM (releases the ZEDs).
#   2. Reap stale eval procs in rlinf-eval         — eval_embodied_agent / ray,
#      (frees ZEDs)                                   the things a crashed run leaks.
#   3. Halt motion + stop franka-robot-server on    — /move/joint_velocity_stop, then
#      NUC1 (RELEASES FCI)                            `systemctl stop`; verify :4242 closed.
#
# Leaves the polymetis container DOWN and does NOT touch Desk — re-lock the
# joints / re-Activate FCI manually next time (or just run eval.sh again).
#
#   LAUNCH_DRY_RUN=1 ./eval-stop.sh   # print actions, change nothing
set -euo pipefail
_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$_DIR/lib.sh"

RS_URL="http://$NUC1_HOST:4242"
ensure_root "$0" "$@"

# ── 1. Halt any in-flight motion FIRST (a blocking chunk keeps moving) ───────
step "0/3 — halt in-flight motion on robot_server ($RS_URL)"
desk "curl -fsS --max-time 3 -X POST $RS_URL/move/joint_velocity_stop >/dev/null 2>&1 || true"
desk "curl -fsS --max-time 3 -X POST $RS_URL/stop                     >/dev/null 2>&1 || true"

# ── 1. Desktop eval dashboard (SIGTERM — never -9 a ZED holder) ──────────────
step "1/3 — stop Desktop eval dashboard (rlinf.py)"
desk "pkill -TERM -f '[/_]rlinf[.]py' || true"
[[ -z "$LAUNCH_DRY_RUN" ]] && sleep 2   # let it release the cameras on the way out

# ── 2. In-container eval procs (release ZEDs) ────────────────────────────────
step "2/3 — reap stale eval procs in rlinf-eval (frees ZEDs)"
desk "docker exec rlinf-eval bash -lc 'pkill -9 -f eval_embodied_agent.py; pkill -9 -f \"[r]ay::\"; pkill -9 -f \"[r]aylet\"; /opt/venv/openpi/bin/ray stop --force >/dev/null 2>&1; rm -rf /tmp/ray; true' || true"

# ── 3. NUC1 franky robot_server down over the robot net (releases FCI) ────────
step "3/3 — stop franka-robot-server ($NUC1_SSH) — releases FCI"
if [[ -z "$LAUNCH_DRY_RUN" ]]; then
  ssh -t "$NUC1_SSH" 'sudo systemctl stop franka-robot-server' \
    || warn "NUC1 teardown command failed (already down? ssh unreachable?) — verifying…"
  sleep 2
  if timeout 3 bash -c "cat < /dev/null > /dev/tcp/$NUC1_HOST/4242" 2>/dev/null; then
    warn "robot_server :4242 STILL OPEN — FCI NOT released. Fix on NUC1:
      ssh $NUC1_SSH 'sudo systemctl stop franka-robot-server ; ss -tlnp | grep :4242'"
  else
    ok "robot_server down — FCI released"
  fi
else
  echo "DRY: ssh -t $NUC1_SSH 'sudo systemctl stop franka-robot-server' + verify :4242 closed"
fi

# ── verify ──────────────────────────────────────────────────────────────────
if [[ -z "$LAUNCH_DRY_RUN" ]]; then
  if ss -ltn 2>/dev/null | grep -q ':8003'; then
    warn ":8003 STILL listening — a dashboard didn't exit. Check: ps -eo pid,user,args | grep rlinf.py"
  else
    ok ":8003 free"
  fi
fi

ok "EVAL STOPPED — FCI released. Re-lock joints in Desk when done; run eval.sh to restart."
