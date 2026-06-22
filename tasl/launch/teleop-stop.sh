#!/usr/bin/env bash
# Crash-safe teleop teardown — reverses teleop.sh. RUN ON THE DESKTOP (TASL-1):
#
#     ./teleop-stop.sh
#
# Does three things, all idempotent (safe to run anytime, even half-up):
#   1. Stop the Desktop dashboard(s)        — SIGTERM (lets them release the ZEDs).
#   2. Reap stale collection procs in        — collect_real_data / PolymetisController
#      rlinf-eval (frees ZEDs + GELLO)         / ray, the things a crashed run leaks.
#   3. Bring the NUC1 controller down        — `docker compose down` over the robot net,
#      (RELEASES FCI)                           which frees the FR3's FCI lock.
#
# Leaves the franky robot_server STOPPED and does NOT touch Desk — re-lock the
# joints / re-Activate FCI manually next time (or just run teleop.sh again).
#
#   LAUNCH_DRY_RUN=1 ./teleop-stop.sh   # print actions, change nothing
set -euo pipefail
_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$_DIR/lib.sh"

ensure_root "$0" "$@"

# ── 1. Desktop dashboards (SIGTERM — never -9 a ZED holder) ──────────────────
step "1/3 — stop Desktop dashboards (collect + openpi)"
desk "pkill -TERM -f '[/_]collect[.]py' || true"
desk "pkill -TERM -f '[/_]openpi[.]py' || true"
[[ -z "$LAUNCH_DRY_RUN" ]] && sleep 2   # let them release the cameras on the way out

# ── 2. In-container collection procs (release ZEDs + GELLO) ──────────────────
step "2/3 — reap stale collection procs in rlinf-eval (frees ZEDs + GELLO)"
desk "docker exec rlinf-eval bash -lc 'pkill -9 -f \"[r]ay::DataCollector\"; pkill -9 -f \"[c]ollect_real_data\"; pkill -9 -f \"[P]olymetisController\"; /opt/venv/openpi/bin/ray stop --force 2>/dev/null; rm -rf /tmp/ray; true' || true"

# ── 3. NUC1 controller down over the robot net (releases FCI) ────────────────
step "3/3 — bring NUC1 controller down ($NUC1_SSH) — releases FCI"
if [[ -z "$LAUNCH_DRY_RUN" ]]; then
  ssh -t "$NUC1_SSH" 'cd ~/polymetis_fr3 && docker compose down' \
    || warn "NUC1 teardown failed (already down? ssh unreachable?) — FCI may still be held; check Desk."
else
  echo "DRY: ssh -t $NUC1_SSH 'cd ~/polymetis_fr3 && docker compose down'"
fi

# ── verify ──────────────────────────────────────────────────────────────────
if [[ -z "$LAUNCH_DRY_RUN" ]]; then
  if ss -ltn 2>/dev/null | grep -q ':8004'; then
    warn ":8004 STILL listening — a dashboard didn't exit. Check: ps -eo pid,user,args | grep collect.py"
  else
    ok ":8004 free"
  fi
fi

ok "TELEOP STOPPED — FCI released. Re-lock joints in Desk when done; run teleop.sh to restart."
