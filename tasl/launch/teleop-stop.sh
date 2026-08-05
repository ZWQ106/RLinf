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
#   ./teleop-stop.sh --keep-dashboard   # everything except stopping collect.py
#   LAUNCH_DRY_RUN=1 ./teleop-stop.sh   # print actions, change nothing
#
# --keep-dashboard exists so the collect dashboard can invoke this script on
# itself (the 结束会话 button): stage 1 would otherwise SIGTERM the very process
# serving the request, and the operator would lose the UI mid-teardown.
set -euo pipefail
_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$_DIR/lib.sh"

KEEP_DASHBOARD=""
for _arg in "$@"; do
  case "$_arg" in
    --keep-dashboard) KEEP_DASHBOARD=1 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) die "unknown arg: $_arg (see --help)" ;;
  esac
done

ensure_root "$0" "$@"

# ── 1. Desktop dashboards (SIGTERM — never -9 a ZED holder) ──────────────────
if [[ -n "$KEEP_DASHBOARD" ]]; then
  step "1/3 — stop the OTHER dashboard only (--keep-dashboard: collect stays up)"
  desk "pkill -TERM -f '[/_]openpi[.]py' || true"
else
  step "1/3 — stop Desktop dashboards (collect + openpi)"
  desk "pkill -TERM -f '[/_]collect[.]py' || true"
  desk "pkill -TERM -f '[/_]openpi[.]py' || true"
fi
[[ -z "$LAUNCH_DRY_RUN" ]] && sleep 2   # let them release the cameras on the way out

# ── 2. In-container collection procs (release ZEDs + GELLO) ──────────────────
step "2/3 — reap stale collection procs in rlinf-eval (frees ZEDs + GELLO)"
# ray stop prints a huge zombie dump on stdout (already-dead procs it can't
# re-kill) — silenced inside the helper; zombies are harmless and clear on the
# next rlinf-eval (re)start. The pkills are what actually free the ZEDs + GELLO.
reap_collection_procs

# ── 3. NUC1 controller down over the robot net (releases FCI) ────────────────
step "3/3 — bring NUC1 controller down ($NUC1_SSH) — releases FCI"
if [[ -z "$LAUNCH_DRY_RUN" ]]; then
  nuc_ssh 'cd ~/polymetis_fr3 && docker compose down' \
    || warn "NUC1 teardown command failed (already down? ssh unreachable?) — verifying…"
  # VERIFY FCI actually released: the controller's :4242 must close. Don't just
  # assume — a silent compose-down failure leaves the robot locked to FCI.
  sleep 2
  if timeout 3 bash -c "cat < /dev/null > /dev/tcp/$NUC1_HOST/4242" 2>/dev/null; then
    warn "controller :4242 STILL OPEN — FCI NOT released. The NUC container didn't stop. Fix on NUC1:
      ssh $NUC1_SSH 'docker ps | grep droid ; cd ~/polymetis_fr3 && docker compose down'"
  else
    ok "controller down — FCI released"
  fi
else
  echo "DRY: ssh -t $NUC1_SSH 'cd ~/polymetis_fr3 && docker compose down' + verify :4242 closed"
fi

# ── verify ──────────────────────────────────────────────────────────────────
if [[ -z "$LAUNCH_DRY_RUN" && -z "$KEEP_DASHBOARD" ]]; then
  if ss -ltn 2>/dev/null | grep -q ':8004'; then
    warn ":8004 STILL listening — a dashboard didn't exit. Check: ps -eo pid,user,args | grep collect.py"
  else
    ok ":8004 free"
  fi
fi

if [[ -n "$KEEP_DASHBOARD" ]]; then
  ok "SESSION STOPPED — FCI released, cameras + GELLO free. Dashboard still up on :8004.
   Re-lock the joints in Desk when done; run teleop.sh to start a new session."
else
  ok "TELEOP STOPPED — FCI released. Re-lock joints in Desk when done; run teleop.sh to restart."
fi
