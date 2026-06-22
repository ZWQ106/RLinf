#!/usr/bin/env bash
# Single-entry GELLO teleop launcher. RUN ON THE DESKTOP (TASL-1):
#
#     ./teleop.sh                 # re-execs itself under sudo (cameras + uinput)
#
# Takes you from cold to "ready to record" in four stages:
#
#   1. NUC1 backend   — ssh over the ROBOT network ($NUC1_SSH): stop the franky
#                       robot_server (FCI exclusivity) + `docker compose up` the
#                       droid-nuc-fr3 polymetis container (zerorpc :4242, lazy).
#   2. MANUAL FCI GATE — you unlock joints + Activate FCI in Desk
#                       (https://172.16.0.1, reachable from this Desktop), Enter.
#   3. Collect dashboard :8004 — free ZEDs, launch the GELLO teleop dashboard
#                       (collect_real_data.py polymetis_jointvel, /dev/gello).
#   4. Bootstrap + AUTO-HOME — recover (launch the controller against the now
#                       -live FCI), then leashed home to the DROID anchor pose.
#
# Then: open :8004, move the GELLO leader, press Start to record an episode.
#
# Everything talks to NUC1 over the robot net by default ($NUC1_HOST=172.16.0.2).
# Override: NUC1_HOST=… ./teleop.sh   /   LAUNCH_DRY_RUN=1 ./teleop.sh (no-op).
set -euo pipefail
_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$_DIR/lib.sh"

DASH="http://127.0.0.1:8004"
export NUC1_HOST          # so the dashboard (collect.py) inherits the same host

ensure_root "$0" "$@"

# ── Stage 1/4 — NUC1 controller backend (over robot net) ────────────────────
step "Stage 1/4 — NUC1 backend over robot net ($NUC1_SSH)"
if [[ -z "$LAUNCH_DRY_RUN" ]]; then
  ssh -t "$NUC1_SSH" 'set -e
    echo "[NUC1] stop franky robot_server (FCI exclusivity)"
    sudo systemctl stop franka-robot-server 2>/dev/null || true
    echo "[NUC1] bring up droid-nuc-fr3 polymetis container"
    cd ~/polymetis_fr3 && docker compose up -d' \
    || die "NUC1 bring-up failed — is $NUC1_SSH reachable on the robot net? sudo pw?"
else
  echo "DRY: ssh -t $NUC1_SSH 'sudo systemctl stop franka-robot-server; cd ~/polymetis_fr3 && docker compose up -d'"
fi

step "  waiting for zerorpc server @ $ZERORPC_ADDR (server up; controller stays lazy until FCI)"
if [[ -n "$LAUNCH_DRY_RUN" ]]; then
  echo "DRY: tcp connect probe $NUC1_HOST:4242"
else
  for i in $(seq 1 20); do
    if timeout 3 bash -c "cat < /dev/null > /dev/tcp/$NUC1_HOST/4242" 2>/dev/null; then
      ok "zerorpc server reachable"; break
    fi
    [[ $i -eq 20 ]] && die "zerorpc server never opened $NUC1_HOST:4242 (container up?)"
    sleep 2
  done
fi

# ── Stage 2/4 — manual FCI gate ─────────────────────────────────────────────
warn "Stage 2/4 — ACTIVATE FCI MANUALLY (Desk is reachable from this Desktop)"
cat <<'EOF'

    ┌────────────────────────────────────────────────────────────┐
    │  1. Open Franka Desk:   https://172.16.0.1                  │
    │  2. Unlock the joints   (brakes click open)                │
    │  3. Activate FCI  +  put the robot in execution mode        │
    └────────────────────────────────────────────────────────────┘
EOF
if [[ -z "$LAUNCH_DRY_RUN" ]]; then
  read -r -p "  Press Enter once FCI is ACTIVE and the brakes are OFF… " _
else
  echo "DRY: (would block for Enter — manual FCI gate)"
fi

# ── Stage 3/4 — Desktop collect dashboard (GELLO teleop) ────────────────────
step "Stage 3/4 — collect dashboard :8004"
step "  rlinf-eval container up"
desk "docker start rlinf-eval"
step "  mutual exclusion: stop the other dashboard"
kill_other_dashboard collect
step "  cameras: both ZEDs free ($ZED_EXTERIOR exterior, $ZED_WRIST wrist)"
zeds_free || die "ZEDs not both free/present — reseat the exterior 2i USB; ensure no dashboard/collection holds them."
step "  reap stale collection procs in rlinf-eval"
desk "docker exec rlinf-eval bash -lc 'pkill -9 -f \"[r]ay::DataCollector\"; pkill -9 -f \"[c]ollect_real_data\"; pkill -9 -f \"[P]olymetisController\"; /opt/venv/openpi/bin/ray stop --force 2>/dev/null; rm -rf /tmp/ray; true'"
step "  stop any existing collect dashboard (idempotent)"
desk "pkill -TERM -f '[/_]collect[.]py' || true"
[[ -z "$LAUNCH_DRY_RUN" ]] && sleep 2
step "  launch collect dashboard :8004 (--no-cam-on-start)"
desk "cd $TASL && ulimit -n 8192 && PYTHONPATH=$TASL:$SITE_PKGS NUC1_HOST=$NUC1_HOST setsid /usr/bin/python3 dashboards/collect.py --port 8004 --no-cam-on-start </dev/null >> $TASL/logs/collect.log 2>&1 &"
wait_http 8004 || die "collect dashboard did not answer on :8004 (see $TASL/logs/collect.log)"
step "  vkbd handoff: docker restart rlinf-eval (rebind uinput 'c'/'s' listener)"
desk "docker restart rlinf-eval"
ok "dashboard up on :8004"

# ── Stage 4/4 — bootstrap controller + auto-home ────────────────────────────
step "Stage 4/4 — bootstrap controller (recover) + auto-home"
step "  recover: launch the controller against the now-live FCI (~10s)"
desk "curl -fsS -X POST $DASH/api/robot/recover >/dev/null" \
  || die "recover failed — controller could not bootstrap. Is FCI active + brakes off in Desk?"
step "  home: leashed move to the DROID anchor pose"
if [[ -z "$LAUNCH_DRY_RUN" ]]; then
  resp=$(curl -fsS -X POST "$DASH/api/robot/home") || die "home request failed (dashboard :8004)"
  echo "    $resp"
  grep -q '"ok": *true' <<<"$resp" \
    || warn "home did NOT converge — check Desk / E-stop, then press Recover in the UI before teleop."
else
  echo "DRY: curl -X POST $DASH/api/robot/home"
fi

ok "READY — open http://$TS_IP:8004 (laptop, via Tailscale), move the GELLO leader, press Start to record."
