#!/usr/bin/env bash
# Bring up the FR3 data-collect dashboard (:8004) and its full chain.
#
# RUN THIS ON THE DESKTOP (TASL-1). No prior context needed:
#     ./start_collect.sh            # it re-runs itself under sudo
# It checks the robot, frees the cameras, (re)starts the rlinf-eval container,
# and launches the collect dashboard. Then open the Tailscale URL printed at the end.
#
# Prereqs it does NOT do for you (it will tell you if missing):
#   • FR3 powered + FCI active in Desk
#   • NUC1 droid-nuc-fr3 container up (zerorpc :4242)  — see the error if down
set -euo pipefail
_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$_DIR/lib.sh"

ensure_root "$0" "$@"

preflight_robot collect

step "Backend: rlinf-eval container up"
ensure_rlinf_container
ok "rlinf-eval up"

step "Reap stale collection processes in rlinf-eval"
desk "docker exec rlinf-eval bash -lc 'pkill -9 -f \"[r]ay::DataCollector\"; pkill -9 -f \"[c]ollect_real_data\"; pkill -9 -f \"[P]olymetisController\"; /opt/venv/openpi/bin/ray stop --force 2>/dev/null; rm -rf /tmp/ray; true'"
ok "clean slate"

step "Stop any existing collect dashboard (idempotent re-run)"
desk "pkill -TERM -f '[/_]collect[.]py' || true"
if [[ -z "$LAUNCH_DRY_RUN" ]]; then sleep 2; fi

step "Launch collect dashboard :8004 (--no-cam-on-start)"
desk "cd $TASL && ulimit -n 8192 && PYTHONPATH=$TASL:$SITE_PKGS setsid /usr/bin/python3 dashboards/collect.py --port 8004 --no-cam-on-start </dev/null >> $TASL/logs/collect.log 2>&1 &"
wait_http 8004 || die "collect dashboard did not answer on :8004 (see $TASL/logs/collect.log)"
ok "dashboard up — http://$TS_IP:8004 (laptop, via Tailscale; robot-net IP if TS offline)"

step "vkbd handoff: docker restart rlinf-eval so the in-container 's'/'c' listener binds the new uinput"
desk "docker restart rlinf-eval"
ok "vkbd handoff done"

ok "READY — collect dashboard at http://$TS_IP:8004 (Tailscale; robot-net IP if TS offline)"
