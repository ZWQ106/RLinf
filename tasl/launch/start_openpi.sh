#!/usr/bin/env bash
# Bring up the FR3 openpi-eval dashboard (:8003) and its full chain.
#
# RUN THIS ON THE DESKTOP (TASL-1). No prior context needed:
#     ./start_openpi.sh             # it re-runs itself under sudo
# It checks the robot, guards the GPU, ensures the openpi serve_policy (:8000)
# is up, and launches the eval dashboard. Then open the Tailscale URL printed at the end.
#
# Prereqs it does NOT do for you (it will tell you if missing):
#   • FR3 powered + FCI active in Desk
#   • NUC1 droid-nuc-fr3 container up (zerorpc :4242)  — see the error if down
set -euo pipefail
_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$_DIR/lib.sh"

ensure_root "$0" "$@"

preflight_robot openpi

step "Backend: serve_policy :8000"
if port_open 8000; then
  ok "serve_policy already up on :8000 (its GPU use is expected — no guard needed)"
else
  step "  :8000 down — GPU guard before starting serve (serve + training co-resident froze the box)"
  gpu_free || die "a compute process is using the Desktop GPU — do NOT start serve_policy on top of training (it froze the box). Stop training (or run it on the cluster), then retry."
  ok "GPU free — starting serve_policy (pi05_droid)"
  desk "cd $DESKTOP_HOME/work/openpi && setsid .venv/bin/python scripts/serve_policy.py --port 8000 policy:checkpoint --policy.config=pi05_droid --policy.dir=$DESKTOP_HOME/.cache/openpi/openpi-assets/checkpoints/pi05_droid </dev/null >> $DESKTOP_HOME/_serve_policy.log 2>&1 &"
  wait_http 8000 60 || die "serve_policy did not come up on :8000 within ~2min (check $DESKTOP_HOME/_serve_policy.log)"
  ok "serve_policy up on :8000"
fi

step "Stop any existing openpi dashboard (idempotent re-run)"
desk "pkill -TERM -f '[/_]openpi[.]py' || true"
if [[ -z "$LAUNCH_DRY_RUN" ]]; then sleep 2; fi

step "Launch openpi dashboard :8003 (--policy-host 127.0.0.1 --policy-port 8000)"
desk "cd $TASL && ulimit -n 8192 && PYTHONPATH=$TASL:$SITE_PKGS:$DESKTOP_HOME/work/openpi/packages/openpi-client/src setsid /usr/bin/python3 dashboards/openpi.py --port 8003 --policy-host 127.0.0.1 --policy-port 8000 </dev/null >> $TASL/logs/openpi.log 2>&1 &"
wait_http 8003 || die "openpi dashboard did not answer on :8003 (see $TASL/logs/openpi.log)"
ok "READY — openpi dashboard at http://$TS_IP:8003 (Tailscale; robot-net IP if TS offline)"
