#!/usr/bin/env bash
# Stop FR3 dashboards cleanly (SIGTERM — never -9 a ZED holder) + reap stale.
#
# RUN THIS ON THE DESKTOP (TASL-1):  ./stop.sh [collect|openpi|all]   (default: all)
# It re-runs itself under sudo (dashboards run as root).
set -euo pipefail
_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$_DIR/lib.sh"

ensure_root "$0" "$@"

what="${1:-all}"

stop_collect() {
  step "Stop collect dashboard + reap collection procs"
  desk "pkill -TERM -f '[/_]collect[.]py' || true"
  desk "docker exec rlinf-eval bash -lc 'pkill -9 -f \"[r]ay::DataCollector\"; pkill -9 -f \"[c]ollect_real_data\"; pkill -9 -f \"[P]olymetisController\"; /opt/venv/openpi/bin/ray stop --force 2>/dev/null; rm -rf /tmp/ray; true' || true"
  ok "collect stopped"
}
stop_openpi() {
  step "Stop openpi dashboard"
  desk "pkill -TERM -f '[/_]openpi[.]py' || true"
  ok "openpi stopped"
}

case "$what" in
  collect) stop_collect ;;
  openpi)  stop_openpi ;;
  all)     stop_collect; stop_openpi ;;
  *) die "usage: stop.sh [collect|openpi|all]" ;;
esac
ok "done"
