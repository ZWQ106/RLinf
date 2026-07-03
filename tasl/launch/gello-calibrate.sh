#!/usr/bin/env bash
# Convenient GELLO calibration + verification. RUN ON THE DESKTOP.
#
#   ./gello-calibrate.sh            # 2-pose calibrate -> write fr3_gello_config.py -> verify
#   ./gello-calibrate.sh --verify   # just re-verify the CURRENT config against the robot
#   ./gello-calibrate.sh --no-write # calibrate + print, do NOT edit the config
#
# Prereqs: the controller is live and the arm is homed (run teleop.sh first);
# don't press Start (keeps /dev/gello free). The arm MOVES to two poses
# (speed-bounded) — KEEP A HAND ON THE E-STOP.
#
# One run does it all: reaps any stale port holder (the stuck-process trap),
# drives to two poses for you to match the leader, solves BOTH sign + offset,
# writes them into fr3_gello_config.py, then re-homes and verifies the arm tracks
# the leader. No manual copy-paste.
set -euo pipefail
_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$_DIR/lib.sh"

ARGS="$*"

step "Reap stale GELLO/port holders in $RLINF_CONTAINER (avoids a busy /dev/gello)"
desk "docker exec $RLINF_CONTAINER bash -lc 'pkill -9 -f \"[c]alibrate_2pose\"; pkill -9 -f \"[g]ello_vs_robot\"; pkill -9 -f \"[D]ynamixel\"; sleep 1; true'"

warn "The arm will MOVE to two poses (speed-bounded). KEEP A HAND ON THE E-STOP."
step "GELLO calibrate/verify ${ARGS:+($ARGS)}"
desk "docker exec -e ZERORPC_ADDR=$ZERORPC_ADDR $RLINF_CONTAINER bash -lc 'cd /workspace/rlinf && PYTHONPATH=/workspace/rlinf /opt/venv/openpi/bin/python -m rlinf.envs.realworld.common.gello.calibrate_2pose $ARGS'"

ok "done. If calibration wrote fr3_gello_config.py, commit it. Then teleop-test:
     press Start in the dashboard and drive gently (E-stop ready)."
