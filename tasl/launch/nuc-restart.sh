#!/usr/bin/env bash
# Recover a wedged NUC1 polymetis controller (symptom: portal shows robot +
# gripper DOWN, Go home does nothing, every zerorpc call times out — the
# single-threaded run_server.py is stuck in a blocking call). Restarts the
# droid-nuc-fr3 container and re-bootstraps the driver. FCI must be ACTIVE in
# Desk. Run as the desktop user (no sudo): uses ~/.ssh/id_ed25519_frankanuc.
# If the container was removed (docker compose down), falls back to
# `docker compose up -d` in ~/polymetis_fr3 — same as start_openpi_full.sh
# stage 1 minus the `systemctl stop franka-robot-server` (needs sudo).
set -euo pipefail
_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUC1_HOST="${NUC1_HOST:-172.16.0.2}"
KEY="${NUC1_SSH_KEY:-$HOME/.ssh/id_ed25519_frankanuc}"
echo "▶ restarting droid-nuc-fr3 on $NUC1_HOST"
ssh -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 "tasl@$NUC1_HOST" \
  'if docker inspect droid-nuc-fr3 >/dev/null 2>&1; then docker restart droid-nuc-fr3 >/dev/null; else echo "  container missing → docker compose up -d in ~/polymetis_fr3"; cd ~/polymetis_fr3 && docker compose up -d >/dev/null; fi && sleep 3 && docker ps --format "  {{.Names}} {{.Status}}"'
for i in $(seq 1 20); do
  timeout 2 bash -c "cat < /dev/null > /dev/tcp/$NUC1_HOST/4242" 2>/dev/null && { echo "✓ zerorpc :4242 open"; break; }
  [[ $i -eq 20 ]] && { echo "✗ :4242 never opened"; exit 1; }
  sleep 2
done
echo "▶ bootstrap driver (needs FCI active)"
PYTHONPATH="$_DIR/..:$HOME/.local/lib/python3.10/site-packages" timeout 120 /usr/bin/python3 - <<PY 2>&1 | grep -v DeprecationWarning
from clients.droid_client import DroidLikeClient
c = DroidLikeClient(address="tcp://$NUC1_HOST:4242", timeout=60)
c.bootstrap()
s = c.get_robot_state()
print("✓ robot state OK, joints:", [round(float(x), 3) for x in s["joint_positions"]], "grip:", round(float(s["gripper_position"]), 3))
PY
echo "done — refresh the portal; it reconnects by itself."
