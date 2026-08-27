#!/usr/bin/env bash
# Pin the polymetis real-time threads on NUC1 to P-cores (bug-1 fix step 1,
# saved_demo/bug/BUGLOG.md §1.7). Live and reversible; no robot command is sent.
#   franka_panda_client FIFO-99 thread -> CPU 3, its other threads -> CPU 2
#   run_server (all threads)           -> CPUs 6-11
# Usage: tasl/launch/nuc-pin-rt.sh [--undo]      (undo = allow CPUs 0-19 again)
set -euo pipefail
NUC=${NUC1_SSH:-tasl@172.16.0.2}
# Works under sudo too (the dashboard runs as root): key lives in the desktop user's home.
OWNER_HOME=$( [ -n "${SUDO_USER:-}" ] && echo "/home/$SUDO_USER" || echo "$HOME" )
KEY=${NUC1_SSH_KEY:-$OWNER_HOME/.ssh/id_ed25519_frankanuc}
UNDO=${1:-}
ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "$NUC" "docker exec -i droid-nuc-fr3 python3 - '$UNDO'" <<'PY'
import os, sys
undo = sys.argv[1] == "--undo"
def cmd(p):
    try: return open(f"/proc/{p}/cmdline","rb").read().replace(b"\0",b" ").decode(errors="replace")
    except OSError: return ""
def rtprio(t):
    with open(f"/proc/{t}/stat") as f: return int(f.read().rsplit(")",1)[1].split()[37])
pids = [p for p in os.listdir("/proc") if p.isdigit()]
cl = [p for p in pids if "build/franka_panda_client" in cmd(p) and "sudo" not in cmd(p)]
sv = [p for p in pids if "build/run_server" in cmd(p) and "sudo" not in cmd(p)]
if len(cl) != 1 or len(sv) != 1: sys.exit(f"expected exactly one client/server, got {cl} {sv}")
ALL = set(range(os.cpu_count()))
for t in os.listdir(f"/proc/{cl[0]}/task"):
    cpus = ALL if undo else ({3} if rtprio(t) >= 99 else {2})
    os.sched_setaffinity(int(t), cpus)
    print(f"client tid {t:>8} rtprio={rtprio(t):>2} -> {sorted(os.sched_getaffinity(int(t)))}")
for t in os.listdir(f"/proc/{sv[0]}/task"):
    os.sched_setaffinity(int(t), ALL if undo else {6,7,8,9,10,11})
    print(f"server tid {t:>8} rtprio={rtprio(t):>2} -> {sorted(os.sched_getaffinity(int(t)))}")
print("undone" if undo else "pinned")
PY
