#!/usr/bin/env bash
# Single-entry RLinf pi05_droid EVAL launcher. RUN ON THE DESKTOP (TASL-1):
#
#     ./eval.sh                   # re-execs itself under sudo (cameras + uinput)
#
# This is the MIRROR-OPPOSITE of teleop.sh. Teleop/collect drive the polymetis
# zerorpc container (backend B). The RLinf pi05 eval env (FrankaJointVelEnv /
# realworld_chunk_native) instead talks HTTP to the franky **robot_server.py**
# (backend A). Both bind NUC1 :4242 and are mutually exclusive, so this script
# brings backend A UP and takes backend B DOWN.
#
# Cold → "ready to eval" in four stages:
#
#   1. NUC1 backend   — ssh over the ROBOT network ($NUC1_SSH): `docker compose
#                       down` the polymetis container (release :4242) + start the
#                       franky robot_server (HTTP :4242). Wait until /state answers.
#   2. MANUAL FCI GATE — you unlock joints + Activate FCI in Desk
#                       (https://172.16.0.1, reachable from this Desktop), Enter.
#   3. Eval dashboard :8003 — free ZEDs, launch the RLinf eval dashboard
#                       (rlinf.py --mode host). It OWNS the cameras and serves
#                       frames over HTTP to the in-container env (no cam handoff).
#   4. Bootstrap + AUTO-HOME — recover the franky controller, then home to the
#                       DROID anchor pose, via the dashboard (proxies to :4242).
#
# Then: open :8003, type a task prompt, press Start to run one eval episode.
# The eval itself runs `eval_embodied_agent.py --config-name realworld_eval_pi05_droid`
# via `docker exec rlinf-eval` against the mounted checkout at /workspace/rlinf.
#
# Everything talks to NUC1 over the robot net by default ($NUC1_HOST=172.16.0.2).
# Override: NUC1_HOST=… ./eval.sh   /   LAUNCH_DRY_RUN=1 ./eval.sh (no-op).
set -euo pipefail
_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$_DIR/lib.sh"

DASH="http://127.0.0.1:8003"
RS_URL="http://$NUC1_HOST:4242"            # franky robot_server HTTP (backend A)
EVAL_CONFIG="realworld_eval_pi05_droid"
# Config + env code the eval needs. They live ONLY on the eval branch
# (franka-fr3/rlinf-pi05-droid-eval / rtc-pi05-droid-deploy), NOT on the
# polymetis collect branch. The container mounts work/rlinf-clone → /workspace/rlinf.
export NUC1_HOST

ensure_root "$0" "$@"

# ── Stage 1/4 — NUC1 controller backend: franky robot_server UP (over robot net) ─
step "Stage 1/4 — NUC1 backend: franky robot_server (HTTP :4242) over robot net ($NUC1_SSH)"
if [[ -z "$LAUNCH_DRY_RUN" ]]; then
  ssh -t "$NUC1_SSH" 'set -e
    echo "[NUC1] bring DOWN the polymetis container (FCI exclusivity — releases :4242)"
    cd ~/polymetis_fr3 && docker compose down 2>/dev/null || true
    echo "[NUC1] start franky robot_server (the systemd unit we keep disabled at boot)"
    sudo systemctl start franka-robot-server' \
    || die "NUC1 bring-up failed — is $NUC1_SSH reachable on the robot net? sudo pw? Is franka-robot-server installed?"
else
  echo "DRY: ssh -t $NUC1_SSH 'cd ~/polymetis_fr3 && docker compose down; sudo systemctl start franka-robot-server'"
fi

step "  waiting for franky robot_server HTTP @ $RS_URL/state"
if [[ -n "$LAUNCH_DRY_RUN" ]]; then
  echo "DRY: curl $RS_URL/state"
else
  ok_rs=0
  for i in $(seq 1 20); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "$RS_URL/state" 2>/dev/null || true)
    # robot_server may answer /state (200) OR only /ping before FCI; accept any
    # real HTTP response (not 000) as "server is up".
    if [[ "$code" =~ ^[0-9]+$ && "$code" -ne 000 ]]; then ok "robot_server up (HTTP $code on /state)"; ok_rs=1; break; fi
    [[ $i -eq 20 ]] && die "franky robot_server never answered $RS_URL/state.
      On NUC1: 'systemctl status franka-robot-server' + 'journalctl -u franka-robot-server -n 50'.
      NOTE: this backend (franky) was set up but not verified end-to-end — it may need fixing."
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

# ── Stage 3/4 — Desktop eval dashboard (rlinf.py, host mode) ────────────────
step "Stage 3/4 — eval dashboard :8003 (rlinf.py --mode host)"
step "  rlinf-eval container up"
desk "docker start rlinf-eval"

step "  preflight: eval config + chunk env present in the mounted checkout (/workspace/rlinf)"
if [[ -z "$LAUNCH_DRY_RUN" ]]; then
  miss=$(docker exec rlinf-eval bash -lc '
    for p in examples/embodiment/config/'"$EVAL_CONFIG"'.yaml \
             examples/embodiment/config/env/realworld_franka_jointvel.yaml \
             rlinf/envs/realworld/realworld_chunk_env.py \
             rlinf/models/embodiment/openpi/dataconfig/droid_dataconfig.py; do
      [ -e "/workspace/rlinf/$p" ] || echo "$p"
    done' 2>/dev/null || echo "DOCKER_EXEC_FAILED")
  if [[ -n "$miss" ]]; then
    die "the mounted checkout (/workspace/rlinf = work/rlinf-clone) is MISSING eval code:
$(printf '      - %s\n' $miss)
    The eval lives on branch franka-fr3/rlinf-pi05-droid-eval (a full checkout already
    exists at ~/work/rlinf-clone-rtc-old). Point the container at the eval checkout
    before running eval — e.g. remount rlinf-eval's /workspace/rlinf at that path, or
    check out the eval branch in work/rlinf-clone. (See docs/EVAL.md.)"
  fi
  ok "eval code present in container"
else
  echo "DRY: docker exec rlinf-eval verify $EVAL_CONFIG.yaml + realworld_chunk_env.py present"
fi

step "  checkpoint /ckpts/pi05_droid_pt mounted"
desk "docker exec rlinf-eval bash -lc 'test -d /ckpts/pi05_droid_pt' || { echo MISSING; exit 1; }" \
  || die "/ckpts/pi05_droid_pt not in the container — the pi05_droid PyTorch checkpoint mount is missing."

step "  mutual exclusion: stop the other dashboards (openpi shares :8003; collect holds cams)"
desk "pkill -TERM -f '[/_]openpi[.]py'  || true"
desk "pkill -TERM -f '[/_]collect[.]py' || true"
desk "pkill -TERM -f '[/_]rlinf[.]py'   || true"
[[ -z "$LAUNCH_DRY_RUN" ]] && sleep 2

step "  cameras: both ZEDs free ($ZED_EXTERIOR exterior, $ZED_WRIST wrist)"
zeds_free || die "ZEDs not both free/present — reseat the exterior 2i USB; ensure no dashboard/eval holds them."

step "  reap stale eval procs in rlinf-eval"
desk "docker exec rlinf-eval bash -lc 'pkill -9 -f eval_embodied_agent.py; pkill -9 -f \"[r]ay::\"; pkill -9 -f \"[r]aylet\"; /opt/venv/openpi/bin/ray stop --force 2>/dev/null; rm -rf /tmp/ray; true'"

step "  launch eval dashboard :8003 (--mode host --robot-server $RS_URL)"
desk "cd $TASL && ulimit -n 8192 && PYTHONPATH=$TASL:$SITE_PKGS NUC1_HOST=$NUC1_HOST setsid /usr/bin/python3 dashboards/rlinf.py --port 8003 --mode host --robot-server $RS_URL </dev/null >> $TASL/logs/eval.log 2>&1 &"
wait_http 8003 || die "eval dashboard did not answer on :8003 (see $TASL/logs/eval.log)"
ok "dashboard up — http://$TS_IP:8003 (laptop, via Tailscale; robot-net IP if TS offline)"

# ── Stage 4/4 — bootstrap controller + auto-home ────────────────────────────
step "Stage 4/4 — recover franky controller + auto-home"
step "  recover: clear reflexes / re-arm the controller against the live FCI"
desk "curl -fsS -X POST $DASH/recover >/dev/null" \
  || warn "recover call failed — if Home below also fails, check FCI active + brakes off in Desk."
step "  home: move to the DROID anchor pose (via robot_server /move/joint)"
if [[ -z "$LAUNCH_DRY_RUN" ]]; then
  resp=$(curl -fsS -X POST "$DASH/home") || die "home request failed (dashboard :8003)"
  echo "    $resp"
  grep -q '"ok": *true' <<<"$resp" \
    || warn "home did NOT converge — check Desk / E-stop, then press Recover in the UI before eval."
else
  echo "DRY: curl -X POST $DASH/home"
fi

ok "READY — open http://$TS_IP:8003, type a task prompt, press Start to run an eval episode.
     SAFETY: first rollout is autonomous (neural net drives the arm). Hand on the
     E-stop, start from home, watch the first chunks before trusting it."
