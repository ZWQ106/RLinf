#!/usr/bin/env bash
# Single-entry RLinf pi05_droid EVAL launcher. RUN ON THE DESKTOP (TASL-1):
#
#     ./eval.sh                   # re-execs itself under sudo (cameras + uinput)
#
# SAME controller as teleop/collect: the polymetis droid-nuc-fr3 container over
# zerorpc (backend B). We use RLinf only for efficient pi05_droid inference +
# infra; the robot is driven by the proven polymetis joint-velocity streaming
# path (RealWorldEnv.chunk_step loops the predicted chunk at 15 Hz — exactly the
# GELLO collect path). NO franky robot_server.
#
# Cold → "ready to eval" in four stages (Stages 1–2 identical to teleop.sh):
#
#   1. NUC1 backend   — ssh over the ROBOT network ($NUC1_SSH): stop the franky
#                       robot_server (FCI exclusivity) + `docker compose up` the
#                       droid-nuc-fr3 polymetis container (zerorpc :4242, lazy).
#   2. MANUAL FCI GATE — you unlock joints + Activate FCI in Desk
#                       (https://172.16.0.1, reachable from this Desktop), Enter.
#   3. Eval dashboard :8003 — free ZEDs, launch the RLinf eval dashboard
#                       (rlinf.py --mode host). It OWNS the cameras and serves
#                       frames over HTTP to the in-container env (no cam handoff).
#   4. Bootstrap + AUTO-HOME — bootstrap the polymetis controller against the
#                       live FCI (zerorpc), then home to the DROID anchor pose.
#
# Then: open :8003, type a task prompt, press Start to run one eval episode. The
# eval runs `eval_embodied_agent.py --config-name realworld_eval_pi05_droid_polymetis`
# via `docker exec rlinf-eval` against the mounted checkout at /workspace/rlinf.
#
# Everything talks to NUC1 over the robot net by default ($NUC1_HOST=172.16.0.2).
# Override: NUC1_HOST=… ./eval.sh   /   LAUNCH_DRY_RUN=1 ./eval.sh (no-op).
set -euo pipefail
_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$_DIR/lib.sh"

DASH="http://127.0.0.1:8003"
EVAL_CONFIG="realworld_eval_pi05_droid_polymetis"
export NUC1_HOST          # so the dashboard (rlinf.py) inherits the same host

ensure_root "$0" "$@"

# ── Stage 1/4 — NUC1 controller backend: polymetis container UP (over robot net) ─
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

# ── Stage 3/4 — Desktop eval dashboard (rlinf.py, host mode) ────────────────
step "Stage 3/4 — eval dashboard :8003 (rlinf.py --mode host)"
step "  rlinf-eval container up"
ensure_rlinf_container

step "  preflight: polymetis eval config + chunk env present in /workspace/rlinf"
if [[ -z "$LAUNCH_DRY_RUN" ]]; then
  miss=$(docker exec rlinf-eval bash -lc '
    for p in examples/embodiment/config/'"$EVAL_CONFIG"'.yaml \
             examples/embodiment/config/env/realworld_franka_jointvel_polymetis.yaml \
             rlinf/envs/realworld/franka/polymetis_controller.py \
             rlinf/models/embodiment/openpi/dataconfig/droid_dataconfig.py; do
      [ -e "/workspace/rlinf/$p" ] || echo "$p"
    done' 2>/dev/null || echo "DOCKER_EXEC_FAILED")
  if [[ -n "$miss" ]]; then
    die "the mounted checkout (/workspace/rlinf = $RLINF_REPO_HOST) is MISSING eval code:
$(printf '      - %s\n' $miss)
    This needs the polymetis-controller branch PLUS the pi05_droid inference port
    (see docs/EVAL.md §1). Mount the unified eval checkout before running."
  fi
  ok "eval code present in container"
else
  echo "DRY: docker exec rlinf-eval verify $EVAL_CONFIG.yaml + polymetis_controller.py present"
fi

step "  checkpoint /ckpts/pi05_droid_pt mounted"
desk "docker exec rlinf-eval bash -lc 'test -d /ckpts/pi05_droid_pt' || { echo MISSING; exit 1; }" \
  || die "/ckpts/pi05_droid_pt not in the container — the pi05_droid PyTorch checkpoint mount is missing."

step "  mutual exclusion: stop the other dashboards (openpi shares :8003; collect holds cams + zerorpc)"
desk "pkill -TERM -f '[/_]openpi[.]py'  || true"
desk "pkill -TERM -f '[/_]collect[.]py' || true"
desk "pkill -TERM -f '[/_]rlinf[.]py'   || true"
[[ -z "$LAUNCH_DRY_RUN" ]] && sleep 2

# Reap FIRST (a prior eval-stop -9'd the in-container ZED holder mid-capture,
# which wedges the ZED USB), THEN auto USB-reset if the SDK can't see them —
# so restarting eval no longer needs a manual `zed-check.sh --reset`.
step "  reap stale eval procs in rlinf-eval (frees any ZED holder)"
desk "docker exec rlinf-eval bash -lc 'pkill -9 -f \"[e]val_embodied_agent.py\"; pkill -9 -f \"[r]ay::\"; pkill -9 -f \"[r]aylet\"; pkill -9 -f \"[P]olymetisController\"; /opt/venv/openpi/bin/ray stop --force 2>/dev/null; rm -rf /tmp/ray; true'"

step "  cameras: ensure both ZEDs free ($ZED_EXTERIOR exterior, $ZED_WRIST wrist) — auto USB-reset if wedged"
zeds_free_or_reset || die "ZEDs still not visible after auto USB-reset. This is now a hardware case: physically unplug/replug the exterior ZED 2i USB and re-run, or reboot (last resort). See $_DIR/zed-check.sh for escalation."

step "  launch eval dashboard :8003 (--mode host, polymetis zerorpc backend)"
desk "cd $TASL && ulimit -n 8192 && PYTHONPATH=$TASL:$SITE_PKGS NUC1_HOST=$NUC1_HOST setsid /usr/bin/python3 dashboards/rlinf.py --port 8003 --mode host </dev/null >> $TASL/logs/eval.log 2>&1 &"
wait_http 8003 || die "eval dashboard did not answer on :8003 (see $TASL/logs/eval.log)"
ok "dashboard up — http://$TS_IP:8003 (laptop, via Tailscale; robot-net IP if TS offline)"

# ── Stage 4/4 — optional pre-eval bootstrap + home (zerorpc, via the dashboard) ─
# NOTE: the in-container eval bootstraps the polymetis controller and resets
# (homes) the arm itself on Start (PolymetisController.launch_controller +
# env.reset). These dashboard calls are a pre-eval convenience and require the
# rlinf.py robot panel to be on zerorpc (EVAL.md §7). Until that swap lands they
# may 404/fail — that's NON-fatal: just press Start and the env will bootstrap.
step "Stage 4/4 — (optional) pre-eval recover + home via dashboard zerorpc API"
if [[ -z "$LAUNCH_DRY_RUN" ]]; then
  if curl -fsS -X POST "$DASH/api/robot/recover" >/dev/null 2>&1; then
    ok "controller recovered"
    resp=$(curl -fsS -X POST "$DASH/api/robot/home" 2>/dev/null || true)
    [[ -n "$resp" ]] && echo "    $resp"
  else
    warn "dashboard robot API not available yet (needs the rlinf.py zerorpc swap, EVAL.md §7).
       Skip — the in-container eval bootstraps + homes on Start anyway."
  fi
else
  echo "DRY: curl -X POST $DASH/api/robot/recover ; curl -X POST $DASH/api/robot/home (best-effort)"
fi

ok "READY — open http://$TS_IP:8003, type a task prompt, press Start to run an eval episode.
     SAFETY: first rollout is autonomous (neural net drives the arm). Hand on the
     E-stop, start from home, watch the first chunks before trusting it."
