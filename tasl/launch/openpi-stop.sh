#!/usr/bin/env bash
# Crash-safe openpi-portal teardown — reverses start_openpi_full.sh. RUN ON THE
# DESKTOP (TASL-1):
#
#     ./openpi-stop.sh
#
# Idempotent — safe to run anytime, even half-up:
#   1. Stop the openpi dashboard (:8003)      — SIGTERM openpi.py + any leftover
#      start_openpi*.sh wrappers.
#   2. Stop serve_policy (:8000)              — SIGTERM (frees the Desktop GPU);
#      verified with nvidia-smi.
#   3. Bring the NUC1 controller down         — `docker compose down` over the
#      (RELEASES FCI)                            robot net; verify :4242 closed.
#
# Leaves the franky robot_server STOPPED and does NOT touch Desk — re-lock the
# joints / re-Activate FCI manually next time (or run start_openpi_full.sh).
#
#   ./openpi-stop.sh --keep-fci       # stages 1+2 only; NUC controller stays up
#                                     # (fast re-test loop: Stage 1 of the next
#                                     # start_openpi_full.sh run will be skipped)
#   LAUNCH_DRY_RUN=1 ./openpi-stop.sh # print actions, change nothing
set -euo pipefail
_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$_DIR/lib.sh"

KEEP_FCI=""
for _arg in "$@"; do
  case "$_arg" in
    --keep-fci) KEEP_FCI=1 ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) die "unknown arg: $_arg (see --help)" ;;
  esac
done

ensure_root "$0" "$@"

# ── 1. Dashboard :8003 (SIGTERM — never -9 a ZED holder) ─────────────────────
step "1/3 — stop openpi dashboard (:8003) + leftover start_openpi wrappers"
desk "pkill -TERM -f '[/_]openpi[.]py' || true"
desk "pkill -TERM -f '[/]start_openpi' || true"
[[ -z "$LAUNCH_DRY_RUN" ]] && sleep 2   # let it release the cameras on the way out

# ── 2. serve_policy :8000 (frees the Desktop GPU) ────────────────────────────
step "2/3 — stop serve_policy (:8000) — frees the Desktop GPU"
desk "pkill -TERM -f '[/]serve_policy(_patched)?[.]py' || true"
if [[ -z "$LAUNCH_DRY_RUN" ]]; then
  # JAX teardown can take a few seconds; poll instead of a fixed sleep.
  for _i in $(seq 1 10); do
    pgrep -f '[/]serve_policy(_patched)?[.]py' >/dev/null || break
    sleep 1
  done
  if pgrep -f '[/]serve_policy(_patched)?[.]py' >/dev/null; then
    warn "serve_policy ignored SIGTERM after 10s — escalating to SIGKILL"
    desk "pkill -KILL -f '[/]serve_policy(_patched)?[.]py' || true"
    sleep 1
  fi
  gpu_free && ok "GPU free" \
    || warn "a compute process still holds the Desktop GPU — check: nvidia-smi"
fi

# ── 3. NUC1 controller down over the robot net (releases FCI) ────────────────
if [[ -n "$KEEP_FCI" ]]; then
  step "3/3 — SKIPPED (--keep-fci): NUC1 controller stays up, FCI stays active"
else
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
fi

# ── verify ──────────────────────────────────────────────────────────────────
if [[ -z "$LAUNCH_DRY_RUN" ]]; then
  for _port in 8003 8000; do
    if ss -ltn 2>/dev/null | grep -q ":$_port "; then
      warn ":$_port STILL listening — something didn't exit. Check: ps -eo pid,user,args | grep -E 'openpi|serve_policy'"
    else
      ok ":$_port free"
    fi
  done
fi

if [[ -n "$KEEP_FCI" ]]; then
  ok "OPENPI PORTAL STOPPED — GPU free; NUC controller + FCI still up.
   Next start_openpi_full.sh will skip Stage 1 and start fast."
else
  ok "OPENPI PORTAL STOPPED — FCI released. Re-lock joints in Desk when done;
   run start_openpi_full.sh to restart."
fi
