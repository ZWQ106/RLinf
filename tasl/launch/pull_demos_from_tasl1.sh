#!/usr/bin/env bash
# Pull rollout videos + saved demos from TASL1 (the FR3 desktop) onto THIS
# machine with rsync over ssh. RUN ON TASL2 (or any box that can ssh to TASL1)
# — TASL1 cannot reach TASL2, so the receiver initiates. Nothing runs on TASL1
# beyond its already-listening sshd; no FTP server needed.
#
#     ./pull_demos_from_tasl1.sh              # incremental pull (safe to repeat)
#     ./pull_demos_from_tasl1.sh --dry-run    # show what would transfer
#
# Sources on TASL1 (both are where the openpi/collect portals write):
#     ~/RLinf/tasl/eval_episodes/   every rollout: <task>/ep_<ts>/{video.mp4,meta.json}
#     ~/RLinf/saved_demo/           manual save-video / save-layout exports
# Destination here (override with DEST):
#     ~/Franka_RealRobot/saved_demo/{eval_episodes,saved_demo}/
#
# TASL1 address: Tailscale IP first (works from anywhere the tailnet does),
# campus LAN as fallback. Override with TASL1_HOST=...
# One-time: put this machine's key on TASL1 →  ssh-copy-id franka_desktop@100.79.65.37
# Automate: crontab -e →  */10 * * * * ~/pull_demos_from_tasl1.sh >> ~/pull_demos.log 2>&1
set -euo pipefail

DRY=""
for _arg in "$@"; do
  case "$_arg" in
    --dry-run) DRY="--dry-run" ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown arg: $_arg (see --help)" >&2; exit 1 ;;
  esac
done

USER_AT="${TASL1_USER:-franka_desktop}"
DEST="${DEST:-$HOME/Franka_RealRobot/saved_demo}"
HOSTS=("${TASL1_HOST:-}" 100.79.65.37 10.12.159.30)   # tailscale, then campus LAN
SSH_OPTS="-o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new"

HOST=""
for h in "${HOSTS[@]}"; do
  [[ -z "$h" ]] && continue
  if ssh $SSH_OPTS "$USER_AT@$h" true 2>/dev/null; then HOST="$h"; break; fi
done
[[ -n "$HOST" ]] || { echo "✗ TASL1 unreachable via: ${HOSTS[*]} (tailscale up? key installed? try: ssh $USER_AT@100.79.65.37)" >&2; exit 1; }
echo "▶ TASL1 = $USER_AT@$HOST  →  $DEST${DRY:+  [DRY RUN]}"
mkdir -p "$DEST/eval_episodes" "$DEST/saved_demo"

# --partial keeps half-copied mp4s for resume; --exclude skips in-progress
# recordings and the no-task bucket; no --delete: nothing here is ever removed
# because it vanished on TASL1.
RS=(rsync -az --partial --info=stats1,progress2 $DRY -e "ssh $SSH_OPTS"
    --exclude 'untagged/' --exclude '.tmp/' --exclude '*.part')
"${RS[@]}" "$USER_AT@$HOST:RLinf/tasl/eval_episodes/" "$DEST/eval_episodes/"
"${RS[@]}" "$USER_AT@$HOST:RLinf/saved_demo/"         "$DEST/saved_demo/"

[[ -n "$DRY" ]] && exit 0
echo "✓ $(find "$DEST/eval_episodes" -name video.mp4 | wc -l) rollout videos + $(find "$DEST/saved_demo" -name '*.mp4' | wc -l) saved demos on this machine ($(du -sh "$DEST" | cut -f1))"
