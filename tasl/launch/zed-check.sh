#!/usr/bin/env bash
# ZED pre-flight for eval / collect: are BOTH cameras visible to the ZED SDK
# inside rlinf-eval? A hot-plug or power-cycle of a ZED under the running
# container leaves it enumerated on USB (lsusb sees it) but invisible to the
# SDK — get_device_list() returns [] on host AND container — which trips the
# `zeds_free` gate in eval.sh / teleop.sh with a misleading "reseat" message.
# This probes the SDK directly and, with --reset, issues a USB reset first
# (usually clears the wedge without a physical reseat or container restart).
#
#   ./zed-check.sh            # probe only; print present/missing serials
#   ./zed-check.sh --reset    # usbreset both ZEDs (needs root), then probe
#   ./zed-check.sh --help
#
# Exit 0 = both cameras visible (safe to run eval/collect); 1 = one/both missing.
set -euo pipefail
_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$_DIR/lib.sh"   # ZED_EXTERIOR/ZED_WRIST, step/ok/warn/die, ensure_root

# STEREOLABS USB ids for the two CAMERA interfaces (not the f6xx HID controls):
#   f880 = ZED 2i (exterior),  f682 = ZED-M (wrist).
ZED_USB_IDS=(2b03:f880 2b03:f682)
CONTAINER="${RLINF_CONTAINER:-rlinf-eval}"
CONTAINER_PY="/opt/venv/openpi/bin/python"

case "${1:-}" in
  -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
  --reset) RESET=1 ;;
  "") RESET=0 ;;
  *) die "unknown arg: $1 (see --help)" ;;
esac

probe() {
  # Print the serials the in-container ZED SDK enumerates (empty on wedge).
  docker exec "$CONTAINER" "$CONTAINER_PY" -c \
    "import pyzed.sl as sl; print([d.serial_number for d in sl.Camera.get_device_list()])" \
    2>/dev/null | tail -1
}

if [[ "${RESET:-0}" == 1 ]]; then
  ensure_root "$0" "$@"
  step "USB reset both ZEDs (clears a post-hot-plug SDK wedge)"
  for id in "${ZED_USB_IDS[@]}"; do
    if usbreset "$id" >/dev/null 2>&1; then ok "reset $id"; else warn "reset $id failed (camera absent / already resetting?)"; fi
  done
  sleep 2   # let the SDK re-enumerate before probing
fi

step "Probe ZED SDK inside $CONTAINER (expect exterior=$ZED_EXTERIOR wrist=$ZED_WRIST)"
serials="$(probe)"
echo "    SDK sees: ${serials:-<none>}"

miss=0
[[ "$serials" == *"$ZED_EXTERIOR"* ]] || { warn "exterior ZED 2i ($ZED_EXTERIOR) NOT visible"; miss=1; }
[[ "$serials" == *"$ZED_WRIST"* ]]    || { warn "wrist ZED-M ($ZED_WRIST) NOT visible"; miss=1; }

if [[ "$miss" == 0 ]]; then
  ok "both ZEDs visible — safe to run eval.sh / collect"
  exit 0
fi

warn "one or both ZEDs not visible to the SDK. Try, in order:
     1. $0 --reset          (USB reset both cameras; needs sudo)
     2. physically unplug/replug the exterior ZED 2i USB, then re-run this
     3. reboot the desktop  (last resort)"
exit 1
