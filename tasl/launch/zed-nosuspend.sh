#!/usr/bin/env bash
# Fix the recurring "exterior ZED 2i not SDK-visible" wedge. RUN ON THE DESKTOP:
#
#     sudo ~/RLinf/tasl/launch/zed-nosuspend.sh
#
# Root cause: kernel USB autosuspend (power/control=auto) suspends the ZED 2i
# 2 s after the last holder releases it; the ZED SDK then fails to probe it,
# while lsusb/UVC still look normal. usbreset does NOT recover it.
# This script: (1) wakes every Stereolabs device now and pins power/control=on,
# (2) installs a udev rule so the fix survives replug/reboot, (3) verifies the
# SDK sees both expected serials.
set -euo pipefail

ZED_VENDOR=2b03
ZED_EXTERIOR=36443134
ZED_WRIST=17150101
RULE=/etc/udev/rules.d/99-zed-nosuspend.rules
CONTAINER="${RLINF_CONTAINER:-rlinf-eval}"

if [[ "$(id -u)" != 0 ]]; then exec sudo "$0" "$@"; fi

echo "▶ 1/3 wake + pin power/control=on for all $ZED_VENDOR (Stereolabs) devices"
found=0
for p in /sys/bus/usb/devices/*; do
  [[ -f "$p/idVendor" ]] || continue
  [[ "$(cat "$p/idVendor")" == "$ZED_VENDOR" ]] || continue
  found=1
  echo on > "$p/power/control"
  echo "    $p ($(cat "$p/idProduct")) -> control=on, status=$(cat "$p/power/runtime_status")"
done
[[ "$found" == 1 ]] || { echo "✗ no Stereolabs USB devices found — check cables"; exit 1; }

echo "▶ 2/3 install udev rule ($RULE) so it survives replug/reboot"
cat > "$RULE" <<EOF
ACTION=="add", SUBSYSTEM=="usb", ATTR{idVendor}=="$ZED_VENDOR", TEST=="power/control", ATTR{power/control}="on"
EOF
udevadm control --reload

echo "▶ 3/3 verify the SDK sees both ZEDs (probe inside $CONTAINER)"
sleep 3
serials="$(docker exec "$CONTAINER" /opt/venv/openpi/bin/python -c \
  "import pyzed.sl as sl; print([d.serial_number for d in sl.Camera.get_device_list()])" 2>/dev/null | tail -1)"
echo "    SDK sees: $serials"
if [[ "$serials" == *"$ZED_EXTERIOR"* && "$serials" == *"$ZED_WRIST"* ]]; then
  echo "✓ both ZEDs visible — run teleop.sh and collect."
else
  echo "! SDK still missing a camera. The suspended state may not clear without a"
  echo "  re-enumeration: REPLUG the missing camera's USB plug (exterior ZED 2i ="
  echo "  the big camera on the frame; its lsusb device number must CHANGE),"
  echo "  then re-run this script to verify. With autosuspend now off it will not"
  echo "  wedge again."
  exit 2
fi
