# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import select
import threading
import time


class KeyboardListener:
    """Headless keyboard listener backed by Linux evdev input devices.

    Resilient to its bound device disappearing at runtime. A collection
    dashboard may drive this listener through an ephemeral uinput *virtual*
    keyboard whose /dev/input/eventN node only exists while the dashboard
    process is alive — and whose event number is NOT stable across dashboard
    restarts. If the listener bound that node by path and the node vanished, a
    naive read loop would die with ``OSError: [Errno 19] No such device`` and
    never recover, silently breaking ALL key input (start / success / discard)
    for the rest of the run. Instead this listener:

      * re-opens on device loss instead of dying, falling back to any readable
        physical keyboard so local key input keeps working, and
      * prefers a device selected by *name* (``RLINF_KEYBOARD_DEVICE_NAME``),
        re-acquiring the dashboard's virtual keyboard automatically when it
        reappears at a new event number after a dashboard restart.

    Device selection order (applied on every open):
      1. ``RLINF_KEYBOARD_DEVICE`` — a hard path override (verified once at
         startup; a bad value raises so a misconfig is caught early).
      2. ``RLINF_KEYBOARD_DEVICE_NAME`` — a soft preference: use the device
         whose evdev name matches, else fall back. Never fatal on its own, so
         a briefly-absent virtual keyboard degrades to the physical keyboard
         rather than killing the run.
      3. First capability-matching keyboard under /dev/input/event*.
    """

    REQUIRED_KEY_NAMES = ("KEY_A", "KEY_B", "KEY_C", "KEY_Q")

    # How often the read loop wakes to (a) notice a vanished device and (b)
    # re-prefer the named device if it reappeared while we read a fallback.
    _PREEMPT_INTERVAL_S = 2.0
    # Backoff between re-open attempts when no keyboard is currently readable.
    _REOPEN_INTERVAL_S = 1.0

    def __init__(self):
        try:
            from evdev import InputDevice, ecodes, list_devices
        except ImportError as exc:
            raise RuntimeError(
                "KeyboardListener requires the 'evdev' package. "
                "Install the real-world extras with evdev support."
            ) from exc

        self._input_device_cls = InputDevice
        self._ecodes = ecodes
        self._list_devices = list_devices

        self._override_path = os.environ.get("RLINF_KEYBOARD_DEVICE") or None
        self._preferred_name = os.environ.get("RLINF_KEYBOARD_DEVICE_NAME") or None

        self.state_lock = threading.Lock()
        self.latest_data = {"key": None}
        self._stop = False

        # Open once at startup so a genuine "no keyboard at all" misconfig is
        # caught immediately (matches the historical fail-fast behavior). The
        # loop then keeps this device alive / re-acquires it as needed.
        self.device = self._open_keyboard_device()

        self.listener = threading.Thread(
            target=self._listen_loop,
            name=f"KeyboardListener:{self.device.path}",
            daemon=True,
        )
        self.listener.start()
        self.last_intervene = 0

    # ── device selection ────────────────────────────────────────────────
    def _open_keyboard_device(self):
        """Open the best available keyboard device (startup path).

        The hard path override is verified strictly (raises on a bad value).
        The name preference and the capability scan are best-effort; only a
        total absence of any readable keyboard raises.
        """
        if self._override_path:
            device = self._open_device(self._override_path, is_override=True)
            if not self._is_keyboard_device(device):
                device.close()
                raise RuntimeError(
                    "KeyboardListener device set by "
                    f"RLINF_KEYBOARD_DEVICE='{self._override_path}' does not look "
                    "like a keyboard device. Point it to the correct "
                    "/dev/input/eventX path."
                )
            return device

        device = self._acquire_device()
        if device is not None:
            return device

        # Nothing openable: distinguish permission from absence for a useful
        # message (mirrors the original diagnostics).
        permission_denied_paths: list[str] = []
        for device_path in sorted(self._list_devices()):
            try:
                self._input_device_cls(device_path).close()
            except PermissionError:
                permission_denied_paths.append(device_path)
            except OSError:
                continue
        if permission_denied_paths:
            denied = ", ".join(permission_denied_paths)
            raise RuntimeError(
                "KeyboardListener could not open any readable keyboard device under "
                f"/dev/input/event*. Permission denied for: {denied}. Grant the runtime "
                "user read access via the input group or udev rules, or set "
                "RLINF_KEYBOARD_DEVICE to a readable keyboard event device."
            )
        raise RuntimeError(
            "KeyboardListener could not find a readable keyboard device under "
            "/dev/input/event*. Ensure a physical keyboard is connected, the runtime "
            "user has access to input devices, or set RLINF_KEYBOARD_DEVICE to the "
            "correct /dev/input/eventX path."
        )

    def _acquire_device(self):
        """Best-effort open using the preference order; None if nothing opens.

        Never raises — used for runtime re-opens where dying is not an option.
        Order: hard path override (if it currently opens) → named preference →
        first capability-matching keyboard.
        """
        # 1. Hard path override, if present and currently openable.
        if self._override_path:
            try:
                device = self._input_device_cls(self._override_path)
                if self._is_keyboard_device(device):
                    return device
                device.close()
            except OSError:
                pass  # fall through to soft selection while it's gone

        # 2. Named preference.
        if self._preferred_name:
            device = self._open_named_device()
            if device is not None:
                return device

        # 3. First capability-matching keyboard.
        for device_path in sorted(self._list_devices()):
            try:
                device = self._input_device_cls(device_path)
            except OSError:
                continue
            if self._is_keyboard_device(device):
                return device
            device.close()
        return None

    def _open_named_device(self):
        """Open the keyboard whose evdev name matches the name preference."""
        for device_path in sorted(self._list_devices()):
            try:
                device = self._input_device_cls(device_path)
            except OSError:
                continue
            try:
                if device.name == self._preferred_name and self._is_keyboard_device(
                    device
                ):
                    return device
            except OSError:
                device.close()
                continue
            device.close()
        return None

    def _preferred_path(self) -> str | None:
        """Path of the currently-available *preferred* device (override path or
        named device), or None. Used to re-prefer it over a fallback we may be
        reading after the preferred device briefly vanished."""
        if self._override_path and os.path.exists(self._override_path):
            return self._override_path
        if self._preferred_name:
            device = self._open_named_device()
            if device is not None:
                path = device.path
                device.close()
                return path
        return None

    def _open_device(self, device_path: str, is_override: bool = False):
        try:
            return self._input_device_cls(device_path)
        except FileNotFoundError as exc:
            if is_override:
                raise RuntimeError(
                    f"KeyboardListener override path '{device_path}' does not exist."
                ) from exc
            raise
        except PermissionError as exc:
            if is_override:
                raise RuntimeError(
                    "KeyboardListener cannot read the device set by "
                    f"RLINF_KEYBOARD_DEVICE='{device_path}'. Grant the runtime user "
                    "read access via the input group or udev rules."
                ) from exc
            raise
        except OSError as exc:
            if is_override:
                raise RuntimeError(
                    "KeyboardListener failed to open the device set by "
                    f"RLINF_KEYBOARD_DEVICE='{device_path}': {exc}"
                ) from exc
            raise RuntimeError(
                f"KeyboardListener failed to open input device '{device_path}': {exc}"
            ) from exc

    def _is_keyboard_device(self, device) -> bool:
        required_codes = {
            getattr(self._ecodes, key_name) for key_name in self.REQUIRED_KEY_NAMES
        }
        capabilities = device.capabilities(verbose=False)
        supported_key_codes = set(capabilities.get(self._ecodes.EV_KEY, []))
        return required_codes.issubset(supported_key_codes)

    # ── read loop ───────────────────────────────────────────────────────
    def _listen_loop(self) -> None:
        try:
            while not self._stop:
                if self.device is None:
                    # No keyboard readable right now (e.g. the dashboard's
                    # virtual keyboard vanished and no physical keyboard is
                    # accessible). Retry rather than exit — input must not
                    # break permanently.
                    time.sleep(self._REOPEN_INTERVAL_S)
                    self.device = self._acquire_device()
                    continue
                try:
                    ready, _, _ = select.select(
                        [self.device.fd], [], [], self._PREEMPT_INTERVAL_S
                    )
                    if not ready:
                        # Idle tick. Two housekeeping checks:
                        #  (a) proactively notice a vanished device even if
                        #      select never flagged its fd (drop + re-acquire),
                        #  (b) if we fell back to a non-preferred device and the
                        #      preferred (named/override) device has reappeared,
                        #      switch back so dashboard injection works again
                        #      after a dashboard restart.
                        if self._current_device_gone():
                            self._close_device()
                            with self.state_lock:
                                self.latest_data["key"] = None
                            continue
                        self._maybe_switch_to_preferred()
                        continue
                    for event in self.device.read():
                        self._handle_event(event)
                except OSError:
                    # Bound device disappeared (dashboard stopped, USB replug,
                    # etc.). Drop it and re-acquire on the next iteration —
                    # falling back to any physical keyboard.
                    self._close_device()
                    with self.state_lock:
                        self.latest_data["key"] = None
        finally:
            with self.state_lock:
                self.latest_data["key"] = None
            self._close_device()

    def _handle_event(self, event) -> None:
        if event.type != self._ecodes.EV_KEY:
            return
        key = self._event_to_key(event.code)
        if key is None:
            return
        if event.value in (1, 2):
            with self.state_lock:
                self.latest_data["key"] = key
        elif event.value == 0:
            with self.state_lock:
                if self.latest_data["key"] == key:
                    self.latest_data["key"] = None

    def _current_device_gone(self) -> bool:
        """True if the bound device's node no longer exists. Cheap path check
        so a vanished device is noticed even when select() never flags its fd."""
        path = getattr(self.device, "path", None)
        return bool(path) and not os.path.exists(path)

    def _on_preferred_device(self) -> bool:
        """Whether the bound device is already the preferred one — a cheap
        check that avoids scanning every input device on each idle tick."""
        if self.device is None:
            return False
        try:
            if self._override_path and self.device.path == self._override_path:
                return True
            if self._preferred_name and self.device.name == self._preferred_name:
                return True
        except OSError:
            return False
        return False

    def _maybe_switch_to_preferred(self) -> None:
        """If reading a fallback while the preferred device is available, drop
        the fallback so the loop re-acquires the preferred device. No-op (and
        no device scan) when already on the preferred device or when no
        preference is configured."""
        if not (self._override_path or self._preferred_name):
            return
        if self._on_preferred_device():
            return
        preferred = self._preferred_path()
        if preferred is not None and preferred != getattr(self.device, "path", None):
            self._close_device()
            with self.state_lock:
                self.latest_data["key"] = None

    def _close_device(self) -> None:
        if self.device is not None:
            try:
                self.device.close()
            except Exception:
                pass
            self.device = None

    def _event_to_key(self, key_code: int) -> str | None:
        key_name = self._ecodes.bytype[self._ecodes.EV_KEY].get(key_code)
        if isinstance(key_name, list):
            key_name = key_name[0]
        if not isinstance(key_name, str):
            return None

        if key_name.startswith("KEY_"):
            normalized_key = key_name.removeprefix("KEY_").lower()
            if len(normalized_key) == 1:
                return normalized_key
            return f"Key.{normalized_key}"
        return key_name.lower()

    def get_key(self) -> str | None:
        """Returns the latest key pressed."""
        with self.state_lock:
            return self.latest_data["key"]

    def stop(self) -> None:
        """Signal the read loop to exit (best-effort; thread is a daemon)."""
        self._stop = True
