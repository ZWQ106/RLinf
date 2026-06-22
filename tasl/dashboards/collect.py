"""TASL FR3 collection dashboard — mission control for RLinf data collection.

Runs on the Desktop HOST (port 8004). It does NOT own the robot or cameras
during collection — the RLinf env process inside the `rlinf-eval` container
does. This dashboard orchestrates:

  * launch/stop of collect_real_data.py inside the container (docker exec)
  * camera discipline: holds the ZEDs while idle (MJPEG preview), releases
    them before a run, reclaims them when the run exits
  * live status by tailing/parsing the collection log
  * mark-success: uinput virtual keyboard injects 'c' for the RLinf
    KeyboardListener (requires root -> launch with sudo)
  * robot panel (idle-only): home / recover / gripper via zerorpc to the
    DROID polymetis container on the NUC
  * dataset manager: scan LeRobot v2.1 datasets on the host bind mount,
    episode playback (MJPEG from embedded parquet images), guarded delete

Lifecycle:
  IDLE:    dashboard owns the ZED cameras, streams MJPEG previews.
  START:   kill legacy _dashboard_openpi.py if running, release cameras,
           docker exec -d the collection process (log -> /tmp/collect_dash.log
           inside the container).
  RUNNING: /api/status polls container pgrep + log tail; camera endpoints
           return 409 (env owns the ZEDs).
  EXIT:    (process ends or Stop pressed) parse final log as last_result,
           ray cleanup, reclaim cameras.

Launch (Desktop host):
    sudo PYTHONPATH=tasl /usr/bin/python3 tasl/dashboards/collect.py --port 8004
(sudo for /dev/uinput; single-tenant lab box, per lab convention)

Open: http://<desktop-tailscale-or-lan-ip>:8004
"""
from __future__ import annotations

import argparse
import base64
import functools
import glob
import json
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time
from typing import Optional

# ── Laptop-compat guards ──────────────────────────────────────────────
# This module must import cleanly on the dev laptop (macOS, no pyzed/flask)
# so the pure log parser is unit-testable. Desktop-only deps are flagged.
try:
    import pyzed.sl  # noqa: F401  (presence check only; used lazily below)
    HAS_PYZED = True
except ImportError:
    HAS_PYZED = False

try:
    from flask import Flask, Response, jsonify, render_template_string, request
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

try:
    import evdev  # noqa: F401  (presence check; UInput/ecodes imported lazily)
    HAS_EVDEV = True
except ImportError:
    HAS_EVDEV = False

try:
    import zerorpc  # noqa: F401  (presence check; imported lazily in DroidClient)
    HAS_ZERORPC = True
except ImportError:
    HAS_ZERORPC = False

try:
    import pyarrow.parquet  # noqa: F401  (presence check; imported lazily)
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

try:
    import cv2  # noqa: F401  (presence check; imported lazily)
    import numpy  # noqa: F401
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

_log = logging.getLogger("collect_dashboard")

# ── Hardware constants (verified 2026-05-27) ─────────────────────────
SN_EXTERIOR = 36443134   # ZED 2i, exterior view
SN_WRIST = 17150101      # ZED Mini, wrist-mounted

CONTAINER = "rlinf-eval"
COLLECT_LOG = "/tmp/collect_dash.log"
CONTAINER_PY = "/opt/venv/openpi/bin/python"
# Host side of the bind mount where the in-container VideoPlayer writes live
# per-camera JPEGs during a collection (container path:
# /workspace/rlinf/outputs/live_cam via RLINF_LIVE_CAM_DIR). LiveCamSource
# reads these to serve a live preview while the env owns the ZED cameras.
LIVE_CAM_DIR_HOST = "/home/franka_desktop/work/rlinf-clone/outputs/live_cam"
# Start-gate sentinel the keyboard wrapper writes ("WAIT"/"RUN") inside the
# container — Ray buffers actor stdout, so the dashboard reads this file
# (docker exec cat) instead of parsing the log to drive the 开始下一条 button.
GATE_FILE = "/tmp/collect_gate"
CONTAINER_RAY = "/opt/venv/openpi/bin/ray"

# DROID polymetis container on NUC1 (zerorpc, FCI on C2 port).
# Reached over the ROBOT network (172.16.0.2) by default — the Desktop is on
# 172.16.0.x; Tailscale (100.75.6.62) is the legacy path. Override: NUC1_HOST=…
NUC1_HOST = os.environ.get("NUC1_HOST", "172.16.0.2")
DROID_ADDR = f"tcp://{NUC1_HOST}:4242"
# DROID reset/home pose (radians), same as the RLinf env uses.
DROID_HOME_Q = [0.0, -0.6283, 0.0, -2.5133, 0.0, 1.8850, 0.0]

VIRTUAL_KBD_NAME = "tasl-collect-dashboard-kbd"

# Dataset manager (Task C). New HF_LEROBOT_HOME bind mount, host path on
# the Desktop. Override with --dataset-roots (comma-separated).
DATASET_ROOTS = ["/home/franka_desktop/work/rlinf-clone/outputs/lerobot"]
# Pre-migration datasets inside the container (notice only, never scanned).
LEGACY_CONTAINER_LEROBOT = "/opt/.cache/huggingface/lerobot"
PLAYBACK_MAX_FPS = 10.0
PLAYBACK_UPSCALE = 2  # 128 -> 256 per camera, INTER_NEAREST


# ─────────────────────────────────────────────────────────────────────
# Log parser — pure function, no I/O (unit-tested)
# ─────────────────────────────────────────────────────────────────────
# Progress bar:  "Collecting Data Episodes::   0%|          | 0/1 [00:00<?, ?it/s]"
_RE_PROGRESS = re.compile(r"Collecting Data Episodes:.*?(\d+)/(\d+)\s*\[")
# Both episode-end variants carry a count:
#   "... Discarded. Total success: 0/1"    "... Total: 1/1"
_RE_TOTAL = re.compile(r"Total(?: success)?:\s*(\d+)/(\d+)")
_RE_DISCARDED = re.compile(r"Episode ended \(reward=([0-9.]+)\)\. Discarded")
_RE_SUCCESS = re.compile(r"Success \(reward=([0-9.]+)")
_RE_SAVED = re.compile(r"Saved episode with (\d+) frames(?:, task: '([^']*)')?")


def _extract_traceback(lines: list[str], start: int, max_extra: int = 5) -> str:
    """Traceback header + up to max_extra following lines.

    The block = indented frame lines + the final non-indented exception
    line. When the block is longer than max_extra, keep the TAIL — the
    exception message is the part worth showing.
    """
    block = [lines[start]]
    j = start + 1
    while j < len(lines):
        ln = lines[j]
        if ln.startswith((" ", "\t")):
            block.append(ln)
            j += 1
            continue
        if ln.strip():
            block.append(ln)  # exception line terminates the block
        break
    body = block[1:]
    if len(body) > max_extra:
        body = body[-max_extra:]
    return "\n".join([block[0]] + body)


def parse_collect_log(text: str) -> dict:
    """{phase, episodes_done, success_count, target, last_event, error}
    phase: "starting" | "collecting" | "error". Caller combines with process
    liveness to derive idle/finished."""
    episodes_done = 0
    success_count = 0
    target: Optional[int] = None
    last_event: Optional[str] = None
    error: Optional[str] = None
    saw_activity = False

    lines = text.splitlines()
    for i, line in enumerate(lines):
        if error is None and "Traceback (most recent call last):" in line:
            error = _extract_traceback(lines, i)
            continue

        m = _RE_PROGRESS.search(line)
        if m:
            saw_activity = True
            target = int(m.group(2))

        m = _RE_TOTAL.search(line)
        if m:
            success_count = int(m.group(1))
            target = int(m.group(2))

        if "Episode ended" in line or "Success (" in line:
            episodes_done += 1
            saw_activity = True

        m = _RE_DISCARDED.search(line)
        if m:
            last_event = f"Episode discarded (reward={m.group(1)})"
        else:
            m = _RE_SUCCESS.search(line)
            if m:
                mt = _RE_TOTAL.search(line)
                if mt:
                    last_event = f"Success {mt.group(1)}/{mt.group(2)}"
                else:
                    last_event = f"Success (reward={m.group(1)})"
        m = _RE_SAVED.search(line)
        if m:
            last_event = f"Saved episode ({m.group(1)} frames)"
            if m.group(2):
                last_event += f", task: '{m.group(2)}'"
            saw_activity = True

    if error is not None:
        phase = "error"
    elif saw_activity:
        phase = "collecting"
    else:
        phase = "starting"
    return {
        "phase": phase,
        "episodes_done": episodes_done,
        "success_count": success_count,
        "target": target,
        "last_event": last_event,
        "error": error,
    }


# Startup milestones, anchored on real collect_real_data.py log strings
# (verified against a successful run 2026-06-15). Ordered; highest reached
# wins. The whole point: tell the operator WHEN teleop becomes live, since
# launch → controllable is a ~40-90s black box (Ray + env + camera + reset).
_RE_PLACEMENT = re.compile(r"Generated \d+ placement")
_STARTUP_STAGES = [
    ("启动采集进程", 5),       # 0 process launched, nothing logged yet
    ("Ray 启动", 20),          # 1
    ("集群 / worker 就绪", 40), # 2
    ("相机打开", 65),          # 3
    ("环境复位 (机械臂回 home)", 85),  # 4
    ("可以遥操", 100),          # 5 = ready
]


def is_waiting_for_start(text: str) -> bool:
    """True if the arm has reset and is blocked at the start gate.

    The keyboard wrapper prints `WAIT_FOR_START:` after each reset and
    `EPISODE_START:` once 's' lands. The latest sentinel wins — if the most
    recent is WAIT_FOR_START (or there's a WAIT with no later START), the
    operator needs to press 开始下一条 / 's'.
    """
    if "WAIT_FOR_START" not in text:
        return False
    return text.rfind("WAIT_FOR_START") > text.rfind("EPISODE_START")


def parse_startup_progress(text: str) -> dict:
    """Map the collection log to a startup progress indicator.

    Returns {percent, stage, label, ready, error, milestones}. `ready` is
    True once the episode loop prints its progress bar — at that point the
    first env reset (arm → home) is done and SpaceMouse input is live.
    `milestones` is the ordered checklist with a `done` flag per stage.
    Only meaningful before the first episode completes; after that the
    normal collection progress takes over.
    """
    error = None
    if "Traceback (most recent call last):" in text:
        error = _extract_traceback(text.splitlines(),
                                   text.splitlines().index(
                                       next(l for l in text.splitlines()
                                            if "Traceback (most recent call last):" in l)))

    ready = bool(_RE_PROGRESS.search(text))
    n_cam = text.count("Camera successfully opened")
    homing = ("passive_env_checker" in text and "reset()" in text)
    has_placement = bool(_RE_PLACEMENT.search(text))
    has_ray = "Started a local Ray instance" in text

    if ready:
        stage = 5
    elif n_cam >= 1 and homing:
        stage = 4
    elif n_cam >= 1:
        stage = 3
    elif has_placement:
        stage = 2
    elif has_ray:
        stage = 1
    else:
        stage = 0

    label, percent = _STARTUP_STAGES[stage]
    milestones = [
        {"label": lbl, "done": i <= stage}
        for i, (lbl, _pct) in enumerate(_STARTUP_STAGES)
    ]
    return {
        "percent": percent,
        "stage": stage,
        "label": label,
        "ready": ready,
        "error": error,
        "milestones": milestones,
    }


# ─────────────────────────────────────────────────────────────────────
# Camera manager (ported from _dashboard_openpi.py — same API)
# ─────────────────────────────────────────────────────────────────────
class CamManager:
    """Owns ZED cameras while the dashboard is idle. stop() releases USB
    so the RLinf env (inside rlinf-eval) gets exclusive ZED access."""

    def __init__(self, serials: dict[str, int], resolution: str = "HD720",
                 jpeg_quality: int = 70):
        self.serials = serials
        self.resolution = resolution
        self.jpeg_quality = jpeg_quality
        self._cams: dict[str, object] = {}
        self._threads: list[threading.Thread] = []
        self._frame_lock = threading.Lock()
        self._latest_jpeg: dict[str, bytes] = {}
        self._running = False
        # True once start() has ever been called — lets the collection manager
        # tell "idle-preview mode" (reclaim cameras after a run) from
        # "--no-cam-on-start mode" (keep them free for a cold env open).
        self.was_started = False
        self.missing_cams: list[tuple[str, int, str]] = []

    def start(self) -> str:
        if self._running:
            return "already running"
        self.was_started = True
        if not HAS_PYZED:
            raise RuntimeError(
                "pyzed not available on this host (Desktop-only dependency); "
                "cannot start cameras"
            )
        import pyzed.sl as sl
        res_map = {
            "HD2K": sl.RESOLUTION.HD2K,
            "HD1080": sl.RESOLUTION.HD1080,
            "HD720": sl.RESOLUTION.HD720,
        }
        # Partial-success policy: any cam that fails to open is logged +
        # skipped but the rest still come up.
        self.missing_cams = []
        opened_any = False
        for name, sn in self.serials.items():
            params = sl.InitParameters()
            params.camera_resolution = res_map[self.resolution]
            params.camera_fps = 30
            params.set_from_serial_number(int(sn))
            params.coordinate_units = sl.UNIT.METER
            params.depth_mode = sl.DEPTH_MODE.NONE
            cam = sl.Camera()
            status = cam.open(params)
            if status != sl.ERROR_CODE.SUCCESS:
                _log.error(f"ZED SN={sn} ({name}) open failed: {status}")
                self.missing_cams.append((name, sn, str(status)))
                continue
            self._cams[name] = cam
            opened_any = True
        if not opened_any:
            return f"no cameras opened (missing: {self.missing_cams})"
        self._running = True
        for name, cam in self._cams.items():
            t = threading.Thread(target=self._grab_loop, args=(name, cam),
                                 name=f"zed-{name}", daemon=True)
            t.start()
            self._threads.append(t)
            _log.info(f"ZED ({name}) grab thread started")
        if self.missing_cams:
            return (f"started with {len(self._cams)} cam(s); "
                    f"missing: {self.missing_cams}")
        return "started"

    def _grab_loop(self, name: str, cam):
        import cv2
        import pyzed.sl as sl
        rt = sl.RuntimeParameters()
        mat = sl.Mat()
        enc_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        while self._running:
            if cam.grab(rt) != sl.ERROR_CODE.SUCCESS:
                time.sleep(0.01)
                continue
            cam.retrieve_image(mat, sl.VIEW.LEFT)
            bgra = mat.get_data()  # (H, W, 4)
            ok, buf = cv2.imencode(".jpg", bgra[:, :, :3], enc_params)
            if not ok:
                continue
            with self._frame_lock:
                self._latest_jpeg[name] = buf.tobytes()
            time.sleep(0.02)

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()
        for cam in self._cams.values():
            try:
                cam.close()
            except Exception:
                pass
        self._cams.clear()
        with self._frame_lock:
            self._latest_jpeg.clear()
        _log.info("CamManager stopped, USB released")

    def is_running(self) -> bool:
        return self._running

    def get_jpeg(self, name: str) -> Optional[bytes]:
        with self._frame_lock:
            return self._latest_jpeg.get(name)


class LiveCamSource:
    """Serves the in-container env's live frames (written to the bind-mounted
    live_cam dir) through the dashboard's existing MJPEG route, during a
    collection. Maps the dashboard's display names to the env's camera-name
    files and returns a placeholder when a feed is stale/absent."""

    NAME_MAP = {"exterior": "wrist_1", "wrist": "wrist_2"}

    def __init__(self, live_dir: str, stale_s: float = 2.0):
        self._dir = live_dir
        self._stale_s = stale_s
        # 1x1 black JPEG placeholder
        import cv2, numpy as np
        ok, buf = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
        self._placeholder = buf.tobytes() if ok else b""

    def get_jpeg(self, name):
        fn = self.NAME_MAP.get(name, name)
        path = os.path.join(self._dir, f"{fn}.jpg")
        try:
            if time.time() - os.path.getmtime(path) <= self._stale_s:
                with open(path, "rb") as f:
                    return f.read()
        except OSError:
            pass
        return self._placeholder


# ─────────────────────────────────────────────────────────────────────
# Collection lifecycle (docker exec into rlinf-eval)
# ─────────────────────────────────────────────────────────────────────
class CollectionManager:
    """Launches/stops/monitors collect_real_data.py inside CONTAINER and
    enforces camera discipline around the run."""

    def __init__(self, cams: CamManager, vkbd: Optional["VirtualKeyboard"] = None):
        self.cams = cams
        self.vkbd = vkbd
        self._lock = threading.Lock()
        self._was_running = False
        self._launched_at = 0.0  # time.monotonic() of last successful start
        self.last_result: Optional[dict] = None
        self.last_launch: Optional[dict] = None
        self.last_action_msg: Optional[str] = None

    # -- docker helpers ------------------------------------------------
    @staticmethod
    def _docker_exec(args: list[str], timeout: float = 15.0,
                     detach: bool = False) -> subprocess.CompletedProcess:
        cmd = ["docker", "exec"] + (["-d"] if detach else []) + [CONTAINER] + args
        return subprocess.run(cmd, capture_output=True, timeout=timeout)

    def _is_alive(self) -> bool:
        # [a] char-class: pgrep's own argv never matches the pattern. With
        # argv-style subprocess there's no shell line to match, but the
        # bracket stays — pattern self-matches have killed our shells before.
        try:
            r = self._docker_exec(["pgrep", "-f", "collect_real_dat[a]"],
                                  timeout=10.0)
            return r.returncode == 0
        except Exception as e:
            _log.warning(f"liveness check failed: {e}")
            return False

    def is_collecting(self) -> bool:
        """Public liveness check for endpoint gating, taken under _lock.
        Honors the same launch grace as status(): robot commands must not
        slip through the start window before the env process appears in
        pgrep — a bare pgrep right after start() would report idle and let
        a zerorpc command fight the freshly launched env."""
        with self._lock:
            if (self._launched_at > 0.0
                    and time.monotonic() - self._launched_at < 8.0):
                return True
            return self._is_alive()

    def _read_gate(self) -> Optional[str]:
        """Read the start-gate sentinel ("WAIT"/"RUN") from the container.
        Returns None if the file is absent (no run yet / cleared) or unreadable
        — the caller then falls back to log parsing."""
        try:
            r = self._docker_exec(["cat", GATE_FILE], timeout=10.0)
            if r.returncode != 0:
                return None
            return r.stdout.decode("utf-8", errors="replace").strip() or None
        except Exception:
            return None

    def _tail_log(self) -> str:
        try:
            r = self._docker_exec(["tail", "-c", "20000", COLLECT_LOG],
                                  timeout=10.0)
            if r.returncode != 0:
                return ""
            return r.stdout.decode("utf-8", errors="replace")
        except Exception as e:
            _log.warning(f"log tail failed: {e}")
            return ""

    def _wait_cameras_released(self, settle_s: float = 3.0,
                               budget_s: float = 40.0) -> None:
        """After the host closes the ZEDs, the env must be able to OPEN them.

        A `get_device_list` 'AVAILABLE' check is NOT sufficient — the device
        reports available while open() still fails during USB teardown
        (observed 2026-06-15: env died with CAMERA FAILED TO SETUP right after
        the dashboard released). So probe by actually opening+closing each
        camera (DEPTH_MODE.NONE, exactly as the env does) from the container,
        retrying inside one process until it succeeds. Only return once the
        cameras are PROVEN openable, so the env's open right after lands on a
        warm device. The dashboard's in-process cam.close() frees the USB less
        cleanly than a process exit, so this can take longer than a few sec.
        """
        time.sleep(settle_s)
        serials = [int(s) for s in self.cams.serials.values()]
        probe = (
            "import pyzed.sl as sl, time, sys\n"
            f"sns={serials}\n"
            f"deadline=time.time()+{budget_s}\n"
            "while time.time()<deadline:\n"
            "    ok=True\n"
            "    for sn in sns:\n"
            "        c=sl.Camera(); ip=sl.InitParameters()\n"
            "        ip.set_from_serial_number(sn); ip.depth_mode=sl.DEPTH_MODE.NONE\n"
            "        st=c.open(ip)\n"
            "        if str(st)=='SUCCESS': c.close()\n"
            "        else: ok=False; break\n"
            "    if ok: print('PROBE_OK'); sys.exit(0)\n"
            "    time.sleep(2)\n"
            "print('PROBE_FAIL'); sys.exit(1)\n"
        )
        try:
            r = self._docker_exec([CONTAINER_PY, "-c", probe],
                                  timeout=budget_s + 25.0)
            if r.returncode == 0 and b"PROBE_OK" in r.stdout:
                _log.info("cameras open-verified from container")
                return
            _log.warning("camera open-probe did not confirm; launching anyway")
        except Exception as e:
            _log.warning(f"camera open-probe failed: {e}; launching anyway")

    # -- camera discipline ----------------------------------------------
    def _reclaim_cameras(self) -> None:
        # Idle camera preview is disabled in the HD-SVO collection workflow
        # (dashboard launched with --no-cam-on-start). Do NOT re-grab the ZEDs
        # after a run: keeping them free lets the next collection's in-container
        # env open them COLD, which is the only reliable handoff at HD720/HD1080
        # (a warm dashboard-release reopen hangs). If idle preview was on, only
        # reclaim then.
        if not self.cams.is_running() and self.cams.was_started:
            try:
                _log.info(f"cameras reclaimed: {self.cams.start()}")
            except Exception as e:
                _log.warning(f"camera reclaim failed: {e}")

    # -- legacy dashboard kill -------------------------------------------
    @staticmethod
    def _kill_legacy_dashboard() -> None:
        """SIGTERM any running _dashboard_openpi.py on this host — it holds
        the ZEDs and the zerorpc socket. The [.] char-class keeps pgrep's
        own argv from matching itself (kept even though argv-style
        subprocess has no shell line — this pattern has bitten us)."""
        try:
            r = subprocess.run(["pgrep", "-f", "[/_]openpi[.]py"],
                               capture_output=True, text=True, timeout=5.0)
        except Exception as e:
            _log.warning(f"legacy dashboard pgrep failed: {e}")
            return
        if r.returncode != 0:
            return
        for pid_s in r.stdout.split():
            try:
                pid = int(pid_s)
                os.kill(pid, signal.SIGTERM)
                _log.info(f"SIGTERM sent to legacy dashboard pid {pid}")
            except (ValueError, ProcessLookupError, PermissionError) as e:
                _log.warning(f"legacy dashboard kill pid={pid_s}: {e}")

    def _kill_orphan_collection(self) -> None:
        """Reap any stray collection processes INSIDE the container before a
        fresh start. A previous run that crashed, was force-killed, or was
        launched directly leaves orphan ``ray::DataCollector`` actors plus the
        ``collect_real_data`` / ``PolymetisController`` procs alive; they keep
        the ZEDs + GELLO held and wedge the next run (camera open() timeout,
        ttyACM busy, or a "refused" guard). SIGKILL is intentional — these are
        already dead-to-us strays, and the kernel releases their USB fds on
        exit so the new env can open the cameras cold."""
        try:
            self._docker_exec(["bash", "-c",
                               "pkill -9 -f ray::DataCollector; "
                               "pkill -9 -f collect_real_data; "
                               "pkill -9 -f PolymetisController; "
                               "true"])
            time.sleep(3.0)  # let the kernel reclaim USB (ZED + ttyACM)
            _log.info("reaped any stray collection processes before start")
        except Exception as e:
            _log.warning(f"orphan collection reap failed: {e}")

    # -- lifecycle --------------------------------------------------------
    def start(self, num_episodes: int, task_description: str) -> str:
        with self._lock:
            # Always start from a clean slate: reap any stray collection inside
            # the container (orphan Ray actors / a leftover collect_real_data
            # from a crash, a refused double-launch, or a direct launch). Strays
            # keep the ZEDs + GELLO held and wedge the new run ("refused" / the
            # env's camera open() failing). This makes start idempotent.
            self._kill_orphan_collection()
            self._kill_legacy_dashboard()

            # RLinf env needs exclusive ZED access — release ours first.
            # cams.stop() closes the sl.Camera handles synchronously, but the
            # ZED USB device needs a few seconds to become reacquirable by
            # another process (the in-container env). Without this settle the
            # env's camera open() races the USB teardown and dies with
            # "Camera opening timeout reached" (observed 2026-06-15; a manual
            # release + 5s wait then opened cleanly from the container).
            # Only release + open-probe if the dashboard is actually holding the
            # ZEDs for the idle preview. With idle preview OFF (--no-cam-on-start),
            # the cameras are already free and the in-container env opens them
            # COLD — a warm reopen right after a dashboard release/probe hangs at
            # HD720/HD1080 ("Camera opening timeout reached"). Cold open is proven
            # reliable at HD720.
            if self.cams.is_running():
                self.cams.stop()
                self._wait_cameras_released()

            # Truncate the previous run's log + clear the start-gate sentinel
            # so a stale WAIT/RUN from a prior run can't mislead the UI.
            try:
                self._docker_exec(
                    ["bash", "-c", f": > {COLLECT_LOG}; rm -f {GATE_FILE}"]
                )
            except Exception as e:
                _log.warning(f"log truncate failed: {e}")

            override = shlex.quote(
                f"env.eval.override_cfg.task_description={task_description}"
            )
            # Unique per-run dataset dir. LeRobotDataset.create() raises
            # FileExistsError if its target dir already exists, so reusing a
            # fixed save_dir crashes every run after the first success
            # (observed 2026-06-15: episode 1 saved, then FileExistsError on
            # outputs/pilot/.../id_0 killed the run + the controller actor).
            # A timestamped log_path makes each Start its own clean dataset.
            run_tag = time.strftime("%Y%m%d_%H%M%S")
            log_path = shlex.quote(f"./outputs/collect_{run_tag}")
            launch_cmd = (
                "cd /workspace/rlinf && "
                "export PYTHONPATH=/workspace/rlinf "
                "EMBODIED_PATH=/workspace/rlinf/examples/embodiment "
                "RLINF_LIVE_CAM_DIR=/workspace/rlinf/outputs/live_cam "
                "HF_LEROBOT_HOME=/workspace/rlinf/outputs/lerobot && "
                f"{CONTAINER_PY} examples/embodiment/collect_real_data.py "
                "--config-name realworld_collect_data_polymetis_jointvel "
                "env.eval.gello_port=/dev/gello "
                # Reach the controller over the robot net, not the config's
                # legacy Tailscale robot_ip (overrides node_groups[0] hardware).
                f"cluster.node_groups.0.hardware.configs.0.robot_ip={NUC1_HOST} "
                f"runner.num_data_episodes={int(num_episodes)} "
                f"runner.logger.log_path={log_path} "
                f"{override} "
                f"> {COLLECT_LOG} 2>&1"
            )
            try:
                r = self._docker_exec(["bash", "-c", launch_cmd], detach=True)
            except Exception as e:
                self._reclaim_cameras()
                return f"launch failed: {e}"
            if r.returncode != 0:
                err = r.stderr.decode("utf-8", errors="replace").strip()[:200]
                self._reclaim_cameras()
                return f"launch failed: {err}"

            self._was_running = True
            self._launched_at = time.monotonic()
            self.last_result = None
            self.last_launch = {
                "num_episodes": int(num_episodes),
                "task_description": task_description,
                "started_at": time.time(),
            }
            msg = (f"started: {num_episodes} episode(s), "
                   f"task={task_description!r}")
            # Fresh launch = the moment to recheck container visibility:
            # collection still starts (physical 'c' always works), but warn
            # if the dashboard's mark-success injection can't land.
            if (self.vkbd is not None and self.vkbd.path is not None
                    and not vkbd_visible_in_container(self.vkbd.path,
                                                      force=True)):
                msg += (" — warning: virtual keyboard not visible inside "
                        "rlinf-eval (/dev is a startup snapshot); "
                        "mark-success injection won't land — use physical "
                        "'c', or docker restart rlinf-eval before next run")
            self.last_action_msg = msg
            _log.info(msg)
            return msg

    def stop(self) -> str:
        with self._lock:
            if not self._is_alive():
                self._was_running = False
                self._reclaim_cameras()
                return "not running"

            # Kill ONLY — NEVER combined with a relaunch in the same shell
            # string (that pattern self-killed us once).
            try:
                self._docker_exec(
                    ["bash", "-c",
                     'pgrep -f "collect_real_dat[a]" | xargs -r kill'],
                    timeout=10.0,
                )
            except Exception as e:
                return f"kill failed: {e}"

            deadline = time.time() + 15.0
            exited = False
            while time.time() < deadline:
                if not self._is_alive():
                    exited = True
                    break
                time.sleep(0.5)
            if not exited:
                return ("kill sent but process still alive after 15s — "
                        "not reclaiming cameras; retry Stop")

            # Process is gone: Ray cleanup, then reclaim cameras.
            try:
                self._docker_exec(
                    ["bash", "-c",
                     f"{CONTAINER_RAY} stop --force; rm -rf /tmp/ray"],
                    timeout=60.0,
                )
            except Exception as e:
                _log.warning(f"ray cleanup failed: {e}")

            self.last_result = parse_collect_log(self._tail_log())
            self._was_running = False
            self._reclaim_cameras()
            msg = "stopped"
            self.last_action_msg = msg
            _log.info("collection stopped + cleaned up")
            return msg

    def status(self) -> dict:
        alive = self._is_alive()
        tail = self._tail_log()
        parsed = parse_collect_log(tail)
        startup = parse_startup_progress(tail)
        # Primary signal = the gate sentinel file (reliable, bypasses Ray's
        # stdout buffering); fall back to log text if the file read fails.
        gate = self._read_gate()
        if gate is not None:
            waiting_for_start = bool(alive and gate == "WAIT")
        else:
            waiting_for_start = bool(alive and is_waiting_for_start(tail))
        with self._lock:
            if self._was_running and not alive:
                # A poll once raced a fresh start(): its pre-lock alive=False
                # sample triggered a false "exited" transition that reclaimed
                # the cameras under the new env. Decide the transition on
                # in-lock state only: re-check liveness + launch grace period.
                in_grace = time.monotonic() - self._launched_at < 8.0
                if not in_grace and not self._is_alive():
                    # genuine running -> exited transition: keep the final
                    # parse, reclaim cameras.
                    self.last_result = parsed
                    self._was_running = False
                    self._reclaim_cameras()
                    _log.info(f"collection exited; final: {parsed}")
            elif alive:
                self._was_running = True
        return {
            "running": alive,
            **parsed,
            "startup": startup,
            "waiting_for_start": waiting_for_start,
            "last_result": self.last_result,
            "last_launch": self.last_launch,
            "last_action_msg": self.last_action_msg,
        }


# ─────────────────────────────────────────────────────────────────────
# Endpoint gating — pure function, no I/O (unit-tested)
# ─────────────────────────────────────────────────────────────────────
def check_gate(collection_alive: bool, require_collecting: bool) -> tuple[bool, str]:
    """Single gating rule for operator endpoints.

    Robot actions (require_collecting=False) are idle-only — the RLinf env
    process owns the robot during a run and concurrent zerorpc commands
    would fight it. Mark-success (require_collecting=True) is the opposite:
    injecting 'c' only makes sense while an episode is being collected.
    """
    if require_collecting and not collection_alive:
        return False, "collection not running"
    if not require_collecting and collection_alive:
        return False, "collection running"
    return True, ""


# ─────────────────────────────────────────────────────────────────────
# Dataset manager — LeRobot v2.1 scan / playback / delete helpers
# ─────────────────────────────────────────────────────────────────────
# Verified layout (pilot dataset, 2026-06-12): meta/info.json,
# meta/episodes.jsonl, meta/episodes_stats.jsonl,
# data/chunk-XXX/episode_XXXXXX.parquet. dtype "image" features are
# embedded PER ROW in the parquet as struct{bytes: binary, path: string}
# holding encoded PNG bytes — there is no images/ or videos/ directory.
#
# dsid scheme: urlsafe base64 (padding stripped) of
# "<root_index>:<name>" — root_index pins the configured root, name is
# the dataset path relative to that root. Encoding the index prevents
# same-named datasets under different roots from shadowing each other
# (first-match resolution once made a delete target the wrong root).
# Base64 survives slashes in nested names without fighting Flask's path
# routing; names may themselves contain ':' (split at first colon only).

def dsid_encode(root_index: int, name: str) -> str:
    raw = f"{root_index}:{name}".encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def dsid_decode(dsid: str) -> tuple[int, str]:
    pad = "=" * (-len(dsid) % 4)
    raw = base64.urlsafe_b64decode(dsid + pad).decode("utf-8")
    idx_s, sep, name = raw.partition(":")
    if not sep:
        raise ValueError(f"malformed dsid payload: {raw!r}")
    return int(idx_s), name


def is_inside_root(path: str, root: str) -> bool:
    """True iff realpath(path) is STRICTLY inside realpath(root).

    Delete-path guard: symlinks are resolved first, so a link pointing
    outside the root does not pass; the root itself does not pass either
    (we never rmtree a configured root).
    """
    rp = os.path.realpath(os.path.expanduser(path))
    rr = os.path.realpath(os.path.expanduser(root))
    try:
        rel = os.path.relpath(rp, rr)
    except ValueError:
        return False
    return rel != "." and not rel.startswith("..") and not os.path.isabs(rel)


def _episode_parquet_path(ds_dir: str, info: dict, episode_index: int) -> str:
    chunks_size = int(info.get("chunks_size") or 1000)
    tmpl = (info.get("data_path")
            or "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    rel = tmpl.format(episode_chunk=episode_index // chunks_size,
                      episode_index=episode_index)
    return os.path.join(ds_dir, rel)


def _image_keys(info: dict) -> list[str]:
    """dtype=='image' feature names; canonical pair first so playback
    hstacks as (image | extra_view_image)."""
    feats = info.get("features") or {}
    keys = [k for k in ("image", "extra_view_image")
            if isinstance(feats.get(k), dict) and feats[k].get("dtype") == "image"]
    keys += sorted(k for k, v in feats.items()
                   if isinstance(v, dict) and v.get("dtype") == "image"
                   and k not in keys)
    return keys


def _successes_from_parquet(ds_dir: str, info: dict) -> Optional[tuple[int, int]]:
    """(n_success, n_episodes) from each episode parquet's LAST-frame
    is_success. Needs pyarrow; None if unavailable/undeterminable."""
    if not HAS_PYARROW:
        return None
    import pyarrow.parquet as pq
    total = int(info.get("total_episodes") or 0)
    if total <= 0:
        return None
    n_success = n_eps = 0
    for i in range(total):
        path = _episode_parquet_path(ds_dir, info, i)
        if not os.path.isfile(path):
            continue
        pf = pq.ParquetFile(path)
        if "is_success" not in pf.schema_arrow.names or pf.metadata.num_rows == 0:
            return None
        col = pf.read(columns=["is_success"]).column("is_success")
        n_eps += 1
        if col[pf.metadata.num_rows - 1].as_py():
            n_success += 1
    return (n_success, n_eps) if n_eps else None


def _count_episode_successes(ds_dir: str, info: dict) -> Optional[tuple[int, int]]:
    """Per-episode success: prefer meta/episodes_stats.jsonl
    is_success.max (cheap — no parquet read, works without pyarrow),
    fall back to last-frame is_success from each episode parquet."""
    stats_path = os.path.join(ds_dir, "meta", "episodes_stats.jsonl")
    if os.path.isfile(stats_path):
        n_eps = n_success = 0
        try:
            with open(stats_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    s = (d.get("stats") or {}).get("is_success")
                    if not isinstance(s, dict) or "max" not in s:
                        n_eps = 0
                        break
                    mx = s["max"]
                    if isinstance(mx, list):
                        mx = mx[0] if mx else None
                    n_eps += 1
                    if mx:
                        n_success += 1
        except (OSError, ValueError):
            n_eps = 0
        if n_eps:
            return n_success, n_eps
    return _successes_from_parquet(ds_dir, info)


def scan_datasets(roots: list[str]) -> list[dict]:
    """Scan roots for LeRobot datasets (any dir holding meta/info.json).

    Per-dataset I/O is wrapped so one corrupt dataset degrades to an
    entry with `error` set instead of killing the scan. Walks the disk —
    call on demand only, never from the 1.5s status poll.
    """
    entries: list[dict] = []
    for root_index, root in enumerate(roots):
        root_r = os.path.realpath(os.path.expanduser(root))
        if not os.path.isdir(root_r):
            continue
        pattern = os.path.join(glob.escape(root_r), "**", "meta", "info.json")
        for info_path in sorted(glob.glob(pattern, recursive=True)):
            ds_dir = os.path.dirname(os.path.dirname(info_path))
            name = os.path.relpath(ds_dir, root_r)
            entry: dict = {
                "name": name,
                "dsid": dsid_encode(root_index, name),
                "root": root_r,
                "total_episodes": None,
                "total_frames": None,
                "fps": None,
                "success_rate": None,
                "n_success": None,
                "size_bytes": None,
                "mtime": None,
                "error": None,
            }
            try:
                with open(info_path) as f:
                    info = json.load(f)
                entry["total_episodes"] = info.get("total_episodes")
                entry["total_frames"] = info.get("total_frames")
                entry["fps"] = info.get("fps")
                size = 0
                mtime = 0.0
                for dirpath, _dirs, files in os.walk(ds_dir):
                    for fn in files:
                        try:
                            st = os.stat(os.path.join(dirpath, fn))
                        except OSError:
                            continue
                        size += st.st_size
                        mtime = max(mtime, st.st_mtime)
                entry["size_bytes"] = size
                entry["mtime"] = mtime or None
                succ = _count_episode_successes(ds_dir, info)
                if succ is not None:
                    n_success, n_eps = succ
                    entry["n_success"] = n_success
                    entry["success_rate"] = n_success / n_eps if n_eps else None
            except Exception as e:
                entry["error"] = f"{e.__class__.__name__}: {e}"[:200]
            entries.append(entry)
    return entries


def _resolve_dataset(dsid: str, roots: list[str]) -> Optional[str]:
    """dsid -> dataset dir, or None. The dsid pins a root index, so a
    same-named dataset under another root can never be targeted by
    mistake; an out-of-range index resolves to None (-> 404). The
    candidate must resolve strictly inside its root (realpath, so no
    ../ or symlink escapes) and hold meta/info.json."""
    try:
        root_index, name = dsid_decode(dsid)
    except Exception:
        return None
    if not name or os.path.isabs(name):
        return None
    if not 0 <= root_index < len(roots):
        return None
    root = roots[root_index]
    try:
        cand = os.path.realpath(os.path.join(os.path.expanduser(root), name))
        if (is_inside_root(cand, root)
                and os.path.isfile(os.path.join(cand, "meta", "info.json"))):
            return cand
    except ValueError:
        # e.g. embedded null byte in the decoded name — not-found, not 500.
        return None
    return None


def legacy_datasets_in_container() -> Optional[bool]:
    """True if pre-migration datasets still exist at the container's old
    HF_LEROBOT_HOME (UI shows a one-line notice; they are never scanned).
    None when the probe itself fails (docker down etc.)."""
    try:
        r = subprocess.run(
            ["docker", "exec", CONTAINER, "test", "-d", LEGACY_CONTAINER_LEROBOT],
            capture_output=True, timeout=10.0)
    except Exception as e:
        _log.warning(f"legacy dataset probe failed: {e}")
        return None
    return r.returncode == 0


def _decode_image_cell(cell) -> Optional["object"]:
    """LeRobot v2.1 image cell -> BGR ndarray. Cell is the HF image
    struct {bytes, path} (verified: PNG bytes embedded per row); plain
    bytes accepted too."""
    if isinstance(cell, dict):
        cell = cell.get("bytes")
    if not cell:
        return None
    import cv2
    import numpy as np
    return cv2.imdecode(np.frombuffer(cell, np.uint8), cv2.IMREAD_COLOR)


def _hstack_frames(tiles: list) -> "object":
    import cv2
    import numpy as np
    h = tiles[0].shape[0]
    norm = []
    for t in tiles:
        if t.shape[0] != h:
            w = max(1, int(round(t.shape[1] * h / t.shape[0])))
            t = cv2.resize(t, (w, h), interpolation=cv2.INTER_NEAREST)
        norm.append(t)
    return norm[0] if len(norm) == 1 else np.hstack(norm)


# ─────────────────────────────────────────────────────────────────────
# Mark-success — uinput virtual keyboard
# ─────────────────────────────────────────────────────────────────────
class VirtualKeyboard:
    """uinput virtual keyboard that injects KEY_C for success labeling.

    INVESTIGATION (vendor/RLinf/rlinf/envs/realworld/common/keyboard/
    keyboard_listener.py, read 2026-06-12) — KeyboardListener device
    selection rule in _open_keyboard_device:
      1. RLINF_KEYBOARD_DEVICE env override wins (capability-checked).
      2. Otherwise it iterates sorted(evdev.list_devices()) — a LEXICOGRAPHIC
         sort of /dev/input/event* paths, so "event10" < "event2" — and binds
         the FIRST device whose EV_KEY capabilities include KEY_A, KEY_B,
         KEY_C, KEY_Q. There is NO name filter; it reads exactly ONE device.

    Design consequences:
      * Our device advertises exactly that capability set, so it satisfies
        the rule. But whether it beats the physical Dell keyboard depends on
        event-node numbering (which we don't control) under a lexicographic
        sort — i.e. selection between the two is effectively arbitrary.
      * The device MUST exist before collection starts: the listener scans
        /dev/input once, at env startup. Hence creation at dashboard startup,
        not lazily on first button press.
      * Deterministic knob: set RLINF_KEYBOARD_DEVICE=<self.path> in the
        collection launch env. Deliberately NOT wired in by default — if the
        container can't see the host-created node, KeyboardListener raises at
        env start and the whole run dies, whereas today the physical keyboard
        always works as fallback. /api/mark_success reports which device the
        listener would pick so the Task D smoke can decide.
      * Container caveat (verified bench fact): rlinf-eval's privileged /dev
        is a SNAPSHOT taken at container start — a /dev/input/eventN node we
        create here afterwards is INVISIBLE inside until
        `docker restart rlinf-eval`. vkbd_visible_in_container() checks this.
    """

    def __init__(self):
        from evdev import UInput, ecodes
        self._ecodes = ecodes
        # A/B/C/Q satisfy the listener's device-detection caps; S is the
        # episode-start key. All injectable.
        caps = {ecodes.EV_KEY: [ecodes.KEY_A, ecodes.KEY_B, ecodes.KEY_C,
                                ecodes.KEY_Q, ecodes.KEY_S]}
        self._ui = UInput(caps, name=VIRTUAL_KBD_NAME)  # needs /dev/uinput (root)
        dev = getattr(self._ui, "device", None)
        self.path: Optional[str] = dev.path if dev is not None else None

    def inject_key(self, key_code: int, hold_s: float = 0.25) -> None:
        """Press + hold + release a key.

        Hold before release: KeyboardListener latches the key on press
        (value 1) and CLEARS it on release (value 0); the env loop only
        polls get_key() at control rate, so an instantaneous press+release
        could be cleared before any poll observes it. 250ms matches a
        deliberate human tap.
        """
        ec = self._ecodes
        self._ui.write(ec.EV_KEY, key_code, 1)
        self._ui.syn()
        time.sleep(hold_s)
        self._ui.write(ec.EV_KEY, key_code, 0)
        self._ui.syn()

    def inject_success(self, hold_s: float = 0.25) -> None:
        """Inject 'c' = label current episode success + done."""
        self.inject_key(self._ecodes.KEY_C, hold_s)

    def inject_start(self, hold_s: float = 0.25) -> None:
        """Inject 's' = start the next episode (release the wait-for-start gate)."""
        self.inject_key(self._ecodes.KEY_S, hold_s)

    def close(self) -> None:
        try:
            self._ui.close()
        except Exception:
            pass


def predict_listener_device() -> Optional[dict]:
    """Replicates KeyboardListener's scan (sorted paths, first device with
    KEY_A/B/C/Q caps) so /api/mark_success can report whether the listener
    would bind OUR virtual device or the physical keyboard. Advisory only:
    runs host-side at request time, while the listener scanned inside the
    container at env start."""
    from evdev import InputDevice, ecodes, list_devices
    required = {ecodes.KEY_A, ecodes.KEY_B, ecodes.KEY_C, ecodes.KEY_Q}
    for path in sorted(list_devices()):
        try:
            dev = InputDevice(path)
        except OSError:
            continue
        try:
            keys = set(dev.capabilities(verbose=False).get(ecodes.EV_KEY, []))
            if required.issubset(keys):
                return {"path": path, "name": dev.name}
        finally:
            dev.close()
    return None


# Cache for vkbd_visible_in_container(): one probe per dashboard lifetime
# unless force=True (start() forces, mark_success reads the cache).
_VKBD_VISIBILITY_CACHE: dict = {"checked": False, "visible": False}


def vkbd_visible_in_container(device_path: Optional[str],
                              force: bool = False) -> bool:
    """Whether the host-created virtual keyboard node exists INSIDE
    rlinf-eval. The container's /dev is a snapshot from container start, so
    nodes created after that are invisible inside until
    `docker restart rlinf-eval` — and the KeyboardListener runs inside, so a
    host-side predict_listener_device() match alone is false confidence.
    Result cached per dashboard lifetime; force=True rechecks on demand
    (a failed probe, e.g. docker timeout, is not cached)."""
    if device_path is None:
        return False
    if _VKBD_VISIBILITY_CACHE["checked"] and not force:
        return _VKBD_VISIBILITY_CACHE["visible"]
    try:
        r = subprocess.run(
            ["docker", "exec", CONTAINER, "test", "-e", device_path],
            capture_output=True, timeout=10.0)
    except Exception as e:
        _log.warning(f"vkbd container-visibility probe failed: {e}")
        return False
    visible = r.returncode == 0
    _VKBD_VISIBILITY_CACHE["checked"] = True
    _VKBD_VISIBILITY_CACHE["visible"] = visible
    return visible


# ─────────────────────────────────────────────────────────────────────
# Robot panel — minimal zerorpc client to the DROID container on NUC1
# ─────────────────────────────────────────────────────────────────────
class DroidClient:
    """Minimal zerorpc client for the robot panel (trimmed port of
    vendor/RLinf/rlinf/envs/realworld/franka/droid_zerorpc_client.py).

    Lifetime: created FRESH inside each Flask request handler and closed in
    a finally. Why not one shared client: zerorpc rides on gevent, whose hub
    is thread-local; Flask threaded=True serves each request on an arbitrary
    thread, and a client created on thread A dies with an opaque LoopExit
    when used from thread B. Per-request connect is plain ZMQ socket setup
    (cheap), and the panel runs at button-press frequency — short-lived
    clients are the simple correct choice over a dedicated owner thread +
    queue. The vendored dual-timeout pair (5s fast / 30s slow) becomes a
    per-instance timeout: state/home/gripper use a short one, recover uses
    30s (launch_controller alone takes ~10s).

    Load-bearing semantics preserved from the vendored client:
      * POSITIONAL args only — kwargs over this zerorpc are silently
        dropped, falling back to action_space="cartesian_velocity" (wrong
        joint motion). Cost 30min of debugging once; never kwargs.
      * get_robot_state() returns a 2-list [state, ts]; unwrap.
      * Joint moves are STREAMED — DROID adaptive_time_to_go() returns 0 for
        small displacements, making a blocking one-shot a silent no-op.
    """

    def __init__(self, address: str = DROID_ADDR, timeout: int = 5):
        import zerorpc as _zerorpc
        self._c = _zerorpc.Client(heartbeat=20, timeout=timeout)
        self._c.connect(address)

    def close(self) -> None:
        try:
            self._c.close()
        except Exception:
            pass

    def get_robot_state(self) -> dict:
        state, _ts = self._c.get_robot_state()  # [state, ts] 2-list
        return {
            "joint_positions": [float(x) for x in state["joint_positions"]],
            "gripper_position": float(state["gripper_position"]),
        }

    def stream_joint_position(self, q7, duration_s: float = 3.5, hz: int = 15,
                              gripper_cmd: float = 0.0) -> None:
        a = [float(x) for x in q7] + [float(gripper_cmd)]
        assert len(a) == 8
        for _ in range(int(duration_s * hz)):
            # POSITIONAL: action, action_space, gripper_action_space, blocking
            self._c.update_command(a, "joint_position", "position", False)
            time.sleep(1.0 / hz)

    def move_to_joint_target(self, target_q7,
                             gripper_cmd: Optional[float] = None,
                             max_step: float = 0.06, hz: int = 15,
                             timeout_s: float = 25.0, tol: float = 0.03) -> float:
        """Closed-loop leashed approach to an absolute joint target (returns
        max joint error rad). Holds current gripper.

        Each tick reads the ACTUAL joint position and commands an impedance
        setpoint at most `max_step` rad ahead toward the target. Because the
        setpoint stays just ahead of the measured pose, the impedance error
        (hence torque, hence speed ~= max_step*hz rad/s) is bounded — no
        full-speed lunge, no Franka velocity/accel reflex.

        Why not the alternatives (both tried 2026-06-15, both failed):
        - streaming the FINAL target (old home): huge error -> impedance lunge
          -> reflex / safety stop;
        - blocking=True (move_to_joint_positions): WEDGED the polymetis gRPC
          server, hung 30s, dropped the controller (needed container restart);
        - open-loop cosine interpolation from start pose: setpoints too small
          per step, arm never tracked (didn't move at all).
        """
        if gripper_cmd is None:
            gripper_cmd = self.get_robot_state()["gripper_position"]
        target = [float(x) for x in target_q7]
        deadline = time.monotonic() + timeout_s
        maxerr = float("inf")
        while True:
            cur = self.get_robot_state()["joint_positions"]
            err = [t - c for t, c in zip(target, cur)]
            maxerr = max(abs(e) for e in err)
            if maxerr < tol or time.monotonic() > deadline:
                return maxerr
            setpoint = [c + max(-max_step, min(max_step, e))
                        for c, e in zip(cur, err)]
            self._c.update_command(setpoint + [float(gripper_cmd)],
                                   "joint_position", "position", False)
            time.sleep(1.0 / hz)

    def update_gripper(self, command: float) -> None:
        cmd = min(max(float(command), 0.0), 1.0)
        # POSITIONAL: command, velocity, blocking
        self._c.update_gripper(cmd, False, False)

    def kill_controller(self) -> None:
        self._c.kill_controller()

    def bootstrap(self, settle_seconds: float = 8.0) -> None:
        """launch_controller (spawn polymetis driver), settle, launch_robot
        (connect RobotInterface). Same sequence as the vendored client."""
        self._c.launch_controller()
        time.sleep(settle_seconds)
        self._c.launch_robot()


# ─────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────
INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TASL FR3 — collection dashboard</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; margin: 18px;
         background: #111; color: #e5e5e5; }
  h1 { font-size: 1.4rem; margin: 0 0 12px 0; }
  h3 { margin-top: 0; }
  .row { display: flex; gap: 16px; align-items: flex-start; flex-wrap: wrap; }
  .col { flex: 1 1 0; min-width: 280px; }
  .cam { background: #000; padding: 6px; border-radius: 6px;
         border: 1px solid #333; }
  .cam img { width: 100%; max-width: 480px; display: block; border-radius: 4px;}
  .cam .label { font-size: 0.85rem; color: #aaa; padding: 4px 0; }
  .cam .placeholder { color: #888; padding: 40px 10px; text-align: center;
                      font-size: 0.9rem; }
  .ctl { background: #1a1a1a; padding: 14px; border-radius: 6px;
         border: 1px solid #2a2a2a; }
  label { font-size: 0.85rem; color: #aaa; display: block; margin-top: 8px; }
  input[type=text], input[type=number] {
         width: 100%; padding: 8px; font-size: 1rem; box-sizing: border-box;
         background: #0c0c0c; color: #eee;
         border: 1px solid #333; border-radius: 4px; }
  button { padding: 10px 18px; font-size: 1rem; cursor: pointer;
           border: 1px solid #333; border-radius: 4px;
           background: #222; color: #eee; margin: 10px 6px 0 0; }
  button:hover { background: #2c2c2c; }
  button:disabled { opacity: 0.35; cursor: not-allowed; }
  button.primary { background: #1f4f1f; }
  button.danger  { background: #5a1f1f; }
  button.marksuccess { display: none; width: 100%; padding: 26px 0;
         font-size: 1.5rem; font-weight: 600; margin-top: 12px;
         background: #1f4f1f; border-radius: 10px; }
  button.startep { display: none; width: 100%; padding: 26px 0;
         font-size: 1.5rem; font-weight: 600; margin-top: 12px;
         background: #1565c0; border-radius: 10px; }
  table { font-size: 0.9rem; border-collapse: collapse; }
  td, th { padding: 4px 8px; border-bottom: 1px solid #2a2a2a;
           text-align: left; vertical-align: top; }
  .err { color: #e58c8c; white-space: pre-wrap; font-family: monospace;
         font-size: 0.8rem; }
  .dot { display: inline-block; width: 8px; height: 8px;
         border-radius: 50%; margin-right: 6px; }
  .ok { background: #4caf50; } .bad { background: #d6322a; }
  .idle { background: #777; }
  button.small { padding: 4px 10px; font-size: 0.85rem; margin: 0 4px 0 0; }
  .notice { font-size: 0.8rem; color: #caa84f; margin: 6px 0; }
  #playOverlay { display: none; position: fixed; inset: 0; z-index: 20;
         background: rgba(0,0,0,0.85); flex-direction: column;
         align-items: center; justify-content: center; }
  #playOverlay img { image-rendering: pixelated; border: 1px solid #444;
         border-radius: 4px; max-width: 92vw; max-height: 80vh; }
  #playTitle { color: #ccc; font-size: 0.9rem; margin-bottom: 8px; }
</style></head>
<body>
<h1>TASL FR3 — 数据采集 dashboard</h1>

<div class="row">
  <div class="col ctl">
    <h3>状态</h3>
    <div id="status">loading...</div>
    <button id="startEpBtn" class="startep"
            onclick="markStartEpisode()">开始下一条 (注入 's')</button>
    <button id="markSuccessBtn" class="marksuccess"
            onclick="markSuccess()">标记成功 (注入 'c')</button>
  </div>
  <div class="col ctl">
    <h3>采集控制</h3>
    <label>Episode 数</label>
    <input type="number" id="numEpisodes" value="10" min="1"/>
    <label>任务描述</label>
    <input type="text" id="taskDesc" value="pick up the cube"/>
    <div>
      <button class="primary" onclick="startCollect()">Start</button>
      <button class="danger" onclick="stopCollect()">Stop</button>
    </div>
    <div id="actionMsg" style="font-size:0.8rem;color:#888;margin-top:8px"></div>
  </div>
</div>

<div class="row" style="margin-top:16px">
  <div class="col cam">
    <div class="label">exterior (ZED 2i)</div>
    <img id="cam-exterior" alt="exterior"/>
    <div id="ph-exterior" class="placeholder" style="display:none"></div>
  </div>
  <div class="col cam">
    <div class="label">wrist (ZED Mini)</div>
    <img id="cam-wrist" alt="wrist"/>
    <div id="ph-wrist" class="placeholder" style="display:none"></div>
  </div>
</div>

<div class="row" style="margin-top:16px">
  <div class="col ctl" id="robot-panel">
    <h3>机械臂</h3>
    <div id="robotState">-</div>
    <div>
      <button id="btnHome" onclick="robotHome()">Home</button>
      <button id="btnRecover" onclick="robotRecover()">Recover</button>
      <button id="btnGripOpen" onclick="gripper('open')">夹爪开</button>
      <button id="btnGripClose" onclick="gripper('close')">夹爪合</button>
    </div>
    <div id="robotMsg" style="font-size:0.8rem;color:#888;margin-top:8px"></div>
  </div>
</div>
<div class="row" style="margin-top:16px">
  <div class="col ctl" id="dataset-panel">
    <h3>数据集</h3>
    <div id="dsNotice" class="notice" style="display:none"></div>
    <div id="dsTable">loading...</div>
    <button onclick="refreshDatasets()">刷新 (重新扫描)</button>
    <div id="dsMsg" style="font-size:0.8rem;color:#888;margin-top:8px"></div>
  </div>
</div>

<div id="playOverlay">
  <div id="playTitle"></div>
  <img id="playImg" alt="episode playback"/>
  <div><button onclick="closePlayback()">关闭</button></div>
</div>

<script>
let camsShown = null;  // null = unknown, force first sync

function setCams(running, collecting) {
  // Show the live MJPEG feed when the dashboard holds the cameras (idle
  // preview) OR when a collection is running (the env writes frames to the
  // bind-mounted live_cam dir, served via the same route). Placeholder only
  // when neither feed is available.
  const show = running || collecting;
  const key = show ? (collecting ? 'live' : 'idle') : 'off';
  if (camsShown === key) return;
  camsShown = key;
  for (const name of ['exterior', 'wrist']) {
    const img = document.getElementById('cam-' + name);
    const ph = document.getElementById('ph-' + name);
    if (show) {
      img.src = '/cam/' + name + '.mjpg?ts=' + Date.now();
      img.style.display = '';
      ph.style.display = 'none';
    } else {
      img.removeAttribute('src');
      img.style.display = 'none';
      ph.textContent = '采集未运行 — 相机空闲';
      ph.style.display = '';
    }
  }
}

async function api(path, body) {
  const r = await fetch(path, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: body ? JSON.stringify(body) : '{}',
  });
  const j = await r.json();
  document.getElementById('actionMsg').textContent = j.msg || JSON.stringify(j);
  return j;
}

async function startCollect() {
  const n = parseInt(document.getElementById('numEpisodes').value, 10);
  const task = document.getElementById('taskDesc').value.trim();
  if (!n || n < 1) { alert('Episode 数必须 >= 1'); return; }
  if (!task) { alert('任务描述不能为空'); return; }
  return api('/api/collect/start', {num_episodes: n, task_description: task});
}

async function stopCollect() {
  if (!confirm('确认停止采集？(kill 进程 + ray cleanup)')) return;
  return api('/api/collect/stop');
}

async function markSuccess() {
  const j = await api('/api/mark_success');
  if (j && j.caveat) {
    document.getElementById('actionMsg').textContent =
      (j.msg || '') + ' — ' + j.caveat;
  }
}

async function markStartEpisode() {
  const j = await api('/api/start_episode');
  if (j && j.caveat) {
    document.getElementById('actionMsg').textContent =
      (j.msg || '') + ' — ' + j.caveat;
  }
}

async function robotApi(path) {
  const el = document.getElementById('robotMsg');
  el.textContent = '...';
  try {
    const r = await fetch(path, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: '{}',
    });
    const j = await r.json();
    el.textContent = j.msg || JSON.stringify(j);
    return j;
  } catch (e) {
    el.textContent = 'error: ' + e;
  }
}

async function robotHome() {
  if (!confirm('确认 Home？机械臂将移动到 DROID home 位姿 (~4s)')) return;
  return robotApi('/api/robot/home');
}

async function robotRecover() {
  if (!confirm('确认 Recover？kill controller + 重新 bootstrap (~15s, 同步等待)')) return;
  return robotApi('/api/robot/recover');
}

async function gripper(action) {
  return robotApi('/api/robot/gripper/' + action);
}

function esc(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
                  .replace(/>/g, '&gt;');
}

function renderRobot(j, running) {
  const waiting = !!(j.collection && j.collection.waiting_for_start);
  // Waiting at the start gate: show 开始下一条; mid-episode: show 标记成功.
  document.getElementById('startEpBtn').style.display =
    waiting ? 'block' : 'none';
  document.getElementById('markSuccessBtn').style.display =
    (running && !waiting) ? 'block' : 'none';
  for (const id of ['btnHome', 'btnRecover', 'btnGripOpen', 'btnGripClose']) {
    document.getElementById(id).disabled = !!running;
  }
  const rs = document.getElementById('robotState');
  if (running) {
    rs.innerHTML = '采集中 — 面板禁用 (env 占用机械臂)';
  } else if (j.robot) {
    if (j.robot.controller_up) {
      rs.innerHTML = '<span class="dot ok"></span>controller up · q=['
        + j.robot.q.map(x => x.toFixed(2)).join(', ') + '] · 夹爪 '
        + (j.robot.gripper_position != null
           ? j.robot.gripper_position.toFixed(2) : '-');
    } else {
      rs.innerHTML = '<span class="dot bad"></span>controller down'
        + (j.robot.msg ? ' — ' + esc(j.robot.msg) : '');
    }
  } else {
    rs.innerHTML = '<span class="dot idle"></span>无状态 (zerorpc 不可用?)';
  }
}

async function refresh() {
  try {
    const r = await fetch('/api/status');
    const j = await r.json();
    const c = j.collection || {};
    let stateLabel, dotCls;
    if (c.running) {
      stateLabel = c.phase === 'error' ? '运行中 (有错误)' : '采集中';
      dotCls = c.phase === 'error' ? 'bad' : 'ok';
    } else if (c.phase === 'error') {
      stateLabel = '已退出 (错误)'; dotCls = 'bad';
    } else if (c.last_result) {
      stateLabel = '已结束'; dotCls = 'idle';
    } else {
      stateLabel = '空闲'; dotCls = 'idle';
    }
    let html = '<table>';
    html += '<tr><td><span class="dot ' + dotCls + '"></span>状态</td><td>'
         + esc(stateLabel) + ' (phase=' + esc(c.phase || '-') + ')</td></tr>';
    // Start gate: arm has reset and is waiting for the operator to begin the
    // (next) episode. Reposition the scene, then press 开始下一条 (or 's').
    if (c.waiting_for_start) {
      const done = c.episodes_done || 0;
      html += '<tr><td>就绪</td><td>'
        + '<b style="color:#1565c0">⏸ 机械臂已复位 — 摆好物体后按「开始下一条」'
        + '(或键盘 s 键)</b><br>'
        + '<span style="font-size:12px;color:#777">已完成 ' + done
        + (c.target ? ' / ' + c.target : '') + ' 条</span></td></tr>';
    }
    // Startup progress: only while a run is spinning up and not yet
    // controllable (env/Ray/camera/reset). Tells the operator when SpaceMouse
    // goes live so they don't poke a not-yet-ready arm.
    const su = c.startup;
    if (c.running && !c.waiting_for_start && su && !c.error
        && (!su.ready || c.episodes_done === 0)) {
      const ready = su.ready;
      const barColor = ready ? '#2e7d32' : '#1565c0';
      const banner = ready
        ? '<b style="color:#2e7d32">✅ 可以遥操了 — SpaceMouse 已生效</b>'
        : '<b style="color:#1565c0">⏳ 启动中,请勿操作 SpaceMouse…</b>';
      let chk = (su.milestones || []).map(function(m) {
        return (m.done ? '✓ ' : '· ') + esc(m.label);
      }).join(' &nbsp; ');
      html += '<tr><td>启动</td><td>' + banner
        + '<div style="margin:4px 0;background:#ddd;border-radius:4px;'
        + 'height:14px;width:100%;overflow:hidden">'
        + '<div style="height:100%;width:' + (su.percent || 0) + '%;'
        + 'background:' + barColor + ';transition:width .4s"></div></div>'
        + '<div style="font-size:12px;color:#555">' + esc(su.label || '')
        + ' (' + (su.percent || 0) + '%)</div>'
        + '<div style="font-size:11px;color:#777;margin-top:2px">' + chk
        + '</div></td></tr>';
    }
    html += '<tr><td>进度</td><td>'
         + 'episodes ' + (c.episodes_done ?? '-')
         + (c.target ? ' / target ' + c.target : '')
         + ' · 成功 ' + (c.success_count ?? '-') + '</td></tr>';
    html += '<tr><td>最近事件</td><td>' + esc(c.last_event || '-') + '</td></tr>';
    if (c.error) {
      html += '<tr><td>错误</td><td><div class="err">' + esc(c.error)
           + '</div></td></tr>';
    }
    if (!c.running && c.last_result) {
      const lr = c.last_result;
      html += '<tr><td>上次结果</td><td>成功 ' + (lr.success_count ?? '-')
           + (lr.target ? '/' + lr.target : '')
           + ' · ' + esc(lr.last_event || '-') + '</td></tr>';
    }
    html += '<tr><td><span class="dot ' + (j.cam_running ? 'ok' : 'bad')
         + '"></span>相机</td><td>'
         + (j.cam_running ? 'dashboard 持有 (idle 预览)' : '已释放 (env 占用)')
         + '</td></tr>';
    html += '</table>';
    document.getElementById('status').innerHTML = html;
    setCams(j.cam_running, !!c.running);
    renderRobot(j, !!c.running);
  } catch (e) {
    document.getElementById('status').innerHTML = 'status error: ' + esc(e);
  }
}
refresh(); setInterval(refresh, 1500);

// -- dataset manager ---------------------------------------------------
// Scan walks the disk: on demand only (page load + 刷新), never in the
// 1.5s poll above.
let dsCache = [];

function fmtSize(b) {
  if (b == null) return '-';
  if (b >= 1e9) return (b / 1e9).toFixed(2) + ' GB';
  if (b >= 1e6) return (b / 1e6).toFixed(1) + ' MB';
  if (b >= 1e3) return (b / 1e3).toFixed(0) + ' KB';
  return b + ' B';
}

function renderDatasets(j) {
  dsCache = j.datasets || [];
  const notice = document.getElementById('dsNotice');
  if (j.legacy_in_container) {
    notice.textContent = '注意: rlinf-eval 容器内 ' +
      '/opt/.cache/huggingface/lerobot 仍有 legacy 数据集（未扫描）';
    notice.style.display = '';
  } else {
    notice.style.display = 'none';
  }
  if (!dsCache.length) {
    document.getElementById('dsTable').innerHTML =
      '无数据集 (roots: ' + esc((j.roots || []).join(', ')) + ')';
    return;
  }
  let html = '<table><tr><th>名称</th><th>episodes</th><th>frames</th>'
    + '<th>成功率</th><th>大小</th><th>修改时间</th><th></th></tr>';
  dsCache.forEach((d, i) => {
    const sr = d.success_rate == null ? '-'
      : (100 * d.success_rate).toFixed(0) + '% (' + d.n_success + '/'
        + d.total_episodes + ')';
    const mt = d.mtime ? new Date(d.mtime * 1000).toLocaleString() : '-';
    html += '<tr><td>' + esc(d.name)
      + (d.error ? '<div class="err">' + esc(d.error) + '</div>' : '')
      + '</td>'
      + '<td>' + (d.total_episodes ?? '-') + '</td>'
      + '<td>' + (d.total_frames ?? '-') + '</td>'
      + '<td>' + esc(sr) + '</td>'
      + '<td>' + fmtSize(d.size_bytes) + '</td>'
      + '<td>' + esc(mt) + '</td>'
      + '<td><button class="small" onclick="playEpisode(' + i + ')">回放</button>'
      + '<button class="small danger" onclick="deleteDataset(' + i + ')">删除'
      + '</button></td></tr>';
  });
  html += '</table>';
  document.getElementById('dsTable').innerHTML = html;
}

async function refreshDatasets() {
  const el = document.getElementById('dsMsg');
  el.textContent = '扫描中...';
  try {
    const r = await fetch('/api/datasets');
    renderDatasets(await r.json());
    el.textContent = '';
  } catch (e) {
    el.textContent = 'scan error: ' + e;
  }
}

function playEpisode(i) {
  const d = dsCache[i];
  if (!d || d.error) return;
  const max = (d.total_episodes || 1) - 1;
  const n = prompt('Episode index (0..' + max + ')', '0');
  if (n === null) return;
  const ep = parseInt(n, 10);
  if (isNaN(ep) || ep < 0 || ep > max) { alert('无效 episode index'); return; }
  document.getElementById('playTitle').textContent =
    d.name + ' / episode ' + ep + ' (播完即止)';
  document.getElementById('playImg').src =
    '/api/dataset/' + d.dsid + '/episode/' + ep + '/play.mjpg?ts=' + Date.now();
  document.getElementById('playOverlay').style.display = 'flex';
}

function closePlayback() {
  document.getElementById('playImg').removeAttribute('src');
  document.getElementById('playOverlay').style.display = 'none';
}

async function deleteDataset(i) {
  const d = dsCache[i];
  if (!d) return;
  if (!confirm('确认删除数据集 ' + d.name + '？')) return;
  if (!confirm('再次确认：' + d.name + ' 将被永久删除 (rmtree)')) return;
  const el = document.getElementById('dsMsg');
  try {
    const r = await fetch('/api/dataset/' + d.dsid + '?confirm=yes',
                          {method: 'DELETE'});
    const j = await r.json();
    el.textContent = j.msg || JSON.stringify(j);
  } catch (e) {
    el.textContent = 'delete error: ' + e;
  }
  refreshDatasets();
}

refreshDatasets();
</script>
</body></html>
"""

CAM_BUSY_MSG = "collection running - cameras owned by env"


def build_app(cams: CamManager, mgr: CollectionManager,
              vkbd: Optional[VirtualKeyboard] = None,
              dataset_roots: Optional[list[str]] = None,
              live_cams: Optional["LiveCamSource"] = None) -> "Flask":
    if not HAS_FLASK:
        raise RuntimeError("flask not installed on this host")
    if dataset_roots is None:
        dataset_roots = list(DATASET_ROOTS)
    app = Flask(__name__)

    # Robot-state cache for the 1.5s status poll: at most one zerorpc
    # round-trip per 3s while idle (12s after a failure — a dead NUC times
    # out at 3s and would otherwise make every poll sluggish). Never queried
    # during collection.
    _robot_cache: dict = {"ts": 0.0, "ttl": 0.0, "data": None}
    _robot_cache_lock = threading.Lock()

    def _fetch_robot_state(timeout: int = 3) -> dict:
        client = DroidClient(timeout=timeout)
        try:
            st = client.get_robot_state()
            return {
                "controller_up": True,
                "q": [round(x, 3) for x in st["joint_positions"]],
                "gripper_position": round(st["gripper_position"], 3),
            }
        except Exception as e:
            msg = str(e).strip() or e.__class__.__name__
            return {"controller_up": False, "msg": msg[:120]}
        finally:
            client.close()

    def _robot_state_for_status() -> Optional[dict]:
        if not HAS_ZERORPC:
            return None
        now = time.monotonic()
        if now - _robot_cache["ts"] < _robot_cache["ttl"]:
            return _robot_cache["data"]
        # Non-blocking lock: if a fetch is already in flight on another
        # request thread, serve the stale value instead of queueing.
        if not _robot_cache_lock.acquire(blocking=False):
            return _robot_cache["data"]
        try:
            data = _fetch_robot_state()
            _robot_cache["data"] = data
            _robot_cache["ts"] = time.monotonic()
            _robot_cache["ttl"] = 3.0 if data["controller_up"] else 12.0
            return data
        finally:
            _robot_cache_lock.release()

    def robot_gated(fn):
        """409 while collection runs (env owns the robot), 503 if zerorpc
        missing on this host."""
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            ok, msg = check_gate(mgr.is_collecting(), require_collecting=False)
            if not ok:
                return jsonify({"ok": False, "msg": msg}), 409
            if not HAS_ZERORPC:
                return jsonify({"ok": False,
                                "msg": "zerorpc not installed on this host"}), 503
            return fn(*args, **kwargs)
        return wrapper

    @app.get("/")
    def index():
        return render_template_string(INDEX_HTML)

    def _cam_src():
        # During a collection the env owns the ZED cameras (CamManager is
        # stopped), so serve the in-container live frames; otherwise serve the
        # dashboard's own idle preview.
        if live_cams is not None and mgr.is_collecting():
            return live_cams
        return cams

    @app.get("/cam/<name>.mjpg")
    def cam_mjpg(name):
        src = _cam_src()
        # Only the dashboard-owned CamManager has a "busy" state to gate on;
        # the live source always has a frame (real or placeholder).
        if src is cams and not cams.is_running():
            return jsonify({"msg": CAM_BUSY_MSG}), 409

        def gen():
            src = _cam_src()  # resolve once per connection (one pgrep, not per-frame)
            boundary = b"--frame\r\n"
            while True:
                if src is cams and not cams.is_running():
                    break
                buf = src.get_jpeg(name)
                if buf is None:
                    time.sleep(0.05)
                    continue
                yield (boundary + b"Content-Type: image/jpeg\r\n\r\n"
                       + buf + b"\r\n")
                time.sleep(0.033)  # ~30 fps cap

        return Response(gen(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/cam/<name>.jpg")
    def cam_jpg(name):
        src = _cam_src()
        if src is cams and not cams.is_running():
            return jsonify({"msg": CAM_BUSY_MSG}), 409
        buf = src.get_jpeg(name)
        if buf is None:
            return Response("no frame", status=404)
        return Response(buf, mimetype="image/jpeg")

    @app.get("/api/status")
    def api_status():
        coll = mgr.status()
        # Piggyback robot state ONLY when idle (never poke zerorpc while the
        # env owns the robot), throttled via the 3s cache above.
        robot = None if coll["running"] else _robot_state_for_status()
        return jsonify({
            "collection": coll,
            "cam_running": cams.is_running(),
            "missing_cams": cams.missing_cams,
            "has_pyzed": HAS_PYZED,
            "robot": robot,
        })

    @app.post("/api/collect/start")
    def api_collect_start():
        body = request.get_json(silent=True) or {}
        try:
            num_episodes = int(body.get("num_episodes", 0))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "msg": "num_episodes must be an int"}), 400
        if num_episodes < 1:
            return jsonify({"ok": False, "msg": "num_episodes must be >= 1"}), 400
        task = str(body.get("task_description", "")).strip()
        if not task:
            return jsonify({"ok": False, "msg": "task_description required"}), 400
        msg = mgr.start(num_episodes, task)
        code = 409 if msg.startswith("refused") else 200
        return jsonify({"ok": code == 200, "msg": msg}), code

    @app.post("/api/collect/stop")
    def api_collect_stop():
        return jsonify({"ok": True, "msg": mgr.stop()})

    # -- mark-success (uinput) ------------------------------------------
    @app.post("/api/mark_success")
    def api_mark_success():
        ok, msg = check_gate(mgr.is_collecting(), require_collecting=True)
        if not ok:
            return jsonify({"ok": False,
                            "msg": f"{msg} - mark-success only valid mid-run"}), 409
        if vkbd is None:
            return jsonify({
                "ok": False,
                "msg": ("uinput virtual keyboard unavailable "
                        "(evdev missing or dashboard not launched with sudo)"),
            }), 503
        try:
            vkbd.inject_success()
        except Exception as e:
            return jsonify({"ok": False, "msg": f"inject failed: {e!r}"[:300]}), 500
        # Advisory: report which device the KeyboardListener's scan rule
        # would pick, so the operator knows whether the injection lands.
        predicted = None
        caveat = None
        try:
            predicted = predict_listener_device()
        except Exception as e:
            _log.warning(f"listener-device prediction failed: {e}")
        # Host-side prediction is necessary but NOT sufficient: the listener
        # runs inside rlinf-eval, whose /dev snapshot may not contain our
        # node at all. Check in-container visibility (cached; start() forces
        # a recheck) — invisibility overrides any optimistic prediction.
        visible = vkbd_visible_in_container(vkbd.path)
        if not visible:
            caveat = (
                "virtual keyboard not visible inside rlinf-eval (container "
                "/dev is a startup snapshot) — run `docker restart "
                "rlinf-eval` before starting collection, or use the "
                "physical keyboard 'c'"
            )
        elif predicted is None or predicted.get("path") != vkbd.path:
            caveat = (
                "listener may be bound to the physical keyboard "
                f"(scan picks {predicted}) - verify in Task D smoke; "
                "physical 'c' is the fallback"
            )
        return jsonify({
            "ok": True,
            "msg": "KEY_C injected",
            "virtual_device": vkbd.path,
            "visible_in_container": visible,
            "listener_predicted_device": predicted,
            "caveat": caveat,
        })

    @app.post("/api/start_episode")
    def api_start_episode():
        # Inject 's' to release the wait-for-start gate (start next episode).
        # Valid only mid-run; harmless if not currently waiting (env ignores
        # 's' outside a reset wait).
        ok, msg = check_gate(mgr.is_collecting(), require_collecting=True)
        if not ok:
            return jsonify({"ok": False,
                            "msg": f"{msg} - start only valid mid-run"}), 409
        if vkbd is None:
            return jsonify({
                "ok": False,
                "msg": ("uinput virtual keyboard unavailable "
                        "(evdev missing or dashboard not launched with sudo)"),
            }), 503
        try:
            vkbd.inject_start()
        except Exception as e:
            return jsonify({"ok": False, "msg": f"inject failed: {e!r}"[:300]}), 500
        visible = vkbd_visible_in_container(vkbd.path)
        caveat = None if visible else (
            "virtual keyboard not visible inside rlinf-eval (container /dev is "
            "a startup snapshot) — docker restart rlinf-eval, or press physical 's'"
        )
        return jsonify({"ok": True, "msg": "KEY_S injected",
                        "visible_in_container": visible, "caveat": caveat})

    # -- robot panel (idle-only) ----------------------------------------
    @app.get("/api/robot/state")
    @robot_gated
    def api_robot_state():
        data = _fetch_robot_state(timeout=5)
        with _robot_cache_lock:
            _robot_cache["data"] = data
            _robot_cache["ts"] = time.monotonic()
            _robot_cache["ttl"] = 3.0 if data["controller_up"] else 12.0
        return jsonify(data)

    @app.post("/api/robot/home")
    @robot_gated
    def api_robot_home():
        client = DroidClient(timeout=30)
        try:
            # Closed-loop leashed move (speed-bounded, no reflex); loops to
            # convergence or its own 25s timeout internally.
            err = client.move_to_joint_target(DROID_HOME_Q)
            converged = err < 0.03
            return jsonify({
                "ok": converged,
                "msg": (f"home done, max joint error {err:.4f} rad"
                        if converged else
                        f"home NOT converged after 15s ({err:.4f} rad) — "
                        "robot locked? Check Desk/E-stop, then Recover."),
                "max_joint_error_rad": round(err, 4),
            })
        except Exception as e:
            return jsonify({"ok": False, "msg": f"home failed: {e!r}"[:300]}), 500
        finally:
            client.close()

    @app.post("/api/robot/recover")
    @robot_gated
    def api_robot_recover():
        # Synchronous on purpose — the operator is waiting at the bench.
        # 30s timeout client: launch_controller alone takes ~10s.
        client = DroidClient(timeout=30)
        try:
            try:
                client.kill_controller()
            except Exception as e:
                # Controller may already be dead — that's what we're recovering.
                _log.warning(f"kill_controller during recover: {e}")
            client.bootstrap()
            return jsonify({"ok": True, "msg": "recover done (controller relaunched)"})
        except Exception as e:
            return jsonify({"ok": False, "msg": f"recover failed: {e!r}"[:300]}), 500
        finally:
            client.close()

    @app.post("/api/robot/gripper/<action>")
    @robot_gated
    def api_robot_gripper(action):
        if action not in ("open", "close"):
            return jsonify({"ok": False, "msg": "action must be open|close"}), 400
        client = DroidClient(timeout=10)
        try:
            client.update_gripper(0.0 if action == "open" else 1.0)
            return jsonify({"ok": True, "msg": f"gripper {action} sent"})
        except Exception as e:
            return jsonify({"ok": False,
                            "msg": f"gripper {action} failed: {e!r}"[:300]}), 500
        finally:
            client.close()

    # -- dataset manager ---------------------------------------------------
    @app.get("/api/datasets")
    def api_datasets():
        # Walks the disk — on demand only (page load / Refresh button),
        # deliberately NOT part of the 1.5s /api/status poll.
        return jsonify({
            "datasets": scan_datasets(dataset_roots),
            "roots": dataset_roots,
            "legacy_in_container": legacy_datasets_in_container(),
        })

    @app.get("/api/dataset/<dsid>/episode/<int:n>/play.mjpg")
    def api_dataset_play(dsid, n):
        # Pure file read — allowed anytime, even mid-collection (does not
        # touch robot or cameras).
        if not (HAS_PYARROW and HAS_CV2):
            return jsonify({"ok": False,
                            "msg": "pyarrow/cv2 not installed on this host"}), 503
        ds_dir = _resolve_dataset(dsid, dataset_roots)
        if ds_dir is None:
            return jsonify({"ok": False, "msg": "dataset not found"}), 404
        try:
            with open(os.path.join(ds_dir, "meta", "info.json")) as f:
                info = json.load(f)
        except Exception as e:
            return jsonify({"ok": False,
                            "msg": f"info.json unreadable: {e!r}"[:300]}), 500
        epath = _episode_parquet_path(ds_dir, info, int(n))
        if not os.path.isfile(epath):
            return jsonify({"ok": False, "msg": f"episode {n} not found"}), 404
        image_keys = _image_keys(info)
        if not image_keys:
            return jsonify({"ok": False,
                            "msg": "dataset has no image features"}), 404
        fps = float(info.get("fps") or PLAYBACK_MAX_FPS)
        period = 1.0 / min(max(fps, 0.1), PLAYBACK_MAX_FPS)

        def gen():
            import cv2
            import pyarrow.parquet as pq
            boundary = b"--frame\r\n"
            pf = pq.ParquetFile(epath)
            try:
                for batch in pf.iter_batches(batch_size=32, columns=image_keys):
                    for row in batch.to_pylist():
                        tiles = []
                        for k in image_keys:
                            img = _decode_image_cell(row.get(k))
                            if img is None:
                                continue
                            tiles.append(cv2.resize(
                                img, None, fx=PLAYBACK_UPSCALE,
                                fy=PLAYBACK_UPSCALE,
                                interpolation=cv2.INTER_NEAREST))
                        if not tiles:
                            continue
                        ok_enc, buf = cv2.imencode(
                            ".jpg", _hstack_frames(tiles),
                            [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                        if not ok_enc:
                            continue
                        yield (boundary + b"Content-Type: image/jpeg\r\n\r\n"
                               + buf.tobytes() + b"\r\n")
                        time.sleep(period)
                # Episode over: generator returns, stream ends (no loop).
            finally:
                # Client disconnects close the generator mid-iteration;
                # release the parquet handle either way.
                pf.close()

        return Response(gen(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/dataset/<dsid>", methods=["DELETE"])
    def api_dataset_delete(dsid):
        if request.args.get("confirm") != "yes":
            return jsonify({"ok": False,
                            "msg": "refused: requires ?confirm=yes"}), 400
        if mgr.is_collecting():
            return jsonify({"ok": False,
                            "msg": "refused: collection running "
                                   "(datasets being written)"}), 409
        ds_dir = _resolve_dataset(dsid, dataset_roots)
        if ds_dir is None:
            return jsonify({"ok": False, "msg": "dataset not found"}), 404
        # Belt-and-suspenders: _resolve_dataset already enforced this.
        if not any(is_inside_root(ds_dir, r) for r in dataset_roots):
            return jsonify({"ok": False,
                            "msg": "refused: path escapes dataset roots"}), 403
        try:
            shutil.rmtree(ds_dir)
        except Exception as e:
            return jsonify({"ok": False,
                            "msg": f"delete failed: {e!r}"[:300]}), 500
        _log.info(f"dataset deleted: {ds_dir}")
        return jsonify({"ok": True, "msg": f"deleted {dsid_decode(dsid)[1]}"})

    return app


def main():
    p = argparse.ArgumentParser(description="TASL FR3 collection dashboard")
    p.add_argument("--port", type=int, default=8004)
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--resolution", default="HD720",
                   choices=["HD2K", "HD1080", "HD720"])
    p.add_argument("--jpeg-quality", type=int, default=70)
    p.add_argument("--no-cam-on-start", action="store_true",
                   help="Don't acquire cameras at startup.")
    p.add_argument("--dataset-roots", default=",".join(DATASET_ROOTS),
                   help="Comma-separated roots scanned by the dataset panel.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cams = CamManager(
        serials={"exterior": SN_EXTERIOR, "wrist": SN_WRIST},
        resolution=args.resolution,
        jpeg_quality=args.jpeg_quality,
    )
    live_cams = LiveCamSource(LIVE_CAM_DIR_HOST)
    if not args.no_cam_on_start:
        try:
            _log.info(f"CamManager.start: {cams.start()}")
        except RuntimeError as e:
            _log.warning(f"cameras not started: {e}")

    # Virtual keyboard at STARTUP (not lazily): KeyboardListener scans
    # /dev/input only once, when the env process starts — the device must
    # pre-exist any collection launch. Needs /dev/uinput access (sudo).
    vkbd = None
    if HAS_EVDEV:
        try:
            vkbd = VirtualKeyboard()
            _log.info(f"virtual keyboard created: {vkbd.path}")
        except Exception as e:
            _log.warning(
                f"virtual keyboard unavailable (launch with sudo for "
                f"/dev/uinput): {e}"
            )
    else:
        _log.warning("evdev not installed - mark-success disabled")

    dataset_roots = [r.strip() for r in args.dataset_roots.split(",")
                     if r.strip()]
    mgr = CollectionManager(cams, vkbd)
    app = build_app(cams, mgr, vkbd, dataset_roots=dataset_roots,
                    live_cams=live_cams)
    _log.info(f"serving on {args.bind}:{args.port}")
    app.run(host=args.bind, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
