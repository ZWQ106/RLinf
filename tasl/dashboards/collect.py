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

from tools.h264_writer import H264Writer  # browser-playable rollout recording

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

# Layout store (stdlib-only, always importable). Two paths because the
# launcher runs `python3 dashboards/collect.py` with PYTHONPATH=tasl (the
# dashboards/openpi.py convention) while unit tests import it as a module.
try:
    from dashboards.layout_store import (
        VIEWS as LAYOUT_VIEWS,
        LayoutError, LayoutStore, dataset_layout_entry, is_usable_snapshot,
        write_dataset_layout,
    )
except ImportError:  # pragma: no cover - direct run from inside dashboards/
    from layout_store import (  # type: ignore[no-redef]
        VIEWS as LAYOUT_VIEWS,
        LayoutError, LayoutStore, dataset_layout_entry, is_usable_snapshot,
        write_dataset_layout,
    )

# Task store — SHARED with the eval dashboard (openpi.py): both portals read
# and write the same tasl/tasks_store.json, so a task defined here is
# immediately selectable for eval and vice versa.
try:
    from dashboards.task_store import TaskStore
except ImportError:  # pragma: no cover - direct run from inside dashboards/
    from task_store import TaskStore  # type: ignore[no-redef]

_log = logging.getLogger("collect_dashboard")

# ── Hardware constants (verified 2026-05-27) ─────────────────────────
SN_EXTERIOR = 36443134   # ZED 2i, exterior view
SN_WRIST = 17150101      # ZED Mini, wrist-mounted

CONTAINER = os.environ.get("RLINF_CONTAINER", "rlinf-eval")
COLLECT_LOG = "/tmp/collect_dash.log"
CONTAINER_PY = "/opt/venv/openpi/bin/python"
# Host side of the bind mount where the in-container VideoPlayer writes live
# per-camera JPEGs during a collection (container path:
# /workspace/rlinf/outputs/live_cam via RLINF_LIVE_CAM_DIR). LiveCamSource
# reads these to serve a live preview while the env owns the ZED cameras.
# Data root (datasets/ + outputs/) — matches lib.sh RLINF_DATA_DIR. Lives OUTSIDE
# the code checkout so swapping the mounted code never touches collected data.
DATA_DIR_HOST = os.environ.get("RLINF_DATA_DIR", "/home/franka_desktop/rlinf_data")
LIVE_CAM_DIR_HOST = f"{DATA_DIR_HOST}/outputs/live_cam"
# Object-placement stencils (九宫格 + markers + reference snapshot), one
# <id>.json (+ <id>.jpg) per layout. Kept beside the data, not in the code
# checkout, for the same reason datasets are: a code swap must not lose them.
LAYOUT_DIR_HOST = os.environ.get("RLINF_LAYOUT_DIR", f"{DATA_DIR_HOST}/layouts")
# Demo exports (save-video / save-layout buttons) — shared with openpi.py.
DEMO_DIR = os.environ.get(
    "RLINF_DEMO_DIR",
    os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "saved_demo"))
# Operator launchers, next door to this file (<repo>/tasl/launch). The session
# endpoints SHELL OUT to these rather than reimplementing the kill set in
# Python: "stopped" has exactly one definition, and the portal cannot drift
# from what teleop-stop.sh does on the command line.
LAUNCH_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "launch")
TELEOP_STOP_SH = os.path.join(LAUNCH_DIR, "teleop-stop.sh")
LAUNCH_LIB_SH = os.path.join(LAUNCH_DIR, "lib.sh")
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

# Dataset manager. Host paths on the Desktop (the rlinf_data bind mount:
# /home/franka_desktop/rlinf_data/{datasets,outputs} -> /workspace/rlinf/{datasets,outputs}).
# Order is significant: root_index 0 = "current" scheme (fixed repo_id/root
# datasets written under datasets/<name>), 1 = "legacy" (old per-run
# timestamped outputs/lerobot/...). The UI groups + labels by this index.
# Override with --dataset-roots (comma-separated, current-first).
DATASET_ROOTS = [
    f"{DATA_DIR_HOST}/datasets",
    f"{DATA_DIR_HOST}/outputs/lerobot",
]
# Container-internal dataset root prefix (mounted as the host DATASET_ROOTS[0]).
# Start passes repo_id/root/svo_dir under this so collect_real_data.py (running
# inside rlinf-eval) writes to the bind-mounted location, visible on the host.
CONTAINER_DATASET_ROOT = "/workspace/rlinf/datasets"
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


class HDRolloutRecorder:
    """Sidecar HD demo recorder for collection episodes.

    The dataset keeps only 224x224 per view — fine for training, useless for
    demo videos. During a recording window (开始记录 → 标记成功/丢弃/Stop)
    this samples the FULL-RES live_cam JPEGs the in-container env already
    writes for the dashboard preview (1280x720 per view), tiles them
    side-by-side and writes a TEMP mp4. Save-video promotes the temp into
    saved_demo/<task>/; starting the next recording quietly discards an
    unpromoted one. Pure sidecar: it drops frames when a read misses and
    never touches the env, the cameras, or the dataset.
    """

    TMP_NAME = "hd_rollout_tmp.mp4"

    def __init__(self, live_source, tmp_dir: str, fps: float = 15.0):
        self.live = live_source
        self.tmp_dir = tmp_dir
        self.fps = fps
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last: Optional[dict] = None
        # A temp from a previous dashboard life is unowned — drop it.
        try:
            os.makedirs(tmp_dir, exist_ok=True)
            for fn in os.listdir(tmp_dir):
                os.unlink(os.path.join(tmp_dir, fn))
        except OSError:
            pass

    def _tmp_path(self) -> str:
        return os.path.join(self.tmp_dir, self.TMP_NAME)

    @staticmethod
    def _decode(buf):
        import cv2
        import numpy as np
        if not buf:
            return None
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        # LiveCamSource hands back an 8x8 placeholder for a stale feed.
        if img is None or img.shape[0] < 100:
            return None
        return img

    def start(self, task_id: str, dataset: str) -> None:
        if not HAS_CV2 or self.live is None:
            return
        with self._lock:
            if self._running:
                return
            self._discard_locked()      # new take replaces an unsaved one
            self._running = True
            self._meta = {"task": task_id or "", "dataset": dataset or "",
                          "started": time.strftime("%Y%m%d_%H%M%S")}
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        import cv2
        import numpy as np
        writer = None
        frames = 0
        # VideoWriter picks the container from the extension — the in-progress
        # name must still end in .mp4.
        path = os.path.join(self.tmp_dir, "in-progress." + self.TMP_NAME)
        dt = 1.0 / self.fps
        nxt = time.time()
        while self._running:
            ext = self._decode(self.live.get_jpeg("exterior"))
            wrist = self._decode(self.live.get_jpeg("wrist"))
            tile = None
            if ext is not None:
                if wrist is not None:
                    h = ext.shape[0]
                    wr = cv2.resize(
                        wrist, (int(wrist.shape[1] * h / wrist.shape[0]), h))
                    tile = np.concatenate([ext, wr], axis=1)
                else:
                    tile = ext
            if tile is not None:
                if writer is None:
                    # H.264 High + faststart — cv2's default mp4v is not
                    # browser-playable (see tasl/tools/h264_writer).
                    writer = H264Writer(
                        path, self.fps, (tile.shape[1], tile.shape[0]))
                    if not writer.isOpened():
                        _log.error("HD rollout VideoWriter open failed")
                        writer = None
                        break
                writer.write(tile)
                frames += 1
            nxt += dt
            delay = nxt - time.time()
            if delay > 0:
                time.sleep(min(delay, 1.0))
            else:
                nxt = time.time()
        if writer is not None:
            writer.release()
        with self._lock:
            if frames > 0:
                try:
                    os.replace(path, self._tmp_path())
                    self._last = dict(self._meta, path=self._tmp_path(),
                                      frames=frames)
                    _log.info(f"HD rollout temp: {frames} frames "
                              f"({self._meta['task']}/{self._meta['dataset']})")
                except OSError as e:
                    _log.warning(f"HD rollout finalize failed: {e}")
            else:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def stop(self) -> None:
        self._running = False
        t = self._thread
        if t is not None:
            t.join(timeout=3.0)
            self._thread = None

    def last(self) -> Optional[dict]:
        with self._lock:
            return dict(self._last) if self._last else None

    def promote(self, dest: str) -> Optional[str]:
        """Move the finalized temp to `dest`; returns dest or None."""
        with self._lock:
            if not self._last:
                return None
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            try:
                shutil.move(self._last["path"], dest)
            except OSError as e:
                _log.warning(f"HD rollout promote failed: {e}")
                return None
            self._last = None
            return dest

    def _discard_locked(self) -> None:
        self._last = None
        try:
            os.unlink(self._tmp_path())
        except OSError:
            pass


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
    def start(self, num_episodes: int, task_description: str,
              dataset_name: Optional[str] = None) -> str:
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
            # Dataset selection: when a name is given, pin the fixed repo_id/root
            # so episodes ACCUMULATE into datasets/<name> across Starts (the new
            # open_or_create writer reopens an existing dataset instead of the old
            # FileExistsError-on-create, which is why the historic timestamped
            # save_dir workaround is no longer needed). svo_dir is the HD sibling.
            ds_override = ""
            if dataset_name:
                if not _valid_dataset_name(dataset_name):
                    return ("refused: invalid dataset name "
                            "(allowed: letters, digits, . _ -)")
                dc = "env.eval.data_collection"
                ds_override = (
                    f"{dc}.repo_id=tasl/{dataset_name} "
                    f"{dc}.root={CONTAINER_DATASET_ROOT}/{dataset_name} "
                    f"{dc}.svo_dir={CONTAINER_DATASET_ROOT}/{dataset_name}_svo "
                )
            # log_path now only holds tensorboard logs (the dataset lands at the
            # fixed root above); a timestamp keeps each run's logs separate.
            run_tag = time.strftime("%Y%m%d_%H%M%S")
            log_path = shlex.quote(f"./outputs/collect_{run_tag}")
            # Deterministic keyboard-device pin. The in-container KeyboardListener
            # (reward_done_wrapper) reads exactly ONE evdev device — the first
            # under /dev/input/event* advertising A/B/C/Q — and BOTH the physical
            # keyboard and our virtual keyboard qualify, so which one it binds is
            # otherwise arbitrary. When it picks the physical keyboard, the
            # dashboard's injected 's'/'c' land on a device nobody reads and the
            # wait-for-start gate never releases (teleop never begins). rlinf-eval
            # is privileged with a live /dev:/dev bind of the host devtmpfs, so a
            # host-created node IS visible inside.
            #
            # Pin by NAME (RLINF_KEYBOARD_DEVICE_NAME), not by the /dev/input/eventN
            # path: the virtual keyboard is EPHEMERAL (it exists only while this
            # dashboard process lives) and its event number is NOT stable across
            # dashboard restarts. A path pin bound the listener to a node that
            # vanished when the dashboard stopped, killing the listener thread and
            # all subsequent key input (2026-07-10 — 's' worked, then the dashboard
            # died mid-episode, event4 disappeared, and 'c'/end-episode was dead).
            # The name pin is a SOFT preference: the (now resilient) listener falls
            # back to the physical keyboard when the virtual one is gone and
            # re-acquires it by name — even at a new event number — when the
            # dashboard comes back, so no in-container visibility gate is needed.
            kbd_visible = (self.vkbd is not None and self.vkbd.path is not None
                           and vkbd_visible_in_container(self.vkbd.path,
                                                         force=True))
            kbd_env = (f"RLINF_KEYBOARD_DEVICE_NAME={shlex.quote(VIRTUAL_KBD_NAME)} "
                       if kbd_visible else "")
            launch_cmd = (
                "cd /workspace/rlinf && "
                "export PYTHONPATH=/workspace/rlinf "
                "EMBODIED_PATH=/workspace/rlinf/examples/embodiment "
                "RLINF_LIVE_CAM_DIR=/workspace/rlinf/outputs/live_cam "
                "HF_LEROBOT_HOME=/workspace/rlinf/outputs/lerobot "
                f"{kbd_env}&& "
                f"{CONTAINER_PY} examples/embodiment/collect_real_data.py "
                "--config-name realworld_collect_data_polymetis_jointvel "
                "env.eval.gello_port=/dev/gello "
                # Reach the controller over the robot net, not the config's
                # legacy Tailscale robot_ip (overrides node_groups[0] hardware).
                f"cluster.node_groups.0.hardware.configs.0.robot_ip={NUC1_HOST} "
                f"runner.num_data_episodes={int(num_episodes)} "
                f"runner.logger.log_path={log_path} "
                f"{override} "
                f"{ds_override}"
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
                "dataset_name": dataset_name,
                "started_at": time.time(),
            }
            msg = (f"started: {num_episodes} episode(s), "
                   f"task={task_description!r}"
                   + (f", dataset={dataset_name!r}" if dataset_name else ""))
            # Report the keyboard-injection path decided above (kbd_visible was
            # probed pre-launch). Collection starts either way — the physical
            # keyboard always works — but tell the operator whether the
            # dashboard's 's'/'c' buttons will actually reach the listener.
            if self.vkbd is not None and self.vkbd.path is not None:
                if kbd_visible:
                    msg += (f" — keyboard pinned by name to {VIRTUAL_KBD_NAME!r} "
                            "(dashboard 's'/'c' injection active; auto-recovers "
                            "across dashboard restarts)")
                else:
                    msg += (" — warning: virtual keyboard not visible inside "
                            "rlinf-eval; dashboard 's'/'c' injection won't land "
                            "— use the physical keyboard, or docker restart "
                            "rlinf-eval before next run")
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

    def after_external_teardown(self) -> None:
        """State fixup after the teardown SCRIPT killed the env behind our back.

        Without this the manager keeps believing a run is live until the next
        poll, and — worse — never reclaims the ZEDs, so the idle preview stays
        dark after a 结束会话.
        """
        with self._lock:
            self.last_result = parse_collect_log(self._tail_log())
            self._was_running = False
            self._launched_at = 0.0
            self._reclaim_cameras()
            self.last_action_msg = "session stopped via teleop-stop.sh"
        _log.info("external teardown: manager state reset, cameras reclaimed")

    def status(self) -> dict:
        alive = self._is_alive()
        tail = self._tail_log()
        parsed = parse_collect_log(tail)
        startup = parse_startup_progress(tail)
        # Primary signal = the gate sentinel file (reliable, bypasses Ray's
        # stdout buffering); fall back to log text if the file read fails.
        gate = self._read_gate()
        if gate is not None:
            # WAIT = homed, arm frozen, waiting to be released.
            # LIVE = released for positioning, teleop live but NOT recording
            #        (two-stage gate only; one-stage goes WAIT -> RUN).
            # RUN  = recording.
            waiting_for_start = bool(alive and gate == "WAIT")
            robot_released = bool(alive and gate == "LIVE")
        else:
            waiting_for_start = bool(alive and is_waiting_for_start(tail))
            robot_released = False
        # Orphan = a run this dashboard never launched. Happens when the
        # dashboard is restarted (or crashes) under a live collection: the run
        # keeps the ZEDs + GELLO, and the next launcher preflight then fails on
        # cameras that look wedged. Surfacing it gives the operator a one-click
        # way out instead of a docker exec.
        with self._lock:
            orphan = bool(alive and self._launched_at == 0.0)
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
            "robot_released": robot_released,
            "orphan": orphan,
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


_DATASET_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _valid_dataset_name(name: str) -> bool:
    """A dataset name is a single path segment of safe chars — no slashes,
    no '..', no leading dot-only. Used wherever a name is interpolated into
    a filesystem path or a Hydra repo_id/root override."""
    return (isinstance(name, str)
            and bool(_DATASET_NAME_RE.match(name))
            and name not in (".", "..")
            and "/" not in name)


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
                "path": ds_dir,
                "category": "current" if root_index == 0 else "legacy",
                "task": None,
                "svo_path": None,
                "total_episodes": None,
                "total_frames": None,
                "fps": None,
                "success_rate": None,
                "n_success": None,
                "size_bytes": None,
                "mtime": None,
                "error": None,
            }
            # Canonical task instruction (first tasks.jsonl line) — drives the
            # Start-panel auto-fill so re-collecting into an existing dataset
            # reuses the exact task string (avoids a duplicate tasks.jsonl row).
            try:
                tasks_path = os.path.join(ds_dir, "meta", "tasks.jsonl")
                with open(tasks_path) as tf:
                    first = tf.readline().strip()
                if first:
                    entry["task"] = json.loads(first).get("task")
            except (OSError, ValueError):
                pass
            # HD SVO archive sibling (current scheme writes <name>_svo).
            svo_sib = ds_dir.rstrip("/") + "_svo"
            if os.path.isdir(svo_sib):
                entry["svo_path"] = svo_sib
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


def list_episodes_meta(ds_dir: str) -> list[dict]:
    """Per-episode rows for the expandable dataset view: {index, task,
    length, success}. Cheap (reads only meta JSONL, no parquet)."""
    succ: dict[int, Optional[bool]] = {}
    spath = os.path.join(ds_dir, "meta", "episodes_stats.jsonl")
    if os.path.isfile(spath):
        try:
            with open(spath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    ei = d.get("episode_index")
                    s = (d.get("stats") or {}).get("is_success") or {}
                    mx = s.get("max")
                    if isinstance(mx, list):
                        mx = mx[0] if mx else None
                    if ei is not None:
                        succ[ei] = bool(mx) if mx is not None else None
        except (OSError, ValueError):
            pass
    eps: list[dict] = []
    epath = os.path.join(ds_dir, "meta", "episodes.jsonl")
    if os.path.isfile(epath):
        try:
            with open(epath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    ei = d.get("episode_index")
                    tasks = d.get("tasks") or []
                    eps.append({"index": ei,
                                "task": tasks[0] if tasks else None,
                                "length": d.get("length"),
                                "success": succ.get(ei)})
        except (OSError, ValueError):
            pass
    return eps


def _svo_files_for(svo_dir: str, ep: int) -> list[str]:
    return sorted(glob.glob(os.path.join(glob.escape(svo_dir),
                                         f"episode_{ep:06d}_*.svo2")))


def delete_episode(ds_dir: str, ep: int) -> str:
    """Delete ONE episode from a LeRobot v2.1 dataset and re-index the rest so
    episode_index stays contiguous (0..N-2).

    Per later episode j>ep: rewrite its parquet (episode_index -> j-1, global
    `index` -= removed-length), move episode_j -> episode_{j-1}, rename its SVO
    sibling, then rebuild episodes.jsonl / episodes_stats.jsonl / info.json /
    svo_index.json. Requires pyarrow. Returns a status string (a 'refused: ' /
    'error: ' prefix on failure). NOTE: assumes a single chunk (<chunks_size
    episodes), which always holds for this bench's datasets."""
    if not HAS_PYARROW:
        return "error: pyarrow not installed on this host"
    import pyarrow as pa
    import pyarrow.parquet as pq
    info_path = os.path.join(ds_dir, "meta", "info.json")
    try:
        with open(info_path) as f:
            info = json.load(f)
    except Exception as e:
        return f"error: info.json unreadable: {e!r}"[:200]
    total = int(info.get("total_episodes") or 0)
    if not 0 <= ep < total:
        return f"refused: episode {ep} out of range (0..{total - 1})"
    chunks_size = int(info.get("chunks_size") or 1000)
    if total > chunks_size:
        return ("refused: multi-chunk dataset (>{} episodes) — per-episode "
                "delete not supported".format(chunks_size))
    eps_path = os.path.join(ds_dir, "meta", "episodes.jsonl")
    try:
        eps = [json.loads(l) for l in open(eps_path) if l.strip()]
    except Exception as e:
        return f"error: episodes.jsonl unreadable: {e!r}"[:200]
    len_ep = int((eps[ep] if ep < len(eps) else {}).get("length") or 0)
    svo_dir = ds_dir.rstrip("/") + "_svo"

    def pq_path(i: int) -> str:
        return _episode_parquet_path(ds_dir, info, i)

    try:
        # 1. drop the target episode's parquet + SVO
        p = pq_path(ep)
        if os.path.isfile(p):
            os.remove(p)
        for s in _svo_files_for(svo_dir, ep):
            try:
                os.remove(s)
            except OSError:
                pass
        # 2. shift every later episode down by one
        for j in range(ep + 1, total):
            src, dst = pq_path(j), pq_path(j - 1)
            if os.path.isfile(src):
                t = pq.read_table(src)
                n = t.num_rows
                ei_i = t.schema.get_field_index("episode_index")
                ix_i = t.schema.get_field_index("index")
                t = t.set_column(ei_i, "episode_index",
                                 pa.array([j - 1] * n,
                                          type=t.schema.field(ei_i).type))
                shifted = [x - len_ep for x in t.column("index").to_pylist()]
                t = t.set_column(ix_i, "index",
                                 pa.array(shifted,
                                          type=t.schema.field(ix_i).type))
                pq.write_table(t, dst)
                os.remove(src)
            for s in _svo_files_for(svo_dir, j):
                cam = os.path.basename(s)[len(f"episode_{j:06d}_"):]
                os.rename(s, os.path.join(svo_dir,
                                          f"episode_{j - 1:06d}_{cam}"))
        # 3. episodes.jsonl + episodes_stats.jsonl (drop ep, renumber > ep)
        for meta_name in ("episodes.jsonl", "episodes_stats.jsonl"):
            mp = os.path.join(ds_dir, "meta", meta_name)
            if not os.path.isfile(mp):
                continue
            rows = [json.loads(l) for l in open(mp) if l.strip()]
            out = []
            for d in rows:
                ei = d.get("episode_index")
                if ei == ep:
                    continue
                if isinstance(ei, int) and ei > ep:
                    d["episode_index"] = ei - 1
                out.append(d)
            with open(mp, "w") as f:
                for d in out:
                    f.write(json.dumps(d) + "\n")
        # 4. info.json counters
        info["total_episodes"] = total - 1
        info["total_frames"] = int(info.get("total_frames") or 0) - len_ep
        info["splits"] = {"train": f"0:{total - 1}"}
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)
        # 5. svo_index.json (renumber keys + filenames)
        sidx_path = os.path.join(svo_dir, "svo_index.json")
        if os.path.isfile(sidx_path):
            try:
                sidx = json.load(open(sidx_path))
                new_sidx = {}
                for k, v in sidx.items():
                    ki = int(k)
                    if ki == ep:
                        continue
                    nk = ki - 1 if ki > ep else ki
                    new_sidx[str(nk)] = [
                        fn.replace(f"episode_{ki:06d}_", f"episode_{nk:06d}_")
                        for fn in v]
                with open(sidx_path, "w") as f:
                    json.dump(new_sidx, f, indent=2)
            except (OSError, ValueError):
                pass
    except Exception as e:
        return f"error: delete failed mid-way: {e!r}"[:250]
    return f"deleted episode {ep}; {total - 1} remain"


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

    KeyboardListener device selection (keyboard_listener.py) prefers, in order:
      1. RLINF_KEYBOARD_DEVICE — a hard path override (capability-checked).
      2. RLINF_KEYBOARD_DEVICE_NAME — a soft preference: the device whose evdev
         name matches (this class's VIRTUAL_KBD_NAME), re-resolved on every
         open so it survives event-number changes.
      3. Otherwise the FIRST device (lexicographic /dev/input/event* scan)
         whose EV_KEY caps include KEY_A/B/C/Q. It reads ONE device at a time,
         but now re-acquires on device loss instead of dying.

    Design consequences:
      * Our device advertises exactly that capability set. Without a pin,
        whether it beats the physical Dell keyboard depends on event-node
        numbering under a lexicographic sort — effectively arbitrary. That bit
        us (2026-07-10): the listener bound the physical keyboard, so injected
        's' never reached the wait-for-start gate.
      * The device MUST exist before collection starts: the listener opens a
        device at env startup. Hence creation at dashboard startup, not lazily
        on first button press.
      * Pin by NAME, not path (CollectionManager.start() exports
        RLINF_KEYBOARD_DEVICE_NAME=VIRTUAL_KBD_NAME). This device is EPHEMERAL
        — it lives only as long as this dashboard process — and its event
        number is not stable across dashboard restarts. A path pin bound the
        listener to a node that vanished when the dashboard stopped mid-run
        (2026-07-10): 's' worked, the dashboard died, event4 disappeared, the
        listener thread crashed on OSError, and 'c'/end-episode was dead for
        the rest of the run. The name pin + the listener's re-acquire-on-loss
        make this self-healing: it falls back to the physical keyboard while
        the virtual one is gone and rebinds by name when the dashboard returns.
      * Container visibility (verified 2026-07-10): rlinf-eval is privileged
        with a LIVE /dev:/dev bind of the host `udev` devtmpfs (per the
        container's /proc/mounts), NOT a Docker-managed snapshot. A
        /dev/input/eventN node created on the host afterwards IS visible
        inside immediately — no `docker restart` needed. (An earlier comment
        here claimed a startup snapshot; that was wrong for this /dev mount.)
        vkbd_visible_in_container() still probes per run to stay honest if the
        container is ever reconfigured with a managed /dev.
    """

    def __init__(self):
        from evdev import UInput, ecodes
        self._ecodes = ecodes
        # A/B/C/Q satisfy the listener's device-detection caps; S is the
        # episode-start key. All injectable.
        # KEY_R drives the two-stage start gate's second step (begin
        # recording). Superset of the {A,B,C,Q} caps KeyboardListener matches
        # on, so device selection is unaffected.
        caps = {ecodes.EV_KEY: [ecodes.KEY_A, ecodes.KEY_B, ecodes.KEY_C,
                                ecodes.KEY_Q, ecodes.KEY_S, ecodes.KEY_R]}
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

    def inject_record(self, hold_s: float = 0.25) -> None:
        """Begin recording — second step of the two-stage start gate."""
        self.inject_key(self._ecodes.KEY_R, hold_s)

    def inject_discard(self, hold_s: float = 0.25) -> None:
        """End the episode WITHOUT saving it (reward -1 + done).

        The cheapest way to 'delete' a bad episode is never to write it: with
        only_success the run drops it, so nothing has to be renumbered out of a
        dataset that LeRobot is still holding open.
        """
        self.inject_key(self._ecodes.KEY_A, hold_s)

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
class LayoutGate:
    """layout-prepare stage: which stencil is armed, and is the scene ready.

    Collection is gated on the operator having (a) selected a layout and
    (b) confirmed the physical objects match it. One confirmation covers a
    whole run — every episode in a run shares the arrangement — and is
    invalidated if the stencil itself is edited underneath (hash change) or
    the operator backs out.

    Also carries the dataset provenance write. `meta/layout.json` cannot be
    written at Start for a brand-new dataset (LeRobotDataset.create refuses a
    non-empty target), so the entry is parked here and flushed by the status
    poll once the dataset directory actually exists.
    """

    def __init__(self, store: "LayoutStore"):
        self.store = store
        self._lock = threading.Lock()
        self._layout: Optional[dict] = None
        self._confirmed_at: Optional[float] = None
        self._pending: Optional[tuple] = None   # (dataset_dir, entry)
        self._last_write: Optional[str] = None

    # -- selection / confirmation -------------------------------------
    def select(self, layout_id: str) -> dict:
        """Arm a layout. Always drops any prior confirmation — a new stencil
        means the scene has not been checked against it yet."""
        layout = self.store.load(layout_id)
        with self._lock:
            self._layout = layout
            self._confirmed_at = None
        return layout

    def refresh(self) -> None:
        """Re-read the armed layout from disk; drop confirmation if the
        stencil moved. Called after a save so nudging a marker while
        'confirmed' cannot smuggle a stale confirmation into a run."""
        with self._lock:
            cur = self._layout
        if not cur:
            return
        try:
            fresh = self.store.load(cur["id"])
        except LayoutError:
            with self._lock:
                self._layout = None
                self._confirmed_at = None
            return
        with self._lock:
            if self._layout and self._layout["id"] == fresh["id"]:
                if fresh["hash"] != self._layout["hash"]:
                    self._confirmed_at = None
                self._layout = fresh

    def confirm(self) -> str:
        with self._lock:
            if not self._layout:
                raise LayoutError("no layout selected — pick or create one first")
            self._confirmed_at = time.time()
            return self._layout["id"]

    def clear(self) -> None:
        with self._lock:
            self._layout = None
            self._confirmed_at = None

    def unconfirm(self) -> None:
        with self._lock:
            self._confirmed_at = None

    def check(self) -> tuple[bool, str]:
        """Gate for /api/collect/start."""
        with self._lock:
            if not self._layout:
                return False, ("refused: layout-prepare 未完成 — 请先选择或新建一个 "
                               "layout")
            if not self._confirmed_at:
                return False, (f"refused: layout '{self._layout['id']}' 已选择但未确认 "
                               "— 摆好物体后点「确认就位」")
            return True, self._layout["id"]

    def current(self) -> Optional[dict]:
        with self._lock:
            return dict(self._layout) if self._layout else None

    def status(self) -> dict:
        with self._lock:
            lay = self._layout
            return {
                "selected": lay["id"] if lay else None,
                "hash": lay["hash"] if lay else None,
                # One layout, two views — the exterior stencil pins the object
                # placement, the wrist (eye-in-hand) one pins the start pose.
                "views": lay["views"] if lay else None,
                "has_snapshot": lay["has_snapshot"] if lay else {},
                "confirmed": self._confirmed_at is not None,
                "confirmed_at": (
                    time.strftime("%H:%M:%S", time.localtime(self._confirmed_at))
                    if self._confirmed_at else None),
                "pending_write": self._pending is not None,
                "last_write": self._last_write,
            }

    # -- dataset provenance -------------------------------------------
    def arm_dataset_write(self, dataset_dir: str, task: str,
                          num_episodes: int) -> None:
        layout = self.current()
        if not layout:
            return
        entry = dataset_layout_entry(layout, task, num_episodes)
        with self._lock:
            self._pending = (dataset_dir, entry)
        self.flush_pending()   # datasets that already exist get it immediately

    def flush_pending(self) -> Optional[str]:
        """Write the parked entry once the dataset dir exists. Cheap enough
        (one isdir) to call from the 1.5 s status poll."""
        with self._lock:
            pending = self._pending
        if not pending:
            return None
        dataset_dir, entry = pending
        if not os.path.isdir(dataset_dir):
            return None
        try:
            path = write_dataset_layout(dataset_dir, entry)
        except OSError as e:
            _log.warning(f"layout metadata write failed for {dataset_dir}: {e}")
            return None
        with self._lock:
            self._pending = None
            self._last_write = path
        _log.info(f"layout provenance written: {path}")
        return path


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
  /* Amber, deliberately NOT green: the robot is live but nothing is being
     recorded yet — the operator must still act. */
  button.startrec { display: none; width: 100%; padding: 26px 0;
         font-size: 1.5rem; font-weight: 600; margin-top: 12px;
         background: #a35b00; border-radius: 10px; }
  /* Deliberately smaller than 标记成功: discarding is the rarer action, and a
     mis-hit costs the whole take. */
  button.discardep { display: none; width: 100%; padding: 12px 0;
         font-size: 1rem; margin-top: 8px;
         background: #4a1f1f; border-radius: 8px; }
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
  .path { font-size: 0.72rem; color: #7a7a7a; font-family: monospace;
          word-break: break-all; }
  details summary { cursor: pointer; color: #aaa; font-size: 0.9rem;
          margin: 8px 0; }
  select, #newDatasetName { width: 100%; box-sizing: border-box;
          margin-bottom: 6px; }
  h4 { margin: 6px 0; font-size: 0.95rem; color: #ddd; }
  .epbox { padding: 6px 10px; background: #161616; border-left: 2px solid #3a6;
           margin: 2px 0 6px 0; }
  .epbox table { width: auto; } .epbox th, .epbox td { padding: 3px 10px; }
  #playOverlay { display: none; position: fixed; inset: 0; z-index: 20;
         background: rgba(0,0,0,0.85); flex-direction: column;
         align-items: center; justify-content: center; }
  #playOverlay img { image-rendering: pixelated; border: 1px solid #444;
         border-radius: 4px; max-width: 92vw; max-height: 80vh; }
  #playTitle { color: #ccc; font-size: 0.9rem; margin-bottom: 8px; }
  /* ---- layout stencil overlay (exterior cam) ---------------------- */
  /* The stencil is composited in the BROWSER, never burned into the MJPEG
     stream — recorded frames stay clean and toggling costs nothing. */
  .camwrap { position: relative; display: block; width: 100%;
             max-width: 480px; line-height: 0; }
  .camwrap img.feed { width: 100%; display: block; border-radius: 4px; }
  /* NB: no `display:none` here — these are toggled from JS with
     style.display = '' , which only strips the INLINE style and would fall
     straight back to a stylesheet rule. Hidden state lives inline in the
     markup instead, matching how every other JS-toggled element here works. */
  .camwrap img.ghost { position: absolute; left: 0; top: 0;
             width: 100%; height: 100%; border-radius: 4px;
             pointer-events: none; }
  .camwrap .stencil { position: absolute; left: 0; top: 0;
             width: 100%; height: 100%;
             pointer-events: none; }
  /* Only edit mode swallows clicks — a live preview stays click-through. */
  .camwrap.editing .stencil { pointer-events: auto; cursor: crosshair; }
  .gline { position: absolute; background: rgba(120,200,255,0.35); }
  .gline.v { top: 0; height: 100%; width: 1px; }
  .gline.h { left: 0; width: 100%; height: 1px; }
  .mk { position: absolute; transform: translate(-50%, -50%);
        width: 14px; height: 14px; border-radius: 50%;
        border: 2px solid #ffb300; background: rgba(255,179,0,0.25);
        box-sizing: border-box; }
  .mk.sel { border-color: #4fc3f7; background: rgba(79,195,247,0.35);
        box-shadow: 0 0 0 3px rgba(79,195,247,0.25); }
  .mk .mklabel { position: absolute; left: 18px; top: -4px;
        white-space: nowrap; font-size: 11px; line-height: 1.4;
        color: #ffe08a; text-shadow: 0 0 3px #000, 0 0 3px #000;
        font-family: monospace; }
  .mk.sel .mklabel { color: #9fe0ff; }
  #layout-panel .kbdhint { font-size: 0.78rem; color: #888; margin-top: 8px;
        line-height: 1.8; }
  kbd { background: #262626; border: 1px solid #3a3a3a; border-radius: 3px;
        padding: 1px 5px; font-family: monospace; font-size: 0.78rem;
        color: #ddd; }
  .lstate { font-size: 0.8rem; padding: 2px 8px; border-radius: 10px;
        margin-left: 8px; vertical-align: middle; }
  .lstate.none { background: #3a2323; color: #e59a9a; }
  .lstate.sel  { background: #3a3423; color: #e5cf8a; }
  .lstate.ok   { background: #1f4f1f; color: #a5e5a5; }
  .objrow { display: flex; align-items: center; gap: 6px; padding: 3px 0;
        font-size: 0.85rem; border-bottom: 1px solid #222; cursor: pointer; }
  .objrow.sel { background: #17242c; }
  .objrow .coord { color: #777; font-family: monospace; font-size: 0.78rem; }
  .objrow .idx { color: #666; font-family: monospace; width: 18px; }
  #sessionOut { background: #0b0b0b; border: 1px solid #262626;
        border-radius: 4px; padding: 8px; margin-top: 8px; max-height: 240px;
        overflow: auto; font-size: 0.75rem; line-height: 1.5; color: #bbb;
        white-space: pre-wrap; word-break: break-all; }
  .warnbox { background: #3a2323; border: 1px solid #5a3030; color: #e5b0b0;
        padding: 8px 10px; border-radius: 4px; font-size: 0.85rem;
        line-height: 1.6; margin-bottom: 8px; }
</style></head>
<body>
<h1>TASL FR3 — 数据采集 dashboard</h1>

<div class="row" style="margin-top:16px">
  <div class="col ctl">
    <h3>状态</h3>
    <div id="status">loading...</div>
    <button id="startEpBtn" class="startep"
            onclick="markStartEpisode()">① 放行机械臂 (注入 's')</button>
    <button id="startRecBtn" class="startrec"
            onclick="markStartRecording()">② 开始记录 (注入 'r')</button>
    <button id="markSuccessBtn" class="marksuccess"
            onclick="markSuccess()">③ 标记成功 (注入 'c')</button>
    <button id="discardEpBtn" class="discardep"
            onclick="markDiscard()">✗ 标记失败/丢弃本条 (注入 'a')</button>
  </div>
  <div class="col ctl">
    <h3>数据采集</h3>
    <label>Episode 数</label>
    <input type="number" id="numEpisodes" value="10" min="1"/>
    <label>数据集 (一个任务一个数据集)</label>
    <select id="datasetSel" onchange="onDatasetSel()">
      <option value="__new__">＋ 新建数据集…</option>
    </select>
    <input type="text" id="newDatasetName"
           placeholder="新数据集名 (字母/数字/._-，如 fr3_stackcube_v1)"/>
    <label>任务描述 (由上方 Task 决定)</label>
    <input type="text" id="taskDesc" readonly placeholder="选择任务后自动填充"/>
    <div>
      <button class="primary" id="btnStart" onclick="startCollect()"
              disabled title="需要先完成 Layout 准备并「确认就位」">Start</button>
      <button class="danger" onclick="stopCollect()">Stop</button>
      <button onclick="robotHome()">⌂ Go home</button>
    </div>
    <div style="margin-top:6px">
      <button id="btnSaveVideo" onclick="saveVideo()" disabled
              title="Stop / 标记成功 / 标记失败 之后可保存,下次开始记录前有效">
        💾 保存 demo 视频</button>
      <button onclick="saveLayoutDemo()">🗺 保存 layout</button>
    </div>
    <div id="actionMsg" style="font-size:0.8rem;color:#888;margin-top:8px"></div>
  </div>
</div>

<div class="row" style="margin-top:16px">
  <div class="col ctl" id="layout-panel" style="flex:1 1 100%">
    <h3>Task + Layout 准备（任务 · 物体摆位）<span id="layoutState"
        class="lstate none">未选择</span></h3>

    <div id="taskArea" style="margin-bottom:10px; padding-bottom:10px;
         border-bottom:1px solid #262626">
      <div style="display:flex; gap:6px; align-items:center">
        <select id="taskSel" onchange="onTaskSelect()" style="flex:1; margin-bottom:0"></select>
        <button class="small primary" onclick="taskNewOpen()">＋ New task</button>
        <button class="small danger" onclick="deleteTask()">删除</button>
      </div>
      <div id="taskInfo" style="font-size:0.8rem;color:#888;margin-top:4px">
        选择任务 → prompt 锁定 + 自动加载最近 layout;一个任务可对应多个
        layout,「确认就位」时自动挂到任务。</div>
      <div id="taskForm" style="display:none; border:1px solid #262626;
           border-radius:6px; padding:8px; margin-top:6px">
        <label>新任务 prompt(语言指令)</label>
        <input type="text" id="taskPrompt" placeholder="如 pick up the cube"/>
        <label>task id(留空自动生成)</label>
        <input type="text" id="taskId" placeholder="如 pick-up-the-cube"/>
        <div style="font-size:0.8rem;color:#888;margin:4px 0">
          layout:在下方新建/加载好后,保存任务时挂上当前 layout 作为第一个;
          之后每次「确认就位」用的 layout 都会自动累积到任务名下。
          任务库与 eval 端共享(tasl/tasks_store.json)。</div>
        <div>
          <button class="small primary" onclick="saveNewTask()">保存新任务</button>
          <button class="small" onclick="taskNewClose()">取消</button>
        </div>
      </div>
    </div>

    <div id="camCtl" style="margin-bottom:10px; padding-bottom:10px;
         border-bottom:1px solid #262626">
      <button id="btnCamOn" onclick="camPreview(true)">▶ 开启相机预览</button>
      <button id="btnCamOff" onclick="camPreview(false)">■ 关闭预览</button>
      <span id="camState" style="font-size:0.82rem; color:#888; margin-left:8px"></span>
      <div style="font-size:0.78rem;color:#777;margin-top:6px;line-height:1.6">
        摆位/标定 layout 需要实时画面。预览由 dashboard 持有 ZED；按 Start 时会自动释放，
        并等到容器内确认能打开相机后才启动采集（几秒）。
      </div>
    </div>

    <div id="layoutPick">
      <label>加载已有 layout</label>
      <div style="display:flex; gap:6px; align-items:center">
        <select id="layoutSel" style="flex:1; margin-bottom:0"></select>
        <button class="small" id="btnLayoutLoad" onclick="layoutLoad()">加载</button>
        <button class="small danger" id="btnLayoutDel" onclick="layoutDelete()">删除</button>
      </div>
      <button class="primary" id="btnLayoutNew" onclick="layoutNew()">＋ 新建 Layout</button>
      <button id="btnLayoutEdit" onclick="layoutEditCurrent()">编辑当前</button>
    </div>

    <div id="layoutEditor" style="display:none">
      <label>Layout 名称（字母/数字/._-）</label>
      <input type="text" id="layoutName" placeholder="如 stackcube_3x3_v1"/>
      <label>正在编辑哪个相机（两个 view 共同定义一个 layout，标记互相独立）</label>
      <div>
        <button class="small" id="tabExterior" onclick="setEditView('exterior')">
          外部 (物体摆位)</button>
        <button class="small" id="tabWrist" onclick="setEditView('wrist')">
          腕部 EIH (起始位姿)</button>
      </div>
      <div style="display:flex; gap:10px">
        <div style="flex:1"><label>网格 行</label>
          <input type="number" id="gridRows" value="3" min="1" max="12"
                 onchange="onGridChange()"/></div>
        <div style="flex:1"><label>网格 列</label>
          <input type="number" id="gridCols" value="3" min="1" max="12"
                 onchange="onGridChange()"/></div>
      </div>
      <label>物体标记 — 添加 → 移到位 → 确定，一个一个来</label>
      <div id="objList" style="max-height:170px; overflow:auto;
           border:1px solid #262626; border-radius:4px; padding:4px 8px"></div>
      <div>
        <button class="small" onclick="objAdd()">① ＋ 添加物体(中心)</button>
        <button class="small primary" onclick="objCommit()">② ✓ 确定此物体</button>
        <button class="small" onclick="objRename()">重命名</button>
        <button class="small danger" onclick="objDelete()">删除选中</button>
      </div>
      <div class="kbdhint">
        <b style="color:#999">移动选中的物体：</b>鼠标直接拖 · 点画面空白处跳过去 ·
        <kbd>←</kbd> <kbd>→</kbd> <kbd>↑</kbd> <kbd>↓</kbd> 微调
        （<kbd>Shift</kbd> 粗调 10× · <kbd>Alt</kbd> 精调）<br>
        <kbd>Tab</kbd> 切换下一个 · <kbd>Del</kbd> 删除 · <kbd>Esc</kbd> 退出编辑 ·
        确定后再点空白处会开始<b>新的</b>物体
      </div>
      <div>
        <button class="primary" onclick="layoutSave(VIEWS)">保存 + 抓两个快照</button>
        <button onclick="layoutSave([editView])">保存 + 只抓当前 view 快照</button>
        <button onclick="layoutSave([])">仅保存标记</button>
        <button onclick="layoutCancelEdit()">取消</button>
      </div>
    </div>

    <div id="layoutView" style="display:none">
      <div style="display:flex; gap:18px; flex-wrap:wrap; align-items:center">
        <label style="margin:0; display:inline">
          <input type="checkbox" id="showGrid" checked onchange="drawStencil()"/>
          显示网格+标记</label>
        <label style="margin:0; display:inline">
          <input type="checkbox" id="showGhost" checked onchange="drawStencil()"/>
          显示参考快照</label>
        <div style="flex:1; min-width:200px">
          <label style="margin:0">ghost 透明度 <span id="ghostVal">45%</span></label>
          <input type="range" id="ghostOp" min="0" max="100" value="45"
                 oninput="drawStencil()" style="width:100%"/>
        </div>
      </div>
      <button class="primary" onclick="layoutConfirm()">✓ 确认就位（解锁 Start）</button>
      <button onclick="layoutClear()">取消选择</button>
    </div>
    <div id="layoutMsg" style="font-size:0.8rem;color:#888;margin-top:8px"></div>
  </div>
</div>

<div class="row" style="margin-top:16px">
  <div class="col cam">
    <div class="label">exterior (ZED 2i)<span id="layoutBadge"></span></div>
    <div class="camwrap" id="camwrap-exterior">
      <img id="cam-exterior" class="feed" alt="exterior"/>
      <img id="ghost-exterior" class="ghost" alt="layout reference"
           style="display:none"/>
      <div id="stencil-exterior" class="stencil" style="display:none"></div>
    </div>
    <div id="ph-exterior" class="placeholder" style="display:none"></div>
  </div>
  <div class="col cam">
    <div class="label">wrist (ZED Mini, eye-in-hand)<span id="layoutBadgeW"></span></div>
    <div class="camwrap" id="camwrap-wrist">
      <img id="cam-wrist" class="feed" alt="wrist"/>
      <img id="ghost-wrist" class="ghost" alt="layout reference"
           style="display:none"/>
      <div id="stencil-wrist" class="stencil" style="display:none"></div>
    </div>
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
  <div class="col ctl" id="session-panel">
    <h3>会话 / 硬件释放</h3>
    <div id="orphanWarn" class="warnbox" style="display:none"></div>
    <div style="font-size:0.82rem;color:#8d8d8d;line-height:1.75">
      <b style="color:#bbb">清理残留</b> — 杀掉容器内的采集进程，释放 ZED + GELLO。
      <u>不动 FCI</u>，可以直接重开下一条。<br>
      <b style="color:#bbb">结束会话</b> — 跑 <code>teleop-stop.sh</code> 全套：再额外关掉
      NUC 控制器、<b>释放 FCI</b>。收工时用。
    </div>
    <div>
      <button onclick="sessionReap()">清理残留进程</button>
      <button class="danger" onclick="sessionStop()">结束会话 (释放 FCI)</button>
    </div>
    <div id="sessionMsg" style="font-size:0.8rem;color:#888;margin-top:8px"></div>
    <pre id="sessionOut" style="display:none"></pre>
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
  <div id="playSpeeds" style="margin-top:8px">
    <span style="color:#999;font-size:0.82rem;margin-right:6px">倍速</span>
    <button class="small" data-s="0.5" onclick="setPlaySpeed(0.5)">0.5×</button>
    <button class="small primary" data-s="1" onclick="setPlaySpeed(1)">1×</button>
    <button class="small" data-s="2" onclick="setPlaySpeed(2)">2×</button>
    <button class="small" data-s="4" onclick="setPlaySpeed(4)">4×</button>
    <button class="small" data-s="8" onclick="setPlaySpeed(8)">8×</button>
    <span style="color:#666;font-size:0.75rem;margin-left:10px">
      改倍速会从头重放</span>
  </div>
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
      // Actionable, not just a statement of fact: the preview is opt-in, so
      // "idle" is a state the operator can leave from right here.
      ph.textContent = '相机空闲 — 点 Layout 面板里的「▶ 开启相机预览」看实时画面';
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

// ── Task 区域(与 eval 端共享任务库)────────────────────────────────
// 选中任务 → prompt 只读锁定 + 自动加载其 layout 蒙版;layout 的
// 新建/编辑/删除只在 New task 模式开放(确认就位不受影响)。
let taskList = [];
let taskNewMode = false;
const jsSlug = s => (s || '').toLowerCase().replace(/[^a-z0-9]+/g, '-')
  .replace(/^-+|-+$/g, '') || '';
function selectedTask() {
  const id = document.getElementById('taskSel').value;
  return taskList.find(t => t.id === id);
}
async function taskRefresh() {
  try {
    const r = await fetch('/api/tasks');
    const j = await r.json();
    taskList = j.tasks || [];
    const sel = document.getElementById('taskSel');
    const cur = sel.value;
    sel.innerHTML = '<option value="">(选择任务)</option>' + taskList.map(t =>
      '<option value="' + esc(t.id) + '">' + esc(t.id) + '</option>').join('');
    if (cur && taskList.some(t => t.id === cur)) sel.value = cur;
  } catch (e) {}
}
function renderTaskInfo(t) {
  const info = document.getElementById('taskInfo');
  if (!t) {
    info.textContent = '选择任务 → prompt 锁定 + 自动加载最近 layout;'
      + '一个任务可对应多个 layout,「确认就位」时自动挂到任务。';
    return;
  }
  const lays = t.layouts || [];
  info.textContent = 'prompt: ' + t.prompt
    + ' · layouts(' + lays.length + '): ' + (lays.join(', ') || '—')
    + ' · datasets: ' + ((t.datasets || []).join(', ') || '—');
}
async function onTaskSelect() {
  taskNewClose();
  const t = selectedTask();
  renderTaskInfo(t);
  if (!t) {
    document.getElementById('taskDesc').value = '';
    loadLayoutList();
    return;
  }
  document.getElementById('taskDesc').value = t.prompt;
  await loadLayoutList();   // 重建分组(本任务的 layouts 排最前)
  if (t.layout) {
    const sel = document.getElementById('layoutSel');
    if ([...sel.options].some(o => o.value === t.layout)) {
      sel.value = t.layout;
      await layApi('/api/layout/select', {id: t.layout});
      refresh();
      layMsg('最近使用的 layout「' + t.layout + '」已加载 — 换一个直接选+加载,'
        + '或新建;开启相机预览可见蒙版,摆好后点「确认就位」');
    } else {
      layMsg('任务最近的 layout「' + t.layout + '」已不存在 — 在下方另选或新建');
    }
  } else {
    layMsg('任务「' + t.id + '」还没有用过 layout — 在下方选择或新建一个,'
      + '「确认就位」时会自动挂到任务');
  }
}
function taskNewOpen() {
  taskNewMode = true;
  document.getElementById('taskSel').value = '';
  document.getElementById('taskForm').style.display = '';
  document.getElementById('taskPrompt').value = '';
  document.getElementById('taskId').value = '';
  document.getElementById('taskDesc').value = '';
  document.getElementById('taskPrompt').focus();
}
function taskNewClose() {
  taskNewMode = false;
  document.getElementById('taskForm').style.display = 'none';
}
async function saveNewTask() {
  const prompt = document.getElementById('taskPrompt').value.trim();
  if (!prompt) { alert('prompt 不能为空'); return; }
  const id = document.getElementById('taskId').value.trim() || jsSlug(prompt);
  if (!id) { alert('task id 不能为空'); return; }
  const layout = (LAY.selected || document.getElementById('layoutSel').value || '');
  const j = await api('/api/task/create', {id, prompt, layout, datasets: []});
  if (j.ok) {
    await taskRefresh();
    document.getElementById('taskSel').value = id;
    onTaskSelect();
  }
}
async function deleteTask() {
  const id = document.getElementById('taskSel').value;
  if (!id) { alert('先选择要删除的任务'); return; }
  if (!confirm('删除任务「' + id + '」?(不影响数据集和 layout 文件)')) return;
  const j = await api('/api/task/delete', {id});
  if (j.ok) {
    await taskRefresh();
    document.getElementById('taskSel').value = '';
    onTaskSelect();
  }
}

// Dataset selector: '__new__' reveals the name input; picking an existing
// dataset hides it. The task instruction comes from the Task 区域 — the old
// per-dataset autofill only kicks in when no task is selected (legacy sets).
function onDatasetSel() {
  const sel = document.getElementById('datasetSel');
  const inp = document.getElementById('newDatasetName');
  if (sel.value === '__new__') {
    inp.style.display = '';
  } else {
    inp.style.display = 'none';
    const d = dsCache.find(x => x.category === 'current' && x.name === sel.value);
    if (d && d.task && !document.getElementById('taskSel').value) {
      document.getElementById('taskDesc').value = d.task;
    }
  }
}

function populateDatasetSelector() {
  const sel = document.getElementById('datasetSel');
  if (!sel) return;
  const prev = sel.value;
  const cur = dsCache.filter(d => d.category === 'current' && !d.error);
  let html = '<option value="__new__">＋ 新建数据集…</option>';
  cur.forEach(d => {
    html += '<option value="' + esc(d.name) + '">' + esc(d.name)
      + ' (' + (d.total_episodes ?? 0) + ' ep)</option>';
  });
  sel.innerHTML = html;
  // Keep the previous selection if it still exists, else stay on 新建.
  sel.value = cur.some(d => d.name === prev) ? prev : '__new__';
  onDatasetSel();
}

function currentDatasetName() {
  const sel = document.getElementById('datasetSel');
  if (sel.value === '__new__') {
    return document.getElementById('newDatasetName').value.trim();
  }
  return sel.value;
}

async function startCollect() {
  const n = parseInt(document.getElementById('numEpisodes').value, 10);
  const tid = document.getElementById('taskSel').value;
  const task = document.getElementById('taskDesc').value.trim();
  const ds = currentDatasetName();
  if (!tid) {
    alert('请先在 Task 区选择任务(或 New task 新建并保存)'); return;
  }
  if (!n || n < 1) { alert('Episode 数必须 >= 1'); return; }
  if (!ds) { alert('请选择或新建一个数据集'); return; }
  if (!/^[A-Za-z0-9._-]+$/.test(ds)) {
    alert('数据集名只能含 字母/数字/._-'); return;
  }
  if (!task) { alert('任务描述不能为空'); return; }
  if (!LAY.confirmed) {
    alert('Layout 准备未完成 — 请先选择/新建 layout，摆好物体后点「确认就位」');
    return;
  }
  const j = await api('/api/collect/start',
                      {num_episodes: n, task_description: task, dataset_name: ds,
                       task_id: tid});
  if (j && j.ok !== false) setSaveVideo(false);
  return j;
}

async function stopCollect() {
  if (!confirm('确认停止采集？(kill 进程 + ray cleanup)')) return;
  const j = await api('/api/collect/stop');
  if (j && j.ok !== false) setSaveVideo(true);
  return j;
}

// ── Demo 导出(RLinf/saved_demo)──────────────────────────────────────
// 保存视频只在 Stop / 标记成功 / 标记失败 之后可点,下次开始记录时重新禁用;
// 保存 layout 全程可点。视频来自数据集最新一条 episode(数据集原生分辨率)。
function setSaveVideo(on) {
  const el = document.getElementById('btnSaveVideo');
  if (el) el.disabled = !on;
}
async function saveVideo() {
  const ds = currentDatasetName();
  if (!ds) { alert('请先选择数据集'); return; }
  return api('/api/save_video',
             {dataset: ds, task_id: document.getElementById('taskSel').value});
}
async function saveLayoutDemo() {
  const lay = LAY.selected || document.getElementById('layoutSel').value;
  if (!lay) { alert('还没有选择 layout'); return; }
  return api('/api/save_layout',
             {layout: lay, task_id: document.getElementById('taskSel').value});
}

async function markSuccess() {
  const j = await api('/api/mark_success');
  if (j && j.ok !== false) setSaveVideo(true);
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

async function markStartRecording() {
  const j = await api('/api/start_recording');
  if (j && j.ok !== false) setSaveVideo(false);
  if (j && j.caveat) {
    document.getElementById('actionMsg').textContent =
      (j.msg || '') + ' — ' + j.caveat;
  }
}

async function markDiscard() {
  if (!confirm('丢弃本条？\\n\\n这一条不会保存，机械臂会复位重来。\\n'
      + '（比录完再删安全：数据集里根本不会出现它）')) return;
  const j = await api('/api/discard_episode');
  if (j && j.ok !== false) setSaveVideo(true);
  if (j && j.caveat) {
    document.getElementById('actionMsg').textContent =
      (j.msg || '') + ' — ' + j.caveat;
  }
}

// ── session teardown ────────────────────────────────────────────────
async function sessionApi(path) {
  const msg = document.getElementById('sessionMsg');
  const out = document.getElementById('sessionOut');
  msg.textContent = '执行中…（NUC ssh 这步可能要十几秒）';
  out.style.display = 'none';
  try {
    const r = await fetch(path, {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}',
    });
    const j = await r.json();
    msg.textContent = j.msg || JSON.stringify(j);
    if (j.output) { out.textContent = j.output; out.style.display = ''; }
    refresh();
    return j;
  } catch (e) {
    msg.textContent = '请求失败: ' + e;
  }
}

// NOTE: INDEX_HTML is a plain (non-raw) Python string, so every backslash is
// consumed by PYTHON before the browser ever sees it. A newline escape must be
// written DOUBLED in the source; written singly it becomes a real line break
// inside the JS string literal, which terminates the string and takes the
// entire <script> block down with it — every panel then sits on "loading...".
async function sessionReap() {
  if (!confirm('清理容器内的采集进程？\\n\\n'
      + '• 释放 ZED 相机 + GELLO\\n'
      + '• 正在进行的采集会被终止，未保存的 episode 丢失\\n'
      + '• FCI / NUC 控制器不受影响，可以直接重开')) return;
  return sessionApi('/api/session/reap');
}

async function sessionStop() {
  if (!confirm('结束整个会话？(teleop-stop.sh)\\n\\n'
      + '• 杀掉容器内采集进程 → 释放相机 + GELLO\\n'
      + '• 关掉 NUC 控制器 → 释放 FCI\\n\\n'
      + '之后要再采集，必须重新跑 teleop.sh 并在 Desk 里重新激活 FCI。')) return;
  return sessionApi('/api/session/stop');
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
  const coll = j.collection || {};
  const waiting = !!coll.waiting_for_start;
  const released = !!coll.robot_released;
  // Exactly one big button per gate state: WAIT -> 放行机械臂,
  // LIVE -> 开始记录, RUN -> 标记成功.
  document.getElementById('startEpBtn').style.display =
    waiting ? 'block' : 'none';
  document.getElementById('startRecBtn').style.display =
    released ? 'block' : 'none';
  const midEpisode = running && !waiting && !released;
  document.getElementById('markSuccessBtn').style.display =
    midEpisode ? 'block' : 'none';
  // Discard shares the mid-episode state: the two ways an episode can end.
  document.getElementById('discardEpBtn').style.display =
    midEpisode ? 'block' : 'none';
  for (const id of ['btnHome', 'btnRecover', 'btnGripOpen', 'btnGripClose']) {
    document.getElementById(id).disabled = !!running;
  }
  // Orphaned run: it owns the ZEDs + GELLO, so the next teleop.sh preflight
  // will fail on "cameras" unless it is reaped first.
  const ow = document.getElementById('orphanWarn');
  if (coll.orphan) {
    ow.style.display = '';
    ow.innerHTML = '⚠ 检测到一个<b>不是本 dashboard 启动</b>的采集进程正在运行'
      + '（dashboard 重启过？）。它占着 ZED + GELLO，会让下次 teleop.sh 的'
      + '相机检查失败 —— 按「清理残留进程」清掉。';
  } else {
    ow.style.display = 'none';
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

// ── layout-prepare ──────────────────────────────────────────────────
// The stencil is composited client-side over the exterior feed: grid lines
// and markers are absolutely-positioned divs in percent coordinates, so they
// track the preview at any size without any aspect-ratio math, and the ghost
// is just the reference JPEG at a chosen opacity.
const VIEWS = ['exterior', 'wrist'];
let LAY = {};            // server gate state, from /api/status
let layEditing = false;  // authoring/editing a stencil
let draft = null;        // {views: {exterior:{grid,objects}, wrist:{...}}, ...}
let editView = 'exterior';  // which camera the editor is acting on
let selIdx = -1;         // selected marker index WITHIN editView
let layCache = [];

function draftView(v) { return draft ? draft.views[v || editView] : null; }

function clamp01(v) { return Math.min(1, Math.max(0, v)); }
function layMsg(t) { document.getElementById('layoutMsg').textContent = t; }

async function camPreview(on) {
  layMsg(on ? '正在打开 ZED…' : '正在释放 ZED…');
  const j = await layApi(on ? '/api/cams/start' : '/api/cams/stop');
  if (j && j.missing_cams && j.missing_cams.length) {
    layMsg('打开失败的相机: ' + JSON.stringify(j.missing_cams));
  }
  camsShown = null;   // force setCams to re-sync the <img> sources
  refresh();
}

function renderCamCtl(camRunning, collecting) {
  const on = document.getElementById('btnCamOn');
  const off = document.getElementById('btnCamOff');
  const st = document.getElementById('camState');
  if (!on) return;
  on.disabled = camRunning || collecting;
  off.disabled = !camRunning || collecting;
  if (collecting) {
    st.innerHTML = '<span class="dot ok"></span>采集中 — 相机由 env 持有';
  } else if (camRunning) {
    st.innerHTML = '<span class="dot ok"></span>预览开启中 (dashboard 持有 ZED)';
  } else {
    st.innerHTML = '<span class="dot idle"></span>相机空闲 — 无实时画面';
  }
}

async function layApi(path, body, method) {
  try {
    const opt = {method: method || 'POST',
                 headers: {'Content-Type': 'application/json'}};
    if (opt.method !== 'DELETE') opt.body = JSON.stringify(body || {});
    const r = await fetch(path, opt);
    const j = await r.json();
    layMsg(j.msg || JSON.stringify(j));
    return j;
  } catch (e) { layMsg('请求失败: ' + e); return {ok: false}; }
}

async function loadLayoutList() {
  try {
    const r = await fetch('/api/layouts');
    const j = await r.json();
    layCache = j.layouts || [];
    const sel = document.getElementById('layoutSel');
    const prev = sel.value;
    const opt = function (l, star) {
      if (l.error) return '<option value="' + esc(l.id) + '">'
                         + esc(l.id) + '  (损坏)</option>';
      const n = l.n_objects || {};
      const snap = l.has_snapshot || {};
      const shots = VIEWS.filter(function (v) { return snap[v]; });
      return '<option value="' + esc(l.id) + '">' + (star ? '★ ' : '')
        + esc(l.id)
        + '  — 外部 ' + (n.exterior || 0) + ' / 腕部 ' + (n.wrist || 0)
        + (shots.length ? ' · 快照 ' + shots.join('+') : '')
        + '</option>';
    };
    // 分组:当前任务用过的 layouts 排最前(★),其余在后。
    const t = selectedTask();
    const mine = new Set((t && t.layouts) || []);
    const ours = layCache.filter(l => mine.has(l.id));
    const rest = layCache.filter(l => !mine.has(l.id));
    let html = '';
    if (ours.length) {
      html += '<optgroup label="本任务用过的 layouts">'
        + ours.map(l => opt(l, true)).join('') + '</optgroup>';
      html += rest.length
        ? '<optgroup label="其他 layouts">'
          + rest.map(l => opt(l, false)).join('') + '</optgroup>'
        : '';
    } else {
      html = layCache.map(l => opt(l, false)).join('');
    }
    sel.innerHTML = html
      || '<option value="">(还没有 layout — 点「新建」)</option>';
    if (layCache.some(l => l.id === prev)) sel.value = prev;
  } catch (e) { layMsg('layout 列表加载失败: ' + e); }
}

// What to draw over ONE camera. Editing shows the draft; otherwise the armed
// layout. The two views are independent: the cameras see completely different
// geometry, so a marker in one has no counterpart in the other.
function stencilFor(view) {
  if (layEditing && draft) {
    const d = draft.views[view];
    return {grid: d.grid, objects: d.objects,
            ghostId: draft.snapId, ghostVer: draft.ghostVer,
            hasSnap: !!(draft.hasSnap || {})[view]};
  }
  if (LAY.selected && LAY.views) {
    const d = LAY.views[view] || {};
    return {grid: d.grid, objects: d.objects || [],
            ghostId: LAY.selected, ghostVer: LAY.hash,
            hasSnap: !!(LAY.has_snapshot || {})[view]};
  }
  return null;
}

function setEditView(v) {
  editView = v;
  selIdx = -1;                    // marker indices are per-view
  const d = draftView();
  if (d) {
    document.getElementById('gridRows').value = d.grid.rows;
    document.getElementById('gridCols').value = d.grid.cols;
  }
  document.getElementById('tabExterior').className =
    'small' + (v === 'exterior' ? ' primary' : '');
  document.getElementById('tabWrist').className =
    'small' + (v === 'wrist' ? ' primary' : '');
  layMsg(v === 'exterior'
    ? '编辑「外部」view — 标物体该摆在哪'
    : '编辑「腕部 EIH」view — 标 episode 起始时夹爪该看到什么');
  drawStencil(); renderObjList();
}

function drawStencil() { VIEWS.forEach(drawOneView); }

function drawOneView(view) {
  const wrap = document.getElementById('camwrap-' + view);
  const sten = document.getElementById('stencil-' + view);
  const ghost = document.getElementById('ghost-' + view);
  if (!wrap || !sten || !ghost) return;
  // Only the view being edited takes mouse input, so a stray click on the
  // other camera can never drop a marker into the wrong view.
  wrap.classList.toggle('editing', layEditing && view === editView);
  const a = stencilFor(view);
  if (!a) {
    sten.style.display = 'none'; sten.innerHTML = '';
    ghost.style.display = 'none'; ghost.removeAttribute('src');
    ghost.dataset.key = '';
    return;
  }
  const cbGrid = document.getElementById('showGrid');
  const cbGhost = document.getElementById('showGhost');
  const opEl = document.getElementById('ghostOp');
  const valEl = document.getElementById('ghostVal');
  if (valEl && opEl) valEl.textContent = opEl.value + '%';
  // While editing, the stencil is always on — you cannot place what you
  // cannot see; the checkboxes only govern the view/confirm step.
  const wantGrid = layEditing || !cbGrid || cbGrid.checked;
  const wantGhost = a.hasSnap && !!a.ghostId && (!cbGhost || cbGhost.checked);

  if (wantGhost) {
    // Cache-bust on the stencil hash: re-saving a layout overwrites the JPEG
    // in place, and a cached ghost would be aligned to the old arrangement.
    const key = '/api/layout/' + encodeURIComponent(a.ghostId)
              + '/snapshot/' + view + '.jpg?v='
              + encodeURIComponent(a.ghostVer || '');
    if (ghost.dataset.key !== key) { ghost.src = key; ghost.dataset.key = key; }
    ghost.style.opacity = opEl ? (parseInt(opEl.value, 10) || 0) / 100 : 0.45;
    ghost.style.display = '';
  } else {
    ghost.style.display = 'none';
  }

  if (!wantGrid) { sten.style.display = 'none'; return; }
  const g = a.grid || {rows: 3, cols: 3};
  const active = layEditing && view === editView;
  let html = '';
  for (let i = 1; i < g.cols; i++) {
    html += '<div class="gline v" style="left:' + (i * 100 / g.cols) + '%"></div>';
  }
  for (let j = 1; j < g.rows; j++) {
    html += '<div class="gline h" style="top:' + (j * 100 / g.rows) + '%"></div>';
  }
  (a.objects || []).forEach(function (o, i) {
    html += '<div class="mk' + (active && i === selIdx ? ' sel' : '')
         + '" data-i="' + i + '" style="left:' + (o.x * 100)
         + '%;top:' + (o.y * 100) + '%">'
         + '<span class="mklabel">' + esc(o.label) + '</span></div>';
  });
  sten.innerHTML = html;
  sten.style.display = '';
}

// Which grid cell a marker LANDED in. Purely informational — positions are
// continuous and never snap; the grid is a visual reference, so an object may
// sit anywhere inside a cell (or straddle a line).
function cellOf(o, g) {
  const c = Math.min(g.cols, Math.floor(o.x * g.cols) + 1);
  const r = Math.min(g.rows, Math.floor(o.y * g.rows) + 1);
  return 'R' + r + 'C' + c;
}

function renderObjList() {
  const el = document.getElementById('objList');
  if (!el) return;
  const d = draftView();
  if (!d || !d.objects.length) {
    el.innerHTML = '<div style="color:#777;font-size:0.85rem;padding:6px 0">'
      + esc(editView) + ' view 还没有标记 — 点击该相机画面添加，或按「＋ 添加物体」'
      + '</div>';
    return;
  }
  const g = d.grid || {rows: 3, cols: 3};
  el.innerHTML = d.objects.map(function (o, i) {
    return '<div class="objrow' + (i === selIdx ? ' sel' : '')
      + '" onclick="objSelect(' + i + ')">'
      + '<span class="idx">' + (i + 1) + '</span>'
      + '<span style="flex:1">' + esc(o.label) + '</span>'
      + '<span class="coord">' + cellOf(o, g) + ' · '
      + o.x.toFixed(4) + ', ' + o.y.toFixed(4)
      + '</span></div>';
  }).join('');
}

function objSelect(i) { selIdx = i; drawStencil(); renderObjList(); }

// Add -> position -> commit, one object at a time. Naming happens at COMMIT,
// not at add: you know what the thing is once it's sitting where it belongs,
// and a prompt up front interrupts the placing rhythm.
function objAdd() {
  const d = draftView();
  if (!d) return;
  d.objects.push({label: 'obj' + (d.objects.length + 1), x: 0.5, y: 0.5});
  selIdx = d.objects.length - 1;
  drawStencil(); renderObjList();
  layMsg('已放到「' + editView + '」画面中心 — 鼠标拖动 或 方向键微调，'
         + '到位后按「✓ 确定此物体」');
}

function objCommit() {
  const d = draftView();
  if (!d || selIdx < 0) { layMsg('没有正在放置的物体'); return; }
  const cur = d.objects[selIdx];
  const label = prompt('这个物体叫什么？', cur.label);
  if (label === null) return;          // cancel keeps it selected, still movable
  if (label.trim()) cur.label = label.trim();
  const where = editView + ' ' + cur.x.toFixed(3) + ', ' + cur.y.toFixed(3);
  // Deselect so the next click on empty canvas starts a NEW object rather
  // than dragging the one just finished.
  selIdx = -1;
  drawStencil(); renderObjList();
  layMsg('已确定 ' + cur.label + ' @ ' + where + ' — 「＋ 添加物体」放下一个，'
         + '或「保存 + 抓参考快照」结束');
}

function objRename() {
  const d = draftView();
  if (!d || selIdx < 0) { layMsg('先选中一个标记'); return; }
  const cur = d.objects[selIdx];
  const label = prompt('新名称：', cur.label);
  if (label === null) return;
  if (label.trim()) { cur.label = label.trim(); drawStencil(); renderObjList(); }
}

function objDelete() {
  const d = draftView();
  if (!d || selIdx < 0) { layMsg('先选中一个标记'); return; }
  d.objects.splice(selIdx, 1);
  selIdx = Math.min(selIdx, d.objects.length - 1);
  drawStencil(); renderObjList();
}

function onGridChange() {
  const d = draftView();
  if (!d) return;
  // Grid is per-view: the two cameras frame the bench completely differently,
  // so one may want 3x3 and the other 2x4.
  d.grid = {rows: parseInt(document.getElementById('gridRows').value, 10) || 3,
            cols: parseInt(document.getElementById('gridCols').value, 10) || 3};
  drawStencil();
}

// Mouse: press a marker to grab it, drag to move; press empty canvas to jump
// the marker being placed straight there (coarse), or to start a new one if
// nothing is selected. Fine positioning is the arrow keys.
let dragIdx = -1;

function stencilPoint(view, e) {
  const r = document.getElementById('stencil-' + view).getBoundingClientRect();
  return {x: clamp01((e.clientX - r.left) / r.width),
          y: clamp01((e.clientY - r.top) / r.height)};
}

function onStencilDown(e) {
  if (!layEditing || !draft) return;
  const view = e.currentTarget.id.replace('stencil-', '');
  // Clicking the other camera switches the editor to it rather than doing
  // nothing — the two views are edited in the same panel.
  if (view !== editView) { setEditView(view); }
  const d = draftView();
  const mk = e.target.closest ? e.target.closest('.mk') : null;
  if (mk) {
    selIdx = parseInt(mk.dataset.i, 10);
    dragIdx = selIdx;
    e.preventDefault();
    drawStencil(); renderObjList();
    return;
  }
  const p = stencilPoint(view, e);
  if (selIdx >= 0 && d.objects[selIdx]) {
    d.objects[selIdx].x = p.x;
    d.objects[selIdx].y = p.y;
  } else {
    d.objects.push({label: 'obj' + (d.objects.length + 1), x: p.x, y: p.y});
    selIdx = d.objects.length - 1;
    layMsg('已放置 — 拖动/方向键微调，到位后按「✓ 确定此物体」');
  }
  dragIdx = selIdx;
  e.preventDefault();
  drawStencil(); renderObjList();
}

function onStencilMove(e) {
  const d = draftView();
  if (dragIdx < 0 || !d || !d.objects[dragIdx]) return;
  const p = stencilPoint(editView, e);
  d.objects[dragIdx].x = p.x;
  d.objects[dragIdx].y = p.y;
  e.preventDefault();
  drawStencil(); renderObjList();
}

function onStencilUp() { dragIdx = -1; }

// Arrow-key nudge. Steps are in normalized image units: ~0.4% of the frame
// per press, 10x with Shift for crossing the scene, 0.1% with Alt to settle
// on an edge. Ignored while a text field has focus.
document.addEventListener('keydown', function (e) {
  if (!layEditing || !draft) return;
  const t = e.target;
  if (t && /^(INPUT|SELECT|TEXTAREA)$/.test(t.tagName)) return;
  const dv = draftView();
  if (!dv) return;
  const step = e.shiftKey ? 0.04 : (e.altKey ? 0.001 : 0.004);
  const o = selIdx >= 0 ? dv.objects[selIdx] : null;
  let handled = true;
  switch (e.key) {
    case 'ArrowLeft':  if (o) o.x = clamp01(o.x - step); break;
    case 'ArrowRight': if (o) o.x = clamp01(o.x + step); break;
    case 'ArrowUp':    if (o) o.y = clamp01(o.y - step); break;
    case 'ArrowDown':  if (o) o.y = clamp01(o.y + step); break;
    case 'Tab':
      if (dv.objects.length) {
        selIdx = (selIdx + (e.shiftKey ? -1 : 1) + dv.objects.length)
                 % dv.objects.length;
      }
      break;
    case 'Delete': case 'Backspace': objDelete(); break;
    case 'Escape': layoutCancelEdit(); return;
    default: handled = false;
  }
  if (!handled) return;
  e.preventDefault();
  drawStencil(); renderObjList();
});

function enterEdit() {
  layEditing = true;
  document.getElementById('layoutEditor').style.display = '';
  document.getElementById('layoutPick').style.display = 'none';
  document.getElementById('layoutView').style.display = 'none';
  renderObjList(); drawStencil();
}

function blankViews() {
  return {exterior: {grid: {rows: 3, cols: 3}, objects: []},
          wrist:    {grid: {rows: 3, cols: 3}, objects: []}};
}

function layoutNew() {
  draft = {views: blankViews(), snapId: null, ghostVer: '', hasSnap: {}};
  selIdx = -1;
  document.getElementById('layoutName').value = '';
  enterEdit();
  setEditView('exterior');
  layMsg('新建 layout：两个相机各标各的 — 外部标物体摆位，腕部标起始位姿。'
         + '保存时建议两个 view 都抓快照');
}

async function layoutEditCurrent() {
  if (!LAY.selected) { layMsg('先加载一个 layout 再编辑'); return; }
  try {
    const r = await fetch('/api/layout/' + encodeURIComponent(LAY.selected));
    const j = await r.json();
    if (!j.ok) { layMsg(j.msg || '加载失败'); return; }
    const L = j.layout;
    const v = {};
    VIEWS.forEach(function (name) {
      const src = L.views[name] || {grid: {rows: 3, cols: 3}, objects: []};
      v[name] = {grid: {rows: src.grid.rows, cols: src.grid.cols},
                 objects: (src.objects || []).map(function (o) {
                   return {label: o.label, x: o.x, y: o.y};
                 })};
    });
    draft = {views: v, snapId: L.id, ghostVer: L.hash,
             hasSnap: L.has_snapshot || {}};
    selIdx = -1;
    document.getElementById('layoutName').value = L.id;
    enterEdit();
    setEditView('exterior');
    layMsg('编辑 ' + L.id + ' — 改动保存后需重新「确认就位」');
  } catch (e) { layMsg('加载失败: ' + e); }
}

function layoutCancelEdit() {
  layEditing = false; draft = null; selIdx = -1;
  document.getElementById('layoutEditor').style.display = 'none';
  document.getElementById('layoutPick').style.display = '';
  layMsg('已退出编辑');
  refresh();
}

async function layoutSave(capture) {
  if (!draft) return;
  const name = document.getElementById('layoutName').value.trim();
  if (!/^[A-Za-z0-9._-]{1,64}$/.test(name)) {
    alert('Layout 名称只能含 字母/数字/._-（最长 64）'); return;
  }
  const total = VIEWS.reduce(function (n, v) {
    return n + draft.views[v].objects.length;
  }, 0);
  if (!total && !confirm('两个 view 都没有物体标记，仍要保存？')) return;
  onGridChange();
  const j = await layApi('/api/layout', {
    id: name, views: draft.views, capture: capture,
  });
  if (!j.ok) return;
  layEditing = false; draft = null; selIdx = -1;
  document.getElementById('layoutEditor').style.display = 'none';
  document.getElementById('layoutPick').style.display = '';
  await loadLayoutList();
  document.getElementById('layoutSel').value = name;
  await layApi('/api/layout/select', {id: name});
  refresh();
}

async function layoutLoad() {
  const id = document.getElementById('layoutSel').value;
  if (!id) { layMsg('没有可加载的 layout'); return; }
  await layApi('/api/layout/select', {id: id});
  refresh();
}

async function layoutDelete() {
  const id = document.getElementById('layoutSel').value;
  if (!id) return;
  if (!confirm('确认删除 layout「' + id + '」？(含参考快照，不可恢复)')) return;
  await layApi('/api/layout/' + encodeURIComponent(id), null, 'DELETE');
  await loadLayoutList();
  refresh();
}

async function layoutConfirm() {
  const tid = document.getElementById('taskSel').value;
  const j = await layApi('/api/layout/confirm', tid ? {task_id: tid} : null);
  if (j && j.ok && tid) {
    await taskRefresh();               // layouts 列表可能新增了这一个
    renderTaskInfo(selectedTask());
    loadLayoutList();
  }
  refresh();
}

async function layoutClear() {
  await layApi('/api/layout/clear');
  refresh();
}

function renderLayout(j) {
  LAY = (j && j.layout) || {};
  const st = document.getElementById('layoutState');
  const badges = {exterior: document.getElementById('layoutBadge'),
                  wrist: document.getElementById('layoutBadgeW')};
  function setBadges(color, suffix) {
    VIEWS.forEach(function (v) {
      if (!badges[v]) return;
      if (!LAY.selected) { badges[v].innerHTML = ''; return; }
      const n = ((LAY.views || {})[v] || {}).objects || [];
      badges[v].innerHTML = ' <span style="color:' + color + '">▣ '
        + esc(LAY.selected) + ' · ' + n.length + ' obj' + suffix + '</span>';
    });
  }
  if (LAY.confirmed) {
    st.className = 'lstate ok';
    st.textContent = '✓ ' + LAY.selected + ' 已确认 ' + (LAY.confirmed_at || '');
    setBadges('#7fd67f', '');
  } else if (LAY.selected) {
    st.className = 'lstate sel';
    st.textContent = LAY.selected + ' — 未确认';
    setBadges('#e5cf8a', ' (未确认)');
  } else {
    st.className = 'lstate none';
    st.textContent = '未选择';
    setBadges('', '');
  }
  if (!layEditing) {
    document.getElementById('layoutView').style.display =
      LAY.selected ? '' : 'none';
    document.getElementById('layoutPick').style.display = '';
  }
  const btn = document.getElementById('btnStart');
  if (btn) {
    btn.disabled = !LAY.confirmed;
    btn.title = LAY.confirmed ? ''
      : '需要先完成 Layout 准备并「确认就位」';
  }
  if (!layEditing) drawStencil();
}

async function refresh() {
  try {
    const r = await fetch('/api/status');
    const j = await r.json();
    const c = j.collection || {};
    let stateLabel, dotCls;
    // waiting_for_start is driven by the unbuffered gate FILE, so it is
    // authoritative even while phase still reads "starting" (Ray buffers the
    // actor stdout the phase parser keys on). Surface it as its own state so
    // "arm reset, waiting for s" doesn't masquerade as a stuck "starting".
    if (c.running && c.waiting_for_start) {
      stateLabel = '等待放行 / waiting for release'; dotCls = 'ok';
    } else if (c.running && c.robot_released) {
      stateLabel = '已放行 · 未记录 / live, NOT recording'; dotCls = 'idle';
    } else if (c.running) {
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
    // Suppress the raw "(phase=starting)" suffix while waiting at the start
    // gate — it's the buffered-log phase and reads as "stuck" there.
    const phaseSuffix = (c.running && (c.waiting_for_start || c.robot_released))
      ? '' : ' (phase=' + esc(c.phase || '-') + ')';
    html += '<tr><td><span class="dot ' + dotCls + '"></span>状态</td><td>'
         + esc(stateLabel) + phaseSuffix + '</td></tr>';
    // Start gate: arm has reset and is waiting for the operator to begin the
    // (next) episode. Reposition the scene, then press 开始下一条 (or 's').
    if (c.waiting_for_start) {
      const done = c.episodes_done || 0;
      html += '<tr><td>就绪</td><td>'
        + '<b style="color:#1565c0">⏸ 机械臂已复位并锁住 — 按 layout 摆好物体后'
        + '按「① 放行机械臂」(或键盘 s 键)</b><br>'
        + '<span style="font-size:12px;color:#777">已完成 ' + done
        + (c.target ? ' / ' + c.target : '') + ' 条</span></td></tr>';
    }
    // Released but not recording: the most dangerous state to misread — the
    // arm follows GELLO and looks exactly like collection, but nothing is
    // being saved. Say so loudly.
    if (c.robot_released) {
      html += '<tr><td>已放行</td><td>'
        + '<b style="color:#e09b3d">▶ 机械臂已放行，GELLO 可以操作 — '
        + '但<u>还没有在记录</u></b><br>'
        + '<span style="font-size:12px;color:#999">把机械臂摆到本条 episode 的'
        + '起始位姿，再按「② 开始记录」(或键盘 r 键)。这段摆位过程不会进数据集。'
        + '</span></td></tr>';
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
    renderCamCtl(!!j.cam_running, !!c.running);
    renderRobot(j, !!c.running);
    renderLayout(j);
  } catch (e) {
    document.getElementById('status').innerHTML = 'status error: ' + esc(e);
  }
}
// mousemove/up on the DOCUMENT so a drag that leaves the image still tracks
// and still releases — otherwise the marker sticks to the cursor.
VIEWS.forEach(function (v) {
  document.getElementById('stencil-' + v)
          .addEventListener('mousedown', onStencilDown);
});
document.addEventListener('mousemove', onStencilMove);
document.addEventListener('mouseup', onStencilUp);
loadLayoutList();
taskRefresh().then(() => renderTaskInfo(selectedTask()));
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

function dsRowsHtml(idxList, withRename) {
  let html = '<table><tr><th>名称</th><th>路径</th><th>episodes</th>'
    + '<th>frames</th><th>成功率</th><th>大小</th><th>修改时间</th><th></th></tr>';
  idxList.forEach(i => {
    const d = dsCache[i];
    const sr = d.success_rate == null ? '-'
      : (100 * d.success_rate).toFixed(0) + '% (' + d.n_success + '/'
        + d.total_episodes + ')';
    const mt = d.mtime ? new Date(d.mtime * 1000).toLocaleString() : '-';
    const svo = d.svo_path
      ? '<div class="path">SVO: ' + esc(d.svo_path) + '</div>' : '';
    html += '<tr><td>' + esc(d.name)
      + (d.task ? '<div class="path">task: ' + esc(d.task) + '</div>' : '')
      + (d.error ? '<div class="err">' + esc(d.error) + '</div>' : '')
      + '</td>'
      + '<td><div class="path">' + esc(d.path || '') + '</div>' + svo + '</td>'
      + '<td>' + (d.total_episodes ?? '-') + '</td>'
      + '<td>' + (d.total_frames ?? '-') + '</td>'
      + '<td>' + esc(sr) + '</td>'
      + '<td>' + fmtSize(d.size_bytes) + '</td>'
      + '<td>' + esc(mt) + '</td>'
      + '<td><button class="small" onclick="toggleEpisodes(' + i + ')">'
      + '<span id="dsx-' + i + '">▶</span> 展开</button>'
      + (withRename
          ? '<button class="small" onclick="renameDataset(' + i + ')">重命名</button>'
          : '')
      + '<button class="small danger" onclick="deleteDataset(' + i + ')">删除'
      + '</button></td></tr>';
    html += '<tr id="dsd-' + i + '" style="display:none"><td colspan="8">'
      + '<div id="dsdc-' + i + '" class="epbox">展开中…</div></td></tr>';
  });
  return html + '</table>';
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
  const cur = [], leg = [];
  dsCache.forEach((d, i) => (d.category === 'current' ? cur : leg).push(i));
  let html = '<h4 style="margin:6px 0">当前数据集 <span class="path">'
    + esc((j.roots || [])[0] || '') + '</span></h4>';
  html += cur.length ? dsRowsHtml(cur, true)
    : '<div class="path" style="margin-bottom:8px">还没有数据集 — 在上面"采集控制"'
      + '里新建一个并开始采集</div>';
  if (leg.length) {
    html += '<details style="margin-top:12px"><summary>Legacy 旧数据 (旧 scheme / '
      + '时间戳，' + leg.length + ' 个) <span class="path">'
      + esc((j.roots || [])[1] || '') + '</span></summary>'
      + dsRowsHtml(leg, false) + '</details>';
  }
  document.getElementById('dsTable').innerHTML = html;
  populateDatasetSelector();
}

async function renameDataset(i) {
  const d = dsCache[i];
  if (!d) return;
  const nn = prompt('重命名数据集 "' + d.name + '" 为:', d.name);
  if (nn === null) return;
  const newName = nn.trim();
  if (!/^[A-Za-z0-9._-]+$/.test(newName)) {
    alert('名称只能含 字母/数字/._-'); return;
  }
  const el = document.getElementById('dsMsg');
  try {
    const r = await fetch('/api/dataset/' + d.dsid + '/rename', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({new_name: newName}),
    });
    const jj = await r.json();
    el.textContent = jj.msg || JSON.stringify(jj);
  } catch (e) {
    el.textContent = 'rename error: ' + e;
  }
  refreshDatasets();
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

// Playback speed. The server paces the MJPEG stream, so changing speed means
// re-requesting it — which restarts the episode from frame 0.
let playSpeed = 1;
let playNow = null;   // {dsid, name, ep} — what is on screen, for re-requests

function setPlaySpeed(s) {
  playSpeed = s;
  const box = document.getElementById('playSpeeds');
  if (box) {
    box.querySelectorAll('button').forEach(function (b) {
      b.className = 'small' + (parseFloat(b.dataset.s) === s ? ' primary' : '');
    });
  }
  if (playNow) openPlayback(playNow.dsid, playNow.name, playNow.ep);
}

function openPlayback(dsid, name, ep) {
  playNow = {dsid: dsid, name: name, ep: ep};
  document.getElementById('playTitle').textContent =
    name + ' / episode ' + ep + '  —  ' + playSpeed + '× (播完即止)';
  document.getElementById('playImg').src =
    '/api/dataset/' + dsid + '/episode/' + ep + '/play.mjpg'
    + '?speed=' + playSpeed + '&ts=' + Date.now();
  document.getElementById('playOverlay').style.display = 'flex';
}

function playEpisode(i) {
  const d = dsCache[i];
  if (!d || d.error) return;
  const max = (d.total_episodes || 1) - 1;
  const n = prompt('Episode index (0..' + max + ')', '0');
  if (n === null) return;
  const ep = parseInt(n, 10);
  if (isNaN(ep) || ep < 0 || ep > max) { alert('无效 episode index'); return; }
  openPlayback(d.dsid, d.name, ep);
}

function closePlayback() {
  document.getElementById('playImg').removeAttribute('src');
  document.getElementById('playOverlay').style.display = 'none';
  playNow = null;
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

async function toggleEpisodes(i) {
  const d = dsCache[i];
  const row = document.getElementById('dsd-' + i);
  const arrow = document.getElementById('dsx-' + i);
  const box = document.getElementById('dsdc-' + i);
  if (!d || !row) return;
  if (row.style.display !== 'none') {
    row.style.display = 'none'; arrow.textContent = '▶'; return;
  }
  row.style.display = ''; arrow.textContent = '▼';
  box.textContent = '展开中…';
  try {
    const r = await fetch('/api/dataset/' + d.dsid + '/episodes');
    const j = await r.json();
    renderEpisodes(i, j.episodes || []);
  } catch (e) {
    box.textContent = 'episode 加载失败: ' + e;
  }
}

function renderEpisodes(i, eps) {
  const box = document.getElementById('dsdc-' + i);
  if (!eps.length) { box.textContent = '无 episode'; return; }
  let h = '<table><tr><th>episode</th><th>task</th><th>frames</th>'
    + '<th>成功</th><th></th></tr>';
  eps.forEach(e => {
    const su = e.success == null ? '-' : (e.success ? '✅' : '❌');
    h += '<tr><td>' + e.index + '</td>'
      + '<td>' + esc(e.task || '-') + '</td>'
      + '<td>' + (e.length ?? '-') + '</td>'
      + '<td>' + su + '</td>'
      + '<td><button class="small" onclick="playEpisodeIdx(' + i + ',' + e.index
      + ')">回放</button>'
      + '<button class="small danger" onclick="deleteEpisode(' + i + ',' + e.index
      + ')">删除</button></td></tr>';
  });
  box.innerHTML = h + '</table>';
}

function playEpisodeIdx(i, ep) {
  const d = dsCache[i];
  if (!d) return;
  openPlayback(d.dsid, d.name, ep);
}

async function deleteEpisode(i, ep) {
  const d = dsCache[i];
  if (!d) return;
  if (!confirm('删除 ' + d.name + ' 的 episode ' + ep
      + ' ？后面的 episode 会自动重新编号 (episode_index 保持连续)。')) return;
  const el = document.getElementById('dsMsg');
  try {
    const r = await fetch('/api/dataset/' + d.dsid + '/episode/' + ep
      + '?confirm=yes', {method: 'DELETE'});
    const j = await r.json();
    el.textContent = j.msg || JSON.stringify(j);
  } catch (e) {
    el.textContent = 'episode delete error: ' + e;
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
              live_cams: Optional["LiveCamSource"] = None,
              layout_gate: Optional["LayoutGate"] = None) -> "Flask":
    if not HAS_FLASK:
        raise RuntimeError("flask not installed on this host")
    if dataset_roots is None:
        dataset_roots = list(DATASET_ROOTS)
    if layout_gate is None:
        layout_gate = LayoutGate(LayoutStore(LAYOUT_DIR_HOST))
    task_store = TaskStore()
    # Sidecar HD demo recorder over the env's live_cam frames; one temp slot,
    # promoted by save-video or replaced by the next recording window.
    hd_rec = HDRolloutRecorder(live_cams, os.path.join(DEMO_DIR, ".tmp"))
    # Task the current run was started under (for the HD temp's metadata).
    run_task = {"id": ""}
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

    # -- idle camera preview (opt-in) -------------------------------------
    # The dashboard is launched --no-cam-on-start on purpose: leaving the ZEDs
    # free lets the next collection's in-container env open them COLD, the only
    # handoff proven reliable at HD720 (see _reclaim_cameras). But the layout
    # stage needs to SEE the bench, so the preview is available on demand —
    # off by default, explicitly turned on, and released again by
    # CollectionManager.start(), which then probes from inside the container
    # until the cameras are proven openable before launching the env.
    @app.post("/api/cams/start")
    def api_cams_start():
        if mgr.is_collecting():
            return jsonify({"ok": False,
                            "msg": "采集进行中 — 相机由 env 持有"}), 409
        if not HAS_PYZED:
            return jsonify({"ok": False, "msg": "pyzed 不可用"}), 503
        try:
            msg = cams.start()
        except Exception as e:
            return jsonify({"ok": False,
                            "msg": f"相机打开失败: {e!r}"[:300]}), 500
        return jsonify({"ok": cams.is_running(), "msg": f"相机预览: {msg}",
                        "missing_cams": cams.missing_cams})

    @app.post("/api/cams/stop")
    def api_cams_stop():
        if mgr.is_collecting():
            return jsonify({"ok": False,
                            "msg": "采集进行中 — 相机由 env 持有"}), 409
        try:
            cams.stop()
        except Exception as e:
            return jsonify({"ok": False,
                            "msg": f"相机关闭失败: {e!r}"[:300]}), 500
        return jsonify({"ok": True, "msg": "相机预览已关闭 — ZED 已释放"})

    @app.get("/api/status")
    def api_status():
        coll = mgr.status()
        # Piggyback robot state ONLY when idle (never poke zerorpc while the
        # env owns the robot), throttled via the 3s cache above.
        robot = None if coll["running"] else _robot_state_for_status()
        # The dataset dir only appears once the env has created it, so the
        # parked layout provenance lands on one of these polls.
        layout_gate.flush_pending()
        return jsonify({
            "collection": coll,
            "cam_running": cams.is_running(),
            "missing_cams": cams.missing_cams,
            "has_pyzed": HAS_PYZED,
            "robot": robot,
            "layout": layout_gate.status(),
        })

    # -- layout-prepare stage -------------------------------------------
    # A layout is an image-space stencil (grid + markers + reference
    # snapshot) the operator matches the physical scene against before
    # recording. Collection is gated on one being selected AND confirmed.
    def _layout_err(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except LayoutError as e:
                return jsonify({"ok": False, "msg": str(e)}), 400
        return wrapper

    @app.get("/api/layouts")
    @_layout_err
    def api_layouts():
        return jsonify({"ok": True, "layouts": layout_gate.store.list(),
                        "dir": layout_gate.store.root})

    @app.get("/api/layout/<layout_id>")
    @_layout_err
    def api_layout_get(layout_id):
        return jsonify({"ok": True, "layout": layout_gate.store.load(layout_id)})

    @app.get("/api/layout/<layout_id>/snapshot/<view>.jpg")
    @_layout_err
    def api_layout_snapshot(layout_id, view):
        buf = layout_gate.store.snapshot_bytes(layout_id, view)
        if buf is None:
            return Response("no snapshot", status=404)
        # Snapshots are overwritten in place on re-save; never let a browser
        # serve a stale ghost against a freshly re-photographed scene.
        return Response(buf, mimetype="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.post("/api/layout")
    @_layout_err
    def api_layout_save():
        body = request.get_json(silent=True) or {}
        layout_id = str(body.get("id", "")).strip()
        # `capture` lists which views to re-photograph now; views left out keep
        # whatever snapshot they already had, so re-nudging markers in one view
        # never drops the other view's ghost.
        capture = body.get("capture") or []
        if not isinstance(capture, list):
            return jsonify({"ok": False, "msg": "capture must be a list"}), 400
        snapshots = {}
        if capture:
            src = _cam_src()
            if src is cams and not cams.is_running():
                return jsonify({"ok": False,
                                "msg": f"{CAM_BUSY_MSG} — 无法抓取参考快照"}), 409
            for view in capture:
                if view not in LAYOUT_VIEWS:
                    return jsonify({"ok": False,
                                    "msg": f"unknown view '{view}'"}), 400
                buf = src.get_jpeg(view)
                if buf is None:
                    return jsonify({"ok": False,
                                    "msg": f"{view} 相机没有可用帧，快照抓取失败"}), 409
                # The live source hands back an 8x8 placeholder when a feed is
                # stale, and that would be stored as a grey "reference" the
                # operator is then asked to align the real bench against.
                if not is_usable_snapshot(buf):
                    return jsonify({
                        "ok": False,
                        "msg": f"{view} 当前是占位帧（相机未就绪或采集未推流）"
                               " — 无法抓取参考快照"}), 409
                snapshots[view] = buf
        layout = layout_gate.store.save(
            layout_id, body.get("views"),
            note=str(body.get("note", "")), snapshots=snapshots)
        # Keep the armed copy in sync; an edit drops any stale confirmation.
        layout_gate.refresh()
        shot = ("（已抓 " + "/".join(snapshots) + " 快照）") if snapshots else ""
        return jsonify({"ok": True, "layout": layout,
                        "msg": f"已保存 layout '{layout['id']}'" + shot})

    @app.route("/api/layout/<layout_id>", methods=["DELETE"])
    @_layout_err
    def api_layout_delete(layout_id):
        cur = layout_gate.current()
        msg = layout_gate.store.delete(layout_id)
        if cur and cur["id"] == layout_id:
            layout_gate.clear()
        return jsonify({"ok": True, "msg": msg})

    @app.post("/api/layout/select")
    @_layout_err
    def api_layout_select():
        body = request.get_json(silent=True) or {}
        layout = layout_gate.select(str(body.get("id", "")).strip())
        return jsonify({"ok": True, "layout": layout,
                        "msg": f"已加载 layout '{layout['id']}' — "
                               "摆好物体后点「确认就位」"})

    @app.post("/api/layout/confirm")
    @_layout_err
    def api_layout_confirm():
        # Confirming mid-run would silently re-arm the gate for a scene the
        # operator has not actually re-checked; the between-episode reposition
        # is driven by the start gate (开始下一条), not by this.
        if mgr.is_collecting():
            return jsonify({"ok": False,
                            "msg": "采集进行中 — 无法重新确认 layout"}), 409
        layout_id = layout_gate.confirm()
        # Confirming under a selected task = "this run of the task uses this
        # layout" — record it (a task accumulates many layouts over time).
        body = request.get_json(silent=True) or {}
        task_id = str(body.get("task_id", "")).strip()
        extra = ""
        if task_id and task_store.add_layout(task_id, layout_id) == "updated":
            extra = f"(已挂到任务 {task_id})"
        return jsonify({"ok": True,
                        "msg": f"layout '{layout_id}' 已确认就位 — "
                               f"Start 已解锁{extra}"})

    @app.post("/api/layout/clear")
    @_layout_err
    def api_layout_clear():
        layout_gate.clear()
        return jsonify({"ok": True, "msg": "已取消 layout 选择"})

    # -- tasks (shared registry with the eval dashboard) ------------------
    @app.get("/api/tasks")
    def api_tasks():
        return jsonify({"tasks": task_store.list()})

    @app.post("/api/task/create")
    def api_task_create():
        body = request.get_json(silent=True) or {}
        msg = task_store.create(body)
        return jsonify({"ok": msg == "created", "msg": msg})

    @app.post("/api/task/delete")
    def api_task_delete():
        body = request.get_json(silent=True) or {}
        msg = task_store.delete(str(body.get("id", "")).strip())
        return jsonify({"ok": msg == "deleted", "msg": msg})

    @app.post("/api/task/update")
    def api_task_update():
        body = request.get_json(silent=True) or {}
        msg = task_store.update(str(body.get("id", "")).strip(), body)
        return jsonify({"ok": msg == "updated", "msg": msg})

    # -- demo exports → RLinf/saved_demo/<task>/ --------------------------
    @app.post("/api/save_video")
    def api_save_video():
        """Save the last episode's demo video.

        Preferred source: the sidecar HD temp (1280x720 per view, sampled
        from the env's live_cam frames during the recording window). Falls
        back to re-encoding the newest dataset episode (224x224 native) when
        no HD temp exists — e.g. the dashboard restarted mid-session.
        """
        if not (HAS_PYARROW and HAS_CV2):
            return jsonify({"ok": False,
                            "msg": "pyarrow/cv2 not installed on this host"}), 503
        body = request.get_json(silent=True) or {}
        dsid = str(body.get("dataset", "")).strip()
        task_id = str(body.get("task_id", "")).strip()
        hd = hd_rec.last()
        if hd is not None:
            folder = task_id or hd["task"] or (hd["dataset"] or dsid or "untagged")
            dest = os.path.join(
                DEMO_DIR, folder,
                f"{(hd['dataset'] or dsid or 'episode')}_{hd['started']}_hd.mp4")
            out = hd_rec.promote(dest)
            if out is not None:
                return jsonify({"ok": True, "path": out,
                                "msg": (f"saved HD demo({hd['frames']} 帧, "
                                        f"720p 双视角) → {out}")})
        # -- fallback: dataset-native re-encode ---------------------------
        # 采集控制面板传的是裸数据集名(不是数据集管理器的编码 dsid)——按
        # 采集写入根目录解析,与 /api/collect/start 的语义一致。
        if not _valid_dataset_name(dsid):
            return jsonify({"ok": False, "msg": f"invalid dataset name '{dsid}'"}), 400
        ds_dir = os.path.join(dataset_roots[0], dsid)
        if not os.path.isfile(os.path.join(ds_dir, "meta", "info.json")):
            return jsonify({"ok": False,
                            "msg": f"数据集 '{dsid}' 不存在或还没有元数据"
                                   "(第一条 episode 保存后才有)"}), 404
        try:
            with open(os.path.join(ds_dir, "meta", "info.json")) as f:
                info = json.load(f)
        except Exception as e:
            return jsonify({"ok": False,
                            "msg": f"info.json unreadable: {e!r}"[:300]}), 500
        total = int(info.get("total_episodes") or 0)
        if total < 1:
            return jsonify({"ok": False, "msg": "数据集还没有已保存的 episode"}), 404
        n = total - 1
        epath = _episode_parquet_path(ds_dir, info, n)
        if not os.path.isfile(epath):
            return jsonify({"ok": False, "msg": f"episode {n} 文件缺失"}), 404
        image_keys = _image_keys(info)
        if not image_keys:
            return jsonify({"ok": False, "msg": "dataset has no image features"}), 404
        fps = float(info.get("fps") or 15)
        out_dir = os.path.join(DEMO_DIR, task_id or dsid.replace("/", "_"))
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir,
                           f"{dsid.replace('/', '_')}_ep{n:03d}.mp4")
        import cv2
        import pyarrow.parquet as pq
        writer = None
        frames = 0
        pf = pq.ParquetFile(epath)
        try:
            for batch in pf.iter_batches(batch_size=32, columns=image_keys):
                for row in batch.to_pylist():
                    tiles = [img for k in image_keys
                             if (img := _decode_image_cell(row.get(k))) is not None]
                    if not tiles:
                        continue
                    frame = _hstack_frames(tiles)
                    if writer is None:
                        # H.264 High + faststart — cv2's default mp4v is not
                        # browser-playable (see tasl/tools/h264_writer).
                        writer = H264Writer(
                            out, fps, (frame.shape[1], frame.shape[0]))
                        if not writer.isOpened():
                            return jsonify({"ok": False,
                                            "msg": "VideoWriter open failed"}), 500
                    writer.write(frame)
                    frames += 1
        finally:
            pf.close()
            if writer is not None:
                writer.release()
        if not frames:
            return jsonify({"ok": False, "msg": f"episode {n} 没有可解码的帧"}), 404
        return jsonify({"ok": True, "path": out,
                        "msg": (f"saved episode {n}({frames} 帧, 数据集原生分辨率)"
                                f" → {out}")})

    @app.post("/api/save_layout")
    def api_save_layout():
        """Export a layout (json + reference JPEGs) as a demo — usable anytime."""
        body = request.get_json(silent=True) or {}
        lay_id = str(body.get("layout", "")).strip()
        task_id = str(body.get("task_id", "")).strip() or "untagged"
        if not lay_id:
            return jsonify({"ok": False, "msg": "还没有选择 layout"}), 400
        try:
            lay = layout_gate.store.load(lay_id)
        except LayoutError as e:
            return jsonify({"ok": False, "msg": str(e)}), 404
        out_dir = os.path.join(DEMO_DIR, task_id)
        os.makedirs(out_dir, exist_ok=True)
        saved = []
        p = os.path.join(out_dir, f"layout_{lay_id}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(lay, f, ensure_ascii=False, indent=2)
        saved.append(os.path.basename(p))
        for view in LAYOUT_VIEWS:
            buf = layout_gate.store.snapshot_bytes(lay_id, view)
            if buf:
                q = os.path.join(out_dir, f"layout_{lay_id}.{view}.jpg")
                with open(q, "wb") as f:
                    f.write(buf)
                saved.append(os.path.basename(q))
        return jsonify({"ok": True, "path": out_dir,
                        "msg": f"saved → {out_dir} ({', '.join(saved)})"})

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
        dataset_name = str(body.get("dataset_name", "")).strip() or None
        if dataset_name is not None and not _valid_dataset_name(dataset_name):
            return jsonify({"ok": False,
                            "msg": "invalid dataset name (allowed: letters, "
                                   "digits, . _ -)"}), 400
        # layout-prepare gate: no recording without a confirmed arrangement,
        # so every episode on disk is traceable to a known scene layout.
        gate_ok, gate_msg = layout_gate.check()
        if not gate_ok:
            return jsonify({"ok": False, "msg": gate_msg}), 409
        msg = mgr.start(num_episodes, task, dataset_name)
        code = 409 if msg.startswith("refused") else 200
        if code == 200 and dataset_name:
            layout_gate.arm_dataset_write(
                os.path.join(dataset_roots[0], dataset_name),
                task, num_episodes)
            # Task provenance: the run's dataset AND layout join the task's
            # record so both portals list what was collected for it.
            task_id = str(body.get("task_id", "")).strip()
            run_task["id"] = task_id
            if task_id:
                task_store.add_dataset(task_id, dataset_name)
                task_store.add_layout(task_id, gate_msg)  # check() ok → id
        return jsonify({"ok": code == 200, "msg": msg}), code

    @app.post("/api/collect/stop")
    def api_collect_stop():
        hd_rec.stop()
        return jsonify({"ok": True, "msg": mgr.stop()})

    # -- session teardown -------------------------------------------------
    # Two levels, mirroring what the launchers do:
    #   reap  = teleop-stop.sh stage 2 — free the ZEDs + GELLO, keep FCI.
    #   stop  = the whole script       — + NUC controller down, FCI released.
    # Both SHELL OUT so the portal and the CLI can never disagree about what
    # "stopped" means.
    def _run_launcher(argv: list[str], timeout: float) -> tuple[bool, str, int]:
        r = subprocess.run(argv, capture_output=True, timeout=timeout)
        out = (r.stdout + b"\n" + r.stderr).decode("utf-8", "replace").strip()
        return r.returncode == 0, out, r.returncode

    def _require_root() -> Optional[tuple]:
        if os.geteuid() != 0:
            return jsonify({
                "ok": False,
                "msg": ("dashboard is not running as root — teardown needs it "
                        "(relaunch via teleop.sh, or run "
                        "`sudo tasl/launch/teleop-stop.sh` yourself)"),
            }), 503
        return None

    @app.post("/api/session/reap")
    def api_session_reap():
        """Free the ZEDs + GELLO by reaping the in-container run. Leaves the
        NUC controller and FCI alone, so the next episode can start without
        re-activating FCI in Desk — this is the fix for an orphaned run."""
        denied = _require_root()
        if denied:
            return denied
        if not os.path.isfile(LAUNCH_LIB_SH):
            return jsonify({"ok": False,
                            "msg": f"lib.sh not found at {LAUNCH_LIB_SH}"}), 500
        try:
            # Call the shell function itself — one definition of the kill set.
            ok_rc, out, rc = _run_launcher(
                ["bash", "-c",
                 f'source {shlex.quote(LAUNCH_LIB_SH)} && reap_collection_procs'],
                timeout=90.0)
        except subprocess.TimeoutExpired:
            return jsonify({"ok": False, "msg": "reap timed out after 90s"}), 504
        mgr.after_external_teardown()
        return jsonify({"ok": ok_rc, "returncode": rc, "output": out[-6000:],
                        "msg": ("已清理容器内采集进程，相机 + GELLO 已释放"
                                "（FCI 保持不变）" if ok_rc
                                else f"reap 返回 {rc} — 见输出")})

    @app.post("/api/session/stop")
    def api_session_stop():
        """Full staged teardown = teleop-stop.sh --keep-dashboard.

        --keep-dashboard because stage 1 SIGTERMs collect.py, which is the
        process serving this very request — without it the operator loses the
        UI halfway through their own teardown.
        """
        denied = _require_root()
        if denied:
            return denied
        if not os.path.isfile(TELEOP_STOP_SH):
            return jsonify({"ok": False,
                            "msg": f"not found: {TELEOP_STOP_SH}"}), 500
        try:
            ok_rc, out, rc = _run_launcher(
                ["bash", TELEOP_STOP_SH, "--keep-dashboard"], timeout=180.0)
        except subprocess.TimeoutExpired:
            return jsonify({
                "ok": False,
                "msg": ("teleop-stop.sh timed out after 180s — the NUC ssh step "
                        "may be blocked; run it in a terminal to see"),
            }), 504
        mgr.after_external_teardown()
        # The script WARNs (does not fail) when :4242 stays open, so grep the
        # transcript: a teardown that silently left FCI held is the one outcome
        # the operator must not mistake for success.
        fci_held = "STILL OPEN" in out
        msg = "会话已结束 — FCI 已释放，相机/GELLO 已free，dashboard 保持运行"
        if fci_held:
            msg = ("⚠ 容器内进程已清理，但 NUC 控制器 :4242 仍然开着 — "
                   "FCI 没有释放。见输出，可能需要手动到 NUC 上 compose down")
        elif not ok_rc:
            msg = f"teleop-stop.sh 返回 {rc} — 见输出"
        return jsonify({"ok": ok_rc and not fci_held, "returncode": rc,
                        "fci_released": not fci_held,
                        "output": out[-8000:], "msg": msg})

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
        hd_rec.stop()   # episode over — finalize the HD demo temp
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

    @app.post("/api/start_recording")
    def api_start_recording():
        # Inject 'r' — second step of the two-stage gate. The robot is already
        # released and following the leader; this is the point where frames
        # start being kept, so everything before it (getting into position) is
        # discarded rather than taught to the policy.
        ok, msg = check_gate(mgr.is_collecting(), require_collecting=True)
        if not ok:
            return jsonify({"ok": False,
                            "msg": f"{msg} - 开始记录 only valid mid-run"}), 409
        if vkbd is None:
            return jsonify({
                "ok": False,
                "msg": ("uinput virtual keyboard unavailable "
                        "(evdev missing or dashboard not launched with sudo)"),
            }), 503
        try:
            vkbd.inject_record()
        except Exception as e:
            return jsonify({"ok": False, "msg": f"inject failed: {e!r}"[:300]}), 500
        # Recording window opens: start the sidecar HD demo recorder (an
        # unsaved temp from the previous take is replaced).
        hd_rec.start(run_task["id"],
                     (mgr.last_launch or {}).get("dataset_name") or "")
        visible = vkbd_visible_in_container(vkbd.path)
        caveat = None if visible else (
            "virtual keyboard not visible inside rlinf-eval (container /dev is "
            "a startup snapshot) — docker restart rlinf-eval, or press physical 'r'"
        )
        return jsonify({"ok": True, "msg": "KEY_R injected — 开始记录",
                        "visible_in_container": visible, "caveat": caveat})

    @app.post("/api/discard_episode")
    def api_discard_episode():
        # Inject 'a' — end this episode with reward -1 so only_success drops
        # it. This is the safe way to get rid of a bad take mid-run: nothing is
        # written, so nothing has to be deleted out from under LeRobot.
        ok, msg = check_gate(mgr.is_collecting(), require_collecting=True)
        if not ok:
            return jsonify({"ok": False,
                            "msg": f"{msg} - 丢弃只在采集中有效"}), 409
        if vkbd is None:
            return jsonify({
                "ok": False,
                "msg": ("uinput virtual keyboard unavailable "
                        "(evdev missing or dashboard not launched with sudo)"),
            }), 503
        try:
            vkbd.inject_discard()
        except Exception as e:
            return jsonify({"ok": False, "msg": f"inject failed: {e!r}"[:300]}), 500
        hd_rec.stop()   # episode over (discarded) — HD temp still savable
        visible = vkbd_visible_in_container(vkbd.path)
        caveat = None if visible else (
            "virtual keyboard not visible inside rlinf-eval (container /dev is "
            "a startup snapshot) — docker restart rlinf-eval, or press physical 'a'"
        )
        return jsonify({"ok": True, "msg": "KEY_A injected — 本条已丢弃，不会保存",
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
        # 90s timeout: a COLD launch_controller (container just up + FCI just
        # activated) can exceed the old 30s and finish server-side AFTER the
        # client gives up. So on a bootstrap RPC error we re-check the state and
        # treat a live controller as success — no false 500.
        client = DroidClient(timeout=90)
        try:
            try:
                client.kill_controller()
            except Exception as e:
                # Controller may already be dead — that's what we're recovering.
                _log.warning(f"kill_controller during recover: {e}")
            try:
                client.bootstrap()
            except Exception as e:
                _log.warning(f"bootstrap RPC error during recover: {e!r}; re-checking state")
                rc = DroidClient(timeout=8)
                try:
                    rc.get_robot_state()  # raises if the controller really isn't up
                except Exception:
                    raise e
                finally:
                    rc.close()
                return jsonify({"ok": True, "msg": "recover done (controller live; bootstrap RPC was slow)"})
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
        # Playback speed. The server paces the MJPEG stream, so speed is a
        # divisor on the inter-frame sleep; at high multiples the sleep hits 0
        # and the real ceiling becomes decode+encode throughput, which is why
        # the effective rate is reported back in a header rather than promised.
        try:
            speed = float(request.args.get("speed", 1.0))
        except (TypeError, ValueError):
            speed = 1.0
        speed = min(max(speed, 0.1), 16.0)
        fps = float(info.get("fps") or PLAYBACK_MAX_FPS)
        period = 1.0 / min(max(fps, 0.1), PLAYBACK_MAX_FPS) / speed

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
                        mimetype="multipart/x-mixed-replace; boundary=frame",
                        headers={"X-Playback-Speed": f"{speed:g}"})

    @app.get("/api/dataset/<dsid>/episodes")
    def api_dataset_episodes(dsid):
        ds_dir = _resolve_dataset(dsid, dataset_roots)
        if ds_dir is None:
            return jsonify({"ok": False, "msg": "dataset not found"}), 404
        return jsonify({"ok": True, "episodes": list_episodes_meta(ds_dir)})

    def _active_dataset_dir() -> Optional[str]:
        """Directory the running collection writes to, or None when idle.

        Returns the ROOT itself when the target is unknown, so an unrecognised
        state refuses every delete rather than guessing.
        """
        if not mgr.is_collecting():
            return None
        name = (mgr.last_launch or {}).get("dataset_name")
        if not name:
            return dataset_roots[0]
        return os.path.abspath(os.path.join(dataset_roots[0], name))

    def _refuse_if_active(ds_dir: str):
        """Deleting renumbers every later episode and rebuilds the meta files.
        Doing that to the dataset LeRobot currently has open — it holds
        total_episodes and the episode list in memory — corrupts it. Datasets
        the run is not touching are perfectly safe to edit mid-collection.
        """
        active = _active_dataset_dir()
        if active is None:
            return None
        if os.path.abspath(ds_dir) == active or active == dataset_roots[0]:
            return jsonify({
                "ok": False,
                "msg": ("refused: 这是当前采集正在写入的数据集 — 删除会重排 "
                        "episode 编号并重建元数据，和 LeRobot 的内存状态冲突。"
                        "坏的一条请用「✗ 丢弃本条」当场丢掉；已保存的等本轮结束再删。"),
            }), 409
        return None

    @app.route("/api/dataset/<dsid>/episode/<int:n>", methods=["DELETE"])
    def api_dataset_episode_delete(dsid, n):
        if request.args.get("confirm") != "yes":
            return jsonify({"ok": False,
                            "msg": "refused: requires ?confirm=yes"}), 400
        ds_dir = _resolve_dataset(dsid, dataset_roots)
        if ds_dir is None:
            return jsonify({"ok": False, "msg": "dataset not found"}), 404
        denied = _refuse_if_active(ds_dir)
        if denied:
            return denied
        msg = delete_episode(ds_dir, int(n))
        if msg.startswith("deleted"):
            _log.info(f"episode delete: {ds_dir} ep={n} -> {msg}")
            return jsonify({"ok": True, "msg": msg}), 200
        code = 409 if msg.startswith("refused") else 500
        return jsonify({"ok": False, "msg": msg}), code

    @app.route("/api/dataset/<dsid>", methods=["DELETE"])
    def api_dataset_delete(dsid):
        if request.args.get("confirm") != "yes":
            return jsonify({"ok": False,
                            "msg": "refused: requires ?confirm=yes"}), 400
        ds_dir = _resolve_dataset(dsid, dataset_roots)
        if ds_dir is None:
            return jsonify({"ok": False, "msg": "dataset not found"}), 404
        denied = _refuse_if_active(ds_dir)
        if denied:
            return denied
        # Belt-and-suspenders: _resolve_dataset already enforced this.
        if not any(is_inside_root(ds_dir, r) for r in dataset_roots):
            return jsonify({"ok": False,
                            "msg": "refused: path escapes dataset roots"}), 403
        try:
            shutil.rmtree(ds_dir)
        except Exception as e:
            return jsonify({"ok": False,
                            "msg": f"delete failed: {e!r}"[:300]}), 500
        # Also remove the HD SVO archive sibling (current scheme), guarded the
        # same way so we never rmtree outside a dataset root.
        svo_sib = ds_dir.rstrip("/") + "_svo"
        svo_msg = ""
        if (os.path.isdir(svo_sib)
                and any(is_inside_root(svo_sib, r) for r in dataset_roots)):
            try:
                shutil.rmtree(svo_sib)
                svo_msg = " + svo"
            except Exception as e:
                svo_msg = f" (svo delete failed: {e!r})"[:80]
        _log.info(f"dataset deleted: {ds_dir}{svo_msg}")
        return jsonify({"ok": True,
                        "msg": f"deleted {dsid_decode(dsid)[1]}{svo_msg}"})

    @app.post("/api/dataset/<dsid>/rename")
    def api_dataset_rename(dsid):
        if mgr.is_collecting():
            return jsonify({"ok": False,
                            "msg": "refused: collection running"}), 409
        body = request.get_json(silent=True) or {}
        new_name = str(body.get("new_name", "")).strip()
        if not _valid_dataset_name(new_name):
            return jsonify({"ok": False,
                            "msg": "invalid new name (allowed: letters, "
                                   "digits, . _ -)"}), 400
        ds_dir = _resolve_dataset(dsid, dataset_roots)
        if ds_dir is None:
            return jsonify({"ok": False, "msg": "dataset not found"}), 404
        parent = os.path.dirname(ds_dir)
        dst = os.path.join(parent, new_name)
        # dst must land strictly inside a configured root, and not already exist.
        if not any(is_inside_root(dst, r) for r in dataset_roots):
            return jsonify({"ok": False,
                            "msg": "refused: target escapes dataset roots"}), 403
        if os.path.exists(dst):
            return jsonify({"ok": False,
                            "msg": f"refused: {new_name} already exists"}), 409
        try:
            os.rename(ds_dir, dst)
        except Exception as e:
            return jsonify({"ok": False,
                            "msg": f"rename failed: {e!r}"[:300]}), 500
        # Rename the HD SVO sibling too, best-effort (keeps name↔svo in sync).
        svo_src = ds_dir.rstrip("/") + "_svo"
        svo_msg = ""
        if os.path.isdir(svo_src):
            svo_dst = dst.rstrip("/") + "_svo"
            if (any(is_inside_root(svo_dst, r) for r in dataset_roots)
                    and not os.path.exists(svo_dst)):
                try:
                    os.rename(svo_src, svo_dst)
                    svo_msg = " + svo"
                except Exception as e:
                    svo_msg = f" (svo rename failed: {e!r})"[:80]
        _log.info(f"dataset renamed: {ds_dir} -> {dst}{svo_msg}")
        return jsonify({"ok": True, "msg": f"renamed -> {new_name}{svo_msg}"})

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
    p.add_argument("--layout-dir", default=LAYOUT_DIR_HOST,
                   help="Directory holding layout stencils (<id>.json/.jpg).")
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
    try:
        os.makedirs(args.layout_dir, exist_ok=True)
    except OSError as e:
        _log.warning(f"layout dir {args.layout_dir} not creatable: {e}")
    layout_gate = LayoutGate(LayoutStore(args.layout_dir))
    _log.info(f"layout dir: {args.layout_dir} "
              f"({len(layout_gate.store.list())} layouts)")
    app = build_app(cams, mgr, vkbd, dataset_roots=dataset_roots,
                    live_cams=live_cams, layout_gate=layout_gate)
    _log.info(f"serving on {args.bind}:{args.port}")
    app.run(host=args.bind, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
