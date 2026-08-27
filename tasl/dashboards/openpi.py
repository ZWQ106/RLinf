"""TASL FR3 openpi-standalone dashboard.

Sister of `_dashboard_rlinf.py` — same UI, but the inference loop runs
**in-process** against an openpi `serve_policy` WebSocket server instead
of going through the RLinf container.

Stack:
  serve_policy.py (Desktop, port 8000)  ◄── WS ──── this dashboard
                                                     │ ZED grab + JPEG
                                                     │ HTTP /state ──► NUC1 robot_server
                                                     │ infer → /move/joint_velocity_chunk
                                                     └ /robotiq/{open,close,move}

Lifecycle:
  • IDLE: dashboard owns the ZED cameras (always), streams MJPEG.
  • START: a background thread connects to ws://localhost:8000, loops:
      cam frames + robot /state  → policy.infer(obs)  → chunk POST to
      robot_server → next iter. Loop exits on stop flag OR iter cap.
  • STOP: sets the flag + POSTs /move/joint_velocity_stop + /stop so
    the in-flight ruckig chunk is preempted within ~50 ms.

Launch (host, system python with pyzed + openpi_client on PYTHONPATH):
    PYTHONPATH=/home/franka_desktop/work/openpi/packages/openpi-client/src \
    PYTHONPATH=tasl /usr/bin/python3 tasl/dashboards/openpi.py --port 8003 \\
        --policy-url ws://localhost:8000

Open: http://<desktop-tailscale-or-lan-ip>:8003
"""
from __future__ import annotations

# zerorpc uses gevent. _loop and Flask's worker threads call zerorpc methods,
# but gevent's hub is thread-local — calling from a thread that didn't create
# the hub raises "This operation would block forever  Hub: Handles: []".
# Patching here makes all threads cooperative so the hub works cross-thread.
# MUST run before any other import (esp. socket / threading / ssl).
import gevent.monkey
gevent.monkey.patch_all()

import argparse
import collections
import io
import json
import logging
import os
import pathlib
import re
import shutil
import signal
import socket
import subprocess
import threading
import time
from typing import Optional

import cv2
import numpy as np
import requests
import yaml
from flask import Flask, Response, jsonify, render_template_string, request, send_file
from PIL import Image

from clients.droid_client import ControllerNotResponding, DroidLikeClient
from rtc import dashboard_hook as _rtc_hook  # real-time chunking (tasl/rtc/), opt-in

# Image preprocessing modes — MUST match how the checkpoint's data was built:
#   "pad"  : aspect-preserving resize + zero-pad to 224² (openpi_client
#            resize_with_pad; DROID / tasl-fr3-10task-250ep / *-v2 ckpts).
#   "crop" : center square crop (1280x720 -> 720x720 @ x=280) then resize to
#            224² — the "pbc" datasets (tasl/tools/make_centercrop_dataset.py
#            crop224 / franka_env._crop_frame crop mode).
IMAGE_MODES = ("pad", "crop")


def policy_resize(arr: np.ndarray, size: int = 224, mode: str = "pad") -> np.ndarray:
    """Turn a raw HD720 RGB frame into the exact 224² image the policy sees."""
    h, w = arr.shape[:2]
    if mode == "crop":
        c = min(h, w)
        y, x = (h - c) // 2, (w - c) // 2
        return cv2.resize(arr[y:y + c, x:x + c], (size, size))  # INTER_LINEAR, as crop224
    scale = min(size / h, size / w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size, arr.shape[2]), dtype=arr.dtype)
    y_off = (size - new_h) // 2
    x_off = (size - new_w) // 2
    out[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return out

# Task-mask store — same LayoutStore the collect dashboard uses for its
# object-placement stencils, but rooted next to the VLA-PatchLen captures so
# eval masks never mix with collection layouts. Here a "mask" is just a
# reference snapshot (no markers): take one per task, then align the bench
# against its semi-transparent ghost before every rollout of a pair.
try:
    from dashboards.layout_store import LayoutError, LayoutStore
    from dashboards.task_store import TaskStore
except ImportError:  # started as `python dashboards/openpi.py`
    from layout_store import LayoutError, LayoutStore  # type: ignore[no-redef]
    from task_store import TaskStore  # type: ignore[no-redef]

# The dashboard runs under sudo (HOME=/root), so `~` expansion is WRONG here —
# derive every default from this file's real location instead.
_TASL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATCH_DIR = os.path.join(_TASL_DIR, "VLA-PatchLen-cp")
_HOME_DIR = os.path.dirname(os.path.dirname(_TASL_DIR))  # /home/franka_desktop

MASK_DIR = os.environ.get("OPENPI_MASK_DIR",
                          os.path.join(_PATCH_DIR, "masks"))

# Task layout masks — SHARED with the collection dashboard (collect.py). A
# task's placement stencil lives in the same directory the collect portal's
# LayoutStore uses, so a task defined on either portal ghosts identically on
# both. The member-area masks above stay private to this portal.
TASK_LAYOUT_DIR = os.environ.get(
    "RLINF_LAYOUT_DIR", os.path.join(_HOME_DIR, "rlinf_data", "layouts"))

# Demo exports (save-video / save-layout buttons) — shared with collect.py.
DEMO_DIR = os.environ.get(
    "RLINF_DEMO_DIR", os.path.join(os.path.dirname(_TASL_DIR), "saved_demo"))

# Where the hooked capture server drops its per-episode dirs — /mark writes
# the operator's success/fail verdict into the newest episode's episode.json
# (the pairing/aggregation scripts filter on it).
ROLLOUTS_ROOT = os.environ.get("PI05_ROLLOUTS_ROOT",
                               os.path.join(_PATCH_DIR, "rollouts"))

# setting1-Rr contract with serve_policy_patched.py: the serve runs with
# --watch-reload on this YAML; the portal rewrites the marked fields
# (inject_layer / target / capture_source_dir) BETWEEN rollouts and the server
# hot-reloads before the next rollout's first inference (new episode dir, step
# counter reset). Applied steering is stamped by the SERVER into each
# episode.json ("steering" field) — the portal keeps no copy of its own.
STEER_CFG_PATH = os.environ.get(
    "OPENPI_STEER_CFG",
    os.path.join(_PATCH_DIR, "configs", "inject_rollout_setting1.yaml"))

# Live blank-filler generation shells out to the openpi venv (the tokenizer
# and its deps live there, not in the dashboard's system python).
OPENPI_VENV_PY = os.environ.get(
    "OPENPI_VENV_PY", os.path.join(_HOME_DIR, "work/openpi/.venv/bin/python"))
FILLER_SCRIPT = os.path.join(_PATCH_DIR, "make_blank_fillers.py")
STEER_PI05_DIR_DEFAULT = os.path.join(_HOME_DIR,
                                      "steer-vla/steer-vla/steer_pi05")

# Experiment modes for the patching runs (recorded verbatim into each capture
# episode's episode.json via `_episode_tag`; the hooked server pops the key,
# the vanilla server's DroidInputs drops it — safe on both):
#   trial        bench exploration while choosing tasks/prompts — excluded
#                from pairing/analysis
#   setting1-Ra  clean-vs-blank pair rollouts, activation capture only
#                (R_action computed off-policy afterwards, no runtime patching)
#   setting2-Ra  R_action with runtime patching
#   setting2-Rr  R_rollout — per-layer patched rollouts scored by success
EXP_MODES = ("trial", "setting1-Ra", "setting1-Rr", "setting2-Ra",
             "setting2-Rr", "none")


# ── Hardware constants (verified 2026-05-27) ──────────────────────────
SN_ZED_2I_RIGHT = 36443134
SN_ZED_MINI_WRIST = 17150101

# DROID reset pose — what pi05_droid saw between every training episode
# (`droid/robot_env.py: self.reset_joints = [0, -pi/5, 0, -4pi/5, 0, 3pi/5, 0]`).
# Using this as deploy reset puts the robot in the model's prior. If
# the user clicks "Set current as home" the value persisted in
# _dashboard_home.json takes over.
import math as _math
HOME_Q_DEFAULT = [
    0.0, -_math.pi / 5, 0.0, -4 * _math.pi / 5,
    0.0, 3 * _math.pi / 5, 0.0,
]
HOME_STORE_PATH = pathlib.Path("/home/franka_desktop/_dashboard_home.json")


class HomeStore:
    """Persistent 7-DOF home pose, file-backed."""

    def __init__(self, default_q):
        self.q = list(default_q)
        self._lock = threading.Lock()
        try:
            data = json.loads(HOME_STORE_PATH.read_text(encoding="utf-8"))
            jq = data.get("joint_q")
            if isinstance(jq, list) and len(jq) == 7:
                self.q = [float(x) for x in jq]
                _log.info(f"home_store loaded from {HOME_STORE_PATH}: {self.q}")
        except FileNotFoundError:
            _log.info("home_store: no saved file, using compiled default")
        except Exception as e:
            _log.warning(f"home_store load failed ({e}); using default")

    def set(self, q):
        with self._lock:
            self.q = [float(x) for x in q]
            try:
                HOME_STORE_PATH.write_text(
                    json.dumps({"joint_q": self.q,
                                "saved_at": time.time()}, indent=2), encoding="utf-8")
                _log.info(f"home_store saved: {self.q}")
            except Exception as e:
                _log.error(f"home_store save failed: {e}")
        return self.q

    def get(self):
        with self._lock:
            return list(self.q)

_log = logging.getLogger("dashboard")


# ─────────────────────────────────────────────────────────────────────
# Camera manager (acquire/release ZEDs on demand)
# ─────────────────────────────────────────────────────────────────────
class CamManager:
    """Owns ZED cameras between eval runs. Releases USB on stop()."""

    def __init__(self, serials: dict[str, int], resolution: str = "HD720",
                 jpeg_quality: int = 70):
        self.serials = serials
        self.resolution = resolution
        self.jpeg_quality = jpeg_quality
        self._cams: dict[str, object] = {}
        self._threads: list[threading.Thread] = []
        self._frame_lock = threading.Lock()
        self._latest_jpeg: dict[str, bytes] = {}
        # Raw BGR copy of the newest frame, camera-native quality. The
        # preview JPEG above is q70 for MJPEG bandwidth; the recorder and
        # mask snapshots read THIS so their quality is not capped by the
        # preview encode.
        self._latest_bgr: dict[str, np.ndarray] = {}
        self._running = False
        self.missing_cams: list[tuple[str, int, str]] = []

    def start(self) -> str:
        if self._running:
            return "already running"
        try:
            import pyzed.sl as sl
        except ImportError as e:
            return f"pyzed import failed: {e}"
        res_map = {
            "HD2K": sl.RESOLUTION.HD2K,
            "HD1080": sl.RESOLUTION.HD1080,
            "HD720": sl.RESOLUTION.HD720,
        }
        # Partial-success policy: any cam that fails to open is logged + skipped
        # but the rest still come up. Caller can check missing_cams to know
        # which views won't appear in the UI.
        self.missing_cams: list[tuple[str, int, str]] = []
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
        # Only flip _running on AFTER at least one cam is open; the grab loops
        # check it as their exit condition.
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
        import pyzed.sl as sl
        rt = sl.RuntimeParameters()
        mat = sl.Mat()
        while self._running:
            if cam.grab(rt) != sl.ERROR_CODE.SUCCESS:
                time.sleep(0.01)
                continue
            cam.retrieve_image(mat, sl.VIEW.LEFT)
            bgra = mat.get_data()
            # sl.Mat's buffer is reused by the next grab — copy out.
            bgr = bgra[:, :, :3].copy()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=self.jpeg_quality)
            with self._frame_lock:
                self._latest_jpeg[name] = buf.getvalue()
                self._latest_bgr[name] = bgr
            time.sleep(0.02)

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()
        for name, cam in self._cams.items():
            try:
                cam.close()
            except Exception:
                pass
        self._cams.clear()
        with self._frame_lock:
            self._latest_jpeg.clear()
            self._latest_bgr.clear()
        _log.info("CamManager stopped, USB released")

    def is_running(self) -> bool:
        return self._running

    def get_jpeg(self, name: str) -> Optional[bytes]:
        with self._frame_lock:
            return self._latest_jpeg.get(name)

    def get_bgr(self, name: str) -> Optional[np.ndarray]:
        """Newest raw BGR frame (camera-native, no JPEG loss). The stored
        array is replaced (never mutated) each grab, so handing out the
        reference is safe for read-only consumers."""
        with self._frame_lock:
            return self._latest_bgr.get(name)

    def get_policy_view_jpeg(self, name: str, size: int = 224,
                             mode: str = "pad") -> Optional[bytes]:
        """Return the exact frame the policy sees (see policy_resize):
        mode="pad" (DROID resize_with_pad) or "crop" (pbc center-crop)."""
        with self._frame_lock:
            jpeg = self._latest_jpeg.get(name)
        if jpeg is None:
            return None
        try:
            arr = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
            small = policy_resize(arr, size, mode)
            buf = io.BytesIO()
            Image.fromarray(small).save(buf, format="JPEG", quality=self.jpeg_quality)
            return buf.getvalue()
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────
# Openpi policy runner (in-thread WS inference loop)
# ─────────────────────────────────────────────────────────────────────
class MotionMonitor:
    """Live answer to "is the arm executing the policy?" — the question the
    30-step watchdog only answers after the fact. Shown as the `motion` chip.

    Every executed tick reports its commanded joint delta (note_cmd); every
    state read reports the RAW joints + polymetis timestamp (note_q). The
    verdict looks at the last WINDOW_S seconds:

      policy_idle    net commanded displacement < CMD_MIN — the policy is not
                     asking for motion (hover / ~zero actions); a still arm is
                     the policy's choice, not a fault.
      executing      command present, joint readings changed, timestamp alive.
      suspect        command present but every joint reading in the window is
                     bit-identical (raw float32) — ticks may be dropped; shown
                     with a running duration, the 30-step watchdog escalates.
      not_executing  the robot-state timestamp stopped advancing — polymetis
                     control loop down; definitive, no ambiguity with hover.

    Calibration (2026-08-25/26 sync episodes, 1.3 s windows): net command is
    > 0.17 rad in 90 % of live windows, yet the arm follows only ~10 % of it
    under DROID's cartesian impedance — so a displacement RATIO cannot tell
    "slow tracking" from "frozen". What separates them is "changed at all" vs
    "bit-identical": frozen windows read exactly 0.0, live ones < 1e-4 in 2 %.
    """
    WINDOW_S = 1.5
    CMD_MIN = 0.03       # rad: net commanded max-joint displacement per window
    MIN_Q_SAMPLES = 3    # state reads needed before judging identity

    def __init__(self):
        self._lock = threading.Lock()
        self._cmd: collections.deque = collections.deque()   # (t, dq[7])
        self._q: collections.deque = collections.deque()     # (t, q[7], ts_ns)
        self._still_since: Optional[float] = None

    def reset(self) -> None:
        with self._lock:
            self._cmd.clear()
            self._q.clear()
            self._still_since = None

    def note_cmd(self, dq) -> None:
        with self._lock:
            self._cmd.append((time.time(), np.asarray(dq, dtype=np.float64)[:7].copy()))

    def note_q(self, q, ts_ns: Optional[int] = None) -> None:
        with self._lock:
            self._q.append((time.time(), np.asarray(q, dtype=np.float64)[:7].copy(), ts_ns))

    def _prune(self, now: float) -> None:
        cut = now - self.WINDOW_S
        while self._cmd and self._cmd[0][0] < cut:
            self._cmd.popleft()
        # Keep one sample from before the window as the displacement reference.
        while len(self._q) > 1 and self._q[1][0] < cut:
            self._q.popleft()

    def verdict(self, running: bool) -> dict:
        now = time.time()
        with self._lock:
            self._prune(now)
            cmds = list(self._cmd)
            qs = list(self._q)
        out = {"state": "idle", "cmd_net": 0.0, "act_net": 0.0, "n_q": len(qs),
               "ts_alive": None, "still_s": 0.0, "window_s": self.WINDOW_S}
        if not running:
            self._still_since = None
            return out
        cmd_net = float(np.abs(sum((d for _, d in cmds), np.zeros(7))).max()) if cmds else 0.0
        out["cmd_net"] = round(cmd_net, 4)
        if len(qs) >= 2:
            out["act_net"] = round(float(np.abs(qs[-1][1] - qs[0][1]).max()), 5)
            ts = [s[2] for s in qs if s[2] is not None]
            if len(ts) >= 2:
                out["ts_alive"] = len(set(ts)) > 1
        if len(qs) < self.MIN_Q_SAMPLES:
            out["state"] = "warming"
            return out
        identical = all(np.array_equal(s[1], qs[0][1]) for s in qs[1:])
        if out["ts_alive"] is False:
            state = "not_executing"
        elif cmd_net < self.CMD_MIN:
            state = "policy_idle"
        elif identical:
            state = "suspect"
        else:
            state = "executing"
        if state in ("suspect", "not_executing"):
            self._still_since = self._still_since or now
            out["still_s"] = round(now - self._still_since, 1)
        else:
            self._still_since = None
        out["state"] = state
        return out


class EvalRunner:
    """In-process openpi pi05_droid inference loop.

    Pulls cam frames from the shared CamManager (which keeps ZEDs open
    permanently), queries the robot state from NUC1 robot_server, calls
    openpi WebsocketClientPolicy, sends the action chunk back. Drops the
    docker/Ray complexity of the RLinf path entirely.
    """

    def __init__(self, cam_mgr: CamManager, home_store: "HomeStore",
                 rs_url: str = "http://100.75.6.62:4242",
                 policy_host: str = "127.0.0.1",
                 policy_port: int = 8000,
                 droid_url: str = "tcp://100.75.6.62:4242",
                 allow_missing_wrist: bool = False):
        # Degraded mode: run the policy with a BLACK wrist view when the wrist
        # ZED is dead (USB3 side gone, HID-only). Fine-manipulation quality
        # drops — bench-diagnostic use only, hence an explicit opt-in flag.
        self.allow_missing_wrist = allow_missing_wrist
        self.cam_mgr = cam_mgr
        self.home_store = home_store
        self.rs_url = rs_url.rstrip("/")
        self.droid_url = droid_url
        self._droid: Optional[DroidLikeClient] = None
        self.policy_host = policy_host
        self.policy_port = policy_port
        self.last_prompt = "grasp"
        # Experiment tag dict ({mode, task, pair, role}) sent as
        # `_episode_tag` with every obs; the hooked capture server records it
        # into the episode's episode.json. None = key not sent.
        self._tag: Optional[dict] = None
        # Per-iter knobs (sane defaults; overridable later via /config).
        self.open_loop_horizon = 4
        # Raw policy chunks (before clip/scale) for the portal's action panel.
        self.action_log: collections.deque = collections.deque(maxlen=40)
        self._action_seq = 0
        # "pad" (DROID-style, default) or "crop" (pbc ckpts) — set per Start.
        self.image_mode = "pad"
        # DROID's controller (droid/franka/robot.py + robot_ik_solver.py):
        #     q_target_step = q_current + action[:7] * delta_scale
        # with delta_scale = 0.2 rad/unit and 15 Hz position-target
        # streaming under impedance. We mirror that here.
        self.delta_scale = 0.2
        self.dynamics_factor = 0.3
        self.max_iterations = 400
        self.control_hz = 15.0
        # State
        # Gripper modes (switchable live via /gripper_mode):
        #  proportional  DROID exact: every iter POST /robotiq/move with
        #                width_m = max * (1-cmd), speed=0.5, force=0.0
        #                matching droid/franka/robot.py:121
        #  raw_binary    legacy: cmd≥threshold → /robotiq/close, debounced
        #  latch_close   once close fires 2/3 frames → stay closed until Start
        #  force_open    diagnostic: gripper held open regardless of policy
        #  raw_inverted  sign-flip A/B test
        #  latch_inverted  latch + inverted threshold
        self.gripper_mode = "proportional"
        self.gripper_threshold = 0.4
        self.gripper_min_switch_s = 0.5
        self.gripper_speed = 0.57     # DROID 0.05 m/s ≈ register 146/255 = 0.573 normalized
        self.gripper_force = 0.0      # DROID Robotiq force (minimum ~20N)
        self.gripper_max_width_m = 0.085
        self.latch_window = 3       # last N raw decisions for latch trigger
        self.latch_min_hits = 2     # need ≥ this many "close" in the window
        self._latched_close = False
        self._close_intent_window: list[bool] = []
        self._last_grip_was_close: Optional[bool] = None
        self._last_grip_switch_t: float = 0.0
        # Audit trail: one JSONL line per iter capturing all four
        # gripper signals so we can see if (a) raw action ever requests
        # close, (b) we sent it, (c) Robotiq executed, (d) state we
        # feed back to model.
        self.audit_path = pathlib.Path("/tmp/_grip_audit.jsonl")
        self._audit_lock = threading.Lock()
        self._audit_ring: list[dict] = []  # last ~40 entries kept in mem
        # State
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._iter = 0
        self._last_error: Optional[str] = None
        self._last_grip_raw: Optional[float] = None
        self._last_dq_max: Optional[float] = None
        self._last_infer_ms: Optional[float] = None
        self._lock = threading.Lock()
        # Episode recorder + per-episode metadata (task/ckpt) set by /start.
        self._recorder: Optional[EvalRecorder] = None
        self._episode_meta: dict = {}
        # Real-time chunking: toggle + knobs + live stats (see tasl/rtc/).
        self.rtc = _rtc_hook.RTCState()
        # True while /home or /eval/mark streams the arm home — Start is
        # refused meanwhile (a Start during the home stream interleaved two
        # command streams on the NUC on 2026-08-25).
        self.homing = False
        # Frozen-arm watchdog state (see watchdog_step).
        self._wd_hist: collections.deque = collections.deque(maxlen=self.WD_STEPS)
        self._wd_ts_last: Optional[int] = None   # last robot-state timestamp (ns)
        self._wd_ts_same = 0                      # consecutive polls with that timestamp
        self._wd_ts_armed = False                 # timestamp seen advancing this episode
        self._wd_ts_first_t = 0.0                 # wall clock when the stale run began
        self._wd_last: dict = {}                  # telemetry for /status
        # Live executing / policy-idle / not-executing verdict (motion chip).
        self.motion = MotionMonitor()

    # Frozen-arm watchdog — "NUC not executing" vs "policy chose not to move".
    # Two rules, evaluated on every policy step:
    #
    # (1) STALE STATE: the polymetis robot-state timestamp stops advancing.
    #     franka_panda_client stamps the state on every 1 kHz tick
    #     (setTimestampToNow); when its control loop is out (libfranka reflex
    #     → automaticErrorRecovery retries, or the driver died) no
    #     ControlUpdate reaches the server and GetRobotState hands back the
    #     last buffered state. DROID's non-blocking update_joints swallows the
    #     grpc error, so every zerorpc call keeps "succeeding" while the arm
    #     ignores the ticks. A hovering policy never trips this — the loop
    #     keeps stamping. Fires after WD_TS_POLLS polls (~2 s sync / ~1.5 s
    #     RTC). Armed only once the timestamp was seen advancing in this
    #     episode, so a driver that never stamps cannot false-trigger.
    #
    # (2) FROZEN JOINTS: commanded motion but bit-identical joint readings for
    #     WD_STEPS steps — backstop for the same fault. Calibrated 2026-08-26
    #     by replaying the 2026-08-25/26 episodes: the real freeze (T5-a
    #     ep_20260826_000452) reads a 30-step spread of exactly 0.0 rad;
    #     hovering under the live controller stays >= 3.5e-4 (RTC) / 2.9e-3
    #     (sync) over 30 steps but dips to 1e-5 over 10 — the window cannot be
    #     shortened, which is what rule (1) is for.
    WD_STEPS = 30
    WD_CMD_MIN = 0.02      # mean |clipped action| that counts as "commanding motion"
    WD_Q_EPS = 5e-5        # rad: joints considered frozen (identical readings) below this
    WD_TS_POLLS = 5        # consecutive polls returning the same robot timestamp
    # A libfranka reflex (communication_constraints_violation — NUC drops FCI
    # packets, measured 2026-08-26) freezes the state for ~1.05 s while
    # polymetis runs automaticErrorRecovery, then the loop resumes by itself;
    # several per episode. Only a loop that stays dead is worth aborting for.
    WD_TS_STALE_S = 3.0    # seconds the timestamp must stay frozen before aborting
    WD_REMEDY = ("NUC polymetis control loop is not executing commands (libfranka reflex → "
                 "auto-recovery). Watch the robot row: if 'loop' turns alive again within "
                 "~10 s, just Go home + Start; if it stays STALE, press 🔧 Reset NUC.")

    def _wd_reset(self) -> None:
        self._wd_hist.clear()
        self._wd_ts_last = None
        self._wd_ts_same = 0
        self._wd_ts_armed = False
        self._wd_ts_first_t = 0.0
        self._wd_last = {}

    def watchdog_step(self, q: np.ndarray, cmd_mag: float,
                      ts_ns: Optional[int] = None) -> Optional[str]:
        """Feed one policy step: RAW joint reading (not rounded), mean
        |clipped action|, robot-state timestamp in ns. Returns an error
        string when the NUC is not executing commands, else None."""
        now = time.time()
        # Rule (1): robot-state timestamp.
        stale_s = 0.0
        if ts_ns is not None:
            if self._wd_ts_last is None or ts_ns != self._wd_ts_last:
                if self._wd_ts_last is not None:
                    self._wd_ts_armed = True
                self._wd_ts_last, self._wd_ts_same, self._wd_ts_first_t = ts_ns, 1, now
            else:
                self._wd_ts_same += 1
                stale_s = now - self._wd_ts_first_t
        # Rule (2): joint spread while commanding motion.
        self._wd_hist.append((np.asarray(q, dtype=np.float64)[:7].copy(), float(cmd_mag)))
        qs = np.stack([h[0] for h in self._wd_hist])
        q_spread = float(np.max(np.abs(qs - qs[0])))
        cmd_mean = float(np.mean([h[1] for h in self._wd_hist]))
        self._wd_last = {
            "n": len(self._wd_hist), "q_spread": round(q_spread, 6),
            "cmd_mean": round(cmd_mean, 4), "ts_armed": self._wd_ts_armed,
            "ts_same": self._wd_ts_same, "ts_stale_s": round(stale_s, 2),
        }
        if self._wd_ts_armed and self._wd_ts_same >= self.WD_TS_POLLS \
                and stale_s >= self.WD_TS_STALE_S:
            self._wd_reset()
            return (f"NUC not executing commands: robot-state timestamp frozen for "
                    f"{stale_s:.1f}s ({self._wd_ts_same} polls) — " + self.WD_REMEDY)
        if len(self._wd_hist) >= self.WD_STEPS and cmd_mean > self.WD_CMD_MIN \
                and q_spread < self.WD_Q_EPS:
            self._wd_reset()
            return (f"controller not executing commands: joint readings identical for "
                    f"{self.WD_STEPS} steps while the policy commanded motion — " + self.WD_REMEDY)
        return None

    def start(self, prompt: str, tag: Optional[dict] = None) -> str:
        with self._lock:
            if self._running:
                return "already running"
            if self.homing:
                return "homing in progress — wait for the arm to reach home, then Start"
            self._wd_reset()
            self.motion.reset()
            self.action_log.clear()
            self._tag = tag
            # Bootstrap the droid_nuc container — first call spawns the
            # polymetis driver + Robotiq driver inside the container; no-op
            # on subsequent calls. Replaces the old /recover handshake.
            try:
                if self._droid is None:
                    self._droid = DroidLikeClient(address=self.droid_url)
                self._droid.bootstrap()
            except Exception as exc:
                _log.error(f"droid bootstrap failed: {exc}")
                return f"bootstrap failed: {exc}"
            self.last_prompt = prompt
            # Episode recording: both views tiled, auto-finalized when the
            # loop exits (Stop or max-iterations).
            meta = dict(self._episode_meta or {})
            self._recorder = EvalRecorder(
                self.cam_mgr,
                task_id=str(meta.get("task") or "untagged"),
                prompt=prompt,
                ckpt=str(meta.get("ckpt") or ""),
                layout=str(meta.get("layout") or ""),
                running_cb=lambda: self._running,
                steps_source=lambda: self._iter,
                abort_source=lambda: self._last_error,
            )
            self._stop.clear()
            self._iter = 0
            self._last_error = None
            self._last_grip_was_close = None
            self._last_grip_switch_t = 0.0
            self._latched_close = False
            self._close_intent_window = []
            self._thread = threading.Thread(
                target=self._loop, args=(prompt,), daemon=True
            )
            self._running = True
            self._thread.start()
            _log.info(f"policy loop started, prompt={prompt!r}")
            return f"started prompt={prompt!r}"

    def stop(self) -> str:
        if not self._running:
            return "not running"
        # Signal loop to exit — checked at the top of each iter, after
        # inference, and between every streamed tick.
        self._stop.set()
        # Once the loop has exited, send one zero-velocity tick so the
        # arm halts at its current pose instead of coasting toward the
        # last streamed setpoint. Keep the last gripper command so a
        # stop mid-grasp doesn't drop the object. Background thread so
        # /stop returns immediately even when the NUC is slow.
        loop_thread = self._thread
        last_grip = self._last_grip_raw

        def _halt():
            if loop_thread is not None:
                loop_thread.join(timeout=5.0)
            try:
                if self._droid is not None and self._droid.bootstrapped:
                    a8 = [0.0] * 8
                    if last_grip is not None:
                        a8[7] = max(0.0, min(1.0, float(last_grip)))
                    self._droid.update_joint_velocity(a8, blocking=False)
                    _log.info("stop: halt tick sent")
            except Exception as exc:
                _log.warning(f"stop: halt tick failed: {exc}")

        threading.Thread(target=_halt, daemon=True).start()
        return "stop signal sent"

    def note_actions(self, actions, src: str, **extra) -> None:
        """Append one raw policy chunk (H, 8) to the action ring buffer shown
        under the console. Values are the policy's own output (joint-velocity
        units in [-1, 1] + gripper [0, 1]), before clip / delta_scale."""
        try:
            a = np.asarray(actions, dtype=np.float64)
            self._action_seq += 1
            rec = {"seq": self._action_seq, "t": round(time.time(), 3), "src": src,
                   "shape": list(a.shape),
                   "values": np.round(a, 3).tolist() if a.ndim == 2 else []}
            for k, v in extra.items():
                rec[k] = round(float(v), 1) if isinstance(v, float) else v
            self.action_log.append(rec)
        except Exception:  # noqa: BLE001 — diagnostics must never break the loop
            pass

    def status(self) -> dict:
        return {
            "running": self._running,
            "iter": self._iter,
            "last_prompt": self.last_prompt,
            "tag": self._tag,
            "last_error": self._last_error,
            "last_grip_raw": self._last_grip_raw,
            "last_dq_max": self._last_dq_max,
            "last_infer_ms": self._last_infer_ms,
            "open_loop_horizon": self.open_loop_horizon,
            "image_mode": self.image_mode,
            "delta_scale": self.delta_scale,
            "dynamics_factor": self.dynamics_factor,
            "control_mode": "joint_position_delta",
            "gripper_mode": self.gripper_mode,
            "gripper_latched": self._latched_close,
            "homing": self.homing,
            "wd": self._wd_last,
            "motion": self.motion.verdict(self._running),
            "rtc": _rtc_hook.status(self),
            "log_tail": [],  # placeholder for build_app compatibility
        }

    def droid_raw(self) -> DroidLikeClient:
        """Lazy client WITHOUT bootstrap — for read-only state polls.
        Bootstrapping as a side effect would spawn the polymetis driver,
        which /status must never do."""
        if self._droid is None:
            self._droid = DroidLikeClient(address=self.droid_url)
        return self._droid

    def get_droid(self) -> DroidLikeClient:
        """Return a ready-to-use DroidLikeClient (lazy create + bootstrap).
        UI handlers (Go Home, Recover, etc.) use this so they don't need
        a separate connection."""
        d = self.droid_raw()
        d.bootstrap()
        return d

    def set_gripper_mode(self, mode: str) -> str:
        valid = {"proportional", "raw_binary", "latch_close", "force_open",
                 "raw_inverted", "latch_inverted"}
        if mode not in valid:
            return f"invalid mode {mode!r}; choose from {sorted(valid)}"
        self.gripper_mode = mode
        # Reset latch / window when switching mode mid-run so old state
        # doesn't bleed into the new policy.
        self._latched_close = False
        self._close_intent_window = []
        _log.info(f"gripper_mode set to {mode!r}")
        return f"set to {mode}"

    def _loop(self, prompt: str):
        try:
            try:
                from openpi_client.websocket_client_policy import (
                    WebsocketClientPolicy,
                )
            except Exception as e:
                self._last_error = f"openpi_client import: {e}"
                _log.error(self._last_error)
                return
            try:
                policy = WebsocketClientPolicy(
                    host=self.policy_host, port=self.policy_port
                )
            except Exception as e:
                self._last_error = f"policy connect: {e}"
                _log.error(self._last_error)
                return

            if _rtc_hook.active(self):
                # Serve was loaded "with RTC" → RTC path (tasl/rtc/): async
                # controller + guided inference.
                # Blocks until the episode stops; the finally below cleans up.
                _rtc_hook.run_episode(self, prompt, policy)
                return

            chunk_dt = 1.0 / self.control_hz
            while self._iter < self.max_iterations and not self._stop.is_set():
                self._iter += 1
                # Frames from CamManager (always running).
                ext_jpeg = self.cam_mgr.get_jpeg("wrist_1")
                wrist_jpeg = self.cam_mgr.get_jpeg("wrist_2")
                if ext_jpeg is None or (wrist_jpeg is None
                                        and not self.allow_missing_wrist):
                    time.sleep(0.05)
                    self._iter -= 1
                    continue
                try:
                    ext_rgb = np.asarray(
                        Image.open(io.BytesIO(ext_jpeg)).convert("RGB")
                    )
                    if wrist_jpeg is None:
                        # --allow-missing-wrist: black frame stands in for the
                        # dead wrist ZED; policy runs half-blind.
                        wrist_rgb = np.zeros_like(ext_rgb)
                    else:
                        wrist_rgb = np.asarray(
                            Image.open(io.BytesIO(wrist_jpeg)).convert("RGB")
                        )
                except Exception as e:
                    _log.warning(f"jpeg decode failed: {e}; skipping iter")
                    continue
                ext_r = self.preprocess_image(ext_rgb)
                wrist_r = self.preprocess_image(wrist_rgb)

                # Robot state.
                # Robot + gripper state via droid_nuc zerorpc — one call
                # replaces both the old /state and /robotiq/state polls.
                try:
                    state = self._droid.get_robot_state()
                except Exception as e:
                    self._last_error = f"state: {e}"
                    _log.error(self._last_error)
                    break
                joint_position = state["joint_positions"].astype(np.float32)
                # polymetis stamps the state every 1 kHz tick — the watchdog's
                # liveness signal (frozen ⇒ the NUC is not executing).
                state_ts_ns = (int(state["timestamp_seconds"]) * 1_000_000_000
                               + int(state["timestamp_nanos"]))
                self.motion.note_q(joint_position, state_ts_ns)
                # DROID's gripper_position is already in [0,1] with 1=close
                # — same convention pi05_droid was trained against in the
                # DROID dataset, so pass through verbatim (no flip).
                gripper_position = np.asarray(
                    [state["gripper_position"]], dtype=np.float32
                )
                # Audit uses these — keep the old names so the audit_write
                # call below doesn't need to change.
                max_w = self.gripper_max_width_m
                gw = max_w * (1.0 - float(state["gripper_position"]))
                gw_norm = max(0.0, min(1.0, gw / max(max_w, 1e-6)))

                obs = {
                    "observation/exterior_image_1_left": ext_r,
                    "observation/wrist_image_left": wrist_r,
                    "observation/joint_position": joint_position,
                    "observation/gripper_position": gripper_position,
                    "prompt": prompt,
                }
                # Experiment tag for the hooked capture server (episode.json);
                # the vanilla server's DroidInputs drops unknown keys.
                if self._tag:
                    obs["_episode_tag"] = self._tag

                t0 = time.perf_counter()
                try:
                    result = policy.infer(obs)
                except Exception as e:
                    self._last_error = f"infer: {e}"
                    _log.error(self._last_error)
                    break
                self._last_infer_ms = (time.perf_counter() - t0) * 1000.0
                actions = np.asarray(result["actions"])  # (H, 8)
                if actions.ndim != 2 or actions.shape[-1] != 8:
                    self._last_error = (
                        f"unexpected action shape {actions.shape}"
                    )
                    _log.error(self._last_error)
                    break
                self.note_actions(actions, "sync", infer_ms=self._last_infer_ms)

                if self._stop.is_set():
                    break

                # DROID position-target streaming: clip raw to [-1, 1],
                # accumulate q_target = q + sum(action[:k] * delta_scale).
                # This matches openpi/examples/droid/main.py + droid/franka/robot.py
                # exactly. Joint-velocity dispatch (our old path) is the WRONG
                # control mode + wrong scale (~3x too slow + no impedance).
                n = min(self.open_loop_horizon, actions.shape[0])
                clipped = np.clip(actions[:n, :7], -1.0, 1.0)
                wd = self.watchdog_step(joint_position, float(np.abs(clipped).mean()),
                                        state_ts_ns)
                if wd:
                    self._last_error = wd
                    _log.error(wd)
                    break
                q_targets: list[list[float]] = []
                q_running = joint_position.astype(np.float64).copy()
                for step in range(n):
                    q_running = q_running + clipped[step] * self.delta_scale
                    q_targets.append([float(x) for x in q_running])
                # DROID action convention: gripper ∈ [0,1], 0=open, 1=close.
                grip_target_raw = float(actions[n - 1, 7])
                self._last_grip_raw = grip_target_raw
                # Diagnostic: largest per-step joint delta we just commanded.
                self._last_dq_max = float(
                    np.max(np.abs(clipped)) * self.delta_scale
                )
                # Trajectory log for the episode recorder (state at this
                # step + the part of the chunk that is about to execute).
                if self._recorder is not None:
                    self._recorder.log_step({
                        "t": round(time.time(), 4),
                        "ts": state_ts_ns,           # robot-state timestamp (NUC loop liveness)
                        "iter": self._iter,
                        "q": [round(float(x), 5) for x in joint_position],
                        "grip": round(float(gripper_position[0]), 4),
                        "infer_ms": round(self._last_infer_ms, 1),
                        "actions": np.round(actions[:n], 4).tolist(),
                        "q_target": [round(x, 5) for x in q_targets[-1]],
                    })

                # Stream 8-D actions to droid_nuc at 15 Hz — DROID's training
                # path exactly. Each tick carries both joint velocity AND
                # gripper command in a single update_command call; polymetis
                # cartesian impedance + Robotiq driver each pick up the
                # next setpoint. blocking=False so we don't stall.
                try:
                    for step in range(n):
                        if self._stop.is_set():
                            break
                        a8 = np.zeros(8)
                        a8[:7] = clipped[step]              # joint vel in [-1, 1]
                        self.motion.note_cmd(clipped[step] * self.delta_scale)
                        a8[7] = float(actions[step, 7])     # gripper 0=open, 1=close
                        self._droid.update_joint_velocity(a8, blocking=False)
                        time.sleep(chunk_dt)
                except Exception as e:
                    self._last_error = f"chunk stream: {e}"
                    _log.error(self._last_error)
                    break
                if self._stop.is_set():
                    break

                # Per-iter gripper decision based on selected mode.
                now = time.time()
                action_taken = None  # "open" | "close" | "move" | None
                reason_skipped = None
                mode = self.gripper_mode

                # --- DROID-exact proportional mode: short-circuit the
                # binary thresholding + debounce entirely. Spawn a tiny
                # background thread that fires N=open_loop_horizon
                # /robotiq/move POSTs at chunk_dt cadence — matches
                # DROID's 15 Hz per-step gripper update inside a chunk,
                # NOT once-per-chunk like the old binary path.
                if mode == "proportional":
                    per_step_cmds = [
                        float(actions[i, 7]) for i in range(n)
                    ]
                    # Gripper was already streamed inside the arm loop
                    # above (each update_joint_velocity carries action[step, 7]).
                    last_cmd = max(0.0, min(1.0, per_step_cmds[-1]))
                    target_w_last = self.gripper_max_width_m * (1.0 - last_cmd)
                    action_taken = "move"
                    # Robotiq readback via droid_nuc state.
                    gw_after = gw
                    try:
                        post_state = self._droid.get_robot_state()
                        gw_after = max_w * (
                            1.0 - float(post_state["gripper_position"])
                        )
                    except Exception:
                        pass
                    self._audit_write({
                        "ts": now, "iter": self._iter, "mode": mode,
                        "raw_action_grip": grip_target_raw,
                        "raw_close": last_cmd >= 0.5,
                        "latched": False,
                        "want_close": last_cmd >= 0.5,
                        "action_taken": action_taken,
                        "reason_skipped": reason_skipped,
                        "gw_pre_m": gw, "gw_post_m": gw_after,
                        "gw_obs_norm": float(gw_norm),
                        "dq_max": self._last_dq_max,
                        "target_w_m": target_w_last,
                        "n_subcmds": n,   # how many gripper POSTs fired this chunk
                    })
                    continue  # skip the legacy binary path below

                # 1. Decide raw intent per mode (legacy binary modes).
                if mode == "force_open":
                    raw_close = False
                elif mode in ("raw_inverted", "latch_inverted"):
                    raw_close = grip_target_raw < self.gripper_threshold
                else:  # raw_binary, latch_close
                    raw_close = grip_target_raw >= self.gripper_threshold

                # 2. Latch logic for latch modes — once we accumulate
                # ≥ latch_min_hits "close" intents in the last latch_window
                # iterations, lock to close until next Start.
                if mode in ("latch_close", "latch_inverted"):
                    self._close_intent_window.append(raw_close)
                    if len(self._close_intent_window) > self.latch_window:
                        self._close_intent_window.pop(0)
                    if (not self._latched_close
                            and sum(self._close_intent_window)
                                >= self.latch_min_hits):
                        self._latched_close = True
                        _log.info(f"gripper LATCHED close at iter={self._iter}")
                    want_close = self._latched_close
                else:
                    want_close = raw_close

                state_changed = (
                    self._last_grip_was_close is None
                    or want_close != self._last_grip_was_close
                )
                cooled_down = (
                    now - self._last_grip_switch_t
                    >= self.gripper_min_switch_s
                )

                if state_changed and cooled_down:
                    ep = "/robotiq/close" if want_close else "/robotiq/open"
                    action_taken = "close" if want_close else "open"
                    try:
                        requests.post(
                            f"{self.rs_url}{ep}",
                            json={"speed": 0.3},
                            timeout=3.0,
                        )
                        self._last_grip_was_close = want_close
                        self._last_grip_switch_t = now
                    except Exception as e:
                        _log.warning(f"gripper: {e}")
                        reason_skipped = f"http_err: {e}"
                else:
                    if not state_changed:
                        reason_skipped = (
                            "latched" if self._latched_close else "same_as_last"
                        )
                    elif not cooled_down:
                        reason_skipped = "cooldown"

                # Robotiq readback AFTER any command — gives the post-cmd
                # state so we can see the motor actually moved.
                gw_after = gw  # fallback to pre-cmd
                try:
                    rg2 = requests.get(
                        f"{self.rs_url}/robotiq/state", timeout=1.0
                    )
                    if rg2.ok:
                        gw_after = float(rg2.json().get("width_m", gw))
                except Exception:
                    pass

                # Audit record: all 4 signals + decision trail + mode/latch.
                self._audit_write({
                    "ts": now,
                    "iter": self._iter,
                    "mode": mode,
                    "raw_action_grip": grip_target_raw,
                    "raw_close": raw_close,
                    "latched": self._latched_close,
                    "want_close": want_close,
                    "action_taken": action_taken,
                    "reason_skipped": reason_skipped,
                    "gw_pre_m": gw,
                    "gw_post_m": gw_after,
                    "gw_obs_norm": float(gw_norm),
                    "dq_max": self._last_dq_max,
                })
            _log.info(f"policy loop exited after {self._iter} iters")
        finally:
            # WebsocketClientPolicy has no close() and its background recv
            # thread pins the connection — close the raw ws explicitly so the
            # capture server sees the episode end promptly.
            try:
                policy._ws.close()  # noqa: SLF001
            except Exception:
                pass
            self._running = False
            self._thread = None

    def _fire_gripper_chunk(self, cmds: list[float], step_dt: float) -> None:
        """Send N gripper /robotiq/move commands at step_dt intervals.

        Matches DROID's 15 Hz gripper command cadence (per-step inside
        a chunk) rather than our old once-per-chunk pattern. Robotiq
        is non-blocking on the server side, so each POST returns ~10ms
        after sending the Modbus frame; the motor physically tracks
        the latest target. The 67ms sleep between commands ensures the
        motor sees a moving setpoint instead of just the final value.
        """
        for i, cmd in enumerate(cmds):
            if self._stop.is_set():
                return
            cmd_clipped = max(0.0, min(1.0, cmd))
            target_w = self.gripper_max_width_m * (1.0 - cmd_clipped)
            try:
                requests.post(
                    f"{self.rs_url}/robotiq/move",
                    json={
                        "width_m": target_w,
                        "speed": self.gripper_speed,
                        "force": self.gripper_force,
                    },
                    timeout=1.0,
                )
            except Exception as exc:
                _log.warning(f"gripper sub-cmd {i}: {exc}")
            if i < len(cmds) - 1:
                time.sleep(step_dt)

    def preprocess_image(self, arr: np.ndarray, size: int = 224) -> np.ndarray:
        """Raw HD720 RGB → the 224² the policy sees, per this run's
        image_mode ("pad" DROID default / "crop" pbc). Shared by the sync
        loop and the RTC hook (tasl/rtc/dashboard_hook.py)."""
        return policy_resize(arr, size, self.image_mode)

    def _audit_write(self, rec: dict) -> None:
        line = json.dumps(rec, default=float)
        with self._audit_lock:
            self._audit_ring.append(rec)
            if len(self._audit_ring) > 40:
                self._audit_ring.pop(0)
            try:
                with self.audit_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass

    def audit_recent(self, n: int = 20) -> list[dict]:
        with self._audit_lock:
            return list(self._audit_ring[-n:])


# Note: log tail / saved-episode parsing helpers are not needed in the
# openpi-standalone dashboard — the inference loop runs in-process and
# exposes its state directly via EvalRunner.status().


# No saved-episode parsing here — the openpi-standalone loop doesn't
# write a LeRobot dataset. EvalRunner.status() exposes per-iter metrics
# directly (last_dq_max, last_grip_raw, last_infer_ms).


# ─────────────────────────────────────────────────────────────────────
# robot_server helpers
# ─────────────────────────────────────────────────────────────────────
class RS:
    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def _get(self, ep: str, t: float = 2.0):
        try:
            r = requests.get(f"{self.url}{ep}", timeout=t)
            return r.json() if r.ok else {"_err": r.status_code, "_body": r.text[:200]}
        except requests.RequestException as e:
            return {"_err": "exc", "_body": repr(e)[:200]}

    def _post(self, ep: str, payload=None, t: float = 30.0):
        try:
            r = requests.post(f"{self.url}{ep}", json=payload or {}, timeout=t)
            return r.json() if r.ok else {"_err": r.status_code, "_body": r.text[:200]}
        except requests.RequestException as e:
            return {"_err": "exc", "_body": repr(e)[:200]}

    def ping(self):
        return self._get("/ping")

    def state(self):
        return self._get("/state")

    def robotiq_state(self):
        return self._get("/robotiq/state")

    def go_home(self, target_q, dynamics_factor: float = 0.05):
        return self._post(
            "/move/joint",
            {"target_q": list(target_q), "dynamics_factor": dynamics_factor},
            t=30.0,
        )

    def stop(self):
        return self._post("/stop", {}, t=3.0)

    def recover(self):
        return self._post("/recover", {}, t=3.0)

    def jog_cartesian(self, dx: float, dy: float, dz: float,
                      drx: float = 0.0, dry: float = 0.0, drz: float = 0.0,
                      dynamics_factor: float = 0.05):
        """One-shot cartesian jog. Euler (xyz, radians) → quaternion delta."""
        # Small-angle Euler → quaternion (intrinsic xyz / rzryrx order matches
        # what FrankaEnv uses elsewhere).
        cx, sx = np.cos(drx / 2), np.sin(drx / 2)
        cy, sy = np.cos(dry / 2), np.sin(dry / 2)
        cz, sz = np.cos(drz / 2), np.sin(drz / 2)
        qx = sx * cy * cz - cx * sy * sz
        qy = cx * sy * cz + sx * cy * sz
        qz = cx * cy * sz - sx * sy * cz
        qw = cx * cy * cz + sx * sy * sz
        payload = {
            "delta_translation": [float(dx), float(dy), float(dz)],
            "delta_quaternion": [float(qx), float(qy), float(qz), float(qw)],
            "dynamics_factor": float(dynamics_factor),
        }
        return self._post("/move/cartesian_relative", payload, t=10.0)

    def gripper(self, action: str, speed: float = 0.3):
        ep = "/robotiq/open" if action == "open" else "/robotiq/close"
        return self._post(ep, {"speed": speed}, t=5.0)

    def freedrive(self, enable: bool):
        ep = "/freedrive/on" if enable else "/freedrive/off"
        return self._post(ep, {}, t=3.0)


# ─────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────
# Policy checkpoint switcher — 10 fine-tuned checkpoints (v2 16-ep +
# calibra50 8-ep), each 5 steps. Loaded NATIVELY by JAX serve_policy
# (the RLinf PyTorch path cannot represent the LoRA arch — conversion
# would silently drop the adapters, so we serve the orbax dirs directly).
# ─────────────────────────────────────────────────────────────────────
# ── Checkpoint registry — auto-discovered, normalized placement ─────
# Convention: every checkpoint lives under CKPT_ROOT (~/ckpts) as
#   <root>/<group>/params            single checkpoint (e.g. pi05_droid)
#   <root>/<group>/<step>/params     per-step checkpoints of a run
# The openpi TrainConfig to serve a group defaults to
# pi05_droid_franka_lora and is overridden by a one-line marker file:
# <group>/config.txt or <root>/<group>.config.txt (e.g. "pi05_droid"
# for the base model). /ckpts re-scans live, so dropping a checkpoint
# into CKPT_ROOT shows up without a dashboard restart.
CKPT_ROOT = os.path.join(_HOME_DIR, "ckpts")
DEFAULT_SERVE_CONFIG = "pi05_droid_franka_lora"


def discover_checkpoints() -> list[dict]:
    """Scan CKPT_ROOT for every serveable checkpoint (orbax params/)."""
    out: list[dict] = []
    root = pathlib.Path(CKPT_ROOT)
    if not root.exists():
        return out
    for g in sorted(root.iterdir()):
        if not g.is_dir():
            continue
        cfg = DEFAULT_SERVE_CONFIG
        for cfg_f in (g / "config.txt", root / f"{g.name}.config.txt"):
            if cfg_f.exists():
                try:
                    cfg = cfg_f.read_text(encoding="utf-8").strip().splitlines()[0].strip() or cfg
                except Exception:
                    pass
                break
        if (g / "params").is_dir():
            out.append({"label": g.name, "dir": str(g), "config": cfg})
        for s in sorted(g.iterdir()):
            if s.is_dir() and (s / "params").is_dir():
                out.append({"label": f"{g.name}/{s.name}",
                            "dir": str(s), "config": cfg})
    return out


SERVE_CMD_DIR = os.path.join(_HOME_DIR, "work/openpi")
SERVE_PY = os.path.join(SERVE_CMD_DIR, ".venv/bin/python")
SERVE_SCRIPT = os.path.join(SERVE_CMD_DIR, "scripts/serve_policy.py")
# "Load with RTC" spawns the RTC-capable drop-in (same CLI) instead — tasl/rtc/.
SERVE_SCRIPT_RTC = _rtc_hook.SERVE_SCRIPT
SERVE_LOG = "/tmp/serve_ckpt.log"

# In-memory ring of recent backend log lines, served to the frontend console
# at /logs (the dashboard log file path varies by launcher — the ring doesn't).
BACKEND_EVENTS: collections.deque = collections.deque(maxlen=120)

# Dashboard process boot timestamp — the frontend watches this to detect
# server restarts and re-attach the MJPEG <img> streams (browsers never
# reconnect a broken multipart stream on their own).
BOOT_TS: float = time.time()


class _RingHandler(logging.Handler):
    """Captures formatted log records into BACKEND_EVENTS.

    Werkzeug access lines ("127.0.0.1 - - [...] GET /status") are skipped —
    the frontend polls 4 endpoints every 1.5s and those lines would drown
    the console in noise.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
            if re.match(r"^[\d.]+ - - \[", msg):
                return
            BACKEND_EVENTS.append(self.format(record))
        except Exception:
            pass


class ServeManager:
    """Owns the openpi serve_policy lifecycle for checkpoint switching.

    switch(): background thread — kill any running serve, spawn serve_policy
    with the chosen --policy.dir, poll until the server answers (weight
    restore + imports ≈ 0.5–2 min). State + log tail surfaced via /ckpts
    and mirrored into /status so the UI shows the loading progress live.
    """

    def __init__(self, port: int = 8000):
        self.port = port
        self.label: Optional[str] = None
        self.dir: Optional[str] = None
        self.state = "idle"          # idle | loading | ready | error
        self.msg = ""
        self.pid: Optional[int] = None
        self.rtc = False             # serve was loaded "with RTC" (tasl/rtc/)
        self._lock = threading.Lock()
        self._recover()

    def _kill_serve(self) -> None:
        try:
            subprocess.run(["pkill", "-f", "scripts/serve_policy.py"],
                           timeout=15, check=False)
        except Exception as exc:
            _log.warning(f"pkill serve_policy failed: {exc}")
        time.sleep(2.0)  # let the port close

    def _spawn(self, ckpt_dir: str, config_name: str, rtc: bool = False) -> None:
        env = dict(os.environ)
        # Under sudo HOME is /root: force the real user's home so openpi's
        # cache (~/.cache/openpi) and the RTC serve's OPENPI_DIR resolve.
        env["HOME"] = _HOME_DIR
        env["OPENPI_DIR"] = SERVE_CMD_DIR
        env.pop("PYTHONPATH", None)
        log_f = open(SERVE_LOG, "w", buffering=1)
        self.pid = subprocess.Popen(
            [SERVE_PY, SERVE_SCRIPT_RTC if rtc else SERVE_SCRIPT, "--port", str(self.port),
             "policy:checkpoint",
             f"--policy.config={config_name}",
             f"--policy.dir={ckpt_dir}"],
            stdout=log_f, stderr=subprocess.STDOUT,
            start_new_session=True, env=env, cwd=SERVE_CMD_DIR,
        ).pid

    def _probe(self) -> bool:
        """TCP-connect probe — proves the WS server is listening WITHOUT
        sending an HTTP request (a plain GET is rejected by the websockets
        server and spams InvalidUpgrade tracebacks into the serve log)."""
        try:
            with socket.create_connection(("127.0.0.1", self.port), timeout=1.0):
                return True
        except Exception:
            return False

    def _recover(self) -> None:
        """If a serve is already listening on our port (e.g. the dashboard was
        restarted while the serve kept running), adopt it: read its
        --policy.dir from the process cmdline and map back to a label."""
        if not self._probe():
            return
        try:
            out = subprocess.run(["pgrep", "-af", "serve_policy.py"],
                                 capture_output=True, text=True,
                                 timeout=5).stdout
        except Exception:
            return
        for line in out.splitlines():
            m = re.search(r"--policy\.dir=(\S+)", line)
            if m:
                for c in discover_checkpoints():
                    if c["dir"] == m.group(1):
                        with self._lock:
                            self.label, self.dir = c["label"], m.group(1)
                            self.rtc = _rtc_hook.is_rtc_serve(line)
                            self.state, self.msg = "ready", "adopted running serve"
                        _log.info(f"ServeManager recovered: {self.label}")
                        return

    def _switch_worker(self, label: str, ckpt_dir: str, config_name: str,
                       rtc: bool = False) -> None:
        with self._lock:
            self.label, self.dir, self.rtc = label, ckpt_dir, rtc
            self.state, self.msg = "loading", "stopping old serve…"
        try:
            self._kill_serve()
            with self._lock:
                self.msg = "restoring weights + JAX warmup (0.5–2 min)…"
            self._spawn(ckpt_dir, config_name, rtc)
        except Exception as exc:
            with self._lock:
                self.state, self.msg = "error", f"spawn failed: {exc}"
            return
        deadline = time.time() + 360.0
        while time.time() < deadline:
            with self._lock:
                if self.state == "idle":  # cancelled via stop()
                    return
            if self._probe():
                with self._lock:
                    self.state, self.msg = "ready", "serve up"
                _log.info(f"ckpt {label} serving on :{self.port}")
                return
            try:
                os.kill(self.pid, 0)
            except (OSError, TypeError):
                with self._lock:
                    self.state, self.msg = "error", "serve exited early — see log"
                return
            time.sleep(3.0)
        with self._lock:
            self.state, self.msg = "error", "timeout (6 min)"

    def switch(self, label: str, rtc: bool = False) -> str:
        entry = next((c for c in discover_checkpoints()
                      if c["label"] == label), None)
        if entry is None:
            return f"unknown checkpoint {label!r}"
        if not pathlib.Path(entry["dir"], "params").exists():
            return f"checkpoint dir missing: {entry['dir']}"
        with self._lock:
            if self.state == "loading":
                return "already switching"
        threading.Thread(target=self._switch_worker,
                         args=(entry["label"], entry["dir"], entry["config"], rtc),
                         daemon=True).start()
        return "switching"

    def stop(self) -> str:
        with self._lock:
            if self.state == "idle":
                return "nothing running"
            self.state = "idle"
            self.label = self.dir = None
            self.pid = None
        self._kill_serve()
        with self._lock:
            self.msg = "stopped"
        return "stopped"

    def status(self) -> dict:
        log_tail = []
        try:
            text = pathlib.Path(SERVE_LOG).read_text(encoding="utf-8")
            log_tail = text[-1500:].splitlines()[-8:]
        except OSError:
            pass
        with self._lock:
            if self.state == "ready" and not self._probe():
                self.state, self.msg = "error", "serve died unexpectedly"
            return {"label": self.label, "state": self.state,
                    "msg": self.msg, "port": self.port, "rtc": self.rtc,
                    "log_tail": log_tail}


# Task store lives in dashboards/task_store.py — SHARED with the
# collection dashboard so both portals see the same task registry.


# ─────────────────────────────────────────────────────────────────────
# Eval episode recorder — every Stop writes the two camera views tiled
# side-by-side (exterior | wrist) into an MP4 under <task>/<ep_id>/video.mp4,
# plus an editable meta.json (time / task / ckpt / steps / mark / note),
# traj.jsonl (one line per policy step: q, gripper, executed action chunk,
# infer latency) and frame_times.json (wall-clock of every video frame) so a
# later pass can draw state/action curves time-aligned with the video.
# ─────────────────────────────────────────────────────────────────────
EPISODES_ROOT = os.path.join(_TASL_DIR, "eval_episodes")


class EvalRecorder:
    """Runs its own sampling thread against the CamManager's latest frames —
    the eval loop itself is untouched. Finalizes when the running callback
    goes False (Stop or max-iterations end)."""

    def __init__(self, cam_mgr, task_id: str, prompt: str, ckpt: str,
                 fps: float = 15.0, running_cb=None, steps_source=None,
                 abort_source=None, layout: str = ""):
        self.cam_mgr = cam_mgr
        self.layout = layout or ""   # task layout id armed at Start ("" = none)
        # Callable → the runner's last_error at finalize time; non-empty
        # means the loop broke on an error (watchdog / state / infer), so the
        # episode is NOT a policy verdict — meta.json carries it as "abort".
        self.abort_source = abort_source
        self.task_id = task_id or "untagged"
        self.prompt = prompt
        self.ckpt = ckpt or ""
        self.fps = fps
        self.running_cb = running_cb
        self.steps_source = steps_source
        self.frames = 0
        self.ep_id = time.strftime("ep_%Y%m%d_%H%M%S")
        self.started = time.time()
        self.out_dir = pathlib.Path(EPISODES_ROOT) / self.task_id / self.ep_id
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._writer = None
        self._closed = False
        self._lock = threading.Lock()
        self._frame_ts: list[float] = []   # wall-clock per written frame
        self._traj_f = None                # traj.jsonl, opened on first step
        self.steps_logged = 0
        threading.Thread(target=self._run, daemon=True).start()

    @staticmethod
    def _tile(ext, wrist):
        """Side-by-side tile from raw BGR frames (no JPEG round-trip)."""
        if ext is None:
            return None
        if wrist is not None:
            h = ext.shape[0]
            wr = cv2.resize(
                wrist, (int(wrist.shape[1] * h / wrist.shape[0]), h))
            return np.concatenate([ext, wr], axis=1)
        return ext

    def _run(self):
        dt = 1.0 / self.fps
        nxt = time.time()
        # The recorder is built BEFORE the runner flips _running=True, so
        # arm on the first True instead of bailing on the initial False
        # (that race finalized every episode at 0 frames). Give up if the
        # loop never starts within 5s (e.g. start failed).
        armed = False
        t0 = time.time()
        while not self._closed:
            live = self.running_cb() if self.running_cb is not None else True
            if not armed:
                if live:
                    armed = True
                    nxt = time.time()
                elif time.time() - t0 < 5.0:
                    time.sleep(0.05)
                    continue
                else:
                    _log.warning("EvalRecorder: loop never started — no video")
                    break
            elif not live:
                break
            ext = self.cam_mgr.get_bgr("wrist_1")
            wrist = self.cam_mgr.get_bgr("wrist_2")
            tile = self._tile(ext, wrist)
            if tile is not None:
                with self._lock:
                    if self._writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        self._writer = cv2.VideoWriter(
                            str(self.out_dir / "video.mp4"), fourcc,
                            self.fps, (tile.shape[1], tile.shape[0]))
                        if not self._writer.isOpened():
                            _log.error("VideoWriter open failed — recording off")
                            self._writer = None
                            return
                    self._writer.write(tile)
                    self.frames += 1
                    self._frame_ts.append(time.time())
            nxt += dt
            delay = nxt - time.time()
            if delay > 0:
                time.sleep(min(delay, 1.0))
            else:
                nxt = time.time()
        self._finalize()

    def log_step(self, rec: dict) -> None:
        """Append one policy step to traj.jsonl (called from the eval loop).

        `rec` should carry at least t / iter / q / grip / actions; anything
        JSON-serialisable is kept verbatim. Silently dropped once finalized.
        """
        line = json.dumps(rec, default=float)
        with self._lock:
            if self._closed:
                return
            try:
                if self._traj_f is None:
                    self._traj_f = (self.out_dir / "traj.jsonl").open("a", encoding="utf-8")
                self._traj_f.write(line + "\n")
                self.steps_logged += 1
            except Exception as exc:
                _log.warning(f"traj write failed: {exc}")

    def _finalize(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._writer is not None:
                try:
                    self._writer.release()
                except Exception:
                    pass
                self._writer = None
            if self._traj_f is not None:
                try:
                    self._traj_f.close()
                except Exception:
                    pass
                self._traj_f = None
            try:
                (self.out_dir / "frame_times.json").write_text(
                    json.dumps([round(t, 4) for t in self._frame_ts]), encoding="utf-8")
            except Exception as exc:
                _log.warning(f"frame_times write failed: {exc}")
        meta = {
            "ep_id": self.ep_id,
            "task": self.task_id,
            "prompt": self.prompt,
            "ckpt": self.ckpt,
            "layout": self.layout,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S",
                                        time.localtime(self.started)),
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_s": round(time.time() - self.started, 1),
            "steps": self.steps_source() if self.steps_source else 0,
            "fps": self.fps,
            "frames": self.frames,
            "t0": round(self.started, 4),          # epoch; traj/frame times are absolute
            "traj_steps": self.steps_logged,
            "abort": str((self.abort_source() if self.abort_source else None) or ""),
            "mark": "",
            "note": "",
        }
        try:
            (self.out_dir / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            _log.info("episode recorded: %s/%s (%d frames, %d steps)",
                      self.task_id, self.ep_id, self.frames, meta["steps"])
        except Exception as exc:
            _log.warning(f"episode meta write failed: {exc}")


INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8">
<title>TASL FR3 — Eval Console</title>
<style>
  :root {
    --bg: #121212; --surface: #1E1E1E; --surface-2: #26262A;
    --border: #38383C; --border-soft: #2C2C30;
    --text: #E6E1E5; --dim: #9E9E9E; --faint: #6E6E73;
    --primary: #8AB4F8; --primary-dim: #16283F;
    --ok: #81C995; --warn: #FFB74D; --err: #F28B82;
    --mono: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace;
  }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue",
         "Noto Sans SC", "PingFang SC", sans-serif;
         margin: 0; padding: 16px 20px 40px; background: var(--bg);
         color: var(--text); font-size: 14px; line-height: 1.45; }
  h1 { font-size: 20px; font-weight: 600; margin: 0; letter-spacing: -0.2px; }
  h3 { font-size: 14px; font-weight: 600; margin: 0 0 8px; }
  .eyebrow { font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px;
             color: var(--dim); margin: 14px 0 6px; }
  .card { background: var(--surface); border: 1px solid var(--border-soft);
          border-radius: 10px; padding: 14px 16px; margin-bottom: 12px; }
  .row { display: flex; gap: 12px; align-items: flex-start; flex-wrap: wrap; }
  .col { flex: 1 1 0; min-width: 300px; }
  /* header */
  header { display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
           margin-bottom: 14px; }
  header .sub { color: var(--dim); font-size: 13px; }
  .chips { display: flex; gap: 8px; flex-wrap: wrap; margin-left: auto; }
  .chip { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px;
          border-radius: 16px; font-size: 12.5px; border: 1px solid var(--border);
          background: var(--surface); color: var(--dim); }
  .chip .dot { width: 8px; height: 8px; border-radius: 50%;
               background: var(--faint); }
  .chip.ok { color: var(--text); border-color: #2E4A36; }
  .chip.ok .dot { background: var(--ok); }
  .chip.bad { color: var(--text); border-color: #4A2E2E; }
  .chip.bad .dot { background: var(--err); }
  .chip.busy { color: var(--text); border-color: #4A3D2E; }
  .chip.busy .dot { background: var(--warn); }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
         margin-right: 6px; }
  .ok-dot { background: var(--ok); } .bad-dot { background: var(--err); }
  /* buttons */
  button { padding: 7px 14px; font-size: 13px; cursor: pointer;
           border-radius: 6px; border: 1px solid var(--border);
           background: var(--surface-2); color: var(--text); margin: 3px 6px 3px 0;
           transition: background 0.12s; }
  button:hover { background: #31313A; }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  button.filled { background: var(--primary-dim); border-color: #2C4A72;
                  color: var(--primary); font-weight: 500; }
  button.filled:hover { background: #1E3554; }
  button.danger { background: #3A2323; border-color: #5A3232; color: var(--err); }
  button.warn   { background: #3A3223; border-color: #5A4A32; color: var(--warn); }
  button.tiny { padding: 4px 10px; font-size: 12px; }
  /* inputs */
  input[type=text], input[type=number], select {
    padding: 7px 9px; font-size: 13px; background: #0F0F0F; color: var(--text);
    border: 1px solid var(--border); border-radius: 6px; }
  input[type=text] { width: 100%; }
  input[type=number] { width: 64px; }
  input[type=range] { accent-color: var(--primary); }
  label { color: var(--dim); font-size: 12.5px; }
  .hint { font-size: 12px; color: var(--faint); margin-top: 4px; }
  /* cameras */
  .cam { background: #000; padding: 8px; border-radius: 10px;
         border: 1px solid var(--border-soft); }
  .cam img { width: 100%; max-width: 520px; display: block; border-radius: 6px; }
  .cam .label { font-size: 12px; color: var(--dim); padding: 5px 0; }
  .ghostwrap { position: relative; width: fit-content; }
  .ghostwrap img.ghost { position: absolute; top: 0; left: 0; width: 100%;
                         height: 100%; opacity: 0.45; pointer-events: none;
                         display: none; }
  .policy-tile { width: 224px; }
  .policy-tile img { width: 224px; height: 224px; image-rendering: pixelated; }
  /* status table */
  table { font-size: 13px; border-collapse: collapse; width: 100%; }
  td, th { padding: 4px 10px 4px 0; text-align: left; vertical-align: top;
           border-bottom: 1px solid var(--border-soft); }
  td:first-child { color: var(--dim); white-space: nowrap; }
  /* console */
  .console { background: #0A0A0A; border: 1px solid var(--border-soft);
             border-radius: 8px; padding: 10px; font-family: var(--mono);
             font-size: 12px; max-height: 320px; overflow: auto;
             color: #C9C9C9; white-space: pre-wrap; word-break: break-all; }
  .console .ts { color: var(--faint); }
  .console .lvl-I { color: #8AB4F8; } .console .lvl-E { color: var(--err); }
  .console .lvl-W { color: var(--warn); }
  /* toasts */
  #toasts { position: fixed; top: 16px; right: 16px; z-index: 50;
            display: flex; flex-direction: column; gap: 8px; max-width: 380px; }
  .toast { padding: 10px 14px; border-radius: 8px; font-size: 13px;
           background: var(--surface-2); border: 1px solid var(--border);
           box-shadow: 0 4px 16px rgba(0,0,0,0.5); animation: slide-in 0.18s; }
  .toast.ok  { border-left: 3px solid var(--ok); }
  .toast.err { border-left: 3px solid var(--err); }
  .toast.info{ border-left: 3px solid var(--primary); }
  @keyframes slide-in { from { transform: translateX(24px); opacity: 0; }
                        to { transform: none; opacity: 1; } }
  /* member lab modules */
  details.member { background: var(--surface); border: 1px solid var(--border-soft);
                   border-radius: 10px; margin-bottom: 12px; overflow: hidden; }
  details.member > summary { cursor: pointer; list-style: none; padding: 12px 16px;
                             font-size: 13.5px; font-weight: 500;
                             color: var(--dim); user-select: none; }
  details.member > summary::-webkit-details-marker { display: none; }
  details.member > summary::before { content: "▸"; margin-right: 8px;
                                     color: var(--faint); }
  details.member[open] > summary::before { content: "▾"; }
  details.member > summary .tag { float: right; font-size: 11px; color: var(--faint);
                                  font-weight: 400; }
  details.member > .member-body { padding: 0 16px 14px; border-top: 1px solid
                                  var(--border-soft); }
  details.member .eyebrow:first-child { margin-top: 10px; }
  /* Episodes: per-task collapsible groups */
  details.epgrp { border: 1px solid var(--border-soft); border-radius: 8px;
                  margin: 6px 0; overflow: hidden; }
  details.epgrp > summary { cursor: pointer; list-style: none; padding: 7px 10px;
                            font-size: 13px; user-select: none;
                            display: flex; gap: 10px; align-items: baseline;
                            flex-wrap: wrap; }
  details.epgrp > summary::-webkit-details-marker { display: none; }
  details.epgrp > summary::before { content: "▸"; color: var(--faint); }
  details.epgrp[open] > summary::before { content: "▾"; }
  details.epgrp > summary .sr { margin-left: auto; font-variant-numeric: tabular-nums;
                                white-space: nowrap; }
  details.epgrp > .epgrp-body { padding: 0 10px 8px; border-top: 1px solid
                                var(--border-soft); overflow-x: auto; }
  details.epgrp .byckpt { font-size: 11.5px; color: var(--faint); padding: 4px 0 2px; }
  .flexbar { display: flex; gap: 6px; align-items: center; flex-wrap: wrap;
             margin: 6px 0; }
</style></head>
<body>
<header>
  <div>
    <h1>TASL FR3 · Eval Console</h1>
    <div class="sub">openpi pi05_droid LoRA · 10 checkpoints · polymetis zerorpc</div>
  </div>
  <div class="chips" id="chips">
    <span class="chip" id="chip-robot"><span class="dot"></span>robot</span>
    <span class="chip" id="chip-gripper"><span class="dot"></span>gripper</span>
    <span class="chip" id="chip-cams"><span class="dot"></span>cams</span>
    <span class="chip" id="chip-eval"><span class="dot"></span>eval</span>
    <span class="chip" id="chip-ckpt"><span class="dot"></span>policy</span>
    <span class="chip" id="chip-motion" title="policy 静止 / 执行中 / 有指令但机械臂没动 — 最近 1.5 s 的下发位移 vs 关节实际位移 + polymetis 时间戳"><span class="dot"></span><span id="chip-motion-txt">motion</span></span>
  </div>
</header>

<div class="card">
  <b>Policy checkpoint</b>
  <div class="flexbar">
    <select id="ckptSel"></select>
    <button class="filled" onclick="switchCkpt(false)">Load w/o RTC</button>
    <button class="filled" onclick="switchCkpt(true)">Load with RTC</button>
    <button class="warn" onclick="api('/ckpt/stop')">Stop serve</button>
    <span id="ckptState" style="font-size:13px;color:var(--primary)"></span>
  </div>
  <pre class="console" id="ckptLog" style="max-height:120px;margin-top:6px;display:none"></pre>
</div>
<!-- rtc:panel -->

<div class="row">
  <div class="col cam">
    <div class="label">exterior — full feed (HD720) [wrist_1]</div>
    <div class="ghostwrap">
      <img src="/cam/wrist_1.mjpg" alt="exterior"/>
      <img id="ghostImg" class="ghost" alt="task mask ghost"/>
    </div>
  </div>
  <div class="col cam">
    <div class="label">wrist — full feed (HD720) [wrist_2]</div>
    <img src="/cam/wrist_2.mjpg" alt="wrist"/>
  </div>
</div>

<div class="row" style="margin-top:12px">
  <div class="col cam policy-tile">
    <div class="label">policy view — exterior (<span class="pbcLabel">224² pad</span>)</div>
    <img id="polExt" src="/cam/wrist_1_policy.mjpg?mode=pad" alt="exterior policy"/>
  </div>
  <div class="col cam policy-tile">
    <div class="label">policy view — wrist (<span class="pbcLabel">224² pad</span>)</div>
    <img id="polWrist" src="/cam/wrist_2_policy.mjpg?mode=pad" alt="wrist policy"/>
  </div>
</div>

<div class="card" id="taskCard">
  <h3>Task</h3>
  <div class="flexbar">
    <select id="taskSel" onchange="onTaskSelect()" style="flex:1;min-width:200px"></select>
    <button class="filled tiny" onclick="taskNewOpen()">＋ New task</button>
    <button class="danger tiny" onclick="deleteTask()">删除</button>
    <span id="taskInfo" style="font-size:12px;color:var(--faint)"></span>
  </div>
  <div class="flexbar" style="margin-top:6px">
    <label style="min-width:56px">Prompt</label>
    <input type="text" id="prompt" readonly placeholder="选择任务后自动填充(仅 New task 可自定义)" style="flex:1"/>
  </div>
  <div class="flexbar">
    <label style="min-width:56px">Layout</label>
    <select id="taskLayoutSel" onchange="taskGhostPick()" style="min-width:160px">
      <option value="">—</option>
    </select>
    <label style="margin-left:10px"><input type="checkbox" id="ghostOn" onchange="ghostUpdate()"/> ghost 蒙版</label>
    <input type="range" id="ghostAlpha" min="10" max="90" value="45"
           style="width:90px" oninput="ghostUpdate()"/>
  </div>
  <div id="taskForm" style="display:none;margin-top:6px;border:1px solid var(--border-soft);border-radius:6px;padding:8px">
    <div class="flexbar">
      <input type="text" id="taskId" placeholder="task id(留空自动生成)" style="width:30%"/>
      <input type="text" id="taskPrompt" placeholder="prompt(语言指令)" style="flex:1"/>
    </div>
    <div class="flexbar">
      <label>layout 蒙版</label>
      <select id="taskLayout"></select>
      <button class="tiny" onclick="takeTaskLayout()">📷 拍当前画面为蒙版</button>
    </div>
    <div class="flexbar">
      <button class="filled tiny" onclick="saveNewTask()">保存新任务</button>
      <button class="tiny" onclick="taskNewClose()">取消</button>
    </div>
    <div class="hint">只有 New task 能自定义 prompt / 拍摆放蒙版。任务库与采集端共享
      (tasl/tasks_store.json;蒙版存 rlinf_data/layouts,两个 portal 同一份)。</div>
  </div>
</div>

<div class="row" style="margin-top:16px;align-items:stretch">
  <div class="col card" style="margin-bottom:0">
    <h3>Eval</h3>
    <div class="flexbar" style="margin-top:8px">
      <button class="filled" onclick="startEval()">▶ Start</button>
      <button class="danger" onclick="stopEval()">■ Stop</button>
      <label title="勾选 = center-crop 224² (pbc 数据训的 ckpt);不勾 = DROID resize+pad (默认)"
             style="margin-left:4px"><input type="checkbox" id="pbcOn" onchange="pbcUpdate()"/> pbc</label>
      <button class="filled" onclick="evalMark('success')">✓ Mark success</button>
      <button class="danger" onclick="evalMark('fail')">✗ Mark fail</button>
      <button onclick="api('/home')">⌂ Go home</button>
      <button class="warn" onclick="nucRestart()" title="机械臂不跟指令 / robot+gripper DOWN 时:重启 NUC 上的 polymetis 容器并重新 bootstrap(约 25 s,需 FCI 已激活)">🔧 Reset NUC</button>
      <button class="warn" onclick="commitGrasp()">Commit grasp</button>
    </div>
    <div class="flexbar">
      <button id="btnSaveVideo" onclick="saveVideo()" disabled
              title="手动导出最新一条录制(Mark ✓/✗ 已经自动导出)— 视频(exterior|wrist)+meta+traj → saved_demo/<task>[-ood]/<layout>_rNN.*">💾 Save demo</button>
      <button onclick="saveLayout()"
              title="导出当前选中 layout 的参考图(exterior/wrist jpg + json)→ saved_demo/<task>/">🖼 Save png</button>
      <button onclick="saveOodLayout()"
              title="把当前相机画面拍成该任务的 OOD layout(<task>-OODn),挂到任务上并作为 ghost;之后的 episode meta 记录该 layout">🌀 Save as OOD layout</button>
      <span id="saveInfo" style="font-size:12px;color:var(--faint)"></span>
    </div>
    <div class="hint">Start 前先在上方选择任务;Mark ✓/✗ 写入最新一条 eval 录制的
      meta 并<b>自动导出</b>视频到 RLinf/saved_demo/&lt;task&gt;[-ood]/&lt;layout&gt;_rNN.*
      (layout 名含 ood 时进 -ood 目录;NN = 该 layout 的第几条 rollout;Stop 不导出,
      aborted 不导出)。Save demo 手动导出最新一条;Save png 导出当前选中
      layout 的参考图到 saved_demo/&lt;task&gt;/。Save as OOD layout
      把当前桌面拍成 &lt;task&gt;-OODn 蒙版并挂到任务上(OOD eval 可复现)。
      Commit grasp = 停策略 → 夹爪闭合 → 上提 5cm(force_open 调试用)</div>
    <span id="evalMarkInfo" style="font-size:12px;color:var(--faint)"></span>

    <div class="eyebrow">Robot</div>
    <div class="flexbar">
      <button class="filled" onclick="setHome()">Set current as home</button>
      <button class="warn" onclick="api('/recover')">Recover</button>
      <button class="danger" onclick="api('/robot_stop')">Robot stop</button>
    </div>
    <div class="flexbar">
      <button class="warn" onclick="api('/freedrive',{enable:true})">Unlock joints</button>
      <button class="filled" onclick="api('/freedrive',{enable:false})">Lock joints</button>
    </div>
    <div class="hint">Unlock 后关节变柔顺可手推;跑 eval 前务必 Lock 回来。</div>

    <details>
      <summary style="cursor:pointer;color:var(--dim);font-size:13px;margin-top:10px">Jog(cartesian,EE 系)</summary>
      <div class="flexbar">
        <label>step</label>
        <select id="jogStep">
          <option value="0.005">5 mm</option>
          <option value="0.01" selected>1 cm</option>
          <option value="0.03">3 cm</option>
          <option value="0.05">5 cm</option>
        </select>
        <label style="margin-left:10px">rot</label>
        <select id="jogRotStep">
          <option value="0.0873">5°</option>
          <option value="0.1745" selected>10°</option>
          <option value="0.3491">20°</option>
        </select>
      </div>
      <div class="flexbar">
        <button class="tiny" onclick="jog('x',+1)">+X</button><button class="tiny" onclick="jog('x',-1)">−X</button>
        <button class="tiny" onclick="jog('y',+1)">+Y</button><button class="tiny" onclick="jog('y',-1)">−Y</button>
        <button class="tiny" onclick="jog('z',+1)">+Z</button><button class="tiny" onclick="jog('z',-1)">−Z</button>
        <button class="tiny" onclick="jogRot('rx',+1)">+rx</button><button class="tiny" onclick="jogRot('rx',-1)">−rx</button>
        <button class="tiny" onclick="jogRot('ry',+1)">+ry</button><button class="tiny" onclick="jogRot('ry',-1)">−ry</button>
        <button class="tiny" onclick="jogRot('rz',+1)">+rz</button><button class="tiny" onclick="jogRot('rz',-1)">−rz</button>
      </div>
    </details>

    <details>
      <summary style="cursor:pointer;color:var(--dim);font-size:13px;margin-top:10px">Gripper</summary>
      <div class="flexbar">
        <select id="gripMode" onchange="setGripperMode()">
          <option value="proportional">proportional(DROID 精确)</option>
          <option value="raw_binary">raw_binary(legacy 二值)</option>
          <option value="latch_close">latch_close</option>
          <option value="force_open">force_open(测试手臂用)</option>
          <option value="raw_inverted">raw_inverted(符号翻转 A/B)</option>
          <option value="latch_inverted">latch_inverted</option>
        </select>
        <button onclick="api('/gripper',{action:'open'})">Open</button>
        <button onclick="api('/gripper',{action:'close'})">Close</button>
      </div>
    </details>
  </div>

  <div class="col card" style="margin-bottom:0">
    <h3>Live status</h3>
    <div id="status">loading…</div>
    <div class="eyebrow">Console(后端日志 + 事件)</div>
    <pre class="console" id="console">— waiting for events —</pre>
  </div>
</div>

<div class="card" id="actionsCard">
  <h3>Action chunks(policy 每次推理输出的原始 chunk)
    <span class="hint" id="actionsHint" style="font-weight:normal;margin-left:8px">— no inference yet —</span></h3>
  <pre class="console" id="actionsBox" style="max-height:280px">— waiting for the first inference —</pre>
</div>

<div class="card">
  <h3>Episodes(eval 视频)
    <span style="font-size:12px;color:var(--faint);font-weight:400">— 每次 Stop 自动录制双视角拼接视频,按 task 分类</span></h3>
  <div class="flexbar">
    <label>task 筛选</label>
    <select id="epTaskFilter" onchange="epRefresh()"></select>
    <button class="tiny" onclick="epRefresh()">↻</button>
    <span id="epCount" style="font-size:12px;color:var(--faint)"></span>
  </div>
  <div id="epTable" style="overflow-x:auto"></div>
  <div id="epEditForm" style="display:none;margin-top:8px;border:1px solid var(--border-soft);border-radius:6px;padding:8px">
    <div class="flexbar">
      <input type="text" id="epEditTask" placeholder="task" style="width:30%"/>
      <input type="text" id="epEditPrompt" placeholder="prompt" style="flex:1"/>
    </div>
    <div class="flexbar">
      <input type="text" id="epEditCkpt" placeholder="ckpt" style="width:30%"/>
      <label style="margin-left:8px">note</label>
      <input type="text" id="epEditNote" placeholder="备注" style="flex:1"/>
    </div>
    <div class="flexbar">
      <button class="filled tiny" onclick="saveEp()">保存 meta</button>
      <button class="tiny" onclick="document.getElementById('epEditForm').style.display='none'">取消</button>
    </div>
  </div>
</div>

<div id="videoModal" style="display:none;position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.85);z-index:99;flex-direction:column;align-items:center;justify-content:center">
  <video id="videoPlayer" controls style="width:min(92vw,1100px);background:#000"></video>
  <button style="margin-top:12px" onclick="closeVideo()">✕ 关闭</button>
</div>

<details class="member">
  <summary>🧪 成员实验区 · 之旭 — Steering(运行时注入实验)
    <span class="tag">折叠不影响共用功能 · 默认收起</span></summary>
  <div class="member-body">
    <div class="eyebrow">Task masks(物品摆放参考快照)</div>
    <div class="flexbar">
      <input type="text" id="maskName" placeholder="mask id (task1…)" style="width:130px"/>
      <button onclick="takeMask()">📷 Take mask</button>
      <select id="maskSel" onchange="ghostUpdate()"><option value="">(no mask)</option></select>
      <button onclick="deleteMask()">🗑</button>
    </div>
    <div class="hint">实验区私有 mask;ghost 开关在上方 Task 区(任务未挂 layout 时回落到这里选的 mask)。</div>

    <div class="eyebrow">Clean/blank 配对(标签进 episode.json,setting1-Ra)</div>
    <div class="flexbar">
      <button onclick="genFiller()">⇢ gen filler</button>
      <button id="useCleanBtn" onclick="usePrompt('clean')" disabled>clean</button>
      <button id="useFillerBtn" onclick="usePrompt('cor')" disabled>blank</button>
      <span id="fillerInfo" style="font-size:12px;color:var(--faint)"></span>
    </div>
    <div class="flexbar">
      <select id="expMode">
        <option value="trial">trial(探索任务—不进分析)</option>
        <option value="setting1-Ra">setting1-Ra(clean/blank 配对,只采集)</option>
        <option value="setting1-Rr">setting1-Rr(blank + 运行时 patch,R_rollout)</option>
        <option value="setting2-Ra">setting2-Ra(运行时 patch,R_action)</option>
        <option value="setting2-Rr">setting2-Rr(逐层 patch,R_rollout)</option>
        <option value="none">no tag</option>
      </select>
      <label>pair</label><input type="number" id="pairNum" min="1" max="9" value="1"/>
      <select id="roleSel">
        <option value="clean">clean</option>
        <option value="cor">cor (blank)</option>
      </select>
    </div>
    <div class="flexbar">
      <button class="filled" onclick="markEp(true)">✓ mark success</button>
      <button class="danger" onclick="markEp(false)">✗ mark fail</button>
      <span id="markInfo" style="font-size:12px;color:var(--faint)"></span>
    </div>
    <div class="hint">把判定写进最新采集 episode 的 episode.json;配对只有 clean✓ + blank✗ 才计数。</div>

    <div class="eyebrow">Runtime patching(setting1-Rr)</div>
    <div id="steerBanner" style="padding:6px 10px;border-radius:6px;background:#26262A;
         color:var(--dim);font-size:12.5px;margin-bottom:6px">steering: loading…</div>
    <div class="flexbar">
      <label>source pair</label>
      <select id="srcEp" style="max-width:320px"></select>
      <button class="tiny" onclick="steerEpsRefresh()">↻</button>
    </div>
    <div class="flexbar">
      <label>layer</label>
      <input type="text" id="injLayer" value="12" size="8"
             title="int, 逗号列表(0,5,12),或 all"/>
      <label>target</label>
      <select id="injTarget">
        <option value="all">all</option>
        <option value="img_tokens">img</option>
        <option value="text_tokens">text(整段 [768:968),含 state token)</option>
      </select>
      <label style="margin-left:6px"><input type="checkbox" id="saveEp"/> save rollout episode</label>
    </div>
    <div class="flexbar">
      <button class="warn" id="applySteerBtn" onclick="applySteer()">Apply steering config</button>
      <span id="steerInfo" style="font-size:12px;color:var(--faint)"></span>
    </div>
    <div class="hint">Apply 改写 watched YAML,下次 Start 生效;rollout 运行中按钮禁用。第一次跑新条件手放 e-stop。</div>
    <div style="margin-top:8px">
      <b style="font-size:13px">Live tally(setting1-Rr)</b>
      <div id="steerTally" style="font-size:12.5px;color:#ccc;margin-top:3px">(no patched rollouts yet)</div>
    </div>

    <div class="eyebrow">Gripper audit(每迭代 4 路信号)</div>
    <pre id="audit" class="console" style="max-height:220px">(idle — start the policy)</pre>
    <div class="hint">列:iter · raw_action_grip · want_close · action_taken · skip_reason · gw_pre_m → gw_post_m · obs_norm</div>
  </div>
</details>

<script>
// 注意:不要给这个变量命名 `prompt` — 会遮蔽 window.prompt 破坏内联 onclick。
const promptInput = document.getElementById('prompt');

// ── 反馈系统:toast + console ──────────────────────────────────────
const CONSOLE_LINES = [];
function consoleLine(text, cls) {
  const ts = new Date().toTimeString().slice(0, 8);
  CONSOLE_LINES.push({ts, text, cls});
  if (CONSOLE_LINES.length > 250) CONSOLE_LINES.shift();
  const el = document.getElementById('console');
  if (!el) return;
  const pinned = el.scrollTop + el.clientHeight >= el.scrollHeight - 24;
  el.innerHTML = CONSOLE_LINES.map(l =>
    '<span class="ts">[' + l.ts + ']</span> ' +
    '<span class="' + (l.cls || '') + '">' + l.text + '</span>').join('\\n');
  if (pinned) el.scrollTop = el.scrollHeight;
}
let _toastTimer = null;
function toast(msg, kind) {
  let box = document.getElementById('toasts');
  if (!box) { box = document.createElement('div'); box.id = 'toasts';
              document.body.appendChild(box); }
  const t = document.createElement('div');
  t.className = 'toast ' + (kind || 'info');
  t.textContent = msg;
  box.appendChild(t);
  while (box.children.length > 4) box.removeChild(box.firstChild);
  setTimeout(() => t.remove(), 4500);
}
function esc(s) {
  return String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

async function api(path, body) {
  const r = await fetch(path, {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: body ? JSON.stringify(body) : '{}',
  });
  let j;
  // 500 会渲染 Flask 的 HTML 错误页 — 转成可读消息而不是让按钮卡在 busy。
  try { j = await r.json(); }
  catch (e) { j = {ok: false, msg: 'HTTP ' + r.status + '(see dashboard log)'}; }
  console.log(path, j);
  const name = path.replace(/^\\//, '');
  if (j.ok === false) toast('✗ ' + name + ': ' + (j.msg || 'failed'), 'err');
  else if (j.msg !== undefined) toast(name + ': ' + j.msg, 'ok');
  else if (j.result !== undefined) toast(name + ': ' + j.result, 'ok');
  else toast(name + ' ✓', 'ok');
  return j;
}

// pbc checkbox: swap the policy-view previews to the same preprocessing the
// next Start will use (pad = DROID default, crop = pbc center-crop).
function pbcUpdate() {
  const crop = document.getElementById('pbcOn').checked;
  const mode = crop ? 'crop' : 'pad';
  document.getElementById('polExt').src = '/cam/wrist_1_policy.mjpg?mode=' + mode;
  document.getElementById('polWrist').src = '/cam/wrist_2_policy.mjpg?mode=' + mode;
  document.querySelectorAll('.pbcLabel').forEach(e => e.textContent = '224² ' + mode);
  try { localStorage.setItem('pbcOn', crop ? '1' : '0'); } catch (e) {}
}
try {
  if (localStorage.getItem('pbcOn') === '1') {
    document.getElementById('pbcOn').checked = true; pbcUpdate();
  }
} catch (e) {}
async function nucRestart() {
  if (!confirm('重启 NUC 上的 polymetis 控制器容器并重新 bootstrap?(约 25 s;确认 Desk 里 FCI 已激活)')) return;
  toast('🔧 NUC controller restarting… (~25 s)', 'ok');
  const j = await api('/nuc_restart');
  if (j && j.ok) consoleLine('nuc_restart: ' + (j.log || '').split(String.fromCharCode(10)).join(' | '), 'lvl-I');
}
async function startEval() {
  if (!document.getElementById('taskSel').value) {
    toast('先在 Task 区选择任务(或 New task 新建)', 'err'); return;
  }
  const p = (promptInput && promptInput.value) ? promptInput.value : 'grasp';
  const j = await api('/start', {
    prompt: p,
    mode: document.getElementById('expMode').value,
    task: document.getElementById('maskSel').value,
    task_id: document.getElementById('taskSel').value,
    layout: document.getElementById('taskLayoutSel').value,
    pair: parseInt(document.getElementById('pairNum').value) || 0,
    role: document.getElementById('roleSel').value,
    pbc: document.getElementById('pbcOn').checked,
  });
  if (j && j.ok !== false) setSaveVideo(false);
  return j;
}

async function stopEval() {
  const j = await api('/stop');
  if (j && j.ok !== false) setSaveVideo(true);
  return j;
}

// ── Demo 导出(RLinf/saved_demo)──────────────────────────────────────
// Save video 只在 Stop / Mark 之后可点,下次 Start 重新禁用;Save layout 全程可点。
function setSaveVideo(on) {
  document.getElementById('btnSaveVideo').disabled = !on;
}
async function saveVideo() {
  const j = await api('/save_video');
  if (j.ok) document.getElementById('saveInfo').textContent = j.msg;
}
async function saveLayout() {
  const t = taskList.find(x => x.id === document.getElementById('taskSel').value);
  if (!t) { toast('先在 Task 区选择任务', 'err'); return; }
  const lay = document.getElementById('taskLayoutSel').value;
  if (!lay) { toast('任务「' + t.id + '」没有 layout', 'err'); return; }
  const j = await api('/save_layout', {layout: lay, task_id: t.id});
  if (j.ok) document.getElementById('saveInfo').textContent = j.msg;
}
// 当前桌面 → 新 OOD layout(<task>-OODn),挂到任务并立即作为 ghost。
async function saveOodLayout() {
  const tid = document.getElementById('taskSel').value;
  if (!tid) { toast('先在 Task 区选择任务', 'err'); return; }
  const j = await api('/save_ood_layout', {task_id: tid});
  if (!j || !j.ok) return;
  await taskRefresh();
  const t = taskList.find(x => x.id === tid);
  const lays = (t && t.layouts && t.layouts.length) ? t.layouts : [j.id];
  const laySel = document.getElementById('taskLayoutSel');
  laySel.innerHTML = lays.map(l => '<option value="' + l + '">' + l + '</option>').join('');
  laySel.value = j.id;
  taskGhostLayout = j.id;
  document.getElementById('ghostOn').checked = true;
  ghostUpdate();
  document.getElementById('taskInfo').textContent = 'layouts: ' + lays.join(', ')
    + ' · datasets: ' + (((t || {}).datasets || []).join(', ') || '—');
  document.getElementById('saveInfo').textContent = j.msg;
  toast('🌀 ' + j.msg, 'ok');
}

// 通用 eval 打分:一键 停止 → 等录制落盘 → 写 mark → 回 home(服务端 /eval/mark)。
// 之旭实验区的 markEp 走 /mark(hooked-capture episode.json),互不影响。
let evalMarkBusy = false;
async function evalMark(mark) {
  if (evalMarkBusy) return;
  evalMarkBusy = true;
  const el = document.getElementById('evalMarkInfo');
  el.textContent = (mark === 'success' ? '✓' : '✗') + ' 停止 + 记录 + 回 home…';
  try {
    const j = await api('/eval/mark', {mark});
    if (j && j.ok) {
      const shown = j.mark === 'aborted'
        ? '⚠ aborted (NUC not executing) — not counted'
        : (mark === 'success' ? '✓ success' : '✗ fail');
      el.textContent = (j.task ? j.task + '/' : '') + j.ep_id + ' → ' + shown
        + (j.stopped ? ' · stopped' : '') + ' · ' + (j.home || '');
      if (j.mark === 'aborted') toast('⚠ episode was aborted by the watchdog → marked "aborted", not ' + mark, 'err');
      if (j.home && j.home !== 'homed') toast('⌂ ' + j.home, 'err');
      const si = document.getElementById('saveInfo');
      if (j.demo && j.demo.error) { si.textContent = '✗ demo: ' + j.demo.error; toast('✗ demo save failed: ' + j.demo.error, 'err'); }
      else if (j.demo) { si.textContent = '💾 ' + j.demo.msg; }
      setSaveVideo(true);
      epRefresh();
    } else {
      el.textContent = 'mark failed: ' + ((j && j.msg) || '');
    }
  } finally { evalMarkBusy = false; }
}

// ── blank-filler ──
let fillerPair = null;
promptInput.addEventListener('input', () => {
  if (!fillerPair) return;
  const v = promptInput.value.trim();
  if (v !== fillerPair.clean && v !== fillerPair.filler) {
    fillerPair = null;
    document.getElementById('useCleanBtn').disabled = true;
    document.getElementById('useFillerBtn').disabled = true;
    document.getElementById('fillerInfo').textContent = 'prompt changed — regenerate filler';
  }
});
async function genFillerFor(p) {
  const el = document.getElementById('fillerInfo');
  el.textContent = 'generating… (tokenizer load, a few s)';
  const j = await api('/filler', {prompt: p});
  if (!j.ok) { el.textContent = 'filler failed: ' + j.msg; return null; }
  fillerPair = {clean: j.prompt, filler: j.filler};
  el.textContent = 'K=' + j.K + ' blank: "' + j.filler + '"';
  document.getElementById('useCleanBtn').disabled = false;
  document.getElementById('useFillerBtn').disabled = false;
  return fillerPair;
}
async function genFiller() {
  const p = promptInput.value.trim();
  if (!p) { alert('先输入 clean prompt'); return; }
  await genFillerFor(p);
}
function usePrompt(which) {
  if (!fillerPair) return;
  promptInput.value = which === 'cor' ? fillerPair.filler : fillerPair.clean;
  document.getElementById('roleSel').value = which;
}
async function markEp(success) {
  const j = await api('/mark', {success});
  const el = document.getElementById('markInfo');
  if (j.ok) {
    const t = j.tag ? ' [' + (j.tag.mode||'') + ' ' + (j.tag.task||'') + ' p' +
               (j.tag.pair||'?') + ' ' + (j.tag.role||'') + ']' : '';
    el.textContent = j.episode + t + ' → ' + (success ? '✓ success' : '✗ fail');
    steerTallyRefresh();
  } else {
    el.textContent = 'mark failed: ' + j.msg;
  }
}

// ── steering(setting1-Rr)─ watched YAML 重写 ──
async function steerEpsRefresh() {
  try {
    const r = await fetch('/steer/eps'); const j = await r.json();
    const sel = document.getElementById('srcEp');
    const cur = sel.value;
    sel.innerHTML = '';
    (j.eps || []).forEach(e => {
      const o = document.createElement('option');
      o.value = e.dir;
      o.textContent = e.task + ' · pair ' + e.pair + ' · ' + (e.n_steps ?? '?') + ' steps';
      sel.appendChild(o);
    });
    if (cur && [...sel.options].some(o => o.value === cur)) sel.value = cur;
  } catch(e) {}
}
async function steerCurrentRefresh() {
  try {
    const r = await fetch('/steer/current'); const j = await r.json();
    const b = document.getElementById('steerBanner');
    if (!j.ok) { b.textContent = 'steering: config unreadable — ' + j.msg; return; }
    if (j.steer_on) {
      b.style.background = '#3A2B10'; b.style.color = '#FFB74D';
      b.innerHTML = '⚡ <b>STEER MODE</b> — L' + JSON.stringify(j.layer)
        + ' · ' + j.target + ' · src ' + j.source_task + ' pair '
        + (j.source_pair ?? '?') + ' (' + (j.source_n_steps ?? '?') + ' steps)'
        + (j.save_episode ? ' · saving episodes' : ' · NOT saving episodes');
    } else {
      b.style.background = '#26262A'; b.style.color = '#9E9E9E';
      b.textContent = 'steering: OFF(当前配置无注入)';
    }
  } catch(e) {}
}
async function steerTallyRefresh() {
  try {
    const r = await fetch('/steer/tally'); const j = await r.json();
    const el = document.getElementById('steerTally');
    if (!j.ok || !(j.rows || []).length) {
      el.textContent = '(no patched rollouts yet)'; return;
    }
    el.innerHTML = j.rows.map(t => {
      const n = t.ok + t.fail;
      return t.task + ' · pair ' + (t.pair ?? '?') + ' · L'
        + JSON.stringify(t.layer) + ':  <b style="color:#81C995">✓' + t.ok
        + '</b> <b style="color:#F28B82">✗' + t.fail + '</b>'
        + (t.unmarked ? ' <span style="color:#E5C07B">·' + t.unmarked
                        + ' unmarked</span>' : '')
        + '  <span style="color:var(--faint)">[' + t.ok + '/' + n + ']</span>';
    }).join('<br>');
  } catch(e) {}
}
async function applySteer() {
  const el = document.getElementById('steerInfo');
  const src = document.getElementById('srcEp').value;
  if (!src) { el.textContent = '先选 source pair'; return; }
  el.textContent = 'applying…';
  const j = await api('/steer/apply', {
    source_dir: src,
    layer: document.getElementById('injLayer').value.trim(),
    target: document.getElementById('injTarget').value,
    save: document.getElementById('saveEp').checked,
  });
  if (!j.ok) { el.textContent = 'apply failed: ' + j.msg; return; }
  document.getElementById('expMode').value = 'setting1-Rr';
  document.getElementById('roleSel').value = 'cor';
  if (j.source_pair) document.getElementById('pairNum').value = j.source_pair;
  el.textContent = 'applied — 下次 Start 生效';
  steerCurrentRefresh();
  if (j.source_prompt) {
    const fp = await genFillerFor(j.source_prompt);
    if (fp) { usePrompt('cor');
              el.textContent = 'applied + filler prompt 已载入 — 可以 Start'; }
  }
}
steerEpsRefresh(); steerCurrentRefresh(); steerTallyRefresh();
setInterval(() => { steerCurrentRefresh(); steerTallyRefresh(); }, 5000);

// ── task masks(摆放参考)──
async function maskRefresh() {
  try {
    const r = await fetch('/mask/list'); const j = await r.json();
    const sel = document.getElementById('maskSel');
    const cur = sel.value;
    sel.innerHTML = '<option value="">(no mask)</option>';
    (j.masks || []).forEach(m => {
      if (m.error) return;
      const o = document.createElement('option');
      o.value = m.id; o.textContent = m.id;
      sel.appendChild(o);
    });
    sel.value = cur && [...sel.options].some(o => o.value === cur) ? cur : '';
  } catch(e) {}
}
async function takeMask() {
  const name = document.getElementById('maskName').value.trim();
  if (!name) { alert('mask id required(e.g. task1)'); return; }
  const j = await api('/mask/take', {id: name});
  if (!j.ok) { alert('take mask failed: ' + j.msg); return; }
  await maskRefresh();
  document.getElementById('maskSel').value = name;
  document.getElementById('ghostOn').checked = true;
  ghostUpdate();
}
async function deleteMask() {
  const id = document.getElementById('maskSel').value;
  if (!id) return;
  if (!confirm('Delete mask "' + id + '"?')) return;
  await api('/mask/delete', {id});
  await maskRefresh(); ghostUpdate();
}
function ghostUpdate() {
  // Ghost 来源优先级:选中任务的 layout 蒙版(共享 store)> 实验区 maskSel。
  const img = document.getElementById('ghostImg');
  const maskId = document.getElementById('maskSel').value;
  const base = taskGhostLayout
    ? '/task_layout/' + encodeURIComponent(taskGhostLayout) + '/'
    : (maskId ? '/mask/' + encodeURIComponent(maskId) + '/' : '');
  const on = document.getElementById('ghostOn').checked && base;
  img.style.opacity = document.getElementById('ghostAlpha').value / 100;
  if (on) {
    if (!img.src.startsWith(location.origin + base))
      img.src = base + 'exterior.jpg?t=' + Date.now();
    img.style.display = 'block';
  } else {
    img.style.display = 'none';
  }
}
maskRefresh();
async function setGripperMode() {
  const m = document.getElementById('gripMode').value;
  return api('/gripper_mode', {mode: m});
}
async function commitGrasp() {
  if (!confirm('Commit grasp:停策略 → 夹爪闭合 → 上提 +Z 5cm。继续?')) return;
  return api('/commit_grasp');
}
async function setHome() {
  if (!confirm('把当前手臂位姿存为 home?之后 Go home / eval reset 会回到这里。')) return;
  return api('/set_home');
}
async function jog(axis, sign) {
  const step = sign * parseFloat(document.getElementById('jogStep').value);
  return api('/jog', {axis, step});
}
async function jogRot(axis, sign) {
  const step = sign * parseFloat(document.getElementById('jogRotStep').value);
  return api('/jog', {axis, step});
}

// ── 状态轮询:chips + 明细 + episode 完成检测 ──────────────────────
let lastEvalRunning = null;
let lastEvalIter = 0;
// 相机流自愈:dashboard 重启后浏览器的 MJPEG <img> 不会自己重连,
// 检测到 boot_ts 变化就重置全部相机流 src。
let pageBootTs = null;
function resetCamSrcs() {
  document.querySelectorAll('img[src*=".mjpg"]').forEach(img => {
    const base = img.getAttribute('src').split('?')[0];
    img.src = base + '?t=' + Date.now();
  });
}
function setChip(id, state) {
  const el = document.getElementById(id);
  if (!el) return;
  el.className = 'chip ' + state;   // ok | bad | busy
}
async function refresh() {
  try {
    const r = await fetch('/status'); const j = await r.json();
    const s = j.robot || {};
    const re = j.eval || {};
    // camera stream self-heal on dashboard restart
    if (j.boot_ts !== undefined) {
      if (pageBootTs === null) pageBootTs = j.boot_ts;
      else if (j.boot_ts !== pageBootTs) {
        pageBootTs = j.boot_ts;
        consoleLine('dashboard server restarted — re-attaching camera streams', 'lvl-W');
        resetCamSrcs();
      }
    }
    const ab = document.getElementById('applySteerBtn');
    if (ab) ab.disabled = !!re.running;
    // header chips
    setChip('chip-robot', j.robot_ok ? 'ok' : (j.freedrive ? 'busy' : 'bad'));
    setChip('chip-gripper', j.gripper_ok ? 'ok' : 'bad');
    setChip('chip-cams', j.cam_running ? 'ok' : 'bad');
    setChip('chip-eval', re.running ? 'busy' : 'ok');
    // motion chip: policy idle vs executing vs commanded-but-not-moving.
    const mo = re.motion || {};
    const fmtN = (x) => (x === null || x === undefined) ? '-' : Number(x).toFixed(4);
    const moMap = {
      executing:     ['ok',   '🟢 执行中'],
      policy_idle:   ['',     '🟡 policy 静止'],
      suspect:       ['busy', '🟠 有指令未动 ' + (mo.still_s || 0) + 's'],
      not_executing: ['bad',  '🔴 NUC 未执行 ' + (mo.still_s || 0) + 's' + ((mo.still_s || 0) < 3 ? ' (reflex 自恢复中)' : '')],
      warming:       ['',     '… 采样中'],
      idle:          ['',     'motion'],
    };
    const mm = moMap[mo.state] || moMap.idle;
    setChip('chip-motion', mm[0]);
    document.getElementById('chip-motion-txt').textContent = mm[1];
    const moDetail = 'cmd_net=' + fmtN(mo.cmd_net) + ' rad → act_net=' + fmtN(mo.act_net)
      + ' rad / ' + (mo.window_s || 1.5) + 's · ts=' + (mo.ts_alive === false ? 'STALE' : (mo.ts_alive ? 'alive' : '-'));
    document.getElementById('chip-motion').title = moDetail;
    // episode lifecycle events
    if (lastEvalRunning === false && re.running === true) {
      consoleLine('eval episode started — prompt: ' + (re.last_prompt || '-'), 'lvl-I');
      toast('Episode started', 'info');
    }
    if (lastEvalRunning === true && re.running === false && lastEvalIter > 0) {
      consoleLine('eval episode finished — ' + lastEvalIter + ' steps', 'lvl-I');
      toast('Episode 完成 — ' + lastEvalIter + ' steps(约 ' +
            Math.round(lastEvalIter / 15) + ' 秒)', 'ok');
      setSaveVideo(true);   // 跑满 max-iterations 自动结束也算 Stop
      epRefresh();   // 新录制的 episode 出现在列表里
    }
    lastEvalRunning = re.running;
    if (re.iter !== undefined) lastEvalIter = re.iter;
    // status detail table
    const dot = (ok) => '<span class="dot ' + (ok ? 'ok-dot' : 'bad-dot') + '"></span>';
    let html = '<table>';
    html += '<tr><td>' + dot(j.robot_ok) + 'robot</td>'
         +  '<td>' + (j.freedrive
                      ? 'freedrive 手推模式 — 推完点 Lock joints 恢复控制'
                      : 'q=' + (s.q ? s.q.map(x => x.toFixed(2)).join(', ') : '?')
                        + '<br>' + (s.err ? esc(String(s.err).slice(0, 160))
                                          : 'controller=' + s.controller
                                            + '  last_cmd_ok=' + s.last_cmd_ok
                                            + '  loop=' + (s.loop_stale_s > 0
                                                ? '<span style="color:var(--err)">STALE ' + s.loop_stale_s + 's (NUC not executing)</span>'
                                                : 'alive')))
         +  '</td></tr>';
    html += '<tr><td>' + dot(j.gripper_ok) + 'gripper</td>'
         +  '<td>width=' + (j.gripper && j.gripper.width_m !== undefined
                            ? j.gripper.width_m.toFixed(3) + 'm' : '?')
         +  '</td></tr>';
    html += '<tr><td>' + dot(re.running) + 'eval</td>'
         +  '<td>' + (re.running ? 'running' : 'idle')
         +  '  · prompt=' + esc(re.last_prompt || '-') + '</td></tr>';
    html += '<tr><td>' + dot(j.cam_running) + 'cams</td>'
         +  '<td>' + (j.cam_running ? 'dashboard 持有' : '已释放(eval 占用)') + '</td></tr>';
    if (re.iter !== undefined) {
      const fmt = (x) => (x === null || x === undefined) ? '-' : Number(x).toFixed(3);
      html += '<tr><td>policy</td><td>'
           +  'iter=' + re.iter
           +  '<br>last infer ms=' + (re.last_infer_ms ? re.last_infer_ms.toFixed(0) : '-')
           +  '<br>last grip cmd(raw,0=open 1=close)=' + fmt(re.last_grip_raw)
           +  '<br>last |dq| max=' + fmt(re.last_dq_max)
           +  '<br>motion: ' + mm[1] + ' <span style="color:var(--faint)">(' + moDetail + ')</span>'
           +  '<br>wd: q_spread=' + fmt(re.wd && re.wd.q_spread) + ' cmd=' + fmt(re.wd && re.wd.cmd_mean)
           +  ' ts=' + (re.wd && re.wd.ts_same > 1
                        ? '<span style="color:var(--err)">STALE×' + re.wd.ts_same + '</span>'
                        : (re.wd && re.wd.ts_armed ? 'alive' : '-'))
           +  (re.homing ? '<br><span style="color:var(--warn)">homing…</span>' : '')
           +  '<br>img=' + (re.image_mode || 'pad')
           +  ' horizon=' + re.open_loop_horizon
           +  ' scale=' + re.action_scale
           +  ' cap=' + re.max_joint_vel;
      if (re.last_error) html += '<br><span style="color:var(--err)">err: '
                                + esc(re.last_error) + '</span>';
      html += '</td></tr>';
    }
    html += '</table>';
    document.getElementById('status').innerHTML = html;
    // gripper audit(之旭模块)
    try {
      const ar = await (await fetch('/audit')).json();
      const lines = (ar.recent || []).map(r => {
        const f = (x, d) => (x === null || x === undefined) ? '-' : Number(x).toFixed(d);
        const skip = r.reason_skipped || '';
        const act = r.action_taken || '';
        return String(r.iter).padStart(4) + ' '
             + 'raw=' + f(r.raw_action_grip, 3).padStart(6) + ' '
             + 'want=' + (r.want_close ? 'CLOSE' : 'open ') + ' '
             + 'did=' + act.padEnd(5) + ' '
             + 'skip=' + skip.padEnd(12) + ' '
             + 'gw ' + f(r.gw_pre_m, 3) + '→' + f(r.gw_post_m, 3) + ' '
             + 'obs=' + f(r.gw_obs_norm, 3);
      });
      document.getElementById('audit').textContent =
        lines.length ? lines.join('\\n') : '(no audit rows yet — start the policy)';
    } catch(e) {}
  } catch (e) {
    document.getElementById('status').innerHTML = 'status error: ' + esc(String(e));
  }
}

// ── Task 区域(共享任务库:选任务锁定 prompt/layout,New task 才可自定义)──
let taskList = [];
let taskGhostLayout = '';   // 选中任务的 layout 蒙版 id('' = 无)
const jsSlug = s => (s || '').toLowerCase().replace(/[^a-z0-9]+/g, '-')
  .replace(/^-+|-+$/g, '') || 'task';
async function taskRefresh() {
  try {
    const j = await (await fetch('/tasks')).json();
    taskList = j.tasks || [];
    const sel = document.getElementById('taskSel');
    const cur = sel.value;
    sel.innerHTML = '<option value="">(选择任务)</option>' + taskList.map(t =>
      '<option value="' + t.id + '">' + t.id + '</option>').join('');
    if (cur && taskList.some(t => t.id === cur)) sel.value = cur;
    const lay = document.getElementById('taskLayout');
    const lcur = lay.value;
    lay.innerHTML = '<option value="">(无摆放蒙版)</option>' + (j.layouts || []).map(m =>
      '<option value="' + m + '">' + m + '</option>').join('');
    if (lcur && (j.layouts || []).indexOf(lcur) >= 0) lay.value = lcur;
  } catch (e) {}
}
function onTaskSelect() {
  const t = taskList.find(x => x.id === document.getElementById('taskSel').value);
  const info = document.getElementById('taskInfo');
  const laySel = document.getElementById('taskLayoutSel');
  taskNewClose();
  if (!t) {
    promptInput.value = '';
    taskGhostLayout = '';
    laySel.innerHTML = '<option value="">—</option>';
    document.getElementById('ghostOn').checked = false;
    ghostUpdate();
    info.textContent = '';
    return;
  }
  promptInput.value = t.prompt;
  // 一个任务可对应多个 layouts;默认展示最近使用的那个(t.layout)。
  const lays = (t.layouts && t.layouts.length) ? t.layouts
    : (t.layout ? [t.layout] : []);
  laySel.innerHTML = lays.length
    ? lays.map(l => '<option value="' + l + '">' + l + '</option>').join('')
    : '<option value="">(无 layout)</option>';
  taskGhostLayout = lays.length ? (t.layout || lays[lays.length - 1]) : '';
  if (taskGhostLayout) laySel.value = taskGhostLayout;
  document.getElementById('ghostOn').checked = !!taskGhostLayout;
  ghostUpdate();
  info.textContent = 'layouts: ' + (lays.join(', ') || '—')
    + ' · datasets: ' + ((t.datasets || []).join(', ') || '—');
  toast('Task ' + t.id + ' → prompt + layout 已锁定', 'info');
}
function taskGhostPick() {
  taskGhostLayout = document.getElementById('taskLayoutSel').value || '';
  if (taskGhostLayout) document.getElementById('ghostOn').checked = true;
  ghostUpdate();
}
function taskNewOpen() {
  document.getElementById('taskForm').style.display = 'block';
  document.getElementById('taskId').value = '';
  document.getElementById('taskPrompt').value = '';
  document.getElementById('taskLayout').value = '';
  taskRefresh();
  document.getElementById('taskPrompt').focus();
}
function taskNewClose() {
  document.getElementById('taskForm').style.display = 'none';
}
async function takeTaskLayout() {
  const tid = document.getElementById('taskId').value.trim()
    || jsSlug(document.getElementById('taskPrompt').value.trim());
  if (!tid || tid === 'task') { toast('先填 prompt 或 task id,蒙版以任务命名', 'err'); return; }
  const j = await api('/task_layout/take', {id: tid});
  if (!j.ok) return;
  await taskRefresh();
  document.getElementById('taskId').value = tid;
  document.getElementById('taskLayout').value = tid;
  toast('蒙版已拍:' + tid + ' — 记得保存新任务', 'ok');
}
async function saveNewTask() {
  const prompt = document.getElementById('taskPrompt').value.trim();
  if (!prompt) { toast('prompt 不能为空', 'err'); return; }
  const body = {
    id: document.getElementById('taskId').value.trim() || jsSlug(prompt),
    prompt,
    layout: document.getElementById('taskLayout').value,
    datasets: [],
  };
  const j = await api('/task/create', body);
  if (j.ok) {
    await taskRefresh();
    document.getElementById('taskSel').value = body.id;
    onTaskSelect();
  }
}
async function deleteTask() {
  const id = document.getElementById('taskSel').value;
  if (!id) { toast('先选择要删除的任务', 'err'); return; }
  if (!confirm('删除任务 ' + id + '?(不影响数据集和蒙版文件)')) return;
  const j = await api('/task/delete', {id});
  if (j.ok) { await taskRefresh(); document.getElementById('taskSel').value = ''; onTaskSelect(); }
}

// ── Episodes:录制视频管理(按 task 分类)─────────────────────────────
let epList = [];
let epEditId = null;
const epUrl = {};   // ep_id -> video url(ep_id 是安全 token,onclick 裸引用)
// 折叠状态跨刷新保留(默认全部展开;用户收起的组记在 epCollapsed)
const epCollapsed = new Set();
function epToggle(el, task) {
  if (el.open) epCollapsed.delete(task); else epCollapsed.add(task);
}
function epTaskLabel(task) {
  const t = taskList.find(x => x.id === task);
  return t && t.prompt ? task + ' · ' + t.prompt : task;
}
function epTally(eps) {
  const t = {ok: 0, fail: 0, unmarked: 0};
  eps.forEach(e => { if (e.mark === 'success') t.ok++;
                     else if (e.mark === 'fail') t.fail++;
                     else if (e.mark === 'aborted' || e.abort) t.aborted = (t.aborted || 0) + 1;
                     else t.unmarked++; });
  t.n = t.ok + t.fail;
  t.sr = t.n ? Math.round(100 * t.ok / t.n) : null;
  return t;
}
function epTallyHtml(t) {
  return '<b style="color:#81C995">✓' + t.ok + '</b> <b style="color:#F28B82">✗'
    + t.fail + '</b>'
    + (t.unmarked ? ' <span style="color:#E5C07B">·' + t.unmarked + ' 未判</span>' : '')
    + ' <span style="color:var(--faint)">SR '
    + (t.sr === null ? '–' : t.ok + '/' + t.n + ' = ' + t.sr + '%') + '</span>';
}
async function epRefresh() {
  try {
    const all = await (await fetch('/episodes')).json();
    const eps = all.episodes || [];       // 服务端已按 start_time 倒序;
                                          // task 为空/untagged 的已按 prompt 映射到任务库 id
    const sel = document.getElementById('epTaskFilter');
    const cur = sel.value;
    // 组顺序:任务库顺序 → 其它(按名字)→ untagged 垫底
    const known = taskList.map(t => t.id);
    const extra = [...new Set(eps.map(e => e.task))]
      .filter(t => known.indexOf(t) < 0 && t !== 'untagged').sort();
    const order = known.concat(extra);
    if (eps.some(e => e.task === 'untagged')) order.push('untagged');
    sel.innerHTML = '<option value="">(全部)</option>' + order.map(t =>
      '<option value="' + esc(t) + '">' + esc(t) + '</option>').join('');
    if (cur && order.indexOf(cur) >= 0) sel.value = cur;
    const filter = sel.value;
    epList = eps.filter(e => !filter || e.task === filter);
    eps.forEach(e => { epUrl[e.ep_id] = e.url; });
    const total = epTally(epList);
    document.getElementById('epCount').innerHTML =
      epList.length ? ('共 ' + epList.length + ' 条 · ' + epTallyHtml(total)) : '(暂无记录)';

    const groups = order.filter(t => !filter || t === filter)
      .map(t => ({task: t, eps: epList.filter(e => e.task === t)}))
      .filter(g => g.eps.length);
    document.getElementById('epTable').innerHTML = groups.map(g => {
      // 组内按时间倒序(最新在上)
      g.eps.sort((a, b) => (b.start_time || '').localeCompare(a.start_time || ''));
      const t = epTally(g.eps);
      // 按 ckpt 细分(同一 task 评多个 step 时一眼看出哪个 ckpt 好)
      const byCk = {};
      g.eps.forEach(e => { const k = e.ckpt || '-'; (byCk[k] = byCk[k] || []).push(e); });
      const ckLine = Object.keys(byCk).length > 1
        ? '<div class="byckpt">by ckpt: ' + Object.keys(byCk).sort().map(k => {
            const c = epTally(byCk[k]);
            return esc(k) + ' <b style="color:#81C995">' + c.ok + '</b>/'
              + c.n + (c.unmarked ? '<span style="color:#E5C07B">+' + c.unmarked + '</span>' : '')
              + (c.sr === null ? '' : ' (' + c.sr + '%)');
          }).join(' · ') + '</div>' : '';
      const rows = g.eps.map(e => {
        const mark = e.mark === 'success' ? '<b style="color:#81C995">✓</b>'
                   : e.mark === 'fail' ? '<b style="color:#F28B82">✗</b>'
                   : (e.mark === 'aborted' || e.abort)
                     ? '<b style="color:var(--warn)" title="' + esc(e.abort || 'aborted by the watchdog') + '">⚠</b>'
                   : '<span style="color:#6E6E73">·</span>';
        const auto = e.task_auto
          ? ' <span title="task 由 prompt 自动匹配" style="color:var(--faint)">(auto)</span>' : '';
        return '<tr>'
          + '<td><button class="tiny" data-ep="' + esc(e.ep_id) + '" onclick="playEpId(this.dataset.ep)">▶</button></td>'
          + '<td>' + esc(e.start_time) + auto
          + (e.demo && e.demo.stem ? '<br><span style="color:var(--faint)" title="saved_demo/' + esc(e.demo.dir.split('/').pop()) + '">💾 ' + esc(e.demo.stem) + '</span>' : '')
          + '</td>'
          + '<td>' + esc(e.ckpt || '-')
          + (e.layout ? '<br><span style="color:var(--faint)" title="layout">' + esc(e.layout) + '</span>' : '')
          + '</td>'
          + '<td>' + e.steps + '步/' + e.duration_s + 's</td>'
          + '<td>' + esc(e.prompt || '').slice(0, 42) + '</td>'
          + '<td>' + mark + '</td>'
          + '<td><button class="tiny" data-ep="' + esc(e.ep_id) + '" data-mark="success" onclick="markEpRec(this.dataset.ep,this.dataset.mark)">✓</button>'
          + '<button class="tiny" data-ep="' + esc(e.ep_id) + '" data-mark="fail" onclick="markEpRec(this.dataset.ep,this.dataset.mark)">✗</button></td>'
          + '<td>' + esc(e.note || '') + '</td>'
          + '<td><button class="tiny" data-ep="' + esc(e.ep_id) + '" onclick="editEp(this.dataset.ep)">✎</button>'
          + '<button class="tiny danger" data-ep="' + esc(e.ep_id) + '" onclick="deleteEp(this.dataset.ep)">🗑</button></td>'
          + '</tr>';
      }).join('');
      return '<details class="epgrp"' + (epCollapsed.has(g.task) ? '' : ' open')
        + ' ontoggle="epToggle(this,' + JSON.stringify(g.task).replace(/"/g, '&quot;') + ')">'
        + '<summary><b>' + esc(epTaskLabel(g.task)) + '</b>'
        + '<span style="color:var(--faint)">' + g.eps.length + ' 条</span>'
        + '<span class="sr">' + epTallyHtml(t) + '</span></summary>'
        + '<div class="epgrp-body">' + ckLine
        + '<table><tr><th></th><th>时间</th><th>ckpt</th><th>长度</th>'
        + '<th>prompt</th><th></th><th>判定</th><th>note</th><th></th></tr>'
        + rows + '</table></div></details>';
    }).join('') || '';
  } catch (e) {}
}
function playEpId(id) {
  const u = epUrl[id];
  if (u) {
    document.getElementById('videoPlayer').src = u;
    document.getElementById('videoModal').style.display = 'flex';
  }
}
function closeVideo() {
  document.getElementById('videoModal').style.display = 'none';
  document.getElementById('videoPlayer').src = '';
}
function editEp(epId) {
  const e = epList.find(x => x.ep_id === epId);
  if (!e) return;
  epEditId = epId;
  document.getElementById('epEditTask').value = e.task || '';
  document.getElementById('epEditPrompt').value = e.prompt || '';
  document.getElementById('epEditCkpt').value = e.ckpt || '';
  document.getElementById('epEditNote').value = e.note || '';
  document.getElementById('epEditForm').style.display = 'block';
}
async function saveEp() {
  if (!epEditId) return;
  const j = await api('/episode/update', {
    ep_id: epEditId,
    task: document.getElementById('epEditTask').value.trim(),
    prompt: document.getElementById('epEditPrompt').value.trim(),
    ckpt: document.getElementById('epEditCkpt').value.trim(),
    note: document.getElementById('epEditNote').value.trim(),
  });
  if (j.ok) { document.getElementById('epEditForm').style.display = 'none'; epRefresh(); }
}
async function markEpRec(epId, mark) {
  const j = await api('/episode/update', {ep_id: epId, mark});
  if (j.ok) epRefresh();
}
async function deleteEp(epId) {
  if (!confirm('删除该 episode 的视频和 meta?')) return;
  const j = await api('/episode/delete', {ep_id: epId});
  if (j.ok) epRefresh();
}

// ── checkpoint 切换器 ──────────────────────────────────────────────
let lastCkptState = null;
// 用户手动改过下拉框后,轮询不再把它强拉回 serving 项;
// 等下一次 Load 完成(选中项 == serving 项)才解除。
let ckptSelDirty = false;
document.getElementById('ckptSel').addEventListener('change', () => {
  ckptSelDirty = true;
});
async function ckptRefresh() {
  try {
    const j = await (await fetch('/ckpts')).json();
    const sel = document.getElementById('ckptSel');
    const cur = sel.value;
    const s = j.serve || {};
    sel.innerHTML = (j.ckpts || []).map(c =>
      '<option value="' + c.label + '">' + c.label + '</option>').join('');
    if ((s.state === 'ready' || s.state === 'loading') && !ckptSelDirty) {
      sel.value = s.label;
    } else if (cur && (j.ckpts || []).some(c => c.label === cur)) {
      sel.value = cur;
    }
    if (s.label && sel.value === s.label) ckptSelDirty = false;
    // chip + state line
    setChip('chip-ckpt', s.state === 'ready' ? 'ok' : (s.state === 'loading' ? 'busy' : 'bad'));
    const el = document.getElementById('ckptState');
    const dot = (ok) => '<span class="dot ' + (ok ? 'ok-dot' : 'bad-dot') + '"></span>';
    let txt, ok;
    if (s.state === 'idle')     { txt = '未加载策略'; ok = false; }
    else if (s.state === 'loading') { txt = '⏳ 加载中 ' + s.label + (s.rtc ? ' [RTC]' : '') + ' — ' + s.msg; ok = false; }
    else if (s.state === 'ready')   { txt = 'serving ' + s.label + (s.rtc ? ' · RTC' : ' · sync'); ok = true; }
    else                        { txt = 'error: ' + s.msg; ok = false; }
    el.innerHTML = dot(ok) + txt;
    // state transitions → console + toast
    if (lastCkptState && lastCkptState !== s.state) {
      if (s.state === 'ready') { consoleLine('checkpoint ready: ' + s.label, 'lvl-I');
                                 toast('Checkpoint 就绪: ' + s.label, 'ok'); }
      if (s.state === 'error') { consoleLine('checkpoint error: ' + s.msg, 'lvl-E');
                                 toast('Checkpoint 错误: ' + s.msg, 'err'); }
    }
    lastCkptState = s.state;
    // serve 日志尾(过滤握手噪音)
    const lg = document.getElementById('ckptLog');
    const noise = /websockets|Traceback|InvalidUpgrade|InvalidMessage|File "|handshake|process_request|await connection|raise |self\\.protocol/;
    const tail = (s.log_tail || []).filter(l => !noise.test(l));
    if (s.state === 'loading' && tail.length) {
      lg.style.display = 'block';
      lg.textContent = tail.join('\\n');
    } else if (s.state === 'error' && tail.length) {
      lg.style.display = 'block';
      lg.textContent = tail.join('\\n');
    } else lg.style.display = 'none';
  } catch (e) {
    document.getElementById('ckptState').innerHTML = 'ckpt status error: ' + esc(String(e));
  }
}
async function switchCkpt(rtc) {
  const label = document.getElementById('ckptSel').value;
  if (!label) return;
  const el = document.getElementById('ckptState');
  el.innerHTML = 'switching…';
  try {
    const r = await fetch('/ckpt/switch', {method: 'POST',
      headers: {'Content-Type': 'application/json'}, body: JSON.stringify({label, rtc: !!rtc})});
    const j = await r.json();
    if (!j.ok) { el.innerHTML = 'switch rejected: ' + j.msg; toast(j.msg, 'err'); return; }
    consoleLine('checkpoint switch requested: ' + label + (rtc ? ' (with RTC)' : ' (w/o RTC)'), 'lvl-I');
  } catch (e) { el.innerHTML = 'switch error: ' + e; return; }
  ckptRefresh();
}

// ── 后端日志 console ───────────────────────────────────────────────
let _seenEvents = 0;
async function logsPoll() {
  try {
    const j = await (await fetch('/logs')).json();
    const evs = j.events || [];
    if (evs.length > _seenEvents) {
      for (let i = _seenEvents; i < evs.length; i++) {
        const m = String(evs[i]).match(/^(\\S+ \\S+) (\\w+) (.*)$/);
        const lvl = m ? m[2] : 'I';
        const txt = m ? m[3] : evs[i];
        const cls = lvl.startsWith('E') ? 'lvl-E' : (lvl.startsWith('W') ? 'lvl-W' : 'lvl-I');
        consoleLine(esc(txt), cls);
      }
      _seenEvents = evs.length;
    } else if (evs.length < _seenEvents) {
      _seenEvents = evs.length;   // dashboard 重启后环被清空
    }
  } catch (e) {}
}

refresh(); ckptRefresh(); logsPoll(); taskRefresh(); maskRefresh(); epRefresh();
// ── Action chunk 面板(policy 原始输出,clip / delta_scale 之前)────────────
let _actSeq = 0, _actChunks = [];
async function actionsPoll() {
  try {
    const j = await (await fetch('/actions?since=' + _actSeq)).json();
    const items = j.items || [];
    if (!items.length) return;
    _actSeq = j.seq;
    _actChunks = _actChunks.concat(items).slice(-3);   // 只留最近 3 个 chunk
    const fmt = v => (v >= 0 ? ' ' : '') + Number(v).toFixed(3);
    const lines = [];
    for (let c = _actChunks.length - 1; c >= 0; c--) {
      const a = _actChunks[c];
      const H = a.shape[0], D = a.shape[1] || 8;
      lines.push('#' + a.seq + '  ' + a.src + (a.iter !== undefined ? '  iter ' + a.iter : '')
        + '  chunk = ' + H + ' × ' + D
        + '  infer ' + (a.infer_ms !== undefined ? a.infer_ms + ' ms' : '–')
        + (a.s !== undefined ? '  rtc s=' + a.s + ' d=' + a.d : '')
        + '  @' + new Date(a.t * 1000).toLocaleTimeString());
      lines.push('  k     j1     j2     j3     j4     j5     j6     j7  |  grip');
      (a.values || []).forEach((row, k) => {
        lines.push(' ' + String(k).padStart(2) + '  ' + row.slice(0, 7).map(fmt).join(' ')
          + '  | ' + fmt(row[7]));
      });
      lines.push('');
    }
    document.getElementById('actionsBox').textContent = lines.join('\\n');
    const last = _actChunks[_actChunks.length - 1];
    document.getElementById('actionsHint').textContent =
      'chunk size ' + last.shape[0] + ' × ' + (last.shape[1] || 8) + ' · ' + _actSeq + ' inferences · 显示最近 3 个';
  } catch (e) { /* keep last render */ }
}
setInterval(actionsPoll, 700);
setInterval(() => { refresh(); ckptRefresh(); logsPoll(); }, 1500);
</script>
</body></html>
"""


def build_app(rs: RS, cams: CamManager, runner: EvalRunner,
              home_store: "HomeStore") -> Flask:
    app = Flask(__name__)

    # ── Policy checkpoint switcher ───────────────────────────────────
    serve = ServeManager(port=runner.policy_port)
    _rtc_hook.register_routes(app, runner, serve)

    @app.get("/ckpts")
    def ckpts():
        return jsonify({"ckpts": discover_checkpoints(), "serve": serve.status()})

    @app.post("/ckpt/switch")
    def ckpt_switch():
        if runner.status()["running"]:
            return jsonify({"ok": False,
                            "msg": "eval running; stop first"}), 409
        body = request.get_json(silent=True) or {}
        label = (body.get("label") or "").strip()
        rtc = bool(body.get("rtc", False))       # "Load with RTC" button
        msg = serve.switch(label, rtc=rtc)
        return jsonify({"ok": msg == "switching", "msg": msg})

    @app.post("/ckpt/stop")
    def ckpt_stop():
        return jsonify({"ok": True, "msg": serve.stop()})

    @app.get("/actions")
    def get_actions():
        """Raw policy chunks since ?since=<seq> (newest last, at most 8)."""
        try:
            since = int(request.args.get("since") or 0)
        except ValueError:
            since = 0
        items = [x for x in list(runner.action_log) if x["seq"] > since]
        return jsonify({"seq": runner._action_seq, "items": items[-8:]})  # noqa: SLF001

    @app.get("/logs")
    def get_logs():
        """Backend log feed for the frontend console: in-memory ring of the
        dashboard's own log records + a tail of the serve_policy log."""
        serve_tail = []
        try:
            text = pathlib.Path(SERVE_LOG).read_text(encoding="utf-8")
            serve_tail = text[-1500:].splitlines()[-6:]
        except OSError:
            pass
        return jsonify({"events": list(BACKEND_EVENTS), "serve": serve_tail})

    @app.get("/")
    def index():
        return render_template_string(_rtc_hook.inject_panel(INDEX_HTML))

    @app.get("/cam/<name>.mjpg")
    def mjpg(name):
        def gen():
            boundary = b"--frame\r\n"
            while True:
                buf = cams.get_jpeg(name)
                if buf is None:
                    time.sleep(0.05)
                    continue
                yield (boundary + b"Content-Type: image/jpeg\r\n\r\n"
                       + buf + b"\r\n")
                time.sleep(0.033)  # ~30 fps cap
        return Response(gen(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/cam/<name>.jpg")
    def single_jpeg(name):
        # Used by env (and anyone who wants a single shot) — same source
        # as the MJPEG stream. Falls back to relay file if cam_mgr idle.
        buf = cams.get_jpeg(name)
        if buf is None:
            return Response("no frame", status=404)
        return Response(buf, mimetype="image/jpeg")

    # ── Task masks: reference snapshots for repositioning objects ─────
    # One mask per real-robot task; the ghost overlay on the exterior feed
    # lets the operator restore the exact initial object placement before
    # every rollout of a clean/cor pair. Backed by the collect dashboard's
    # LayoutStore (markers unused here — snapshot-only masks).
    mask_store = LayoutStore(MASK_DIR)
    task_layout_store = LayoutStore(TASK_LAYOUT_DIR)

    def _hq_jpeg(name: str) -> Optional[bytes]:
        """Snapshot-grade JPEG (q95 from the raw frame); falls back to the
        q70 preview encode if no raw frame is available."""
        bgr = cams.get_bgr(name)
        if bgr is not None:
            ok, buf = cv2.imencode(
                ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if ok:
                return buf.tobytes()
        return cams.get_jpeg(name)

    def _snap_current(store, snap_id: str, note: str = ""):
        """Save the live camera views into `store` as a snapshot-only layout.
        Returns (layout_dict, err_response)."""
        ext = _hq_jpeg("wrist_1")     # exterior ZED 2i
        wrist = _hq_jpeg("wrist_2")   # wrist ZED Mini (may be dead)
        if ext is None:
            return None, (jsonify({"ok": False, "msg": "no exterior frame"}), 503)
        snaps = {"exterior": ext}
        if wrist is not None:
            snaps["wrist"] = wrist
        try:
            return store.save(snap_id, {}, note=note, snapshots=snaps), None
        except LayoutError:
            # Wrist frame may be a placeholder (dead cam) — retry without it.
            try:
                return store.save(snap_id, {}, note=note,
                                  snapshots={"exterior": ext}), None
            except LayoutError as e:
                return None, (jsonify({"ok": False, "msg": str(e)}), 400)

    @app.post("/mask/take")
    def mask_take():
        body = request.get_json(silent=True) or {}
        mask_id = (body.get("id") or "").strip()
        lay, err = _snap_current(mask_store, mask_id)
        if err is not None:
            return err
        return jsonify({"ok": True, "id": lay["id"],
                        "has_snapshot": lay["has_snapshot"]})

    @app.get("/mask/list")
    def mask_list():
        return jsonify({"masks": mask_store.list()})

    @app.get("/mask/<mask_id>/<view>.jpg")
    def mask_jpeg(mask_id, view):
        try:
            buf = mask_store.snapshot_bytes(mask_id, view)
        except LayoutError as e:
            return Response(str(e), status=400)
        if buf is None:
            return Response("no snapshot", status=404)
        return Response(buf, mimetype="image/jpeg")

    # Cache: tokenizer load costs a few seconds per subprocess call, and the
    # operator will regenerate while iterating on prompts.
    _filler_cache: dict = {}

    @app.post("/filler")
    def gen_filler():
        """Length-matched blank filler for the given clean prompt — live
        version of make_blank_fillers.py so prompts can be decided at the
        bench. Runs in the openpi venv (tokenizer deps live there)."""
        body = request.get_json(silent=True) or {}
        prompt = (body.get("prompt") or "").strip()
        if not prompt:
            return jsonify({"ok": False, "msg": "prompt required"}), 400
        if prompt in _filler_cache:
            return jsonify({"ok": True, **_filler_cache[prompt],
                            "cached": True})
        env = dict(os.environ)
        # Explicit HOME/steer paths: under sudo, HOME=/root and the tokenizer
        # cache + steer checkout would resolve to the wrong place. And the
        # dashboard's PYTHONPATH (system-python site-packages) must NOT leak
        # into the venv subprocess — its numpy shadows the venv's.
        env.setdefault("STEER_PI05_DIR", STEER_PI05_DIR_DEFAULT)
        env["HOME"] = _HOME_DIR
        env.pop("PYTHONPATH", None)
        try:
            res = subprocess.run(
                [OPENPI_VENV_PY, FILLER_SCRIPT, "--json", prompt],
                capture_output=True, text=True, timeout=120, env=env)
        except subprocess.TimeoutExpired:
            return jsonify({"ok": False, "msg": "filler generation timed out "
                            "(tokenizer download stuck?)"}), 504
        if res.returncode != 0:
            return jsonify({"ok": False,
                            "msg": res.stderr.strip()[-400:] or "failed"}), 500
        try:
            data = json.loads(res.stdout.strip().splitlines()[-1])
            r = data["results"][0]
        except (ValueError, KeyError, IndexError) as e:
            return jsonify({"ok": False,
                            "msg": f"bad output: {e}: {res.stdout[-200:]}"}), 500
        if "error" in r:
            return jsonify({"ok": False, "msg": r["error"]}), 400
        _filler_cache[prompt] = r
        return jsonify({"ok": True, **r, "cached": False})

    @app.post("/mark")
    def mark_episode():
        """Write the operator's success verdict into the NEWEST capture
        episode's episode.json. Screening rule for setting1: a pair counts
        only if clean succeeded AND blank failed — failed pairs get re-run,
        and the pairing script filters on this field."""
        body = request.get_json(silent=True) or {}
        success = body.get("success")
        if not isinstance(success, bool):
            return jsonify({"ok": False, "msg": "success must be true/false"}), 400
        try:
            runs = sorted(
                (d for d in os.listdir(ROLLOUTS_ROOT)
                 if os.path.isdir(os.path.join(ROLLOUTS_ROOT, d))),
                key=lambda d: os.path.getmtime(os.path.join(ROLLOUTS_ROOT, d)))
        except OSError:
            runs = []
        if not runs:
            return jsonify({"ok": False,
                            "msg": f"no capture runs under {ROLLOUTS_ROOT} — "
                                   "is the hooked server running?"}), 404
        run_dir = os.path.join(ROLLOUTS_ROOT, runs[-1])
        eps = sorted(d for d in os.listdir(run_dir)
                     if d.startswith("ep")
                     and os.path.isdir(os.path.join(run_dir, d)))
        if not eps:
            return jsonify({"ok": False,
                            "msg": f"no episodes yet in {runs[-1]}"}), 404
        ep_json = os.path.join(run_dir, eps[-1], "episode.json")
        try:
            with open(ep_json) as fh:
                meta = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            return jsonify({"ok": False, "msg": f"episode.json: {e}"}), 500
        meta["success"] = success
        meta["success_marked_at"] = time.time()
        with open(ep_json, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        _log.info(f"marked {runs[-1]}/{eps[-1]} success={success}")
        return jsonify({"ok": True, "run": runs[-1], "episode": eps[-1],
                        "success": success, "tag": meta.get("tag"),
                        "prompt": meta.get("prompt")})

    # ── setting1-Rr: portal-driven runtime-patch config ────────────────
    # No new server protocol: the portal rewrites the marked fields of the
    # watch-reloaded YAML (STEER_CFG_PATH); serve_policy_patched re-applies
    # steering + opens a new episode dir before the next rollout's first
    # inference, and stamps the applied params into episode.json ("steering").

    def _pair_task_label(tag: dict) -> str:
        """Human task name for a source pair. tag.task when recorded; else
        inferred from the operator's pair-numbering convention (bench
        2026-08-10: pairs 4-6 were recorded without a mask id but are Task2)."""
        t = (tag.get("task") or "").strip()
        if t:
            return t
        p = tag.get("pair")
        try:
            p = int(p)
        except (TypeError, ValueError):
            return "?"
        return {0: "Task1", 1: "Task2", 2: "Task3"}.get((p - 1) // 3, "?") \
            if 1 <= p <= 9 else "?"

    @app.get("/steer/eps")
    def steer_eps():
        """Inject-source candidates: successful CLEAN setting1-Ra episodes
        with activations on disk, one per pair (newest re-run wins), labeled
        "TaskN · pair P". n_steps is surfaced because shorter sources wrap
        modulo when the live rollout runs longer."""
        try:
            runs = sorted(
                (d for d in os.listdir(ROLLOUTS_ROOT)
                 if os.path.isdir(os.path.join(ROLLOUTS_ROOT, d))),
                key=lambda d: os.path.getmtime(os.path.join(ROLLOUTS_ROOT, d)),
                reverse=True)
        except OSError:
            runs = []
        by_pair: dict = {}
        for run in runs[:6]:
            run_dir = os.path.join(ROLLOUTS_ROOT, run)
            for ep in sorted(os.listdir(run_dir), reverse=True):
                ep_dir = os.path.join(run_dir, ep)
                rs_dir = os.path.join(ep_dir, "residual_stream")
                if not os.path.isdir(rs_dir) or not os.listdir(rs_dir):
                    continue
                try:
                    with open(os.path.join(ep_dir, "episode.json"),
                              encoding="utf-8") as fh:
                        meta = json.load(fh)
                except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                    continue
                tag = meta.get("tag") or {}
                if (tag.get("mode") != "setting1-Ra"
                        or tag.get("role") != "clean"
                        or meta.get("success") is not True):
                    continue
                try:
                    pair = int(tag.get("pair"))
                except (TypeError, ValueError):
                    continue
                if pair in by_pair:      # newest run/ep already claimed it
                    continue
                by_pair[pair] = {
                    "pair": pair, "task": _pair_task_label(tag),
                    "ep": ep, "dir": ep_dir,
                    "n_steps": meta.get("n_steps"),
                    "prompt": meta.get("prompt")}
        eps = [by_pair[p] for p in sorted(by_pair)]
        return jsonify({"ok": True, "eps": eps})

    @app.post("/steer/apply")
    def steer_apply():
        """Rewrite inject_layer / target / capture_source_dir in the watched
        YAML. Refused mid-rollout — the hot-reload would cut a new episode
        under the live run."""
        if getattr(runner, "_running", False):
            return jsonify({"ok": False,
                            "msg": "a rollout is running — stop it first"}), 409
        body = request.get_json(silent=True) or {}
        src = (body.get("source_dir") or "").strip()
        layer_raw = str(body.get("layer") or "").strip().lower()
        target = body.get("target")
        if target not in ("all", "img_tokens", "text_tokens"):
            return jsonify({"ok": False, "msg": f"bad target {target!r}"}), 400
        rs_dir = os.path.join(src, "residual_stream")
        if not os.path.isdir(rs_dir):
            return jsonify({"ok": False,
                            "msg": f"no residual_stream/ under {src}"}), 400
        if layer_raw == "all":
            layer_yaml, probe_layers = "all", [0]
        else:
            try:
                layers = sorted({int(x) for x in layer_raw.split(",")})
            except ValueError:
                return jsonify({"ok": False,
                                "msg": f"bad layer {layer_raw!r} — int, "
                                       "comma list, or all"}), 400
            if not layers or not all(0 <= l <= 17 for l in layers):
                return jsonify({"ok": False,
                                "msg": "layers must be in 0..17"}), 400
            layer_yaml = (str(layers[0]) if len(layers) == 1
                          else "[" + ", ".join(map(str, layers)) + "]")
            probe_layers = layers
        for pl in probe_layers:
            if not os.path.isfile(os.path.join(
                    rs_dir, f"layer_{pl}_step_0000.pt")):
                return jsonify({"ok": False,
                                "msg": f"source has no layer_{pl}_step_0000.pt "
                                       "— wrong capture config?"}), 400
        try:
            # UTF-8 explicitly: under sudo the locale is POSIX/ASCII and the
            # template's unicode comments make a bare open() blow up.
            with open(STEER_CFG_PATH, encoding="utf-8") as fh:
                txt = fh.read()
        except OSError as e:
            return jsonify({"ok": False, "msg": f"{STEER_CFG_PATH}: {e}"}), 500
        save_ep = bool(body.get("save", False))
        subs = {"inject_layer": layer_yaml, "target": target,
                "capture_source_dir": src,
                "save_episode": "true" if save_ep else "false"}
        for key, val in subs.items():
            txt, n = re.subn(rf"(?m)^(\s*{key}:\s*).*$",
                             lambda m, v=val: m.group(1) + v, txt, count=1)
            if n != 1 and key == "save_episode":
                # Field is newer than the template — add it under capture:
                # (after save_io), honored by the server's hot-reload.
                txt, n = re.subn(r"(?m)^(\s*)(save_io:.*)$",
                                 rf"\1\2\n\1save_episode: {val}", txt, count=1)
            if n != 1:
                return jsonify({"ok": False,
                                "msg": f"field {key} not found in "
                                       f"{os.path.basename(STEER_CFG_PATH)} — "
                                       "template drifted?"}), 500
        with open(STEER_CFG_PATH, "w", encoding="utf-8") as fh:
            fh.write(txt)
        src_meta, src_pair = {}, None
        try:
            with open(os.path.join(src, "episode.json"),
                      encoding="utf-8") as fh:
                src_meta = json.load(fh)
            src_pair = (src_meta.get("tag") or {}).get("pair")
        except (OSError, json.JSONDecodeError):
            pass
        _log.info(f"steer apply: L={layer_yaml} target={target} "
                  f"src={os.path.basename(src)}")
        return jsonify({"ok": True, "layer": layer_yaml, "target": target,
                        "source_ep": os.path.basename(src),
                        "source_n_steps": src_meta.get("n_steps"),
                        "source_pair": src_pair,
                        "source_prompt": src_meta.get("prompt"),
                        "save_episode": save_ep,
                        "config": STEER_CFG_PATH})

    @app.get("/steer/current")
    def steer_current():
        """The condition currently in the watched YAML — what the NEXT Start
        will run with. Rendered as the portal's steer-mode banner."""
        try:
            with open(STEER_CFG_PATH, encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
        except (OSError, yaml.YAMLError, UnicodeDecodeError) as e:
            return jsonify({"ok": False, "msg": str(e)}), 500
        rs = (cfg.get("hooks") or {}).get("residual_stream") or {}
        cap = cfg.get("capture") or {}
        src = str(rs.get("capture_source_dir") or "")
        task, pair, n_steps = "?", None, None
        try:
            with open(os.path.join(src, "episode.json"),
                      encoding="utf-8") as fh:
                m = json.load(fh)
            tag = m.get("tag") or {}
            task = _pair_task_label(tag)
            pair = tag.get("pair")
            n_steps = m.get("n_steps")
        except (OSError, json.JSONDecodeError):
            pass
        return jsonify({
            "ok": True,
            "steer_on": (rs.get("mode") in ("write", "read_and_write")
                         and rs.get("steering_mode") == "inject_per_step"),
            "layer": rs.get("inject_layer"), "target": rs.get("target"),
            "source_task": task, "source_pair": pair,
            "source_n_steps": n_steps,
            "save_episode": bool(cap.get("save_episode", True))})

    @app.get("/steer/tally")
    def steer_tally():
        """Live per-condition tally for setting1-Rr: every patched episode is
        grouped by (inject_layer, source pair) from the server-side steering
        stamp; ✓/✗ come from the operator's /mark clicks. Cheap enough to
        poll (episode.json reads only)."""
        try:
            runs = sorted(
                (d for d in os.listdir(ROLLOUTS_ROOT)
                 if os.path.isdir(os.path.join(ROLLOUTS_ROOT, d))),
                key=lambda d: os.path.getmtime(os.path.join(ROLLOUTS_ROOT, d)),
                reverse=True)
        except OSError:
            runs = []
        rows: dict = {}
        for run in runs[:4]:
            run_dir = os.path.join(ROLLOUTS_ROOT, run)
            for ep in sorted(os.listdir(run_dir)):
                ej = os.path.join(run_dir, ep, "episode.json")
                if not os.path.isfile(ej):
                    continue
                try:
                    with open(ej, encoding="utf-8") as fh:
                        meta = json.load(fh)
                except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                    continue
                tag = meta.get("tag") or {}
                if tag.get("mode") != "setting1-Rr":
                    continue
                st = (meta.get("steering") or {}).get("residual") or {}
                layer = st.get("inject_layer")
                src = os.path.basename(str(st.get("capture_source_dir") or ""))
                key = (json.dumps(layer), tag.get("pair"), src)
                r = rows.setdefault(key, {
                    "layer": layer, "pair": tag.get("pair"),
                    "task": _pair_task_label(tag), "src": src,
                    "ok": 0, "fail": 0, "unmarked": 0})
                s = meta.get("success")
                if s is True:
                    r["ok"] += 1
                elif s is False:
                    r["fail"] += 1
                else:
                    r["unmarked"] += 1
        out = sorted(rows.values(),
                     key=lambda r: (str(r["task"]), str(r["pair"]),
                                    str(r["layer"])))
        return jsonify({"ok": True, "rows": out})

    @app.post("/mask/delete")
    def mask_delete():
        body = request.get_json(silent=True) or {}
        try:
            msg = mask_store.delete((body.get("id") or "").strip())
        except LayoutError as e:
            return jsonify({"ok": False, "msg": str(e)}), 400
        return jsonify({"ok": True, "msg": msg})

    # ── Task store: prompt + layout mask + collected datasets ───────
    # One record per bench task. Selecting a task in the UI auto-fills the
    # prompt and re-enables the placement ghost (layout mask) so the
    # operator aligns the objects before every rollout.
    task_store = TaskStore()

    @app.get("/tasks")
    def get_tasks():
        return jsonify({
            "tasks": task_store.list(),
            "layouts": [m["id"] for m in task_layout_store.list()
                        if not m.get("error")],
        })

    @app.post("/task_layout/take")
    def task_layout_take():
        """Snapshot the live views into the SHARED layout dir (New task 拍蒙版)."""
        body = request.get_json(silent=True) or {}
        lay_id = (body.get("id") or "").strip()
        lay, err = _snap_current(task_layout_store, lay_id)
        if err is not None:
            return err
        return jsonify({"ok": True, "id": lay["id"],
                        "has_snapshot": lay["has_snapshot"]})

    @app.get("/task_layout/<layout_id>/<view>.jpg")
    def task_layout_jpeg(layout_id, view):
        try:
            buf = task_layout_store.snapshot_bytes(layout_id, view)
        except LayoutError as e:
            return Response(str(e), status=400)
        if buf is None:
            return Response("no snapshot", status=404)
        return Response(buf, mimetype="image/jpeg")

    # ── Demo exports → RLinf/saved_demo/<task>/ ──────────────────────
    # ── Demo exports → RLinf/saved_demo/<task>[-ood]/<layout>_rNN.* ───────
    # Naming: directory <task>-ood when the layout id contains "ood", else
    # <task>; stem = <layout id>_r<NN> (the task id when the episode ran
    # without a layout), NN = rollout number counted per layout id. The
    # episode's meta.json remembers {"demo": {"dir", "stem"}} so delete /
    # re-mark / compact can find the export again.
    _DEMO_SIDE = (("video.mp4", ".mp4"), ("traj.jsonl", ".traj.jsonl"),
                  ("frame_times.json", ".frames.json"))

    def _demo_target(m: dict):
        task = (m.get("task") or "untagged").strip()
        lay = (m.get("layout") or "").strip()
        sub = f"{task}-ood" if "ood" in lay.lower() else task
        return pathlib.Path(DEMO_DIR) / sub, (lay or task)

    def _demo_rollout_no(out_dir: pathlib.Path, base: str) -> int:
        pat = re.compile(rf"^{re.escape(base)}_r(\d+)$")
        n = 0
        for j in out_dir.glob(f"{base}_r*.json"):
            mm = pat.match(j.stem)
            if mm:
                n = max(n, int(mm.group(1)))
        return n + 1

    def _demo_files(out_dir: pathlib.Path, stem: str):
        return [out_dir / f"{stem}{suf}" for _n, suf in _DEMO_SIDE] + [out_dir / f"{stem}.json"]

    def _write_meta(meta_f: pathlib.Path, m: dict) -> None:
        meta_f.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")

    def _export_episode(meta_f: pathlib.Path, m: dict) -> dict:
        """Copy video + sidecars + meta of one episode into saved_demo with the
        <layout>_rNN naming; re-exporting an already exported episode keeps
        its number. Raises FileNotFoundError when the episode has no video."""
        src = meta_f.parent / "video.mp4"
        if not src.exists():
            raise FileNotFoundError(f"{m.get('ep_id')} 没有 video.mp4")
        out_dir, base = _demo_target(m)
        out_dir.mkdir(parents=True, exist_ok=True)
        prev = m.get("demo") or {}
        if (prev.get("dir") == str(out_dir) and prev.get("stem")
                and (out_dir / f"{prev['stem']}.json").exists()):
            stem = prev["stem"]
        else:
            stem = f"{base}_r{_demo_rollout_no(out_dir, base):02d}"
        saved = []
        for name, suf in _DEMO_SIDE:
            side = meta_f.parent / name
            if side.exists():
                shutil.copy2(side, out_dir / f"{stem}{suf}")
                saved.append(f"{stem}{suf}")
        m["demo"] = {"dir": str(out_dir), "stem": stem}
        (out_dir / f"{stem}.json").write_text(
            json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
        saved.append(f"{stem}.json")
        _write_meta(meta_f, m)
        return {"dir": str(out_dir), "stem": stem, "files": saved,
                "path": str(out_dir / f"{stem}.mp4"),
                "msg": f"saved → {out_dir.name}/{stem}.mp4 (+{len(saved) - 1} sidecars)"}

    @app.post("/save_video")
    def save_video():
        """Manual export of the newest eval recording (same naming as the
        automatic export done by Mark ✓/✗)."""
        if runner.status()["running"]:
            return jsonify({"ok": False, "msg": "eval 进行中 — 先 Stop"}), 409
        newest = None
        root = pathlib.Path(EPISODES_ROOT)
        if root.exists():
            metas = []
            for meta_f in root.rglob("meta.json"):
                try:
                    m = json.loads(meta_f.read_text(encoding="utf-8"))
                except Exception:
                    continue
                metas.append((m.get("start_time", ""), meta_f, m))
            if metas:
                metas.sort(key=lambda t: t[0])
                newest = metas[-1][1:]
        if newest is None:
            return jsonify({"ok": False, "msg": "还没有 eval 录制"}), 404
        meta_f, m = newest
        try:
            r = _export_episode(meta_f, m)
        except FileNotFoundError as e:
            return jsonify({"ok": False, "msg": str(e)}), 404
        return jsonify({"ok": True, **r})

    @app.post("/save_layout")
    def save_layout():
        """Export a task's layout mask (json + reference JPEGs) as a demo."""
        body = request.get_json(silent=True) or {}
        lay_id = (body.get("layout") or "").strip()
        task = (body.get("task_id") or "").strip() or "untagged"
        if not lay_id:
            return jsonify({"ok": False, "msg": "当前任务没有 layout 蒙版"}), 400
        try:
            lay = task_layout_store.load(lay_id)
        except LayoutError as e:
            return jsonify({"ok": False, "msg": str(e)}), 404
        out_dir = pathlib.Path(DEMO_DIR) / task
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        p = out_dir / f"layout_{lay_id}.json"
        p.write_text(json.dumps(lay, ensure_ascii=False, indent=2), encoding="utf-8")
        saved.append(p.name)
        for view in ("exterior", "wrist"):
            buf = task_layout_store.snapshot_bytes(lay_id, view)
            if buf:
                q = out_dir / f"layout_{lay_id}.{view}.jpg"
                q.write_bytes(buf)
                saved.append(q.name)
        return jsonify({"ok": True, "path": str(out_dir),
                        "msg": f"saved → {out_dir} ({', '.join(saved)})"})

    @app.post("/save_ood_layout")
    def save_ood_layout():
        """Snapshot the CURRENT bench scene as a new OOD layout of the selected
        task — id `<task>-OOD<n>` in the shared layout dir — and attach it to
        the task (task_store.add_layout → it becomes the armed ghost), so an
        out-of-distribution arrangement is reproducible later and every
        episode's meta.json names the layout it ran under."""
        body = request.get_json(silent=True) or {}
        task = (body.get("task_id") or "").strip()
        trec = task_store.get(task) if task else None
        if trec is None:
            return jsonify({"ok": False, "msg": "先在 Task 区选择任务"}), 400
        pat = re.compile(rf"^{re.escape(task)}-OOD(\d+)$")
        known = ([m["id"] for m in task_layout_store.list()]
                 + list(trec.get("layouts") or []))
        n = 1 + max([int(mo.group(1)) for mo in map(pat.match, known) if mo],
                    default=0)
        lay_id = f"{task}-OOD{n}"
        lay, err = _snap_current(
            task_layout_store, lay_id,
            note=f"OOD layout of {task} — live scene snapshot from the eval portal")
        if err is not None:
            return err
        msg = task_store.add_layout(task, lay_id)
        return jsonify({"ok": True, "id": lay["id"], "task": task,
                        "has_snapshot": lay["has_snapshot"],
                        "msg": f"OOD layout {lay_id} 已拍并挂到 {task}({msg})"})

    @app.post("/task/create")
    def task_create():
        body = request.get_json(silent=True) or {}
        msg = task_store.create(body)
        return jsonify({"ok": msg == "created", "msg": msg})

    @app.post("/task/update")
    def task_update():
        body = request.get_json(silent=True) or {}
        tid = (body.get("id") or "").strip()
        msg = task_store.update(tid, body)
        return jsonify({"ok": msg == "updated", "msg": msg})

    @app.post("/task/delete")
    def task_delete():
        body = request.get_json(silent=True) or {}
        tid = (body.get("id") or "").strip()
        msg = task_store.delete(tid)
        return jsonify({"ok": msg == "deleted", "msg": msg})

    # ── Episode store: recorded evals grouped by task ────────────────
    # Episode identity is the ep_id (timestamp-unique); the meta.json
    # "task" field is editable, so lookups resolve by ep_id, not path.
    @staticmethod
    def _find_ep_meta(ep_id: str):
        root = pathlib.Path(EPISODES_ROOT)
        if not root.exists():
            return None
        for meta_f in root.rglob("meta.json"):
            try:
                m = json.loads(meta_f.read_text(encoding="utf-8"))
            except Exception:
                continue
            if m.get("ep_id") == ep_id:
                return meta_f, m
        return None

    def _norm_prompt(p) -> str:
        return " ".join(str(p or "").lower().split()).rstrip(".!")

    @app.get("/episodes")
    def episodes_list():
        task = (request.args.get("task") or "").strip()
        prompt2task = {_norm_prompt(t.get("prompt")): t["id"]
                       for t in task_store.list() if t.get("prompt")}
        eps = []
        root = pathlib.Path(EPISODES_ROOT)
        if root.exists():
            for meta_f in root.rglob("meta.json"):
                try:
                    m = json.loads(meta_f.read_text(encoding="utf-8"))
                except Exception:
                    continue
                # Episodes started without a task (or before the task
                # library existed) are mapped to the task whose prompt
                # matches — read-side only, meta.json is left untouched.
                if (m.get("task") or "untagged") == "untagged":
                    hit = prompt2task.get(_norm_prompt(m.get("prompt")))
                    if hit:
                        m["task"], m["task_auto"] = hit, True
                m.setdefault("task", "untagged")
                if task and m.get("task") != task:
                    continue
                m["has_video"] = (meta_f.parent / "video.mp4").exists()
                m["url"] = f"/episodes/video/{m.get('ep_id')}.mp4"
                eps.append(m)
        eps.sort(key=lambda e: e.get("start_time", ""), reverse=True)
        return jsonify({"episodes": eps})

    @app.get("/episodes/video/<ep_id>.mp4")
    def episode_video(ep_id):
        hit = _find_ep_meta(ep_id)
        if hit is None:
            return Response("no video", status=404)
        meta_f, _m = hit
        p = meta_f.parent / "video.mp4"
        if not p.exists():
            return Response("no video", status=404)
        return send_file(str(p), mimetype="video/mp4", conditional=True)

    @app.post("/episode/update")
    def episode_update():
        body = request.get_json(silent=True) or {}
        ep = (body.get("ep_id") or "").strip()
        hit = _find_ep_meta(ep)
        if hit is None:
            return jsonify({"ok": False, "msg": "episode not found"}), 404
        meta_f, meta = hit
        for k in ("mark", "note", "task", "ckpt", "prompt"):
            if k in body and body[k] is not None:
                meta[k] = body[k]
        meta_f.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        demo = meta.get("demo") or {}
        if demo.get("dir") and demo.get("stem"):
            jf = pathlib.Path(demo["dir"]) / f"{demo['stem']}.json"
            if jf.exists():
                jf.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return jsonify({"ok": True, "msg": "updated"})

    @app.post("/episode/delete")
    def episode_delete():
        body = request.get_json(silent=True) or {}
        ep = (body.get("ep_id") or "").strip()
        hit = _find_ep_meta(ep)
        if hit is None:
            return jsonify({"ok": False, "msg": "episode not found"}), 404
        meta_f, _m = hit
        shutil.rmtree(meta_f.parent)
        return jsonify({"ok": True, "msg": "deleted"})

    # Robot/gripper panel data comes from droid_nuc zerorpc (the old
    # robot_server HTTP on :4242 is gone — that port is now the zerorpc
    # socket, and HTTP GETs against it just time out). One state call
    # covers both rows; 1s TTL cache so browser polls don't hammer the
    # NUC during eval.
    _status_cache = {"t": 0.0, "robot": None, "gripper": None,
                     "ok": (False, False), "freedrive": False}

    def _poll_robot_status():
        now = time.time()
        if now - _status_cache["t"] < 1.0:
            return
        try:
            st = runner.droid_raw().get_robot_state()
            # polymetis control-loop liveness: the state timestamp advances
            # every 1 kHz tick, so an unchanged value across two polls (>= 1 s
            # apart) means the loop is down (libfranka reflex recovery) and
            # the arm ignores commands — see EvalRunner.watchdog_step.
            ts_ns = (int(st["timestamp_seconds"]) * 1_000_000_000
                     + int(st["timestamp_nanos"]))
            if ts_ns != _status_cache.get("ts_ns"):
                _status_cache.update(ts_ns=ts_ns, ts_since=now)
            robot = {
                "q": [round(float(x), 4) for x in st["joint_positions"]],
                "controller": "up",
                "last_cmd_ok": bool(st["prev_command_successful"]),
                "loop_stale_s": round(now - _status_cache.get("ts_since", now), 1),
            }
            gpos = float(st["gripper_position"])  # DROID: 1=closed
            gripper = {
                "width_m": runner.gripper_max_width_m * (1.0 - gpos),
                "position_norm": round(gpos, 3),
            }
            ok = (True, True)
        except Exception as e:
            msg = repr(e)
            if "UNAVAILABLE" in msg or "failed to connect" in msg:
                msg = "controller not launched (Start or Recover boots it)"
            robot = {"err": msg[:200]}
            gripper = {"err": msg[:200]}
            ok = (False, False)
        # While the franky freedrive sidecar holds FCI, the polymetis robot
        # state reads fail — surface that as a distinct, non-alarming state.
        freedrive = False
        if not ok[0]:
            try:
                fr = requests.get(
                    "http://172.16.0.2:4243/ping", timeout=0.8)
                freedrive = fr.ok and bool(fr.json().get("in_freedrive"))
            except Exception:
                pass
        _status_cache.update(t=now, robot=robot, gripper=gripper, ok=ok,
                             freedrive=freedrive)

    @app.get("/status")
    def status():
        _poll_robot_status()
        robot_ok, gripper_ok = _status_cache["ok"]
        return jsonify({
            "freedrive": _status_cache["freedrive"],
            "robot_ok": robot_ok,
            "robot": _status_cache["robot"],
            "gripper_ok": gripper_ok,
            "gripper": _status_cache["gripper"],
            "cam_running": cams.is_running(),
            "eval": runner.status(),
            "ckpt": serve.status(),
            "boot_ts": BOOT_TS,
        })

    @app.post("/start")
    def post_start():
        body = request.get_json(silent=True) or {}
        prompt = (body.get("prompt") or "grasp").strip() or "grasp"
        # Experiment tag → hooked capture server's episode.json. mode="none"
        # (or missing) sends no tag at all.
        tag = None
        mode = (body.get("mode") or "").strip()
        if mode and mode != "none":
            if mode not in EXP_MODES:
                return jsonify({"ok": False,
                                "msg": f"unknown mode {mode!r}"}), 400
            tag = {"mode": mode}
            task = (body.get("task") or "").strip()
            if task:
                tag["task"] = task[:64]
            try:
                pair = int(body.get("pair"))
                if pair > 0:
                    tag["pair"] = pair
            except (TypeError, ValueError):
                pass
            role = (body.get("role") or "").strip()
            if role in ("clean", "cor"):
                tag["role"] = role
        # Image preprocessing: pbc checkbox → center-crop, else DROID pad.
        runner.image_mode = "crop" if bool(body.get("pbc", False)) else "pad"
        # Episode recording metadata: which task + which checkpoint.
        runner._episode_meta = {
            "task": (body.get("task_id") or "").strip(),
            "layout": (body.get("layout") or "").strip(),
            "ckpt": ((serve.status() or {}).get("label") or "")
                    + (" +rtc" if serve.rtc else ""),
            "image_mode": runner.image_mode,
        }
        msg = runner.start(prompt, tag=tag)
        return jsonify({"ok": True, "msg": msg, "prompt": prompt, "tag": tag})

    @app.post("/stop")
    def post_stop():
        return jsonify({"ok": True, "msg": runner.stop()})

    @app.post("/eval/mark")
    def eval_mark():
        """One-click verdict: stop the rollout, wait for the recorder to
        flush meta.json, write the mark, then send the arm home (blocking).
        Mirrors the collect portal's mark flow. If nothing is running the
        newest episode gets the mark and the arm still homes."""
        body = request.get_json(silent=True) or {}
        mark = (body.get("mark") or "").strip()
        if mark not in ("success", "fail"):
            return jsonify({"ok": False, "msg": "mark must be success|fail"}), 400
        rec = runner._recorder  # noqa: SLF001
        was_running = runner.status()["running"]
        if was_running:
            runner.stop()
        # Recorder finalizes (writes meta.json) once the loop exits.
        meta_f = (rec.out_dir / "meta.json") if rec is not None else None
        if meta_f is not None:
            deadline = time.time() + 15.0
            while not meta_f.exists() and time.time() < deadline:
                time.sleep(0.2)
            if not meta_f.exists():
                return jsonify({"ok": False, "ep_id": rec.ep_id,
                                "msg": "recorder did not flush meta.json in 15s"}), 500
        else:
            hit = None
            root = pathlib.Path(EPISODES_ROOT)
            if root.exists():
                metas = sorted(root.rglob("meta.json"),
                               key=lambda p: p.parent.name, reverse=True)
                hit = metas[0] if metas else None
            if hit is None:
                return jsonify({"ok": False, "msg": "no eval episode to mark"}), 404
            meta_f = hit
        try:
            meta = json.loads(meta_f.read_text(encoding="utf-8"))
            if meta.get("abort") and not body.get("force"):
                # Watchdog/error-aborted episode: the NUC dropped the ticks (or
                # the loop died), so the verdict says nothing about the policy
                # — record "aborted", never success/fail (pairing/aggregation
                # filter on those). {"force": true} overrides.
                mark = "aborted"
            meta["mark"] = mark
            meta_f.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            return jsonify({"ok": False, "msg": f"mark write failed: {e}"}), 500
        ep_id = meta.get("ep_id") or meta_f.parent.name
        _log.info(f"eval mark: {meta.get('task')}/{ep_id} → {mark}")
        # Auto-export the recording (video + sidecars) for success / fail;
        # aborted episodes are not counted, so they are not exported either.
        demo = None
        if mark in ("success", "fail"):
            try:
                demo = _export_episode(meta_f, meta)
                _log.info(f"demo saved: {demo['dir']}/{demo['stem']}")
            except Exception as e:  # noqa: BLE001 — the verdict is already on disk
                demo = {"error": str(e)}
                _log.warning(f"eval mark: demo save failed: {e}")
        # Let /stop's halt tick land before commanding the home move.
        if was_running:
            time.sleep(0.7)
        home_msg = "homed"
        runner.homing = True
        try:
            # Streamed (non-blocking ticks), NOT update_joint_position(blocking=True):
            # the blocking move wedged the NUC zerorpc server after RTC episodes.
            runner.get_droid().stream_to_joint_position(
                home_store.get(), gripper_cmd=0.0)
        except ControllerNotResponding as e:
            home_msg = f"home failed: {e}"
            runner._last_error = str(e)  # noqa: SLF001 — surface in the status table
            _log.error(f"eval mark: {home_msg}")
        except Exception as e:
            home_msg = f"home failed: {e}"
            _log.warning(f"eval mark: {home_msg}")
        finally:
            runner.homing = False
        return jsonify({"ok": True, "ep_id": ep_id, "task": meta.get("task"),
                        "mark": mark, "stopped": was_running, "home": home_msg,
                        "demo": demo})

    @app.post("/home")
    def post_home():
        """Move to home_store.get() — DROID's reset pose by default
        (overridable via /set_home) — by streaming interpolated setpoints
        through droid_nuc (see DroidLikeClient.stream_to_joint_position; the
        blocking variant wedged the NUC server). Returns once the arm has
        been driven to the target."""
        if runner.status()["running"]:
            return jsonify({"ok": False, "msg": "eval running; stop first"}), 409
        runner.homing = True
        try:
            home_q = home_store.get()
            r = runner.get_droid().stream_to_joint_position(
                home_q, gripper_cmd=0.0
            )
            return jsonify({"ok": True, "home_q": home_q,
                            "result": f"moved to home (residual {r['residual']:.3f} rad)"})
        except ControllerNotResponding as e:
            runner._last_error = str(e)  # noqa: SLF001
            _log.error(f"home: {e}")
            return jsonify({"ok": False, "msg": str(e)}), 500
        except Exception as e:
            return jsonify({"ok": False, "msg": str(e)}), 500
        finally:
            runner.homing = False

    @app.post("/nuc_restart")
    def post_nuc_restart():
        """Recover a frozen/wedged NUC controller: restart the droid-nuc-fr3
        container and re-bootstrap the polymetis driver (tasl/launch/
        nuc-restart.sh, ~25 s). FCI must be active in Desk."""
        if runner.status()["running"]:
            return jsonify({"ok": False, "msg": "eval running; stop first"}), 409
        script = os.path.join(_TASL_DIR, "launch", "nuc-restart.sh")
        env = dict(os.environ, HOME=_HOME_DIR)   # dashboard runs as root: use the desktop user's key/site-packages
        try:
            # encoding: the dashboard runs under sudo with LANG=C — the
            # script prints UTF-8 (▶ ✓), which text=True would decode as ASCII.
            out = subprocess.run(["bash", script], env=env, capture_output=True,
                                 encoding="utf-8", errors="replace", timeout=240)
        except subprocess.TimeoutExpired:
            return jsonify({"ok": False, "msg": "nuc-restart.sh timed out (240s)"}), 500
        tail = "\n".join((out.stdout + out.stderr).strip().splitlines()[-6:])
        _log.info(f"nuc_restart rc={out.returncode}: {tail}")
        if out.returncode != 0:
            return jsonify({"ok": False, "msg": f"nuc-restart failed (rc={out.returncode}): {tail}"}), 500
        runner._last_error = None  # noqa: SLF001
        return jsonify({"ok": True, "msg": "NUC controller restarted + bootstrapped — press Go home", "log": tail})

    @app.post("/set_home")
    def post_set_home():
        if runner.status()["running"]:
            return jsonify({"ok": False,
                            "msg": "eval running; stop first"}), 409
        try:
            st = runner.get_droid().get_robot_state()
            new_q = [float(x) for x in st["joint_positions"]][:7]
            home_store.set(new_q)
            return jsonify({"ok": True, "home_q": new_q})
        except Exception as e:
            return jsonify({"ok": False, "msg": str(e)}), 500

    @app.post("/recover")
    def post_recover():
        """Clear FCI errors by re-bootstrapping. DROID's polymetis handles
        recover internally on launch_controller."""
        try:
            runner.get_droid().bootstrap()
            return jsonify({"ok": True, "result": "bootstrapped"})
        except Exception as e:
            return jsonify({"ok": False, "msg": str(e)}), 500

    @app.post("/robot_stop")
    def post_robot_stop():
        """Tear down polymetis driver — next motion needs Recover/Start."""
        try:
            d = runner.get_droid()
            d.kill_controller()
            return jsonify({"ok": True, "result": "controller killed"})
        except Exception as e:
            return jsonify({"ok": False, "msg": str(e)}), 500

    @app.get("/cam/<name>_policy.mjpg")
    def policy_view(name):
        mode = request.args.get("mode", "pad")
        if mode not in IMAGE_MODES:
            mode = "pad"

        def gen():
            boundary = b"--frame\r\n"
            while True:
                buf = cams.get_policy_view_jpeg(name, size=224, mode=mode)
                if buf is None:
                    time.sleep(0.05)
                    continue
                yield (boundary + b"Content-Type: image/jpeg\r\n\r\n"
                       + buf + b"\r\n")
                time.sleep(0.066)  # ~15fps, matches DROID control rate
        return Response(gen(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.post("/jog")
    def post_jog():
        if runner.status()["running"]:
            return jsonify({"ok": False,
                            "msg": "eval running; stop first"}), 409
        body = request.get_json(silent=True) or {}
        axis = body.get("axis", "")
        step = float(body.get("step", 0.01))
        # Cartesian jog via zerorpc EE pose (the legacy HTTP robot_server is
        # stopped — rs.jog_cartesian would just time out).
        mapping = {"x": 0, "y": 1, "z": 2, "rx": 3, "ry": 4, "rz": 5}
        if axis not in mapping:
            return jsonify({"ok": False, "msg": f"unknown axis {axis!r}"}), 400
        try:
            pose = runner.get_droid().get_ee_pose()
            pose[mapping[axis]] += step
            runner.get_droid().update_pose(pose, blocking=True)
            return jsonify({"ok": True, "msg": f"jog {axis} {step:+.3f}"})
        except Exception as exc:
            return jsonify({"ok": False, "msg": str(exc)}), 500

    @app.post("/gripper")
    def post_gripper():
        body = request.get_json(silent=True) or {}
        action = body.get("action", "")
        if action not in ("open", "close"):
            return jsonify({"ok": False, "msg": "action must be open|close"}), 400
        try:
            cmd = 0.0 if action == "open" else 1.0
            runner.get_droid().update_gripper(cmd, blocking=True)
            return jsonify({"ok": True, "result": f"gripper {action}"})
        except Exception as e:
            return jsonify({"ok": False, "msg": str(e)}), 500

    # One freedrive operation at a time — a double-click on Lock would run
    # two concurrent bootstraps and they'd tear down each other's driver.
    _freedrive_lock = threading.Lock()

    @app.post("/freedrive")
    def post_freedrive():
        """FCI is exclusive: the polymetis controller and the franky freedrive
        sidecar (NUC1 :4243) cannot hold it at the same time.

        Unlock: kill polymetis -> sidecar grabs FCI in low-impedance mode
                (arm becomes hand-pushable).
        Lock:   sidecar restores defaults + releases FCI -> re-bootstrap
                polymetis (takes ~15-30s; the console shows the steps).
        """
        if runner.status()["running"]:
            return jsonify({"ok": False,
                            "msg": "eval running; stop first"}), 409
        if not _freedrive_lock.acquire(blocking=False):
            return jsonify({"ok": False,
                            "msg": "freedrive 操作进行中 — 等它完成再点"}), 409
        try:
            body = request.get_json(silent=True) or {}
            enable = bool(body.get("enable", False))
            if enable:
                # Kill the polymetis driver WITHOUT bootstrapping first —
                # get_droid() would re-launch it (~15s) and then we'd kill
                # it again, racing franky for the FCI handoff.
                d = runner.droid_raw()
                try:
                    d.kill_controller()
                except Exception as exc:
                    _log.warning("freedrive: kill_controller failed "
                                 "(driver already down?): %s", str(exc)[:120])
                time.sleep(2.5)  # let the driver fully release FCI
                r = requests.post(
                    "http://172.16.0.2:4243/freedrive/on", timeout=20.0)
            else:
                r = requests.post(
                    "http://172.16.0.2:4243/freedrive/off", timeout=20.0)
                runner.get_droid().bootstrap()  # re-grab FCI
            j = r.json()
            return jsonify({"ok": r.ok,
                            "msg": str(j.get("mode") or j)})
        except Exception as exc:
            return jsonify({"ok": False, "msg": str(exc)}), 500
        finally:
            _freedrive_lock.release()

    @app.get("/audit")
    def get_audit():
        return jsonify({"recent": runner.audit_recent(40)})

    @app.post("/gripper_mode")
    def post_gripper_mode():
        body = request.get_json(silent=True) or {}
        mode = (body.get("mode") or "").strip()
        return jsonify({"ok": True, "msg": runner.set_gripper_mode(mode),
                        "now": runner.gripper_mode})

    @app.post("/commit_grasp")
    def post_commit_grasp():
        """Human-in-the-loop grasp primitive (zerorpc path — the legacy HTTP
        robot_server is stopped, so all motion goes through DroidLikeClient).

        Use after running the policy in force_open mode: when the arm
        has reached a good pre-grasp pose, click this to:
          1) stop the policy loop
          2) halt any in-flight motion (zero joint velocities)
          3) close Robotiq
          4) cartesian +Z 5cm to lift
        """
        body = request.get_json(silent=True) or {}
        lift_m = float(body.get("lift_m", 0.05))
        close_settle_s = float(body.get("close_settle_s", 0.8))
        droid = runner.get_droid()
        # 1. Stop the policy loop (idempotent if not running).
        stop_msg = runner.stop()
        # 2. Halt in-flight motion: one zero-velocity tick (mirrors /stop).
        try:
            droid.update_joint_velocity([0.0] * 8, blocking=False)
        except Exception as exc:
            _log.warning(f"commit: halt tick failed: {exc}")
        # Small settle before commanding new motion.
        time.sleep(0.15)
        # 3. Close Robotiq.
        try:
            droid.update_gripper(1.0, blocking=True)
        except Exception as exc:
            return jsonify({"ok": False, "step": "close",
                            "err": repr(exc)}), 500
        time.sleep(close_settle_s)
        # 4. Lift +Z (cartesian via EE pose, zerorpc).
        try:
            droid.lift_ee(dz=lift_m, blocking=True)
        except Exception as exc:
            return jsonify({"ok": False, "step": "lift",
                            "err": repr(exc)}), 500
        return jsonify({"ok": True,
                        "msg": f"grasped + lifted {lift_m}m",
                        "stop_msg": stop_msg,
                        "lift_m": lift_m})

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8003)
    p.add_argument("--bind", default="0.0.0.0")
    # NUC1 is reachable on the robot machine network (172.16.0.2), not the
    # legacy Tailscale address (100.75.6.62) — that tailnet IP is long gone.
    _nuc_host = os.environ.get("NUC1_HOST", "172.16.0.2")
    p.add_argument("--robot-server", default=f"http://{_nuc_host}:4242")
    p.add_argument("--droid-url", default=f"tcp://{_nuc_host}:4242",
                   help="polymetis zerorpc address on NUC1 (machine network)")
    p.add_argument("--policy-host", default="127.0.0.1",
                   help="openpi serve_policy host")
    p.add_argument("--policy-port", type=int, default=8000,
                   help="openpi serve_policy port")
    p.add_argument("--resolution", default="HD720",
                   choices=["HD2K", "HD1080", "HD720"])
    p.add_argument("--jpeg-quality", type=int, default=70)
    p.add_argument("--no-cam-on-start", action="store_true",
                   help="Don't acquire cameras at startup.")
    p.add_argument("--allow-missing-wrist", action="store_true",
                   help="Run the policy with a BLACK wrist view when the "
                        "wrist ZED is absent (degraded: half-blind policy).")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger().addHandler(_RingHandler())

    # Use the canonical env yaml names (wrist_1 = exterior right ZED 2i,
    # wrist_2 = wrist-mounted ZED Mini) so the same /cam/<name>.jpg URL
    # works for both the UI tiles and the env's HTTP frame source.
    cams = CamManager(
        serials={"wrist_1": SN_ZED_2I_RIGHT, "wrist_2": SN_ZED_MINI_WRIST},
        resolution=args.resolution,
        jpeg_quality=args.jpeg_quality,
    )
    if not args.no_cam_on_start:
        msg = cams.start()
        _log.info(f"CamManager.start: {msg}")

    rs = RS(args.robot_server)
    home_store = HomeStore(HOME_Q_DEFAULT)
    runner = EvalRunner(
        cams, home_store,
        rs_url=args.robot_server,
        droid_url=args.droid_url,
        policy_host=args.policy_host,
        policy_port=args.policy_port,
        allow_missing_wrist=args.allow_missing_wrist,
    )
    if args.allow_missing_wrist:
        _log.warning("--allow-missing-wrist: wrist view will be BLACK if the "
                     "wrist ZED is absent — policy runs half-blind")
    app = build_app(rs, cams, runner, home_store)
    _log.info(f"serving on {args.bind}:{args.port}")
    app.run(host=args.bind, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
