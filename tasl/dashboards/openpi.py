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
import io
import json
import logging
import os
import pathlib
import signal
import subprocess
import threading
import time
from typing import Optional

import cv2
import numpy as np
import requests
from flask import Flask, Response, jsonify, render_template_string, request
from PIL import Image

from clients.droid_client import DroidLikeClient


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
            data = json.loads(HOME_STORE_PATH.read_text())
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
                                "saved_at": time.time()}, indent=2)
                )
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
            # bgra is (H, W, 4); convert to RGB for JPEG
            rgb = cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=self.jpeg_quality)
            with self._frame_lock:
                self._latest_jpeg[name] = buf.getvalue()
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
        _log.info("CamManager stopped, USB released")

    def is_running(self) -> bool:
        return self._running

    def get_jpeg(self, name: str) -> Optional[bytes]:
        with self._frame_lock:
            return self._latest_jpeg.get(name)

    def get_policy_view_jpeg(self, name: str, size: int = 224) -> Optional[bytes]:
        """Return the exact frame the policy sees: aspect-preserving resize
        + zero-pad to (size, size). Matches openpi_client.image_tools
        resize_with_pad — what pi05_droid was trained on."""
        with self._frame_lock:
            jpeg = self._latest_jpeg.get(name)
        if jpeg is None:
            return None
        try:
            arr = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
            h, w = arr.shape[:2]
            scale = min(size / h, size / w)
            new_h = int(round(h * scale))
            new_w = int(round(w * scale))
            resized = cv2.resize(arr, (new_w, new_h),
                                 interpolation=cv2.INTER_AREA)
            small = np.zeros((size, size, arr.shape[2]), dtype=arr.dtype)
            y_off = (size - new_h) // 2
            x_off = (size - new_w) // 2
            small[y_off:y_off + new_h, x_off:x_off + new_w] = resized
            buf = io.BytesIO()
            Image.fromarray(small).save(buf, format="JPEG", quality=self.jpeg_quality)
            return buf.getvalue()
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────
# Openpi policy runner (in-thread WS inference loop)
# ─────────────────────────────────────────────────────────────────────
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
                 droid_url: str = "tcp://100.75.6.62:4242"):
        self.cam_mgr = cam_mgr
        self.home_store = home_store
        self.rs_url = rs_url.rstrip("/")
        self.droid_url = droid_url
        self._droid: Optional[DroidLikeClient] = None
        self.policy_host = policy_host
        self.policy_port = policy_port
        self.last_prompt = "grasp"
        # Per-iter knobs (sane defaults; overridable later via /config).
        self.open_loop_horizon = 4
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

    def start(self, prompt: str) -> str:
        with self._lock:
            if self._running:
                return "already running"
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

    def status(self) -> dict:
        return {
            "running": self._running,
            "iter": self._iter,
            "last_prompt": self.last_prompt,
            "last_error": self._last_error,
            "last_grip_raw": self._last_grip_raw,
            "last_dq_max": self._last_dq_max,
            "last_infer_ms": self._last_infer_ms,
            "open_loop_horizon": self.open_loop_horizon,
            "delta_scale": self.delta_scale,
            "dynamics_factor": self.dynamics_factor,
            "control_mode": "joint_position_delta",
            "gripper_mode": self.gripper_mode,
            "gripper_latched": self._latched_close,
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

            chunk_dt = 1.0 / self.control_hz
            while self._iter < self.max_iterations and not self._stop.is_set():
                self._iter += 1
                # Frames from CamManager (always running).
                ext_jpeg = self.cam_mgr.get_jpeg("wrist_1")
                wrist_jpeg = self.cam_mgr.get_jpeg("wrist_2")
                if ext_jpeg is None or wrist_jpeg is None:
                    time.sleep(0.05)
                    self._iter -= 1
                    continue
                try:
                    ext_rgb = np.asarray(
                        Image.open(io.BytesIO(ext_jpeg)).convert("RGB")
                    )
                    wrist_rgb = np.asarray(
                        Image.open(io.BytesIO(wrist_jpeg)).convert("RGB")
                    )
                except Exception as e:
                    _log.warning(f"jpeg decode failed: {e}; skipping iter")
                    continue
                ext_r = self._resize_with_pad(ext_rgb, 224)
                wrist_r = self._resize_with_pad(wrist_rgb, 224)

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

                if self._stop.is_set():
                    break

                # DROID position-target streaming: clip raw to [-1, 1],
                # accumulate q_target = q + sum(action[:k] * delta_scale).
                # This matches openpi/examples/droid/main.py + droid/franka/robot.py
                # exactly. Joint-velocity dispatch (our old path) is the WRONG
                # control mode + wrong scale (~3x too slow + no impedance).
                n = min(self.open_loop_horizon, actions.shape[0])
                clipped = np.clip(actions[:n, :7], -1.0, 1.0)
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

    @staticmethod
    def _resize_with_pad(arr: np.ndarray, size: int) -> np.ndarray:
        """Aspect-preserving resize + zero-pad to (size, size).
        Mirrors openpi_client.image_tools.resize_with_pad — what
        pi05_droid was trained + serve_policy expects. 1280×720 →
        224×126 + 49px top/bottom zero pad. Center-cropping instead
        discards the left/right ~280px the model needs for context.
        """
        h, w = arr.shape[:2]
        scale = min(size / h, size / w)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        resized = cv2.resize(arr, (new_w, new_h),
                             interpolation=cv2.INTER_AREA)
        out = np.zeros((size, size, arr.shape[2]), dtype=arr.dtype)
        y_off = (size - new_h) // 2
        x_off = (size - new_w) // 2
        out[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return out

    def _audit_write(self, rec: dict) -> None:
        line = json.dumps(rec, default=float)
        with self._audit_lock:
            self._audit_ring.append(rec)
            if len(self._audit_ring) > 40:
                self._audit_ring.pop(0)
            try:
                with self.audit_path.open("a") as f:
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
INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8">
<title>TASL FR3 — openpi dashboard</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; margin: 18px;
         background: #111; color: #e5e5e5; }
  h1 { font-size: 1.4rem; margin: 0 0 12px 0; }
  .row { display: flex; gap: 16px; align-items: flex-start; flex-wrap: wrap; }
  .col { flex: 1 1 0; min-width: 280px; }
  .cam { background: #000; padding: 6px; border-radius: 6px;
         border: 1px solid #333; }
  .cam img { width: 100%; max-width: 480px; display: block; border-radius: 4px;}
  .cam .label { font-size: 0.85rem; color: #aaa; padding: 4px 0; }
  .ctl { background: #1a1a1a; padding: 14px; border-radius: 6px;
         border: 1px solid #2a2a2a; }
  input[type=text] { width: 100%; padding: 8px; font-size: 1rem;
                     background: #0c0c0c; color: #eee;
                     border: 1px solid #333; border-radius: 4px; }
  button { padding: 8px 14px; font-size: 0.95rem; cursor: pointer;
           border: 1px solid #333; border-radius: 4px;
           background: #222; color: #eee; margin: 4px 4px 0 0; }
  button:hover { background: #2c2c2c; }
  button.primary { background: #1f4f1f; }
  button.danger  { background: #5a1f1f; }
  button.warn    { background: #5a4f1f; }
  pre { background: #0c0c0c; padding: 10px; border-radius: 4px;
        border: 1px solid #2a2a2a; max-height: 280px; overflow: auto;
        font-size: 0.78rem; color: #ccc; }
  table { font-size: 0.85rem; border-collapse: collapse; }
  td, th { padding: 3px 8px; border-bottom: 1px solid #2a2a2a;
           text-align: left; vertical-align: top; }
  .dot { display: inline-block; width: 8px; height: 8px;
         border-radius: 50%; margin-right: 6px; }
  .ok { background: #4caf50; } .bad { background: #d6322a; }
</style></head>
<body>
<h1>TASL FR3 — openpi eval dashboard</h1>
<div class="row">
  <div class="col cam">
    <div class="label">exterior — full feed (HD720) [wrist_1]</div>
    <img src="/cam/wrist_1.mjpg" alt="exterior"/>
  </div>
  <div class="col cam">
    <div class="label">wrist — full feed (HD720) [wrist_2]</div>
    <img src="/cam/wrist_2.mjpg" alt="wrist"/>
  </div>
</div>
<div class="row" style="margin-top:8px">
  <div class="col cam" style="flex:0 1 auto">
    <div class="label">policy view — exterior (224×224 center-crop)</div>
    <img src="/cam/wrist_1_policy.mjpg" alt="exterior policy"
         style="width:224px;height:224px;image-rendering:pixelated"/>
  </div>
  <div class="col cam" style="flex:0 1 auto">
    <div class="label">policy view — wrist (224×224)</div>
    <img src="/cam/wrist_2_policy.mjpg" alt="wrist policy"
         style="width:224px;height:224px;image-rendering:pixelated"/>
  </div>
</div>
<div class="row" style="margin-top:16px">
  <div class="col ctl">
    <h3>Run eval</h3>
    <label>Prompt</label>
    <input type="text" id="prompt" value="grasp"/>
    <div style="margin-top:8px">
      <button class="primary" onclick="startEval()">Start eval</button>
      <button class="danger" onclick="api('/stop')">Stop eval</button>
      <button class="warn" onclick="commitGrasp()">Commit grasp</button>
    </div>
    <div style="font-size:0.78rem;color:#888;margin-top:2px">
      Commit grasp = stop policy → close gripper → lift +Z 5cm.
      Use after force_open lets the arm reach a good pre-grasp pose.
    </div>
    <div style="margin-top:10px">
      <label>Gripper mode</label>
      <select id="gripMode" onchange="setGripperMode()">
        <option value="proportional">proportional (DROID exact — default)</option>
        <option value="raw_binary">raw_binary (legacy binary)</option>
        <option value="latch_close">latch_close (close stays closed)</option>
        <option value="force_open">force_open (test arm only)</option>
        <option value="raw_inverted">raw_inverted (sign flip A/B)</option>
        <option value="latch_inverted">latch_inverted (latch + flip)</option>
      </select>
    </div>
    <h3 style="margin-top:18px">Robot</h3>
    <button onclick="api('/home')">Go home</button>
    <button class="primary" onclick="setHome()">Set current as home</button>
    <button class="warn" onclick="api('/recover')">Recover</button>
    <button class="danger" onclick="api('/robot_stop')">Robot stop</button>
    <div style="font-size:0.78rem;color:#888;margin-top:4px">
      Set current as home = save the arm's current q to dashboard_home.json
      so future Go home / eval reset comes here.
    </div>
    <h3 style="margin-top:18px">Jog (cartesian, EE frame)</h3>
    <div>
      <label>step</label>
      <select id="jogStep">
        <option value="0.005">5 mm</option>
        <option value="0.01" selected>1 cm</option>
        <option value="0.03">3 cm</option>
        <option value="0.05">5 cm</option>
      </select>
      <label style="margin-left:12px">rot</label>
      <select id="jogRotStep">
        <option value="0.0873">5°</option>
        <option value="0.1745" selected>10°</option>
        <option value="0.3491">20°</option>
      </select>
    </div>
    <table style="margin-top:8px"><tr>
      <td>
        <div><button onclick="jog('x',+1)">+X (fwd)</button>
             <button onclick="jog('x',-1)">−X (back)</button></div>
        <div><button onclick="jog('y',+1)">+Y (left)</button>
             <button onclick="jog('y',-1)">−Y (right)</button></div>
        <div><button onclick="jog('z',+1)">+Z (up)</button>
             <button onclick="jog('z',-1)">−Z (down)</button></div>
      </td>
      <td style="padding-left:12px">
        <div><button onclick="jogRot('rx',+1)">+rx</button>
             <button onclick="jogRot('rx',-1)">−rx</button></div>
        <div><button onclick="jogRot('ry',+1)">+ry</button>
             <button onclick="jogRot('ry',-1)">−ry</button></div>
        <div><button onclick="jogRot('rz',+1)">+rz</button>
             <button onclick="jogRot('rz',-1)">−rz</button></div>
      </td>
    </tr></table>
    <h3 style="margin-top:18px">Gripper</h3>
    <button onclick="api('/gripper',{action:'open'})">Open</button>
    <button onclick="api('/gripper',{action:'close'})">Close</button>
    <h3 style="margin-top:18px">Manual guidance (Desk-style)</h3>
    <div style="font-size:0.78rem;color:#aaa;margin-bottom:6px">
      Unlock makes joints compliant so you can push the arm by hand.
      Always Lock back before running eval or /move commands.
    </div>
    <button class="warn" onclick="api('/freedrive',{enable:true})">Unlock joints</button>
    <button class="primary" onclick="api('/freedrive',{enable:false})">Lock joints</button>
  </div>
  <div class="col ctl">
    <h3>Status</h3>
    <div id="status">loading…</div>
  </div>
</div>
<h3 style="margin-top:18px">Gripper audit (4 signals per iter)</h3>
<pre id="audit" style="max-height:300px">(idle — start the policy)</pre>
<div style="font-size:0.78rem;color:#888;margin-top:-6px">
  Columns: iter · raw_action_grip · want_close · action_taken · skip_reason · gw_pre_m → gw_post_m · obs_norm
  &nbsp;&nbsp;Also written to <code>/tmp/_grip_audit.jsonl</code> for offline grep.
</div>
<script>
// Note: do NOT name this `prompt` — that shadows window.prompt and
// breaks inline onclick handlers in some browsers / cache states.
const promptInput = document.getElementById('prompt');
async function api(path, body) {
  const r = await fetch(path, {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: body ? JSON.stringify(body) : '{}',
  });
  const j = await r.json();
  console.log(path, j);
  return j;
}
async function startEval() {
  const p = (promptInput && promptInput.value) ? promptInput.value : 'grasp';
  return api('/start', {prompt: p});
}
async function setGripperMode() {
  const m = document.getElementById('gripMode').value;
  return api('/gripper_mode', {mode: m});
}
async function commitGrasp() {
  if (!confirm('Commit grasp: stop policy → close gripper → lift +Z 5cm. Continue?')) return;
  return api('/commit_grasp');
}
async function setHome() {
  if (!confirm('Save the arm\\'s current pose as home? Future Go home / eval reset will move here.')) return;
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
async function refresh() {
  try {
    const r = await fetch('/status'); const j = await r.json();
    const s = j.robot || {};
    const re = j.eval || {};
    const dot = (ok) => '<span class="dot ' + (ok?'ok':'bad') + '"></span>';
    let html = '<table>';
    html += '<tr><td>'+dot(j.robot_ok)+'robot</td>'
         +  '<td>q=' + (s.q ? s.q.map(x=>x.toFixed(2)).join(', ') : '?')
         +  '<br>' + (s.err ? s.err
                            : 'controller=' + s.controller
                              + '  last_cmd_ok=' + s.last_cmd_ok)
         +  '</td></tr>';
    html += '<tr><td>'+dot(j.gripper_ok)+'gripper</td>'
         +  '<td>width=' + (j.gripper && j.gripper.width_m !== undefined
                            ? j.gripper.width_m.toFixed(3)+'m' : '?')
         +  '</td></tr>';
    html += '<tr><td>'+dot(re.running)+'eval</td>'
         +  '<td>pid=' + (re.pid||'-')
         +  '  rc=' + (re.returncode===null?'-':re.returncode)
         +  '<br>prompt=' + (re.last_prompt || '-') + '</td></tr>';
    html += '<tr><td>'+dot(j.cam_running)+'cams</td>'
         +  '<td>' + (j.cam_running ? 'dashboard holds' : 'released (eval or idle)') + '</td></tr>';
    // In-process loop stats (openpi-standalone)
    if (re.iter !== undefined) {
      const fmt = (x) => (x === null || x === undefined) ? '-' : Number(x).toFixed(3);
      html += '<tr><td>policy</td><td>'
           +  'iter=' + re.iter
           +  '<br>last infer ms=' + (re.last_infer_ms ? re.last_infer_ms.toFixed(0) : '-')
           +  '<br>last grip cmd (raw, 0=open 1=close)=' + fmt(re.last_grip_raw)
           +  '<br>last |dq| max=' + fmt(re.last_dq_max)
           +  '<br>horizon=' + re.open_loop_horizon
           +  ' scale=' + re.action_scale
           +  ' cap=' + re.max_joint_vel;
      if (re.last_error) html += '<br><span style="color:#e58c8c">err: '
                                + re.last_error + '</span>';
      html += '</td></tr>';
    }
    html += '</table>';
    document.getElementById('status').innerHTML = html;
    // Audit panel: pull /audit and render fixed-width table
    try {
      const ar = await (await fetch('/audit')).json();
      const lines = (ar.recent||[]).map(r => {
        const f = (x, d) => (x === null || x === undefined) ? '-' : Number(x).toFixed(d);
        const skip = r.reason_skipped || '';
        const act = r.action_taken || '';
        return `${String(r.iter).padStart(4)} `
             + `raw=${f(r.raw_action_grip,3).padStart(6)} `
             + `want=${r.want_close?'CLOSE':'open '} `
             + `did=${act.padEnd(5)} `
             + `skip=${skip.padEnd(12)} `
             + `gw ${f(r.gw_pre_m,3)}→${f(r.gw_post_m,3)} `
             + `obs=${f(r.gw_obs_norm,3)}`;
      });
      document.getElementById('audit').textContent =
        lines.length ? lines.join('\\n') : '(no audit rows yet — start the policy)';
    } catch(e) {}
  } catch (e) {
    document.getElementById('status').innerHTML = 'status error: ' + e;
  }
}
refresh(); setInterval(refresh, 1500);
</script>
</body></html>
"""


def build_app(rs: RS, cams: CamManager, runner: EvalRunner,
              home_store: "HomeStore") -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template_string(INDEX_HTML)

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

    # Robot/gripper panel data comes from droid_nuc zerorpc (the old
    # robot_server HTTP on :4242 is gone — that port is now the zerorpc
    # socket, and HTTP GETs against it just time out). One state call
    # covers both rows; 1s TTL cache so browser polls don't hammer the
    # NUC during eval.
    _status_cache = {"t": 0.0, "robot": None, "gripper": None,
                     "ok": (False, False)}

    def _poll_robot_status():
        now = time.time()
        if now - _status_cache["t"] < 1.0:
            return
        try:
            st = runner.droid_raw().get_robot_state()
            robot = {
                "q": [round(float(x), 4) for x in st["joint_positions"]],
                "controller": "up",
                "last_cmd_ok": bool(st["prev_command_successful"]),
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
        _status_cache.update(t=now, robot=robot, gripper=gripper, ok=ok)

    @app.get("/status")
    def status():
        _poll_robot_status()
        robot_ok, gripper_ok = _status_cache["ok"]
        return jsonify({
            "robot_ok": robot_ok,
            "robot": _status_cache["robot"],
            "gripper_ok": gripper_ok,
            "gripper": _status_cache["gripper"],
            "cam_running": cams.is_running(),
            "eval": runner.status(),
        })

    @app.post("/start")
    def post_start():
        body = request.get_json(silent=True) or {}
        prompt = (body.get("prompt") or "grasp").strip() or "grasp"
        msg = runner.start(prompt)
        return jsonify({"ok": True, "msg": msg, "prompt": prompt})

    @app.post("/stop")
    def post_stop():
        return jsonify({"ok": True, "msg": runner.stop()})

    @app.post("/home")
    def post_home():
        """Blocking move to home_store.get() — DROID's reset pose by default
        (overridable via /set_home). Routes through droid_nuc, not the old
        robot_server."""
        try:
            home_q = home_store.get()
            runner.get_droid().update_joint_position(
                home_q, gripper_cmd=0.0, blocking=True
            )
            return jsonify({"ok": True, "home_q": home_q,
                            "result": "moved to home"})
        except Exception as e:
            return jsonify({"ok": False, "msg": str(e)}), 500

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
        def gen():
            boundary = b"--frame\r\n"
            while True:
                buf = cams.get_policy_view_jpeg(name, size=224)
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
        kwargs = {k: 0.0 for k in ("dx", "dy", "dz", "drx", "dry", "drz")}
        mapping = {"x": "dx", "y": "dy", "z": "dz",
                   "rx": "drx", "ry": "dry", "rz": "drz"}
        if axis not in mapping:
            return jsonify({"ok": False, "msg": f"unknown axis {axis!r}"}), 400
        kwargs[mapping[axis]] = step
        return jsonify({"ok": True, "result": rs.jog_cartesian(**kwargs)})

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

    @app.post("/freedrive")
    def post_freedrive():
        if runner.status()["running"]:
            return jsonify({"ok": False,
                            "msg": "eval running; stop first"}), 409
        body = request.get_json(silent=True) or {}
        enable = bool(body.get("enable", False))
        return jsonify({"ok": True,
                        "enabled": enable,
                        "result": rs.freedrive(enable)})

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
        """Human-in-the-loop grasp primitive.

        Use after running the policy in force_open mode: when the arm
        has reached a good pre-grasp pose, click this to:
          1) stop the policy loop
          2) preempt any in-flight motion (jvel_stop)
          3) close Robotiq
          4) cartesian +Z 5cm to lift
        """
        body = request.get_json(silent=True) or {}
        lift_m = float(body.get("lift_m", 0.05))
        close_settle_s = float(body.get("close_settle_s", 0.8))
        # 1. Stop the policy loop (idempotent if not running).
        stop_msg = runner.stop()
        # 2. Halt any in-flight ruckig chunk.
        try:
            requests.post(
                f"{runner.rs_url}/move/joint_velocity_stop", timeout=2.0
            )
        except Exception as exc:
            _log.warning(f"commit: jvel_stop failed: {exc}")
        # Small settle before commanding new motion.
        time.sleep(0.15)
        # 3. Close Robotiq.
        try:
            requests.post(
                f"{runner.rs_url}/robotiq/close",
                json={"speed": 0.3}, timeout=5.0,
            )
        except Exception as exc:
            return jsonify({"ok": False, "step": "close",
                            "err": repr(exc)}), 500
        time.sleep(close_settle_s)
        # 4. Lift +Z.
        try:
            rs_lift = rs.jog_cartesian(
                dx=0.0, dy=0.0, dz=lift_m, dynamics_factor=0.1
            )
        except Exception as exc:
            return jsonify({"ok": False, "step": "lift",
                            "err": repr(exc)}), 500
        return jsonify({"ok": True,
                        "stop_msg": stop_msg,
                        "lift_m": lift_m,
                        "lift_result": rs_lift})

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8003)
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--robot-server", default="http://100.75.6.62:4242")
    p.add_argument("--policy-host", default="127.0.0.1",
                   help="openpi serve_policy host")
    p.add_argument("--policy-port", type=int, default=8000,
                   help="openpi serve_policy port")
    p.add_argument("--resolution", default="HD720",
                   choices=["HD2K", "HD1080", "HD720"])
    p.add_argument("--jpeg-quality", type=int, default=70)
    p.add_argument("--no-cam-on-start", action="store_true",
                   help="Don't acquire cameras at startup.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

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
        policy_host=args.policy_host,
        policy_port=args.policy_port,
    )
    app = build_app(rs, cams, runner, home_store)
    _log.info(f"serving on {args.bind}:{args.port}")
    app.run(host=args.bind, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
