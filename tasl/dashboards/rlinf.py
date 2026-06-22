"""TASL FR3 RLinf eval dashboard.

Same UX shell as the 2026-05-28 openpi-standalone dashboard, but wired
to the RLinf eval pipeline instead of openpi `serve_policy`.

Lifecycle:
  • IDLE: dashboard owns the two ZED cameras (right ZED 2i + wrist ZED
    Mini), streams MJPEG; status panel polls NUC1 robot_server.
  • START: dashboard releases cameras (closes pyzed handles → frees
    USB), spawns `eval_embodied_agent.py` with `task_description`
    overridden by the user's prompt. Eval's EnvWorker then opens the
    cameras through RLinf's own pipeline.
  • STOP: kills the eval Python process tree (Ray workers + main).
    Once it exits, dashboard re-opens cameras and returns to IDLE.
  • HOME: POST NUC1 /move/joint to the calibration anchor pose.

Designed to run INSIDE the long-lived `rlinf-eval` docker container so
it shares the openpi venv (pyzed + rlinf + ray) and reaches NUC1 over
`--network host`. Source is mounted at /workspace/rlinf.

Launch (host side):
    docker exec -d rlinf-eval bash -c \
        "cd /workspace/rlinf && PYTHONPATH=/workspace/rlinf \
        PYTHONPATH=/workspace/rlinf/tasl /opt/venv/openpi/bin/python \
        /workspace/rlinf/tasl/dashboards/rlinf.py --port 8003"

Open: http://<desktop-tailscale-or-lan-ip>:8003
"""
from __future__ import annotations

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


# ── Hardware constants (verified 2026-05-27) ──────────────────────────
SN_ZED_2I_RIGHT = 36443134
SN_ZED_MINI_WRIST = 17150101

# Default home pose from tools/calibration/anchors/eth_right_cam.yaml.
# Overridden at runtime by `home_store` if a user has clicked "Set home"
# before — that captures whatever the arm's current q is and persists.
HOME_Q_DEFAULT = [-0.049, 0.004, 0.532, -1.752, 0.341, 2.118, -0.281]
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

# Path layout — defaults assume running in container; flip via --mode host
# at launch time. Container paths are mounted-in; host paths are real host fs.
CONTAINER_NAME = "rlinf-eval"
DEFAULT_REPO_PATH_CONTAINER = "/workspace/rlinf"
DEFAULT_REPO_PATH_HOST = "/home/franka_desktop/work/rlinf-clone"
# Resolved at launch (set in main()):
REPO_PATH: pathlib.Path = pathlib.Path(DEFAULT_REPO_PATH_CONTAINER)
CONFIG_DIR: pathlib.Path = REPO_PATH / "examples/embodiment/config"
EVAL_SCRIPT: pathlib.Path = REPO_PATH / "examples/embodiment/eval_embodied_agent.py"
SAVE_DIR: pathlib.Path = pathlib.Path("/tmp/rlinf_pi05_droid_eval")
EVAL_LOG: pathlib.Path = pathlib.Path("/workspace/rlinf/_dashboard_eval.log")
# When dashboard runs on host, eval still runs in the container, so we
# need to (a) prefix the launch with docker exec and (b) read log/parquet
# files through docker exec because they live inside the container's /tmp.
HOST_MODE: bool = False

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
        """Return the exact frame the policy would see: square center-crop + resize.

        Mirrors FrankaJointVelEnv._center_crop_resize so the policy view is
        WYSIWYG with what RLinf's EnvWorker feeds the model.
        """
        with self._frame_lock:
            jpeg = self._latest_jpeg.get(name)
        if jpeg is None:
            return None
        try:
            arr = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
            h, w = arr.shape[:2]
            side = min(h, w)
            sy = (h - side) // 2
            sx = (w - side) // 2
            crop = arr[sy:sy + side, sx:sx + side]
            small = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
            buf = io.BytesIO()
            Image.fromarray(small).save(buf, format="JPEG", quality=self.jpeg_quality)
            return buf.getvalue()
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────
# Eval runner (spawn / kill RLinf eval subprocess)
# ─────────────────────────────────────────────────────────────────────
class EvalRunner:
    """Spawns the RLinf eval as a child process; tracks PID + log tail."""

    def __init__(self, cam_mgr: CamManager, home_store: "HomeStore",
                 rs_url: str = "http://100.75.6.62:4242"):
        self.cam_mgr = cam_mgr
        self.home_store = home_store
        self.rs_url = rs_url.rstrip("/")
        self.proc: Optional[subprocess.Popen] = None
        self.config_name = "realworld_eval_pi05_droid_polymetis"
        self.last_prompt = "grasp"
        self._lock = threading.Lock()

    def start(self, prompt: str) -> str:
        with self._lock:
            if self.proc is not None and self.proc.poll() is None:
                return "already running"
            self.last_prompt = prompt

            # NOTE: dashboard now owns cameras permanently. Env consumes
            # frames via HTTP (/cam/<name>.jpg) — no cam handoff, no USB
            # tantrums on start/stop. Do NOT call cam_mgr.stop() here.

            # 1. Clean old data dir (LeRobot writer refuses to overwrite).
            if HOST_MODE:
                subprocess.run(
                    ["docker", "exec", CONTAINER_NAME, "rm", "-rf",
                     str(SAVE_DIR)],
                    check=False,
                )
            else:
                try:
                    if SAVE_DIR.exists():
                        import shutil
                        shutil.rmtree(SAVE_DIR)
                except Exception as e:
                    _log.warning(f"failed to clean {SAVE_DIR}: {e}")

            # 3. Build the eval command. We use Hydra overrides so we
            # don't need to mutate the YAML on disk.
            ts = time.strftime("%Y%m%d-%H:%M:%S-dashboard")
            container_repo = pathlib.Path(DEFAULT_REPO_PATH_CONTAINER)
            log_dir_inside = container_repo / "logs" / ts
            # Hydra-formatted joint list (no spaces — hydra parses).
            home_q = self.home_store.get()
            home_q_str = "[" + ",".join(f"{v:.6f}" for v in home_q) + "]"
            inner_cmd = (
                f"mkdir -p {log_dir_inside} && "
                f"cd {container_repo} && "
                f"PATH=/opt/venv/openpi/bin:$PATH "
                f"PYTHONPATH={container_repo} "
                f"/opt/venv/openpi/bin/python "
                f"{container_repo}/examples/embodiment/eval_embodied_agent.py "
                f"--config-path={container_repo}/examples/embodiment/config/ "
                f"--config-name={self.config_name} "
                f"runner.logger.log_path={log_dir_inside} "
                f"++env.train.override_cfg.task_description='{prompt}' "
                f"++env.eval.override_cfg.task_description='{prompt}' "
                f"++env.train.override_cfg.joint_reset_qpos={home_q_str} "
                f"++env.eval.override_cfg.joint_reset_qpos={home_q_str} "
                f"> {container_repo}/_dashboard_eval.log 2>&1"
            )
            if HOST_MODE:
                # docker exec without -d so we own the PID + can SIGTERM it.
                cmd = ["docker", "exec", CONTAINER_NAME, "bash", "-c", inner_cmd]
                _log.info(f"eval cmd (host mode): docker exec {CONTAINER_NAME} ...")
            else:
                cmd = ["bash", "-c", inner_cmd]
                _log.info("eval cmd (in-container mode)")
            # Eval log: in host mode the file lives inside the container,
            # so we cannot open() it directly on the host. Pipe through a
            # local-side log on host; the inner_cmd already redirects the
            # container-side stdout to `{REPO}/_dashboard_eval.log`.
            local_log_path = pathlib.Path(
                "/tmp/_dashboard_eval_local.log" if HOST_MODE else EVAL_LOG
            )
            try:
                local_log_path.parent.mkdir(parents=True, exist_ok=True)
                log_f = open(local_log_path, "w", buffering=1)
            except Exception as e:
                _log.error(f"failed to open local log {local_log_path}: {e}")
                # Re-open cameras since we won't actually launch.
                threading.Thread(
                    target=self.cam_mgr.start, daemon=True
                ).start()
                return f"failed to open log {local_log_path}: {e}"
            try:
                self.proc = subprocess.Popen(
                    cmd,
                    stdout=log_f, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except Exception as e:
                _log.error(f"failed to spawn eval: {e}")
                threading.Thread(
                    target=self.cam_mgr.start, daemon=True
                ).start()
                return f"spawn failed: {e}"
            _log.info(f"eval spawned, pid={self.proc.pid}, prompt={prompt!r}")
            threading.Thread(
                target=self._wait_for_exit, daemon=True
            ).start()
            return f"started pid={self.proc.pid}"

    def _wait_for_exit(self):
        if self.proc is None:
            return
        self.proc.wait()
        # Cams stay open across eval cycles now; nothing to do here beyond
        # logging the exit code.
        _log.info(f"eval exited rc={self.proc.returncode}")

    def stop(self) -> str:
        with self._lock:
            if self.proc is None or self.proc.poll() is not None:
                return "not running"
            # 1. Kill the eval FIRST (stops its zerorpc velocity stream), then
            # settle the arm to zero velocity below. Order matters: halting while
            # the eval still streams would race two clients on the controller.

            # 2. Kill python in container (covers all Ray workers too).
            if HOST_MODE:
                try:
                    subprocess.run(
                        ["docker", "exec", CONTAINER_NAME, "bash", "-c",
                         "pkill -9 -f eval_embodied_agent.py; "
                         "pkill -9 -f 'ray'; "
                         "pkill -9 -f 'raylet'; true"],
                        check=False, timeout=5,
                    )
                except Exception as exc:
                    _log.warning(f"in-container pkill failed: {exc}")
            # 3. Kill local subprocess group.
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            # 4. Settle: stream a few zero-velocity ticks so the arm doesn't
            # coast on the last commanded velocity (impedance holds last setpoint).
            try:
                c = DroidClient(timeout=5)
                try:
                    c.halt()
                finally:
                    c.close()
            except Exception as exc:
                _log.warning(f"post-stop halt failed: {exc}")
            return "stopped"

    def status(self) -> dict:
        with self._lock:
            running = self.proc is not None and self.proc.poll() is None
            return {
                "running": running,
                "pid": self.proc.pid if self.proc else None,
                "returncode": self.proc.returncode if self.proc else None,
                "last_prompt": self.last_prompt,
                "log_tail": _read_log_tail(EVAL_LOG, n=30),
            }


def _read_log_tail(path: pathlib.Path, n: int = 30) -> list[str]:
    """Read last n lines of `path`. In HOST_MODE, the log lives inside the
    container, so we shell out via docker exec."""
    if HOST_MODE:
        try:
            r = subprocess.run(
                ["docker", "exec", CONTAINER_NAME, "tail", "-n", str(n),
                 str(path)],
                capture_output=True, timeout=3, text=True,
            )
            return r.stdout.splitlines()
        except Exception:
            return []
    if not path.exists():
        return []
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            block = 8192
            data = b""
            while size > 0 and data.count(b"\n") <= n:
                read = min(block, size)
                size -= read
                f.seek(size)
                data = f.read(read) + data
        lines = data.decode(errors="replace").splitlines()
        return lines[-n:]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────
# Action stats (last episode written to LeRobot dataset)
# ─────────────────────────────────────────────────────────────────────
def latest_action_stats() -> dict:
    """Compute action magnitude stats from the most recent saved episode.

    In HOST_MODE the parquet lives inside the container; we cat it through
    docker exec into a tmpfile and parse on host.
    """
    try:
        import pandas as pd
        ep_dir_str = str(SAVE_DIR / "rank_0" / "id_0" / "data" / "chunk-000")
        if HOST_MODE:
            r = subprocess.run(
                ["docker", "exec", CONTAINER_NAME, "bash", "-c",
                 f"ls {ep_dir_str}/episode_*.parquet 2>/dev/null | sort"],
                capture_output=True, timeout=3, text=True,
            )
            files = r.stdout.strip().splitlines()
            if not files:
                return {"available": False}
            inside_path = files[-1]
            r = subprocess.run(
                ["docker", "exec", CONTAINER_NAME, "cat", inside_path],
                capture_output=True, timeout=10,
            )
            if r.returncode != 0:
                return {"available": False}
            df = pd.read_parquet(io.BytesIO(r.stdout))
            ep_name = pathlib.Path(inside_path).name
        else:
            ep_dir = pathlib.Path(ep_dir_str)
            if not ep_dir.exists():
                return {"available": False}
            eps = sorted(ep_dir.glob("episode_*.parquet"))
            if not eps:
                return {"available": False}
            df = pd.read_parquet(eps[-1])
            ep_name = eps[-1].name
        if "actions" not in df.columns:
            return {"available": False}
        a = np.stack(df["actions"].to_list())
        return {
            "available": True,
            "episode_file": ep_name,
            "n_steps": int(a.shape[0]),
            "abs_mean_per_dim": [round(float(x), 3)
                                 for x in np.abs(a).mean(axis=0)],
            "abs_max_per_dim": [round(float(x), 3)
                                for x in np.abs(a).max(axis=0)],
            "clamp_hit_pct_joints": round(
                float(np.sum(np.abs(a[:, :7]) >= 0.499)) /
                float(max(a.size - a.shape[0], 1)) * 100, 2
            ),
        }
    except Exception as e:
        return {"available": False, "error": repr(e)}


# ─────────────────────────────────────────────────────────────────────
# robot control — zerorpc to the DROID polymetis container (backend B)
# ─────────────────────────────────────────────────────────────────────
# NUC1 polymetis DROID container (zerorpc :4242). Robot net by default; the
# launcher exports NUC1_HOST so the dashboard and the eval env agree on the host.
NUC1_HOST = os.environ.get("NUC1_HOST", "172.16.0.2")
DROID_ADDR = f"tcp://{NUC1_HOST}:4242"
# DROID home (the env's joint_reset_qpos default). go_home streams to this.
DROID_HOME_Q = [0.0, -0.6283, 0.0, -2.5133, 0.0, 1.8850, 0.0]


class DroidClient:
    """Per-request zerorpc client to the DROID polymetis container (backend B).

    Faithful port of collect.py's robot-panel client. Created FRESH inside each
    Flask handler and closed in a finally: zerorpc rides on gevent whose hub is
    thread-local, and Flask threaded=True serves each request on an arbitrary
    thread — a client made on thread A dies with an opaque LoopExit when used
    from thread B. Per-request connect is cheap ZMQ setup, and the panel runs at
    button-press frequency. Load-bearing semantics preserved from the vendored
    client: POSITIONAL args only (kwargs are silently dropped → wrong action
    space); get_robot_state() returns a [state, ts] 2-list; joint moves are
    STREAMED (DROID adaptive_time_to_go() returns 0 for small deltas, so a
    blocking one-shot is a silent no-op).
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

    def move_to_joint_target(self, target_q7, gripper_cmd: Optional[float] = None,
                             max_step: float = 0.06, hz: int = 15,
                             timeout_s: float = 25.0, tol: float = 0.03) -> float:
        """Closed-loop leashed approach to an absolute joint target (returns max
        joint error rad). Each tick reads the ACTUAL pose and commands a setpoint
        at most max_step rad ahead — speed-bounded, no reflex/lunge. Holds the
        current gripper unless gripper_cmd given."""
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

    def halt(self, ticks: int = 8, hz: int = 15) -> None:
        """Stream a few zero joint-velocity ticks to settle the arm (used after
        killing the eval, which stops its own stream)."""
        grip = self.get_robot_state()["gripper_position"]
        zero = [0.0] * 7 + [float(grip)]
        for _ in range(ticks):
            self._c.update_command(zero, "joint_velocity", "position", False)
            time.sleep(1.0 / hz)

    def update_gripper(self, command: float) -> None:
        cmd = min(max(float(command), 0.0), 1.0)
        self._c.update_gripper(cmd, False, False)  # POSITIONAL: command, velocity, blocking

    def kill_controller(self) -> None:
        self._c.kill_controller()

    def bootstrap(self, settle_seconds: float = 8.0) -> None:
        self._c.launch_controller()
        time.sleep(settle_seconds)
        self._c.launch_robot()


class RS:
    """Robot control facade over the polymetis zerorpc controller. Keeps the
    method names the Flask routes already call, but each method opens a fresh
    gevent-safe DroidClient (see above) instead of HTTP to a franky robot_server.
    Returns dicts with `_err` on failure so the existing handlers are unchanged.
    """

    def __init__(self, addr: str = DROID_ADDR):
        # `addr` may be a tcp://host:port (preferred) or a stale http://host:port
        # from an old default — coerce the latter to the zerorpc address.
        if addr.startswith("http://"):
            host = addr[len("http://"):].split(":")[0]
            addr = f"tcp://{host}:4242"
        self.addr = addr

    def state(self):
        c = DroidClient(self.addr, timeout=5)
        try:
            st = c.get_robot_state()
            # `q` kept for /set_home compatibility (was franka_server's key).
            st["q"] = st["joint_positions"]
            return st
        except Exception as e:
            return {"_err": "exc", "_body": repr(e)[:200]}
        finally:
            c.close()

    def robotiq_state(self):
        c = DroidClient(self.addr, timeout=5)
        try:
            return {"gripper_position": c.get_robot_state()["gripper_position"]}
        except Exception as e:
            return {"_err": "exc", "_body": repr(e)[:200]}
        finally:
            c.close()

    def go_home(self, target_q, dynamics_factor: float = 0.05):
        c = DroidClient(self.addr, timeout=30)
        try:
            err = c.move_to_joint_target(list(target_q))
            ok = err < 0.03
            return {"ok": ok, "max_joint_error_rad": round(err, 4),
                    "msg": (f"home done, max joint error {err:.4f} rad" if ok else
                            f"home NOT converged ({err:.4f} rad) — Desk/E-stop? then Recover.")}
        except Exception as e:
            return {"_err": "exc", "_body": repr(e)[:200]}
        finally:
            c.close()

    def stop(self):
        c = DroidClient(self.addr, timeout=10)
        try:
            c.halt()
            return {"ok": True, "msg": "halted (zero velocity)"}
        except Exception as e:
            return {"_err": "exc", "_body": repr(e)[:200]}
        finally:
            c.close()

    def recover(self):
        # Synchronous; 90s timeout — a COLD launch_controller (container just up
        # + FCI just activated) can exceed 30s and finish server-side after the
        # client gives up, so on a bootstrap RPC error re-check state and treat a
        # live controller as success (no false 500). Mirrors collect.py.
        c = DroidClient(self.addr, timeout=90)
        try:
            try:
                c.kill_controller()
            except Exception as e:
                _log.warning(f"kill_controller during recover: {e}")
            try:
                c.bootstrap()
            except Exception as e:
                _log.warning(f"bootstrap RPC error during recover: {e!r}; re-checking state")
                rc = DroidClient(self.addr, timeout=8)
                try:
                    rc.get_robot_state()
                except Exception:
                    return {"_err": "exc", "_body": repr(e)[:200]}
                finally:
                    rc.close()
                return {"ok": True, "msg": "recover done (controller live; bootstrap RPC was slow)"}
            return {"ok": True, "msg": "recover done (controller relaunched)"}
        finally:
            c.close()

    def gripper(self, action: str, speed: float = 0.3):
        c = DroidClient(self.addr, timeout=10)
        try:
            c.update_gripper(0.0 if action == "open" else 1.0)
            return {"ok": True, "msg": f"gripper {action} sent"}
        except Exception as e:
            return {"_err": "exc", "_body": repr(e)[:200]}
        finally:
            c.close()

    def jog_cartesian(self, *args, **kwargs):
        # Not supported on the polymetis backend (no relative-cartesian RPC here).
        return {"_err": "unsupported",
                "_body": "cartesian jog not available on the polymetis backend"}

    def freedrive(self, enable: bool):
        return {"_err": "unsupported",
                "_body": "freedrive not available on the polymetis backend"}


# ─────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────
INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8">
<title>TASL FR3 — RLinf dashboard</title>
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
<h1>TASL FR3 — RLinf eval dashboard</h1>
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
<h3 style="margin-top:18px">Eval log (tail)</h3>
<pre id="logtail">(idle)</pre>
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
         +  '<br>in_control=' + s.is_in_control
         +  '  has_errors=' + s.has_errors + '</td></tr>';
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
    if (j.last_episode && j.last_episode.available) {
      const le = j.last_episode;
      html += '<tr><td>last ep</td><td>'
           +  le.episode_file + '<br>steps=' + le.n_steps
           +  '<br>|dq| mean per-dim: [' + le.abs_mean_per_dim.join(', ') + ']'
           +  '<br>|dq| max per-dim:  [' + le.abs_max_per_dim.join(', ') + ']'
           +  '<br>joint clamp hit %: ' + le.clamp_hit_pct_joints
           +  '</td></tr>';
    }
    html += '</table>';
    document.getElementById('status').innerHTML = html;
    document.getElementById('logtail').textContent =
      (re.log_tail||[]).join('\\n');
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

    @app.get("/status")
    def status():
        rs_state = rs.state()
        robot_ok = "_err" not in rs_state
        gripper = rs.robotiq_state()
        gripper_ok = "_err" not in gripper
        return jsonify({
            "robot_ok": robot_ok,
            "robot": rs_state if robot_ok else {"err": rs_state},
            "gripper_ok": gripper_ok,
            "gripper": gripper if gripper_ok else {"err": gripper},
            "cam_running": cams.is_running(),
            "eval": runner.status(),
            "last_episode": latest_action_stats(),
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
        return jsonify({"ok": True, "home_q": home_store.get(),
                        "result": rs.go_home(home_store.get())})

    @app.post("/set_home")
    def post_set_home():
        if runner.status()["running"]:
            return jsonify({"ok": False,
                            "msg": "eval running; stop first"}), 409
        st = rs.state()
        if "_err" in st or "q" not in st:
            return jsonify({"ok": False,
                            "msg": "couldn't read robot state",
                            "robot": st}), 500
        new_q = list(st["q"])[:7]
        home_store.set(new_q)
        return jsonify({"ok": True, "home_q": new_q})

    @app.post("/recover")
    def post_recover():
        return jsonify({"ok": True, "result": rs.recover()})

    @app.post("/robot_stop")
    def post_robot_stop():
        return jsonify({"ok": True, "result": rs.stop()})

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
        return jsonify({"ok": True, "result": rs.gripper(action)})

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

    return app


def main():
    global HOST_MODE, REPO_PATH, CONFIG_DIR, EVAL_SCRIPT, SAVE_DIR, EVAL_LOG
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8003)
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--robot-server", default=DROID_ADDR,
                   help="zerorpc address of the DROID polymetis controller "
                        "(tcp://host:4242). Defaults to NUC1_HOST on the robot net.")
    p.add_argument("--resolution", default="HD720",
                   choices=["HD2K", "HD1080", "HD720"])
    p.add_argument("--jpeg-quality", type=int, default=70)
    p.add_argument("--no-cam-on-start", action="store_true",
                   help="Don't acquire cameras at startup (useful if "
                        "RLinf eval is already running).")
    p.add_argument("--mode", choices=["container", "host"], default="container",
                   help="container: dashboard runs inside rlinf-eval and "
                        "spawns the eval via local python. host: dashboard "
                        "runs on Desktop host, spawns eval via docker exec.")
    args = p.parse_args()

    HOST_MODE = (args.mode == "host")
    REPO_PATH = pathlib.Path(
        DEFAULT_REPO_PATH_HOST if HOST_MODE else DEFAULT_REPO_PATH_CONTAINER
    )
    CONFIG_DIR = REPO_PATH / "examples/embodiment/config"
    EVAL_SCRIPT = REPO_PATH / "examples/embodiment/eval_embodied_agent.py"
    # Eval log + save dir always live inside the container (eval's POV).
    EVAL_LOG = pathlib.Path(DEFAULT_REPO_PATH_CONTAINER) / "_dashboard_eval.log"

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    _log.info(f"mode={'host' if HOST_MODE else 'container'}, "
              f"REPO_PATH={REPO_PATH}, EVAL_LOG(container-side)={EVAL_LOG}")

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
    runner = EvalRunner(cams, home_store, rs_url=args.robot_server)
    app = build_app(rs, cams, runner, home_store)
    _log.info(f"serving on {args.bind}:{args.port}")
    app.run(host=args.bind, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
