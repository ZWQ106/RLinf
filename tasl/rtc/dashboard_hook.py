"""Glue between dashboards/openpi.py and the RTC package.

Everything RTC-specific that the eval portal needs lives here, so the portal
itself only carries a handful of one-line hooks:

    _rtc_hook.RTCState()                      EvalRunner.__init__  -> self.rtc
    _rtc_hook.active(runner)                  EvalRunner._loop: take the RTC path?
    _rtc_hook.run_episode(runner, prompt, policy)
                                              EvalRunner._loop, when active
    _rtc_hook.status(runner)                  EvalRunner.status()["rtc"]
    _rtc_hook.register_routes(app, runner, serve)
                                              build_app  (GET/POST /rtc)
    _rtc_hook.inject_panel(INDEX_HTML)        index route (UI card)
    _rtc_hook.SERVE_SCRIPT                    ServeManager "Load with RTC" target
    _rtc_hook.is_rtc_serve(cmdline)           ServeManager._recover: which kind is running

The switch is the checkpoint loader: "Load w/o RTC" spawns openpi's own
scripts/serve_policy.py (portal = the old synchronous loop, byte for byte);
"Load with RTC" spawns the RTC-capable drop-in and the eval loop takes the
RTC path. `active(runner)` simply reports which serve is loaded.
"""

from __future__ import annotations

import io
import json
import logging
import os
import pathlib
import time
from typing import Optional

import numpy as np
from flask import jsonify, request
from PIL import Image

from rtc.executor import RTCConfig, RTCExecutor

_log = logging.getLogger("rtc.hook")

SERVE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "serve_policy.py")


class RTCState:
    """Per-runner RTC state: knobs (editable via /rtc), the ServeManager that
    knows which serve is loaded, the live executor while an RTC episode runs,
    and the stats of the last one."""

    def __init__(self) -> None:
        self.cfg = RTCConfig()
        self.serve = None                       # set by register_routes
        self.executor: Optional[RTCExecutor] = None
        self.last_stats: dict = {}
        self.last_episode: Optional[str] = None


def is_rtc_serve(cmdline: str) -> bool:
    """True if a serve_policy.py process command line is the RTC-capable script."""
    return "rtc/scripts/serve_policy.py" in cmdline


def active(runner) -> bool:
    """RTC is used iff the checkpoint was loaded 'with RTC'."""
    serve = getattr(runner.rtc, "serve", None)
    return bool(serve is not None and getattr(serve, "rtc", False))


class ObsNotReady(RuntimeError):
    pass


def capture_obs(runner, prompt: str, wait_s: float = 1.0,
                meta: Optional[dict] = None) -> dict:
    """Build one policy observation. Mirrors the obs block of
    EvalRunner._loop (dashboards/openpi.py) — keep the two in sync."""
    deadline = time.monotonic() + wait_s
    while True:
        ext_jpeg = runner.cam_mgr.get_jpeg("wrist_1")
        wrist_jpeg = runner.cam_mgr.get_jpeg("wrist_2")
        if ext_jpeg is not None and (wrist_jpeg is not None or runner.allow_missing_wrist):
            break
        if time.monotonic() > deadline:
            raise ObsNotReady("camera frames not available")
        time.sleep(0.02)
    ext_rgb = np.asarray(Image.open(io.BytesIO(ext_jpeg)).convert("RGB"))
    if wrist_jpeg is None:  # --allow-missing-wrist: black frame stands in
        wrist_rgb = np.zeros_like(ext_rgb)
    else:
        wrist_rgb = np.asarray(Image.open(io.BytesIO(wrist_jpeg)).convert("RGB"))
    ext_r = runner.preprocess_image(ext_rgb)      # pad / crop per runner.image_mode
    wrist_r = runner.preprocess_image(wrist_rgb)
    state = runner._droid.get_robot_state()            # noqa: SLF001
    joint_position = np.asarray(state["joint_positions"], dtype=np.float32)
    ts_ns = (int(state["timestamp_seconds"]) * 1_000_000_000
             + int(state["timestamp_nanos"]))
    runner.motion.note_q(joint_position, ts_ns)   # live motion chip
    if meta is not None:
        # Side channel for the watchdog / traj log — NOT part of the obs sent
        # to the policy: raw (unrounded) joints + polymetis state timestamp.
        meta["q_raw"] = joint_position.copy()
        meta["ts_ns"] = ts_ns
    # DROID gripper_position is already in [0,1] with 1=close (pi05_droid convention).
    gripper_position = np.asarray([state["gripper_position"]], dtype=np.float32)
    obs = {
        "observation/exterior_image_1_left": ext_r,
        "observation/wrist_image_left": wrist_r,
        "observation/joint_position": joint_position,
        "observation/gripper_position": gripper_position,
        "prompt": prompt,
    }
    if runner._tag:  # noqa: SLF001
        obs["_episode_tag"] = runner._tag  # noqa: SLF001
    return obs


def run_episode(runner, prompt: str, policy) -> None:
    """Drive one RTC episode on `runner` (an EvalRunner whose droid client is
    bootstrapped and whose `_stop` event ends the episode). Blocks until the
    episode stops; the caller's finally-block handles ws close / flags."""
    st: RTCState = runner.rtc
    cfg = st.cfg
    droid = runner._droid  # noqa: SLF001
    recorder = runner._recorder  # noqa: SLF001
    last_grip = [0.0]

    def infer(obs: dict, req: dict):
        o = dict(obs)
        o["rtc"] = req
        res = policy.infer(o)
        if "actions_model" not in res:
            raise RuntimeError(
                "policy server is not RTC-capable (no `actions_model` in the response) — "
                "reload the checkpoint from the dashboard so it is served by tasl/rtc/scripts/serve_policy.py")
        A = np.asarray(res["actions"])
        if A.ndim != 2 or A.shape[-1] != 8:
            raise RuntimeError(f"unexpected action shape {A.shape}")
        runner._last_infer_ms = float(res.get("policy_timing", {}).get("infer_ms") or 0.0)  # noqa: SLF001
        runner._iter += 1  # noqa: SLF001
        if runner._iter >= runner.max_iterations:  # noqa: SLF001
            runner._stop.set()  # noqa: SLF001
        return A, np.asarray(res["actions_model"], dtype=np.float32)

    def send_action(a: Optional[np.ndarray]) -> None:
        # Same wire format as the synchronous path: 8-D joint-velocity tick in
        # [-1,1] + absolute gripper command (0=open, 1=close), 15 Hz.
        a8 = np.zeros(8)
        if a is None:            # chunk exhausted -> hold (zero velocity, keep gripper)
            a8[7] = last_grip[0]
        else:
            a8[:7] = np.clip(a[:7], -1.0, 1.0)
            a8[7] = float(a[7])
            last_grip[0] = max(0.0, min(1.0, a8[7]))
            runner._last_grip_raw = float(a[7])  # noqa: SLF001
            runner._last_dq_max = float(np.max(np.abs(a8[:7])) * runner.delta_scale)  # noqa: SLF001
        runner.motion.note_cmd(a8[:7] * runner.delta_scale)   # live motion chip
        droid.update_joint_velocity(a8, blocking=False)

    # Last observation fed to the policy — logged with each inference so the
    # RTC traj.jsonl carries q/grip like the sync loop's (needed to diagnose
    # "model doesn't move" episodes: off-home start pose vs. policy hover).
    last_obs: dict = {}

    def get_obs() -> dict:
        o = capture_obs(runner, prompt, meta=last_obs)   # fills q_raw / ts_ns
        last_obs["q"] = np.round(o["observation/joint_position"], 5).tolist()
        last_obs["grip"] = round(float(o["observation/gripper_position"][0]), 4)
        return o

    def on_inference(rec: dict) -> None:
        # Frozen-arm watchdog (same rules as the sync loop): stale robot
        # timestamp, or commanded motion with frozen joints → stop. Feed the
        # RAW joints — the rounded copy above has 1e-5 resolution, coarser
        # than the watchdog's 5e-5 bar.
        if last_obs.get("q_raw") is not None:
            wd = runner.watchdog_step(last_obs["q_raw"],
                                      float(np.abs(np.clip(np.asarray(rec["actions"])[:, :7], -1, 1)).mean()),
                                      last_obs.get("ts_ns"))
            if wd:
                runner._last_error = wd  # noqa: SLF001
                _log.error(wd)
                runner._stop.set()  # noqa: SLF001
        if recorder is None:
            return
        recorder.log_step({
            "t": round(time.time(), 4),
            "ts": last_obs.get("ts_ns"),   # robot-state timestamp (NUC loop liveness)
            "iter": rec["n_infer"],
            "q": last_obs.get("q"),
            "grip": last_obs.get("grip"),
            "rtc": {k: rec[k] for k in ("s", "d", "elapsed", "guided", "ticks", "starved_ticks")},
            "infer_ms": rec["infer_ms"],
            "actions": np.round(rec["actions"], 4).tolist(),
        })

    ex = RTCExecutor(
        cfg,
        infer=infer,
        get_obs=get_obs,
        send_action=send_action,
        stop_event=runner._stop,  # noqa: SLF001
        on_inference=on_inference,
    )
    st.executor = ex
    sidecar = _sidecar_path(recorder)
    _write_sidecar(sidecar, {"cfg": cfg.to_dict(), "prompt": prompt, "started": time.strftime("%Y-%m-%d %H:%M:%S")})
    _log.info(f"RTC episode start: {cfg.to_dict()}")
    try:
        ex.run()
    finally:
        st.last_stats = ex.stats()
        st.last_episode = getattr(recorder, "ep_id", None)
        st.executor = None
        if ex.error:
            runner._last_error = f"rtc: {ex.error}"  # noqa: SLF001
        _write_sidecar(sidecar, {"stats": st.last_stats, "ended": time.strftime("%Y-%m-%d %H:%M:%S")})
        _log.info(f"RTC episode end: {st.last_stats}")


def _sidecar_path(recorder) -> Optional[pathlib.Path]:
    out_dir = getattr(recorder, "out_dir", None)
    return pathlib.Path(out_dir) / "rtc.json" if out_dir else None


def _write_sidecar(path: Optional[pathlib.Path], update: dict) -> None:
    if path is None:
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        data.update(update)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        _log.warning(f"rtc sidecar write failed: {exc}")


def status(runner) -> dict:
    st: RTCState = runner.rtc
    live = st.executor.stats() if st.executor is not None else st.last_stats
    return {"active": active(runner), "cfg": st.cfg.to_dict(), "live": live,
            "last_episode": st.last_episode}


def register_routes(app, runner, serve=None) -> None:
    runner.rtc.serve = serve

    @app.get("/rtc")
    def rtc_get():
        return jsonify({"ok": True, **status(runner)})

    @app.post("/rtc")
    def rtc_post():
        if runner._running:  # noqa: SLF001
            return jsonify({"ok": False, "msg": "eval running; stop first"}), 409
        body = request.get_json(silent=True) or {}
        try:
            runner.rtc.cfg = runner.rtc.cfg.updated(body)
        except (ValueError, TypeError) as exc:
            return jsonify({"ok": False, "msg": str(exc)}), 400
        _log.info(f"rtc config: {runner.rtc.cfg.to_dict()}")
        return jsonify({"ok": True, "msg": "rtc knobs updated", **status(runner)})


PANEL_MARKER = "<!-- rtc:panel -->"

# Raw string: the portal renders INDEX_HTML through Jinja, so no "{{" here.
PANEL_HTML = r"""
<!-- RTC panel — tasl/rtc/dashboard_hook.py -->
<div class="card" id="rtcCard">
  <b>Real-time chunking (RTC)</b>
  <span style="font-size:12px;color:var(--faint)"> — async inference + guided inpainting (arXiv:2506.07339). Switch = <b>Load with RTC</b> / <b>Load w/o RTC</b> above; knobs here.</span>
  <div class="flexbar" style="margin-top:4px">
    <label>s_min <input type="number" id="rtcSmin" min="1" max="15" style="width:52px"/></label>
    <label>delay_init <input type="number" id="rtcDelay" min="0" max="15" style="width:52px"/></label>
    <label>&beta; <input type="number" id="rtcBeta" step="0.5" min="0.5" style="width:60px"/></label>
    <label>schedule <select id="rtcSched"><option>exp</option><option>linear</option><option>zeros</option><option>ones</option></select></label>
    <label><input type="checkbox" id="rtcWarmup"/> warm-up</label>
    <button class="filled" onclick="rtcApply()">Apply</button>
    <span id="rtcState" style="font-size:13px;color:var(--primary)"></span>
  </div>
  <div id="rtcLive" style="font-size:12.5px;color:var(--faint);margin-top:4px"></div>
</div>
<script>
let rtcDirty = false, rtcLoaded = false;
['rtcSmin','rtcDelay','rtcBeta','rtcSched','rtcWarmup'].forEach(id =>
  document.getElementById(id).addEventListener('input', () => { rtcDirty = true; }));
function rtcFill(c) {
  document.getElementById('rtcSmin').value = c.s_min;
  document.getElementById('rtcDelay').value = c.delay_init;
  document.getElementById('rtcBeta').value = c.max_guidance_weight;
  document.getElementById('rtcSched').value = c.schedule;
  document.getElementById('rtcWarmup').checked = !!c.warmup;
}
async function rtcRefresh() {
  try {
    const j = await (await fetch('/rtc')).json();
    const c = j.cfg || {}, l = j.live || {};
    if (!rtcDirty || !rtcLoaded) { rtcFill(c); rtcLoaded = true; }
    const st = document.getElementById('rtcState');
    st.textContent = j.active ? 'RTC ACTIVE (serve loaded with RTC)' : 'RTC inactive (sync chunks — load a checkpoint "with RTC" to use it)';
    st.style.color = j.active ? 'var(--ok, #2e7d32)' : 'var(--faint)';
    const live = document.getElementById('rtcLive');
    if (l && l.state) {
      live.textContent = 'state=' + l.state + '  infer#=' + l.n_infer + '  ticks=' + l.ticks
        + '  starved=' + l.starved_ticks + '  last: s=' + l.last_s + ' d=' + l.last_d
        + ' elapsed=' + l.last_elapsed + ' infer=' + l.last_infer_ms + 'ms'
        + '  delays=[' + (l.delays || []).join(',') + ']'
        + (l.error ? '  ERR: ' + l.error : '');
    } else {
      live.textContent = j.active ? 'no RTC episode yet' : '';
    }
  } catch (e) { /* dashboard restarting */ }
}
async function rtcApply() {
  const body = {
    s_min: parseInt(document.getElementById('rtcSmin').value, 10),
    delay_init: parseInt(document.getElementById('rtcDelay').value, 10),
    max_guidance_weight: parseFloat(document.getElementById('rtcBeta').value),
    schedule: document.getElementById('rtcSched').value,
    warmup: document.getElementById('rtcWarmup').checked,
  };
  try {
    const r = await fetch('/rtc', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                   body: JSON.stringify(body)});
    const j = await r.json();
    if (j.ok === false) { toast('rtc: ' + j.msg, 'err'); return; }
    toast(j.msg || 'rtc updated', 'ok');
    rtcDirty = false;
    rtcRefresh();
  } catch (e) { toast('rtc: ' + e, 'err'); }
}
setInterval(rtcRefresh, 1500);
rtcRefresh();
</script>
"""


def inject_panel(index_html: str) -> str:
    if PANEL_MARKER not in index_html:
        _log.warning("RTC panel marker missing from INDEX_HTML; panel not shown")
        return index_html
    return index_html.replace(PANEL_MARKER, PANEL_HTML)
