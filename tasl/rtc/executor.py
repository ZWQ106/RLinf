"""Real-Time Chunking (RTC) executor — client side of paper Algorithm 1.

"Real-Time Execution of Action Chunking Flow Policies" (Black, Galliker,
Levine, arXiv:2506.07339), Algorithm 1:

  * a CONTROLLER thread ticks at a fixed rate (15 Hz for pi05_droid) and
    consumes one action per tick from the current chunk A_cur; it never waits
    for the policy (a chunk that runs out -> "hold" ticks);
  * the INFERENCE loop waits until s >= s_min actions of A_cur were consumed,
    grabs a fresh observation, asks the policy for the next chunk while
    passing (a) the remaining actions of A_cur shifted into the new chunk's
    frame and (b) the expected inference delay d (max of the last few
    observed delays), then swaps A_cur for the new chunk and re-indexes t so
    execution continues at the matching action.

The policy side (rtc_policy.RTCPolicy, request key `rtc`) does the guided
inpainting; this module only owns timing and bookkeeping and is
model-agnostic. Because the first d actions of every new chunk are frozen to
what the controller already executed, chunk switches produce no jump.

Wiring is by callbacks so the executor is testable without hardware
(tests/test_rtc_executor.py):

    infer(obs, rtc_request) -> (actions[H, n], actions_model[H, m])
    get_obs()               -> obs dict for the policy
    send_action(a | None)   -> one controller tick; None = chunk exhausted,
                               caller decides how to hold
"""

from __future__ import annotations

import collections
import dataclasses
import logging
import threading
import time
from typing import Any, Callable, Optional

import numpy as np

_log = logging.getLogger("rtc.executor")

SCHEDULES = ("zeros", "linear", "exp", "ones")


@dataclasses.dataclass
class RTCConfig:
    """User-facing RTC knobs (dashboard /rtc endpoint). Whether RTC is *used*
    is decided at checkpoint-load time (portal "Load with RTC" → RTC-capable
    serve); these only tune the executor."""

    # Minimum execution horizon s_min: consume at least this many actions of
    # the current chunk before starting the next inference. Paper: d <= s <= H-d.
    s_min: int = 4
    # Initial inference-delay estimate d_init (controller ticks). Replaced by the
    # max over the last `delay_buffer` observed delays once running.
    delay_init: int = 3
    delay_buffer: int = 5
    # Guidance clip beta (paper: 5.0) and soft-mask schedule (paper: "exp").
    max_guidance_weight: float = 5.0
    schedule: str = "exp"
    # Controller rate. DROID / pi05_droid stream at 15 Hz.
    control_hz: float = 15.0
    # One throwaway guided inference before moving: compiles the guided JAX
    # trace while the arm is still (cheap if the server already warmed up).
    warmup: bool = True

    def validate(self, horizon: Optional[int] = None) -> None:
        if self.s_min < 1:
            raise ValueError("s_min must be >= 1")
        if self.delay_init < 0 or self.delay_buffer < 1:
            raise ValueError("delay_init must be >= 0 and delay_buffer >= 1")
        if self.max_guidance_weight <= 0:
            raise ValueError("max_guidance_weight must be > 0")
        if self.schedule not in SCHEDULES:
            raise ValueError(f"schedule must be one of {SCHEDULES}")
        if not (1.0 <= self.control_hz <= 100.0):
            raise ValueError("control_hz out of range")
        if horizon is not None and self.s_min + self.delay_init > horizon:
            raise ValueError(
                f"s_min + delay_init = {self.s_min + self.delay_init} exceeds the chunk "
                f"length H={horizon}; the paper needs d <= H - s")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def updated(self, changes: dict) -> "RTCConfig":
        """Return a validated copy with `changes` applied (unknown keys rejected)."""
        cur = self.to_dict()
        for k, v in changes.items():
            if k not in cur:
                raise ValueError(f"unknown rtc field {k!r}")
            if isinstance(cur[k], bool):
                cur[k] = v.lower() in ("1", "true", "on", "yes") if isinstance(v, str) else bool(v)
            elif isinstance(cur[k], int):
                cur[k] = int(v)
            elif isinstance(cur[k], float):
                cur[k] = float(v)
            else:
                cur[k] = str(v)
        new = RTCConfig(**cur)
        new.validate()
        return new


def shift_chunk(actions_model: np.ndarray, executed: int) -> np.ndarray:
    """Drop the `executed` leading actions and right-pad with zeros back to the
    original length (Algorithm 1 lines 15 + 24). Zero is a fill value only:
    the soft mask gives those rows weight 0."""
    h = actions_model.shape[0]
    executed = int(min(max(executed, 0), h))
    out = np.zeros_like(actions_model)
    if executed < h:
        out[: h - executed] = actions_model[executed:]
    return out


class RTCExecutor:
    """Runs one episode with real-time chunking. `run()` blocks until
    `stop_event` is set (or the policy/robot raises) and owns the controller
    thread for that duration."""

    def __init__(
        self,
        cfg: RTCConfig,
        *,
        infer: Callable[[dict, dict], tuple[np.ndarray, np.ndarray]],
        get_obs: Callable[[], dict],
        send_action: Callable[[Optional[np.ndarray]], None],
        stop_event: threading.Event,
        on_inference: Optional[Callable[[dict], None]] = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        cfg.validate()
        self.cfg = cfg
        self._infer_cb = infer
        self._get_obs = get_obs
        self._send_action = send_action
        self._stop = stop_event
        self._on_inference = on_inference
        self._sleep = sleep
        self._clock = clock

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._A: Optional[np.ndarray] = None       # current chunk, executable space
        self._Am: Optional[np.ndarray] = None      # same chunk, model space
        self._t = 0                                # actions consumed from _A
        self.horizon: Optional[int] = None
        # stats (read by the dashboard /status, /rtc)
        self.state = "init"
        self.error: Optional[str] = None
        self.n_infer = 0
        self.ticks = 0
        self.starved_ticks = 0
        self.last_infer_ms = 0.0
        self.last_d = cfg.delay_init
        self.last_s = 0
        self.last_elapsed = 0
        self.last_guided = False
        self.delays: collections.deque = collections.deque([cfg.delay_init], maxlen=cfg.delay_buffer)

    # ── public ────────────────────────────────────────────────────────
    def stats(self) -> dict:
        with self._lock:
            return {
                "state": self.state,
                "error": self.error,
                "n_infer": self.n_infer,
                "ticks": self.ticks,
                "starved_ticks": self.starved_ticks,
                "t": self._t,
                "horizon": self.horizon,
                "last_infer_ms": round(self.last_infer_ms, 1),
                "last_d": self.last_d,
                "last_s": self.last_s,
                "last_elapsed": self.last_elapsed,
                "last_guided": self.last_guided,
                "delays": list(self.delays),
            }

    def run(self) -> None:
        cfg = self.cfg
        ctrl: Optional[threading.Thread] = None
        try:
            # Initial chunk (plain sampling) + optional guided warm-up while the
            # arm is still. The warm-up result is discarded; it exists to trigger
            # the compile of the guided trace before the controller starts.
            obs = self._get_obs()
            A, Am = self._infer(obs, prev=None, d=0, s=0)
            self.horizon = int(A.shape[0])
            cfg.validate(self.horizon)
            if cfg.warmup:
                self.state = "warmup"
                self._infer(obs, prev=shift_chunk(Am, cfg.s_min), d=cfg.delay_init, s=cfg.s_min)
                if self._stop.is_set():
                    return
                obs = self._get_obs()
                A, Am = self._infer(obs, prev=None, d=0, s=0)
            with self._lock:
                self._A, self._Am, self._t = A, Am, 0
                self.state = "running"
            ctrl = threading.Thread(target=self._controller, name="rtc-controller", daemon=True)
            ctrl.start()
            self._inference_loop()
        except Exception as exc:  # policy / robot / obs failure -> end the episode
            self.error = f"{type(exc).__name__}: {exc}"
            _log.error(f"rtc executor stopped: {self.error}")
        finally:
            self._stop.set()
            with self._cond:
                self._cond.notify_all()
            if ctrl is not None:
                ctrl.join(timeout=3.0)
            self.state = "stopped"

    # ── internals ─────────────────────────────────────────────────────
    def _infer(self, obs: dict, *, prev: Optional[np.ndarray], d: int, s: int):
        H = self.horizon
        req: dict[str, Any] = {
            "enabled": True,
            "prev_actions": None if prev is None else np.asarray(prev, dtype=np.float32),
            "inference_delay": int(d),
            "prefix_attention_horizon": int((H - s) if H is not None else 0),
            "schedule": self.cfg.schedule,
            "max_guidance_weight": float(self.cfg.max_guidance_weight),
        }
        t0 = self._clock()
        A, Am = self._infer_cb(obs, req)
        ms = (self._clock() - t0) * 1000.0
        A = np.asarray(A)
        Am = np.asarray(Am, dtype=np.float32)
        if A.ndim != 2 or Am.ndim != 2 or A.shape[0] != Am.shape[0]:
            raise ValueError(f"policy returned actions {A.shape} / actions_model {Am.shape}")
        if H is not None and A.shape[0] != H:
            raise ValueError(f"chunk length changed: {A.shape[0]} != {H}")
        with self._lock:
            self.last_infer_ms = ms
            self.last_guided = prev is not None
        return A, Am

    def _inference_loop(self) -> None:
        cfg = self.cfg
        H = self.horizon
        assert H is not None
        while not self._stop.is_set():
            with self._cond:
                while not self._stop.is_set() and self._t < cfg.s_min:
                    self._cond.wait(timeout=0.25)
                if self._stop.is_set():
                    return
            # Observation is captured outside the lock (jpeg decode + robot state
            # take tens of ms); the reference tick s is read right after, so the
            # new chunk's frame is anchored to the tick the obs belongs to.
            obs = self._get_obs()
            with self._lock:
                s = self._t
                Am_cur = self._Am
            d = int(min(max(self.delays), max(H - s, 0)))
            # s >= H: chunk fully consumed (inference slower than H ticks),
            # nothing left to condition on -> plain sample.
            prev = None if s >= H else shift_chunk(Am_cur, s)
            A_new, Am_new = self._infer(obs, prev=prev, d=d, s=min(s, H))
            with self._cond:
                elapsed = self._t - s          # ticks consumed while inferring
                self._A, self._Am = A_new, Am_new
                self._t = elapsed              # continue at the matching index
                self.n_infer += 1
                self.last_d, self.last_s, self.last_elapsed = d, s, elapsed
                self.delays.append(elapsed)
                self._cond.notify_all()
            if self._on_inference is not None:
                try:
                    self._on_inference({
                        "n_infer": self.n_infer, "s": s, "d": d, "elapsed": elapsed,
                        "infer_ms": round(self.last_infer_ms, 1), "guided": prev is not None,
                        "starved_ticks": self.starved_ticks, "ticks": self.ticks,
                        "actions": A_new,
                    })
                except Exception as exc:
                    _log.warning(f"on_inference callback failed: {exc}")

    def _controller(self) -> None:
        dt = 1.0 / self.cfg.control_hz
        next_t = self._clock()
        while not self._stop.is_set():
            with self._cond:
                i = self._t
                a = self._A[i] if (self._A is not None and i < len(self._A)) else None
                self._t += 1
                self.ticks += 1
                if a is None:
                    self.starved_ticks += 1
                self._cond.notify_all()
            try:
                self._send_action(a)
            except Exception as exc:
                self.error = f"send_action: {exc}"
                _log.error(self.error)
                self._stop.set()
                with self._cond:
                    self._cond.notify_all()
                return
            next_t += dt
            delay = next_t - self._clock()
            if delay > 0:
                self._sleep(delay)
            else:
                next_t = self._clock()   # fell behind: resync instead of bursting
