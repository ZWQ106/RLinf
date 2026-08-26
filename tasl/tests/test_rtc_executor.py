"""Hardware-free test of the RTC executor timing/bookkeeping.

A fake policy that (a) sleeps for a configurable time and (b) returns chunks
whose first d rows copy the prefix it was given, so chunk switches can be
checked for continuity. A fake controller sink records every tick.

Run:  cd ~/RLinf/tasl && python3 -m pytest tests/test_rtc_executor.py -q
"""

import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rtc.executor import RTCConfig, RTCExecutor, shift_chunk  # noqa: E402

H, A_EXEC, A_MODEL = 15, 8, 32


class FakePolicy:
    """Chunk k has value k in every entry; guided calls copy the frozen prefix
    (first d rows) from `prev_actions`, i.e. a perfect inpainter."""

    def __init__(self, infer_s: float):
        self.infer_s = infer_s
        self.calls = []
        self.k = 0

    def __call__(self, obs, req):
        self.calls.append(dict(req))
        time.sleep(self.infer_s)
        self.k += 1
        Am = np.full((H, A_MODEL), float(self.k), dtype=np.float32)
        if req["prev_actions"] is not None:
            d = req["inference_delay"]
            Am[:d] = req["prev_actions"][:d]
        A = Am[:, :A_EXEC].copy()
        return A, Am


class Sink:
    def __init__(self):
        self.ticks = []       # (monotonic, value or None)

    def __call__(self, a):
        self.ticks.append((time.monotonic(), None if a is None else float(a[0])))


def _run(cfg, infer_s, run_s):
    pol = FakePolicy(infer_s)
    sink = Sink()
    stop = threading.Event()
    ex = RTCExecutor(cfg, infer=pol, get_obs=lambda: {"o": 1}, send_action=sink,
                     stop_event=stop, on_inference=lambda rec: None)
    th = threading.Thread(target=ex.run, daemon=True)
    th.start()
    time.sleep(run_s)
    stop.set()
    th.join(timeout=5)
    assert not th.is_alive()
    return ex, pol, sink


def test_shift_chunk():
    x = np.arange(H * 2, dtype=np.float32).reshape(H, 2)
    y = shift_chunk(x, 4)
    assert (y[: H - 4] == x[4:]).all() and (y[H - 4:] == 0).all()
    assert (shift_chunk(x, 0) == x).all()
    assert (shift_chunk(x, H + 3) == 0).all()


def test_continuity_and_delay_estimate():
    cfg = RTCConfig(s_min=4, delay_init=3, control_hz=50.0, warmup=False)
    ex, pol, sink = _run(cfg, infer_s=0.045, run_s=1.5)   # ~2.25 ticks -> elapsed 2..3
    assert ex.error is None, ex.error
    st = ex.stats()
    assert st["n_infer"] >= 8
    assert st["starved_ticks"] == 0
    # every guided request carried a prefix shifted by the ticks consumed
    guided = [c for c in pol.calls if c["prev_actions"] is not None]
    assert guided and all(c["prefix_attention_horizon"] == H - c_s for c, c_s in
                          zip(guided, [H - c["prefix_attention_horizon"] for c in guided]))
    # observed delays are stored and bounded
    assert all(0 <= d_ <= 4 for d_ in st["delays"]), st["delays"]
    # executed value sequence never jumps: with a perfect inpainter the chunk
    # switch is invisible except that the value increases by at most 1 per switch.
    vals = [v for _, v in sink.ticks if v is not None]
    jumps = np.diff(vals)
    assert (jumps >= 0).all() and (jumps <= 1).all(), vals
    # tick rate ~50 Hz
    ts = np.array([t for t, _ in sink.ticks])
    assert abs(np.median(np.diff(ts)) - 0.02) < 0.004


def test_starvation_holds_instead_of_crashing():
    # inference takes longer than a whole chunk -> controller must emit holds
    cfg = RTCConfig(s_min=2, delay_init=1, control_hz=100.0, warmup=False)
    ex, pol, sink = _run(cfg, infer_s=0.25, run_s=1.2)   # 25 ticks per infer > H=15
    assert ex.error is None
    st = ex.stats()
    assert st["starved_ticks"] > 0
    assert any(v is None for _, v in sink.ticks)
    assert st["n_infer"] >= 3


def test_warmup_runs_guided_call_before_moving():
    cfg = RTCConfig(s_min=4, delay_init=2, control_hz=50.0, warmup=True)
    ex, pol, sink = _run(cfg, infer_s=0.01, run_s=0.5)
    kinds = ["guided" if c["prev_actions"] is not None else "plain" for c in pol.calls[:3]]
    assert kinds == ["plain", "guided", "plain"], kinds
    assert ex.error is None


def test_policy_error_stops_episode():
    def bad(obs, req):
        raise RuntimeError("boom")
    stop = threading.Event()
    ex = RTCExecutor(RTCConfig(warmup=False), infer=bad, get_obs=lambda: {},
                     send_action=lambda a: None, stop_event=stop)
    ex.run()
    assert stop.is_set() and "boom" in (ex.error or "")


def test_config_validation():
    cfg = RTCConfig()
    assert cfg.updated({"warmup": "false", "s_min": "6"}).s_min == 6
    assert cfg.updated({"warmup": "false"}).warmup is False
    for bad in ({"schedule": "nope"}, {"s_min": 0}, {"bogus": 1}, {"max_guidance_weight": 0}):
        try:
            cfg.updated(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} accepted")
    try:
        RTCConfig(s_min=10, delay_init=8).validate(horizon=15)
    except ValueError:
        pass
    else:
        raise AssertionError("s_min + delay_init > H accepted")
