"""Dashboard glue test: /rtc routes + panel injection, with a fake runner.

Run:  cd ~/RLinf/tasl && /usr/bin/python3 -m pytest tests/test_rtc_hook.py -q   (needs flask)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flask import Flask  # noqa: E402

from rtc import dashboard_hook as hook  # noqa: E402


class FakeServe:
    def __init__(self, rtc=False):
        self.rtc = rtc


class FakeRunner:
    def __init__(self):
        self.rtc = hook.RTCState()
        self._running = False


def _client(rtc_serve=False):
    app = Flask("t")
    runner = FakeRunner()
    hook.register_routes(app, runner, FakeServe(rtc_serve))
    return app.test_client(), runner


def test_get_and_set_config():
    c, runner = _client()
    j = c.get("/rtc").get_json()
    assert j["ok"] and j["active"] is False and j["live"] == {}
    j = c.post("/rtc", json={"s_min": 5, "max_guidance_weight": 3.0, "schedule": "linear"}).get_json()
    assert j["ok"] and j["cfg"]["s_min"] == 5 and j["cfg"]["schedule"] == "linear"
    assert runner.rtc.cfg.max_guidance_weight == 3.0


def test_invalid_config_rejected_and_unchanged():
    c, runner = _client()
    r = c.post("/rtc", json={"schedule": "bogus"})
    assert r.status_code == 400 and runner.rtc.cfg.schedule == "exp"
    r = c.post("/rtc", json={"s_min": 0})
    assert r.status_code == 400


def test_rejected_while_running():
    c, runner = _client()
    runner._running = True
    assert c.post("/rtc", json={"s_min": 6}).status_code == 409
    assert runner.rtc.cfg.s_min == 4


def test_panel_injection():
    page = "<html><body><div>x</div>\n" + hook.PANEL_MARKER + "\n<script>function toast(){}</script></body></html>"
    out = hook.inject_panel(page)
    assert 'id="rtcCard"' in out and hook.PANEL_MARKER not in out
    assert "{{" not in hook.PANEL_HTML and "{%" not in hook.PANEL_HTML  # Jinja-safe
    assert hook.inject_panel("<html></html>") == "<html></html>"       # marker missing -> untouched


def test_status_shape_and_active_follows_serve():
    runner = FakeRunner()
    s = hook.status(runner)
    assert set(s) == {"active", "cfg", "live", "last_episode"} and s["active"] is False
    c, runner = _client(rtc_serve=True)
    assert hook.active(runner) and c.get("/rtc").get_json()["active"] is True
    assert hook.is_rtc_serve("python /x/tasl/rtc/scripts/serve_policy.py --port 8000")
    assert not hook.is_rtc_serve("python /home/u/work/openpi/scripts/serve_policy.py --port 8000")
