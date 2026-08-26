"""Real-Time Chunking (RTC) for the FR3 openpi eval portal — self-contained.

Server side (runs inside openpi's venv, nothing in ~/work/openpi is modified):
    rtc_math.py          soft mask + pseudoinverse guidance (JAX, model-agnostic)
    rtc_policy.py        RTCPolicy wrapper around openpi.policies.Policy + the
                         guided Pi0/Pi05 sampler
    scripts/serve_policy.py   drop-in replacement for openpi/scripts/serve_policy.py
                         (same CLI); vanilla requests behave exactly as before
Client side (dashboard process):
    executor.py          RTCConfig + RTCExecutor (paper Algorithm 1, threads)
    dashboard_hook.py    glue for dashboards/openpi.py: episode runner, /rtc
                         routes, status, UI panel
Tools / tests:
    smoke.py             offline check against a running :8000 (no robot)
    ../tests/test_rtc_executor.py, ../tests/test_rtc_math.py
"""
