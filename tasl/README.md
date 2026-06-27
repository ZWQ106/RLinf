# TASL FR3 bench tooling

Operator-side tooling for the TASL Franka FR3 bench, consolidated from loose
files that used to live in `/home/franka_desktop`. This code *drives* RLinf /
openpi / the NUC robot controller — it is not part of the RLinf framework
itself (the framework patches live under `rlinf/` and `examples/`).

## Layout

```
tasl/
  dashboards/        web "mission control" UIs (one per stack)
    collect.py       RLinf data collection  (:8004, host, sudo — uinput)
    openpi.py        openpi serve_policy inference loop (:8003, host)
    rlinf.py         RLinf eval pipeline (:8003, runs INSIDE rlinf-eval container)
  clients/
    droid_client.py        DroidLikeClient — zerorpc wrapper for the NUC DROID/polymetis container
    smoke_droid_client.py  standalone smoke test for the above
  tools/
    zed_viewer.py    dual ZED 2i MJPEG viewer (:8002)
  launch/            self-contained Desktop launchers (lib.sh + start_*.sh + stop.sh)
  logs/              runtime logs (gitignored)
  docs/              handoff notes
```

## Launching (Desktop / TASL-1)

```
tasl/launch/start_collect.sh    # collect dashboard :8004  (re-execs under sudo)
tasl/launch/start_openpi.sh     # openpi  dashboard :8003
tasl/launch/stop.sh
```

Dashboards run from the repo root with `PYTHONPATH=tasl` so `clients.droid_client`
resolves. The collect dashboard kills a running openpi dashboard (and vice-versa)
by matching `dashboards/<name>.py` in the process table — keep that path shape if
you rename anything.

`rlinf.py` is executed inside the `rlinf-eval` container, where this repo is
mounted at `/workspace/rlinf`, so it self-locates at
`/workspace/rlinf/tasl/dashboards/rlinf.py` — no separate copy/mount needed.

## ⚠️ Operational paths NOT yet repointed

These still reference the *old* `work/rlinf-clone` checkout and the existing
container/host conventions. They were left as-is during the file move and must
be reviewed before this becomes the live checkout:

- `dashboards/collect.py`: `CONTAINER = "rlinf-eval"`,
  `LIVE_CAM_DIR_HOST` and `DATASET_ROOTS` → `…/work/rlinf-clone/outputs/...`
- `dashboards/rlinf.py`: `DEFAULT_REPO_PATH_HOST = …/work/rlinf-clone`,
  `EVAL_SCRIPT` under it.
- `dashboards/{openpi,rlinf}.py`: `HOME_STORE_PATH = /home/franka_desktop/_dashboard_home.json`
  (shared runtime home-pose state, lives at host home — intentionally outside the repo).

Decide whether the `rlinf-eval` container should mount `~/RLinf` instead of
`~/work/rlinf-clone`; if so, update the paths above to match.
