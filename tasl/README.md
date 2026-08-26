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
    layout_from_dataset.py  derive a layout (ghost snapshots) from a dataset episode's SVO frame, link it to a task
  controller/        NUC polymetis controller config (compose, parameters.py, Dockerfile, conf/) — see docs/CONTROLLER.md
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

## Paths & configuration (consolidated 2026-07-03)

The `rlinf-eval` container now mounts **`~/RLinf`** at `/workspace/rlinf` (one
checkout for host tooling *and* the in-container framework), with data on a
separate **`~/rlinf_data`** mount so a code swap never touches collected data.
Host paths are **env-driven** (single source of truth in `launch/lib.sh`, matching
defaults derived in Python) — nothing is hard-coded to a specific checkout:

- `launch/lib.sh`: `RLINF_REPO_HOST` (repo, auto-derived from the script location),
  `RLINF_DATA_DIR` (`~/rlinf_data`), `RLINF_CONTAINER`, `RLINF_IMAGE`, `CKPT_HOST`.
  `ensure_rlinf_container` is the version-controlled `docker run` recipe;
  `ensure_container_deps` reinstalls the GELLO stack when missing.
- `dashboards/rlinf.py`: `DEFAULT_REPO_PATH_HOST` = `RLINF_REPO_HOST` or
  `Path(__file__).parents[2]`; `CONTAINER_NAME` = `RLINF_CONTAINER`.
- `dashboards/collect.py`: `DATA_DIR_HOST` = `RLINF_DATA_DIR`; `LIVE_CAM_DIR_HOST`
  and `DATASET_ROOTS` derive from it.
- `dashboards/{openpi,rlinf}.py`: `HOME_STORE_PATH = /home/franka_desktop/_dashboard_home.json`
  (shared runtime home-pose state at host home — intentionally outside the repo).

To repoint the container mounts: `docker rm -f rlinf-eval`, then re-run any
launcher (it recreates the container via `ensure_rlinf_container`). `~/work/rlinf-clone`
is retired.
