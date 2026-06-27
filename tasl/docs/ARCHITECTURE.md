# TASL FR3 bench — architecture

Orientation doc for the whole bench: which codebases exist, what each owns,
which machine they run on, and how they connect. The `tasl/` tooling in this
repo is the *control plane* — it orchestrates everything below it but owns no
robot/camera/policy logic itself.

## System diagram (two machines)

```
        DESKTOP  (TASL-1, RTX 4090, ~/ )                         NUC1  (RT kernel, cabled to FR3)
 ┌───────────────────────────────────────────────┐        ┌────────────────────────────────────────┐
 │  tasl/  (operator tooling — this repo)         │        │  Controller backend (ONE at a time):     │
 │   dashboards: collect:8004 openpi:8003 rlinf   │        │                                          │
 │   clients/ (DroidLikeClient)  launch/  tools/  │        │  (A) franky robot_server.py  FastAPI     │
 │        │            │            │             │        │      systemd franka-robot-server  :4242  │
 │        │            │            │             │        │        └─ libfranka 0.18 ── FCI ──┐      │
 │        ▼            ▼            ▼             │        │                                   │      │
 │  openpi serve   RLinf eval   RLinf collect    │        │  (B) droid-nuc-fr3 container      │      │
 │  _policy :8000  (Ray, in     (collect_real_   │        │      polymetis + zerorpc :4242    │      │
 │   (VLA WS)      container)    data.py)         │        │        └─ franka_panda_client ────┤      │
 │        │            │            │             │        │                                   ▼      │
 │  ZED SDK / pyzed  ──┴────────────┘             │        │                            FR3 + Desk    │
 │   2× ZED 2i + wrist (USB3)                     │        │                       172.16.0.1 (FCI)   │
 └───────────────────────────────────────────────┘        └────────────────────────────────────────┘
        HTTP/WS over Tailscale/LAN  ───────────────►  controller :4242
```

## Codebases

| Codebase | Where | What it owns | Runtime / interface |
|---|---|---|---|
| **`tasl/` (this dir)** | `~/RLinf/tasl` | 3 dashboards, NUC client wrapper, launch scripts, ZED viewer. Control plane only. | Flask `:8003/:8004`, `http.server` MJPEG `:8002` |
| **openpi** | `~/work/openpi` | VLA policy: `serve_policy.py` serves a pi05_droid checkpoint over WebSocket; `openpi-client` is the inference client `dashboards/openpi.py` imports. | Python WS server `:8000` |
| **RLinf** | `~/RLinf`, `~/work/rlinf-clone` | Embodied RL/data framework: `collect_real_data.py`, `eval_embodied_agent.py`, `rlinf/envs/realworld/franka/*` (env, polymetis controller, GELLO teleop), configs. Runs in the `rlinf-eval` container. | Python + Ray; `examples/embodiment/*` |
| **lerobot** | `~/work/lerobot` | Dataset format (LeRobot v2.1/v3) — what collection writes, what playback/openpi read. | Library; datasets under `…/outputs/lerobot` |
| **franky `robot_server`** *(backend A)* | NUC1 (v2 plan) | Sole modern libfranka client; HTTP/WS arm+gripper API. `dashboards/openpi.py`'s `RS` client targets this. | FastAPI `:4242`, systemd `franka-robot-server` |
| **DROID / polymetis** *(backend B)* | NUC1 `droid-nuc-fr3` container (current) | The controller collect/rlinf + `DroidLikeClient` use today; zerorpc `run_server.py` → `franka_panda_client`. | zerorpc `:4242` |
| **ZED SDK / pyzed** | Desktop `/usr/local/zed` | Cameras; each dashboard's `CamManager` opens/holds/releases them. | USB3; pyzed 5.3 |
| **FR3 + Desk** | robot `172.16.0.1` | Hardware + FCI activation/brakes. | Desk HTTPS |

## Cross-cutting constraints

1. **Two controller backends, one port, mutually exclusive.** Both franky
   `robot_server` (A) and the DROID container (B) bind `:4242` and are FCI
   clients — only one runs at a time. Today: `openpi` dashboard → (A) franky;
   `collect`/`rlinf` dashboards + `DroidLikeClient` → (B) DROID. **This split is
   the in-flight v1→v2 pivot, not a finished design** (see README's "not yet
   repointed" list).
2. **One FCI client total** — only one dashboard owns the robot at a time;
   `launch/lib.sh kill_other_dashboard` enforces it on the Desktop side. On NUC1,
   the franky `franka-robot-server` systemd unit must be STOPPED while the DROID
   container holds FCI.
3. **One camera owner at a time** — dashboards hold the ZEDs while idle (MJPEG
   preview) and release them so the RLinf env can grab them during a run.
4. **Machine split** — GPU / policy / cameras / dashboards on the Desktop;
   libfranka / FCI on NUC1. They meet only over `:4242` (controller) and `:8000`
   (policy).

## FCI / Desk access (quick ref)

Desk is served by the FR3 Control at `https://172.16.0.1`. Only NUC1
(`172.16.0.2`) is on that link; from the Desktop, tunnel:
`ssh -L 8443:172.16.0.1:443 tasl@100.75.6.62` → browse `https://localhost:8443`.
Unlock joints → sidebar **Activate FCI**. Bring-up order: NUC1 release/grab FCI →
Desk Activate FCI → dashboards.

## Known architectural debt

- **Backend split** (A vs B above) — collect/rlinf should eventually move to the
  franky `robot_server` so there's one controller path.
- **Path drift** — `dashboards/collect.py` / `rlinf.py` still reference the old
  `~/work/rlinf-clone` checkout and the `rlinf-eval` container, not `~/RLinf`.
- **Diverged branch** — fork `tasl-bench-polymetis-controller` (`6640d523`,
  teleop/GELLO) vs local `~/work/rlinf-clone` (`a253363f`) need reconciliation.
