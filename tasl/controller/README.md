# NUC controller — repo-managed config

The essential files for the **`droid-nuc-fr3` polymetis controller** that runs on
NUC1 (`~/polymetis_fr3/`). Kept here so the controller config is version-managed
instead of copied between machines. Full guide: [../docs/CONTROLLER.md](../docs/CONTROLLER.md).

```
controller/
├── docker-compose.yaml        # starts the droid-nuc-fr3 container
├── parameters.py              # mounted in — overrides DROID's default robot IPs (secret scrubbed)
├── Dockerfile.droid-0181      # builds droid-nuc-fr3:0.18.1 (libfranka 0.13.5 → 0.18.1)
└── conf/
    ├── franka_hardware.yaml   # robot_client: robot_ip, gains, safety envelope
    └── franka_panda.yaml      # robot_model: joints, rest pose, limits
```

## Deploy to a NUC

These are the entire working tree of `~/polymetis_fr3/` minus the deprecated
`wrapper/`, `scripts/`, `fairo/`, and the old `Dockerfile`. To stand up a NUC:

```bash
git clone https://github.com/tasl-lab/RLinf ~/RLinf      # or pull
cp -r ~/RLinf/tasl/controller ~/polymetis_fr3            # or symlink / run in place
cd ~/polymetis_fr3
# image: docker load a transferred droid-nuc-fr3:0.18.1, OR build with Dockerfile.droid-0181
docker compose up -d
```

Edit `parameters.py` / `conf/franka_hardware.yaml` if the bench IPs differ, then
commit the change here so every NUC stays in sync.

## Update a running NUC's parameters — `deploy.sh`

To push config changes (impedance gains, limits, IPs) to an already-running NUC
and restart the controller so it re-reads them, run **from the Desktop**:

```bash
cd ~/RLinf/tasl/controller
./deploy.sh                                  # push conf -> NUC1, restart, wait :4242
./deploy.sh --kq "20 15 25 12 18 12 5"       # set joint stiffness first, then deploy
./deploy.sh --no-restart                     # copy only (apply on next restart)
LAUNCH_DRY_RUN=1 ./deploy.sh                  # print actions only
```

`--kq` rewrites `default_Kq` in `conf/franka_hardware.yaml` (lower = softer/more
compliant — the arm yields on contact; gravity is compensated so it still holds
pose). The config files are bind-mounted (see `docker-compose.yaml`), so the
restart re-reads them. **A restart drops FCI** — afterwards re-Activate FCI in
Desk and Recover/Home in the dashboard, or just run `sudo ../launch/teleop.sh`.
The NUC host defaults to `$NUC1_HOST` (`172.16.0.2`); override with an env var.

## Notes

- **Only `parameters.py` is a real override.** `conf/franka_hardware.yaml` and
  `conf/franka_panda.yaml` are **byte-identical to the copies baked into the
  image** (verified) — the mounts are redundant but harmless, and keeping them
  here makes the config visible/editable without cracking the image open.
- **Secret scrubbed:** the live `parameters.py` hard-coded `sudo_password`; the
  repo copy reads it from `$SUDO_PASSWORD` instead (unused on the root-container
  path). Don't commit the real password back.
- The image itself (~18 GB) is **not** in the repo — `docker save | docker load`
  it, or rebuild from `Dockerfile.droid-0181`. See CONTROLLER.md §3 / §1.5.
