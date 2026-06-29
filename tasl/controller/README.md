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
