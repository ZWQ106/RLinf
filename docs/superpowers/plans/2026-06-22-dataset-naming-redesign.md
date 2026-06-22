# Dataset Naming / Storage Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make FR3 GELLO collection write ONE accumulating LeRobotDataset with a fixed semantic repo_id, SVO aligned by episode_index, training frames at 224, and no redundant `.pt` dump.

**Architecture:** Config-driven changes isolated to the collection pipeline. `CollectEpisode` opens-or-creates a fixed dataset (`repo_id` + explicit `root`); SVO named by the LeRobot `episode_index`; `franka_env` obs resolution config-driven; `.pt` dump gated off. Spec: `docs/superpowers/specs/2026-06-22-dataset-naming-redesign-design.md`.

**Tech Stack:** Python, RLinf, LeRobot 0.1.0 (`LeRobotDataset.create/__init__/add_frame/save_episode`), pytest.

**Working tree:** laptop `vendor/RLinf` on branch `tasl-bench-dataset-naming` (off eval, baseline == container). Deploy each changed file to Desktop `~/work/rlinf-clone` (the `rlinf-eval` bind-mount) for the hardware task. Do NOT push without asking.

---

### Task 1: `LeRobotDatasetWriter` — explicit root + open-or-create

**Files:**
- Modify: `rlinf/data/lerobot_writer.py`
- Test: `rlinf/data/tests/test_lerobot_writer_open.py`

- [ ] **Step 1: Write the failing test**

```python
# rlinf/data/tests/test_lerobot_writer_open.py
import numpy as np, pytest
from rlinf.data.lerobot_writer import LeRobotDatasetWriter

FEATS_KW = dict(robot_type="panda", fps=10, state_dim=4, action_dim=4, has_image=False)

def _add_ep(w, val, n=3):
    for _ in range(n):
        w.add_frame({"observation.state": np.full(4, val, np.float32),
                     "action": np.full(4, val, np.float32), "task": "pick up the cube"})
    w.save_episode()

def test_open_or_create_accumulates_across_instances(tmp_path):
    root = str(tmp_path / "ds"); rid = "tasl/ds_test"
    w1 = LeRobotDatasetWriter(); w1.open_or_create(repo_id=rid, root=root, **FEATS_KW)
    _add_ep(w1, 1.0); _add_ep(w1, 2.0)
    assert w1.dataset.meta.total_episodes == 2
    del w1
    w2 = LeRobotDatasetWriter(); w2.open_or_create(repo_id=rid, root=root, **FEATS_KW)
    assert w2.dataset.meta.total_episodes == 2          # reopened existing
    _add_ep(w2, 3.0)
    assert w2.dataset.meta.total_episodes == 3          # appended, not overwritten
    import json, pathlib
    info = json.loads((pathlib.Path(root) / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run (in container): `docker exec rlinf-eval bash -lc "cd /workspace/rlinf && /opt/venv/openpi/bin/python -m pytest rlinf/data/tests/test_lerobot_writer_open.py -q"`
Expected: FAIL with `AttributeError: 'LeRobotDatasetWriter' object has no attribute 'open_or_create'`.

- [ ] **Step 3: Implement `open_or_create` + `root`**

In `rlinf/data/lerobot_writer.py`: add `root: str | None = None` to `create()` and forward it to `LeRobotDataset.create(..., root=root)`. Add:

```python
def open_or_create(self, repo_id: str, root: str | None = None, **create_kwargs) -> None:
    """Open an existing LeRobotDataset at `root` if present, else create it.

    Enables cross-session accumulation: reopening reads meta.total_episodes so
    save_episode continues the episode_index instead of overwriting.
    """
    import os
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    if root is not None and os.path.isfile(os.path.join(root, "meta", "info.json")):
        self.dataset = LeRobotDataset(repo_id, root=root)
    else:
        self.create(repo_id=repo_id, root=root, **create_kwargs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: same pytest command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rlinf/data/lerobot_writer.py rlinf/data/tests/test_lerobot_writer_open.py
git commit -m "feat(lerobot): open-or-create writer with explicit root for cross-session accumulation"
```

---

### Task 2: `CollectEpisode` — fixed repo_id/root + accumulating writer

**Files:**
- Modify: `rlinf/envs/wrappers/collect_episode.py` (`__init__` 64-132; `_get_lerobot_writer` ~553-576)

- [ ] **Step 1: Add constructor params**

In `__init__` signature (after `record_svo`): add
```python
        repo_id: Optional[str] = None,
        root: Optional[str] = None,
        svo_dir: Optional[str] = None,
```
and store after `self.save_dir = save_dir`:
```python
        self.repo_id = repo_id
        self.root = root
```

- [ ] **Step 2: Use the fixed root for SVO dir**

Replace line 100:
```python
        self._svo_dir = os.path.join(os.path.dirname(save_dir.rstrip("/")), "svo")
```
with:
```python
        # SVO archive tied to the dataset root (parallel dir, same prefix), so it
        # accumulates across sessions alongside the parquet. Falls back to the
        # legacy sibling-of-save_dir when no fixed dataset root is configured.
        if svo_dir is not None:
            self._svo_dir = svo_dir
        elif root is not None:
            self._svo_dir = f"{root.rstrip('/')}_svo"
        else:
            self._svo_dir = os.path.join(os.path.dirname(save_dir.rstrip("/")), "svo")
```

- [ ] **Step 3: Switch the writer to open-or-create at the fixed repo_id**

In `_get_lerobot_writer` (the block that calls `self._lerobot_writer.create(repo_id=os.path.join(self.save_dir, f"rank_{self.rank}", f"id_{self._episodes_written}"), ...)`), replace the `create(...)` call with:

```python
            repo_id = self.repo_id or os.path.join(
                self.save_dir, f"rank_{self.rank}", f"id_{self._episodes_written}"
            )
            self._lerobot_writer.open_or_create(
                repo_id=repo_id,
                root=self.root,
                robot_type=self.robot_type,
                fps=self.fps,
                image_shape=first["image"].shape if "image" in first else None,
                state_dim=int(first["state"].shape[-1]),
                action_dim=int(first["actions"].shape[-1]),
                has_image="image" in first,
                wrist_image_keys=wrist_image_keys,
                extra_view_image_keys=extra_view_image_keys,
                has_intervene_flag="intervene_flag" in first,
            )
```
(When `self.repo_id` is None the legacy path is preserved exactly.)

- [ ] **Step 4: Smoke-import**

Run: `python3 -m py_compile rlinf/envs/wrappers/collect_episode.py` → no error.

- [ ] **Step 5: Commit**

```bash
git add rlinf/envs/wrappers/collect_episode.py
git commit -m "feat(collect): fixed repo_id/root accumulation + dataset-tied SVO dir"
```

---

### Task 3: `CollectEpisode` — SVO named by episode_index

**Files:**
- Modify: `rlinf/envs/wrappers/collect_episode.py` (`_finalize_svo` ~391-427)

- [ ] **Step 1: Replace the per-run `_svo_kept` counter with the LeRobot episode_index**

In `_finalize_svo`, replace:
```python
            idx = self._svo_kept
            self._svo_kept += 1
```
with:
```python
            # Index SVO by the LeRobot episode_index so parquet and SVO are
            # aligned by construction and accumulate across sessions without
            # collision. total_episodes is the next index to be written.
            with self._lerobot_lock:
                idx = self._episodes_written
```
and change the filename template at the `dst = os.path.join(...)` line to 6-digit zero-pad:
```python
                dst = os.path.join(self._svo_dir, f"episode_{idx:06d}_{name}.svo2")
```

- [ ] **Step 2: Verify ordering note**

Confirm `_finalize_svo(kept=True)` runs AFTER `_flush_episode` increments `_episodes_written` for the same episode (read `_maybe_flush`/the done-handling block ~341-364). If `_episodes_written` is incremented inside the executor write (`_write_lerobot_episode`), use the value captured BEFORE the async submit instead (pass it into `_finalize_svo`). Adjust so the SVO idx equals the parquet `episode_index` for the SAME episode.

- [ ] **Step 3: Smoke-import**

Run: `python3 -m py_compile rlinf/envs/wrappers/collect_episode.py` → no error.

- [ ] **Step 4: Commit**

```bash
git add rlinf/envs/wrappers/collect_episode.py
git commit -m "fix(collect): index SVO by lerobot episode_index (no cross-session overwrite)"
```

---

### Task 4: `collect_real_data.py` — pass new params + gate `.pt` dump

**Files:**
- Modify: `examples/embodiment/collect_real_data.py` (CollectEpisode construction ~54-73; ReplayBuffer ~79-93)

- [ ] **Step 1: Forward repo_id/root/svo_dir to CollectEpisode**

In the `CollectEpisode(...)` kwargs, add:
```python
                repo_id=getattr(self.cfg.env.eval.data_collection, "repo_id", None),
                root=getattr(self.cfg.env.eval.data_collection, "root", None),
                svo_dir=getattr(self.cfg.env.eval.data_collection, "svo_dir", None),
```

- [ ] **Step 2: Gate the `.pt` replay-buffer dump**

Wrap the `TrajectoryReplayBuffer(... auto_save=True ... trajectory_format="pt")` so it only auto-saves when configured:
```python
        save_pt = getattr(self.cfg.env.eval.get("data_collection", {}) or {}, "save_pt_trajectory", False) \
            if self.cfg.env.eval.get("data_collection", None) else False
        self.buffer = TrajectoryReplayBuffer(
            seed=self.cfg.seed if hasattr(self.cfg, "seed") else 1234,
            enable_cache=False,
            auto_save=save_pt,
            auto_save_path=buffer_path,
            trajectory_format="pt",
        )
```

- [ ] **Step 3: Smoke-import**

Run: `python3 -m py_compile examples/embodiment/collect_real_data.py` → no error.

- [ ] **Step 4: Commit**

```bash
git add examples/embodiment/collect_real_data.py
git commit -m "feat(collect): forward repo_id/root, gate redundant .pt trajectory dump off by default"
```

---

### Task 5: `franka_env.py` — config-driven obs image size

**Files:**
- Modify: `rlinf/envs/realworld/franka/franka_env.py` (obs space ~626; resize ~787-812; read the cfg in `__init__`)

- [ ] **Step 1: Read `obs_image_size` from cfg in `__init__`**

Add (near where other cfg fields are read):
```python
        # pi0/pi05 resize to 224 internally; 224 is the no-loss/no-waste default.
        # Kept configurable; default 128 preserves the eval path's behavior.
        self._obs_image_size = int(getattr(cfg, "obs_image_size", 128))
```

- [ ] **Step 2: Use it for the observation_space shape**

Replace the hardcoded `shape=(128, 128, 3)` at ~626 with:
```python
                            0, 255, shape=(self._obs_image_size, self._obs_image_size, 3), dtype=np.uint8
```

- [ ] **Step 3: Smoke-import**

Run: `python3 -m py_compile rlinf/envs/realworld/franka/franka_env.py` → no error.

- [ ] **Step 4: Commit**

```bash
git add rlinf/envs/realworld/franka/franka_env.py
git commit -m "feat(franka-env): config-driven obs_image_size (default 128)"
```

---

### Task 6: Collect config — fixed dataset + 224 + no `.pt`

**Files:**
- Modify: `examples/embodiment/config/realworld_collect_data_polymetis_jointvel.yaml`
- Modify: `examples/embodiment/config/env/realworld_franka_jointvel_polymetis.yaml`

- [ ] **Step 1: Point collection at a fixed dataset + drop `.pt`**

In the `data_collection:` block, add:
```yaml
      repo_id: "tasl/fr3_pickcube_v1"
      root: "/home/franka_desktop/work/datasets/fr3_pickcube_v1"
      svo_dir: "/home/franka_desktop/work/datasets/fr3_pickcube_v1_svo"
      save_pt_trajectory: False
```

- [ ] **Step 2: Set training-frame resolution to 224**

In `env/realworld_franka_jointvel_polymetis.yaml`, add under the env block:
```yaml
  obs_image_size: 224
```

- [ ] **Step 3: Commit**

```bash
git add examples/embodiment/config/realworld_collect_data_polymetis_jointvel.yaml examples/embodiment/config/env/realworld_franka_jointvel_polymetis.yaml
git commit -m "config(collect): fixed fr3_pickcube_v1 dataset, 224 obs, no .pt dump"
```

---

### Task 7: Hardware smoke (in `rlinf-eval` container)

**Pre-req:** deploy the 5 changed source files to `~/work/rlinf-clone`, clear pyc; NUC polymetis `:4242` up + FCI active; GELLO on `/dev/ttyACM0`; object in scene.

- [ ] **Step 1: Deploy + clear pyc**

```bash
for f in rlinf/data/lerobot_writer.py rlinf/envs/wrappers/collect_episode.py \
         examples/embodiment/collect_real_data.py rlinf/envs/realworld/franka/franka_env.py \
         examples/embodiment/config/realworld_collect_data_polymetis_jointvel.yaml \
         examples/embodiment/config/env/realworld_franka_jointvel_polymetis.yaml; do
  scp "$f" tasl-desktop:~/work/rlinf-clone/"$f"; done
ssh tasl-desktop 'docker exec -u root rlinf-eval bash -lc "find /workspace/rlinf -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null; echo cleared"'
```

- [ ] **Step 2: Session A — collect 2 success episodes** (via dashboard or `run` script), label success with keyboard.
  Expected: `~/work/datasets/fr3_pickcube_v1/data/chunk-000/episode_000000.parquet` + `episode_000001.parquet`;
  `meta/info.json` `total_episodes==2`; `fr3_pickcube_v1_svo/episode_000000_*.svo2` + `episode_000001_*.svo2`;
  NO `demos/*.pt`.

- [ ] **Step 3: Verify resolution + alignment**

```bash
ssh tasl-desktop 'docker exec rlinf-eval /opt/venv/openpi/bin/python -c "
import json,pyarrow.parquet as pq,io,numpy as np
from PIL import Image
root=\"/home/franka_desktop/work/datasets/fr3_pickcube_v1\"
info=json.load(open(root+\"/meta/info.json\")); print(\"total_episodes\",info[\"total_episodes\"])
print(\"image feat\",info[\"features\"][\"image\"][\"shape\"])   # expect [224,224,3]
"'
```
Expected: `total_episodes 2`, `image feat [224, 224, 3]`.

- [ ] **Step 4: Session B — restart collection, collect 1 more** (fresh process to prove cross-session append).
  Expected: `episode_000002.parquet` appears, `total_episodes==3`, `episode_000002_*.svo2` present, episode_000000/1 untouched.

- [ ] **Step 5: Confirm SVO opens valid** (depth NONE) and frame count matches the parquet episode length.

- [ ] **Step 6: Commit any fixes found during smoke; record results in the plan checkboxes.**

---

## Self-Review notes
- Task 3 Step 2 is the one real risk: the SVO `idx` must equal the parquet `episode_index` for the SAME episode. Confirm whether `_episodes_written` is incremented synchronously in `_flush_episode` or inside the async `_write_lerobot_episode`; capture the index at flush time and thread it into `_finalize_svo` if needed.
- `only_success: True` → only labeled-success episodes are written; SVO `_finalize_svo(kept=...)` already keys off the same success flag, so SVO and parquet stay in lockstep.
- Default `obs_image_size=128` keeps the eval path unchanged; only the collect config opts into 224.
