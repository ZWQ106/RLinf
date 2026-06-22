# Dataset Naming / Storage Redesign — Design

**Date:** 2026-06-22
**Branch:** `tasl-bench-dataset-naming` (off `franka-fr3/eval-pi05-polymetis`)
**Scope:** RLinf real-world GELLO collection pipeline (FR3 bench)

## Problem

The current collection scheme fragments data and will make downstream
experiments (pi05_droid fine-tune, RECAP RL, dataset merge, versioning) hard:

1. **Per-run timestamped dirs.** The dashboard sets `log_path=collect_<YYYYMMDD>_<HHMMSS>`;
   each run becomes its own dataset. 10 rollouts → 10 fragmented 1-episode datasets.
2. **repo_id is a relative path.** `collect_episode.py` builds
   `repo_id = os.path.join(save_dir, "rank_R", "id_I")`. `LeRobotDataset.create`
   then nests it under `HF_LEROBOT_HOME`, so the real data lands in
   `<HF_LEROBOT_HOME>/outputs/<run>/collected_data/rank_0/id_0/` while the run
   dir's own `collected_data/` stays empty. Confusing; easy to look in the wrong place.
3. **SVO index drift.** SVO files use a per-run `_svo_kept` counter
   (`episode_0_*.svo2`) that resets every run, while the parquet uses the
   LeRobot `episode_index`. Across sessions the second run's `episode_0_*.svo2`
   **overwrites** the first run's.
4. **Redundant `.pt` copy.** A `TrajectoryReplayBuffer(trajectory_format="pt")`
   writes a third copy to `demos/trajectory_*.pt`, parallel to the LeRobot parquet.
5. **Training frames are 128×128.** pi0/pi05 resize every image to 224×224
   internally (`resize_with_pad`), so 128 is upscaled → lossy. 224 is the
   no-loss / no-waste sweet spot most openpi fine-tunes use.

## Verified facts (this session)

- **LeRobot 0.1.0 supports clean cross-session append.** Empirical test:
  `create()` 2 episodes → `del` → reopen `LeRobotDataset(repo_id, root)` (reads
  `total_episodes=2`) → `save_episode()` continues `episode_index` 2,3,…; disk
  shows `episode_000000/1/2.parquet`, `meta/episodes.jsonl` clean. No `resume=`
  flag needed; `add_episode/consolidate/finalize` do **not** exist at 0.1.0 and
  are **not** required.
- `LeRobotDataset.create(repo_id, fps, root, robot_type, features, use_videos…)`
  takes an explicit `root` → setting it kills the `HF_LEROBOT_HOME` double-nesting.
- Standard layout (research-confirmed, 24/25 claims): `data/chunk-{NNN}/episode_{NNNNNN}.parquet`,
  `videos/chunk-{NNN}/{video_key}/episode_{NNNNNN}.mp4`, `meta/{info,episodes,tasks,stats,episodes_stats}`.
  repo_id convention is `<org>/<dataset_name>`.

## Design

Config-driven, isolated to the collection pipeline. The HD 720p SVO is kept as a
separate experiment archive (NOT folded into the LeRobot dataset).

### 1. Fixed semantic repo_id + explicit root
`data_collection` config gains `repo_id` and `root`. When `repo_id` is set,
`CollectEpisode` writes to that fixed dataset at that fixed root — independent of
the dashboard's `log_path`. Example:
```yaml
data_collection:
  repo_id: "tasl/fr3_pickcube_v1"
  root:    "/home/franka_desktop/work/datasets/fr3_pickcube_v1"
```

### 2. Open-if-exists else create (accumulation)
`CollectEpisode._get_lerobot_writer` opens an existing dataset at `root` when one
is present (`meta/info.json` exists), else `create()`. Same task across sessions
→ episodes accumulate as `episode_000000..N`. **Requirement:** the task string
must be byte-identical across sessions (it is config-driven via `task_description`,
reused by `tasks.jsonl`).

### 3. Training frames 224×224
`franka_env` gains a config `obs_image_size` (default kept at **128** to preserve
the eval path's current behavior); the collect config sets it to **224**. Drives
the `observation_space["frames"]` shape and the `cv2.resize` target.

### 4. SVO indexed by episode_index, fixed sibling dir
SVO files are named by the LeRobot `episode_index` (`episode_{NNNNNN}_<cam>.svo2`)
in a fixed dir tied to the dataset (config `svo_dir`, default `<root>_svo`), with
`index.json` keyed by `episode_index`. Replaces the `_svo_kept` per-run counter →
no cross-session overwrite, parquet↔SVO always aligned by construction.

### 5. Drop the redundant `.pt` dump
`collect_real_data.py` gates the `TrajectoryReplayBuffer` auto-save behind a
config flag `save_pt_trajectory` (default **False**). LeRobot parquet (+ optional
SVO) is the single source of truth.

### Final on-disk shape
```
~/work/datasets/
  fr3_pickcube_v1/                       # LeRobot dataset (training, parquet @224 + meta)
  │   ├─ data/chunk-000/episode_000000.parquet ...
  │   └─ meta/{info,episodes,tasks,stats,episodes_stats}
  └─ fr3_pickcube_v1_svo/                # HD master (experiments), parallel + same prefix
      ├─ episode_000000_exterior.svo2
      ├─ episode_000000_wrist.svo2
      └─ index.json                      # {episode_index: [files...], task, date}
```

## What this fixes

| Mess | Fix |
|---|---|
| per-run timestamped fragments | fixed repo_id, episodes accumulate |
| repo_id relative → double-nest, empty run dir | `<org>/<name>` + explicit `root` |
| SVO per-run counter overwrites on append | name by `episode_index`, fixed `svo_dir` |
| redundant `.pt` | gated off by default |
| 128 training frames (lossy for pi05) | config `obs_image_size`, collect uses 224 |

## Out of scope
- Folding HD video into the LeRobot dataset as mp4 (kept as SVO sidecar by user choice).
- Bumping the **eval** path resolution (default stays 128; only collect uses 224).
- Provenance metadata schema beyond what LeRobot already stores (task/success/intervene_flag) — deferred.
- Dataset merge tooling / v3.0 upgrade — deferred (open question).

## Constraints / risks
- Vendored RLinf: work on `tasl-bench-dataset-naming`, test in the `rlinf-eval`
  container (bind-mounts `~/work/rlinf-clone`), commit on the branch; **do not push
  without asking**.
- `only_success: True` means only success-labeled episodes accumulate — operator
  must label with the keyboard or nothing is written.
- Global `stats.json` is written incrementally and may lag; harmless because
  openpi recomputes norm stats via `compute_norm_stats` before training.
- Back-compat: when `repo_id` is absent, `CollectEpisode` keeps the legacy
  `save_dir`-based behavior so other configs don't break.
