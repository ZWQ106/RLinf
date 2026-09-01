---
license: other
pretty_name: FR3 OOD rollouts — pi05-droid-ft-15k
task_categories:
- robotics
tags:
- robotics
- franka
- fr3
- vla
- pi05
- rollouts
- eval
configs:
- config_name: default
  drop_labels: true
  data_files:
  - split: train
    path:
    - metadata.jsonl
    - web/**/*.mp4
---
# pi05-droid-ft-15k — OOD rollouts (T1-a … T5-b)

<!-- gen:summary -->**206 rollouts / 3.8 GB** over **10 tasks** (T1-a … T5-b), ~2 rollouts per OOD layout, recorded 2026-08-27 … 2026-08-28. **117/206 succeeded (56.8%)**.<!-- /gen:summary -->

- ckpt: `pi05_droid_franka_lora_10task_v2/15000 +rtc` (HF `ZhixuLi/pi05-droid-franka-lora-10task-v2`, step 15000, served with RTC)
- Recorded 2026-08-27/28 on the FR3 bench via the eval portal (`tasl/dashboards/openpi.py`)
- `steps` = policy control steps (15 Hz), capped at **1200** by the eval runner. Rollouts at the
  cap are timeouts (`timeout: true`) and are marked ⏱ in the index below; their step and time
  figures are censored.
- **Verdict is `mark`, not the filename suffix.** A few rollouts were re-marked after export and
  kept a stale `_T` / missing suffix (`suffix_stale: true`); they are flagged ⚠️ in the index
  below. The portal now renames on re-mark, so this won't recur. Always filter on `success` /
  `mark`, never on the stem.

## Browsing the rollouts

### Space (works today)

**https://huggingface.co/spaces/axisrobotics/fr3-rollout-browser** — filter by task / layout /
verdict / timeout, search prompts, watch each rollout next to its control trajectory (7 joint
positions, gripper, inference latency, RTC ticks), plus a 6-up grid and per-task success rates.

### Two encodings — use `web/` in a browser

<!-- gen:encodings -->
The recorder wrote **MPEG-4 Part 2** (`mp4v`, Simple Profile) with the `moov` atom
*after* `mdat`. No browser decodes `mp4v`, and the trailing `moov` blocks progressive
playback, so the original files will not play in the Hub preview, the dataset viewer,
or any `<video>` element — they need VLC/ffmpeg.

`web/` mirrors the tree with **H.264 High / yuv420p, `+faststart`, 2 s keyframes** at
the same native resolution and frame rate. Same frames, plays everywhere, ~47% of the size.

| | codec | size | plays in a browser |
|---|---|---|---|
| `<task>-ood/…/<stem>.mp4` | `mp4v` (MPEG-4 Part 2) | 3.8 GB | ❌ |
| `web/<task>-ood/…/<stem>.mp4` | `avc1` (H.264 High) | 1.8 GB | ✅ |
<!-- /gen:encodings -->

`metadata.jsonl` points `file_name` at the `web/` copy, so the viewer, `load_dataset` and the Space
all get the playable one; `source_video` keeps the path to the original.

### Dataset viewer

The repo is laid out as a `videofolder`: root `metadata.jsonl` carries one row per rollout with a
`file_name` pointing at that rollout's mp4, so the Hub's Data Studio renders an inline video player
per row plus filterable columns (`task`, `layout`, `success`, `steps`, …) and a SQL console.

The explicit `configs:` block above is **required**: the repo holds two `.json` and one `.jsonl`
sidecar per mp4, and without it the loader infers the `json` builder from the extension
majority and no video column appears. It also scopes the glob to `web/**/*.mp4` so the unplayable
originals stay out of the viewer.

> **Note:** the Hub dataset viewer does not run on *private* datasets unless the **owning
> organization** is on a Team/Enterprise plan (an individual PRO account covers only its own repos).
> While this repo is private under `TASL-FR3`, use the Space above, the index at the bottom of this
> card, or `load_dataset` locally. The `configs:` block is already in place, so the viewer switches
> on by itself if the repo goes public or the org moves to Team.

### Locally

```python
from datasets import load_dataset  # needs: pip install "datasets" torchcodec

ds = load_dataset("TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k", split="train")
ds = ds.filter(lambda r: r["task"] == "T2-b" and not r["success"])
ds[0]["video"]      # decoded video
ds[0]["prompt"], ds[0]["steps"], ds[0]["mark"]
```

To skip video decoding and just query the eval results:

```python
import pandas as pd
from huggingface_hub import hf_hub_download

df = pd.read_json(hf_hub_download("TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k",
                                  "metadata.jsonl", repo_type="dataset"), lines=True)
df.groupby("task")["success"].agg(["mean", "count"])
```

## Files

One directory per rollout, `<task>-ood/<stem>/<stem>.{mp4,traj.jsonl,frames.json,json}`:

| file | contents |
|---|---|
| `<stem>.mp4` | original rollout video, `mp4v` — **not browser-playable** |
| `web/<same path>` | H.264 + faststart copy, plays in any browser |
| `<stem>.json` | per-rollout sidecar — the source of truth for `metadata.jsonl` |
| `<stem>.frames.json` | flat list of wall-clock timestamps, one per video frame |
| `<stem>.traj.jsonl` | one line per control step: `t`, `ts`, `iter`, `q` (7 joints), `grip`, `infer_ms`, `rtc` (`s`, `d`, `elapsed`, `guided`, `ticks`, `starved_ticks`), `actions` (16×8 chunk) |

`metadata.jsonl` columns beyond the raw sidecar fields:

| column | meaning |
|---|---|
| `file_name` | relative path to the **`web/`** mp4 — links the row to its video for the viewer |
| `source_video` | relative path to the original `mp4v` recording |
| `success` | `mark == "success"`; **use this, not the filename suffix** |
| `timeout` | `steps >= 1200`, i.e. the rollout was cut off by the eval runner's cap |
| `suffix_stale` | filename `_T`/`_F`/`_Q` suffix disagrees with `mark` — always trust `mark` |
| `task_group`, `variant`, `ood_index`, `rollout` | parsed out of the stem for grouping/filtering |
| `traj_file`, `frames_file`, `sidecar_file` | relative paths to the other three per-rollout files |

## Stats

`pi05-droid-ft-15k_ood_stats.xlsx` has two sheets: `per-task` (aggregates + an `ALL` row) and
`rollouts` (one row per rollout); the same numbers are in the shared sheet:
https://docs.google.com/spreadsheets/d/1J_CD5HYlcdYnrfNVtSUw6h_0DI_9c-sIh03hflh5QMA/edit?gid=467699410#gid=467699410

`unsure` rollouts are excluded from `n` and `SR %`.

<!-- gen:stats -->
| task | prompt | n | SR % | steps mean | time mean s |
|------|--------|---|------|-----------|-------------|
| T1-a | pick up the blue cup and place it into the red cup | 20 | 75.0 | 209.9 | 55.9 |
| T1-b | stack the red block on top of the blue block | 24 | 29.2 | 377.4 | 100.8 |
| T2-a | press the blue button | 20 | 90.0 | 105.3 | 28.1 |
| T2-b | close the lid of the wooden shape sorter box | 19 | 100.0 | 432.5 | 115.3 |
| T3-a | align the three colored blocks to the same orientation | 20 | 40.0 | 191.4 | 50.9 |
| T3-b | rotate the red block so that it is perpendicular to the blue block | 20 | 55.0 | 161.9 | 43.1 |
| T4-a | insert the orange block into the wooden shape sorter box | 20 | 45.0 | 172.3 | 45.8 |
| T4-b | insert the book into the black book stand | 22 | 36.4 | 294.5 | 78.8 |
| T5-a | pull the smaller book out of the black book stand | 21 | 71.4 | 114.6 | 30.5 |
| T5-b | pull the small block out from under the large block | 20 | 35.0 | 940.9 | 255.6 |
| **ALL** | | **206** | **56.8** | **300.0** | **80.4** |
<!-- /gen:stats -->

### vs `cotrain-pbc-v2-8000` on the shared tasks

Companion dataset: `TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000`.

<!-- gen:compare -->
| task | cotrain-pbc-v2-8000 | pi05-droid-ft-15k | Δ |
|------|-------------------:|-----------------:|---:|
| T1-a | 65.0 % (n=20) | 75.0 % (n=20) | +10.0 |
| T1-b | 20.0 % (n=20) | 29.2 % (n=24) | +9.2 |
| T2-a | 90.0 % (n=20) | 90.0 % (n=20) | ±0.0 |
| T2-b | 15.0 % (n=20) | 100.0 % (n=19) | +85.0 |
| T3-a | 55.0 % (n=20) | 40.0 % (n=20) | −15.0 |
| T3-b | 33.3 % (n=21) | 55.0 % (n=20) | +21.7 |
| T4-a | 10.0 % (n=20) | 45.0 % (n=20) | +35.0 |
| T4-b | 30.0 % (n=20) | 36.4 % (n=22) | +6.4 |
| T5-a | 75.0 % (n=20) | 71.4 % (n=21) | −3.6 |
| T5-b | 40.0 % (n=20) | 35.0 % (n=20) | −5.0 |
| **overlap** | **43.3 % (n=201)** | **56.8 % (n=206)** | **+13.5** |
<!-- /gen:compare -->

Caveat: the two evals were recorded on different days (08-27/28 vs 08-30/31) with the bench re-set
between them, so per-layout scene state is not identical — treat the Δ as a task-level trend, not a
paired comparison.

## Provenance

Uploaded 2026-08-28 from `~/RLinf/saved_demo` on the FR3 desktop; the repo layout mirrors disk minus
the ckpt folder level. One repo per ckpt keeps the tree identical to `saved_demo/`, so future ckpts
(`cotrain-pbc-v2-8000`, …) get their own repo.

```bash
PY=~/miniconda3/envs/nemo_peft/bin/python
cd ~/RLinf/saved_demo
STAGE=/tmp/hf_stage_pi05-droid-ft-15k && rm -rf $STAGE && mkdir -p $STAGE
for d in T?-?-ood; do ln -s $PWD/$d/pi05-droid-ft-15k $STAGE/$d; done
$PY - <<'PYEOF'
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k", repo_type="dataset",
                private=True, exist_ok=True)
api.upload_large_folder(repo_id="TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k",
                        repo_type="dataset", folder_path="/tmp/hf_stage_pi05-droid-ft-15k")
PYEOF
```

`web/`, `metadata.jsonl`, the stats workbook and the `<!-- gen:* -->` blocks in this card are all
regenerated (and the tree re-uploaded) by one committed tool:

```bash
python tasl/tools/publish_ood_rollouts.py pi05-droid-ft-15k --compare cotrain-pbc-v2-8000
```

It is idempotent — already-encoded rollouts are skipped and the hand-written prose around the
generated blocks is left alone.

## Rollout index

<!-- gen:index -->
All 206 rollouts, grouped by task. Each link opens the browser-playable copy under `web/` on the Hub (you must be signed in — the repo is private).

Verdicts come from `mark`; ❓ unsure rollouts are excluded from the success rates above, ⏱ marks a rollout that hit the step cap, ⚠️ one whose filename suffix is stale.

<details>
<summary><b>T1-a</b> — <i>pick up the blue cup and place it into the red cup</i> — <b>15/20</b> (75%)</summary>

| rollout | verdict | steps | duration | video |
|---|---|---:|---:|---|
| `T1-a-OOD1_r01` | ❌ fail | 677 | 180.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-a-ood/T1-a-OOD1_r01_F/T1-a-OOD1_r01_F.mp4) |
| `T1-a-OOD1_r02` | ✅ success | 87 | 23.0 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-a-ood/T1-a-OOD1_r02_T/T1-a-OOD1_r02_T.mp4) |
| `T1-a-OOD2_r01` | ✅ success | 87 | 23.2 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-a-ood/T1-a-OOD2_r01_T/T1-a-OOD2_r01_T.mp4) |
| `T1-a-OOD2_r02` | ✅ success | 83 | 22.2 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-a-ood/T1-a-OOD2_r02_T/T1-a-OOD2_r02_T.mp4) |
| `T1-a-OOD3_r01` | ✅ success | 118 | 31.3 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-a-ood/T1-a-OOD3_r01_T/T1-a-OOD3_r01_T.mp4) |
| `T1-a-OOD3_r02` | ❌ fail | 179 | 47.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-a-ood/T1-a-OOD3_r02_F/T1-a-OOD3_r02_F.mp4) |
| `T1-a-OOD6_r01` | ✅ success | 165 | 43.8 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-a-ood/T1-a-OOD6_r01_T/T1-a-OOD6_r01_T.mp4) |
| `T1-a-OOD6_r02` | ✅ success | 356 | 94.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-a-ood/T1-a-OOD6_r02_T/T1-a-OOD6_r02_T.mp4) |
| `T1-a-OOD7_r01` | ❌ fail | 115 | 30.4 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-a-ood/T1-a-OOD7_r01_F/T1-a-OOD7_r01_F.mp4) |
| `T1-a-OOD7_r02` | ✅ success | 207 | 55.3 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-a-ood/T1-a-OOD7_r02_T/T1-a-OOD7_r02_T.mp4) |
| `T1-a-OOD8_r01` | ✅ success | 107 | 28.3 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-a-ood/T1-a-OOD8_r01_T/T1-a-OOD8_r01_T.mp4) |
| `T1-a-OOD8_r02` | ✅ success | 95 | 25.4 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-a-ood/T1-a-OOD8_r02_T/T1-a-OOD8_r02_T.mp4) |
| `T1-a-OOD9_r01` | ❌ fail | 102 | 27.0 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-a-ood/T1-a-OOD9_r01_F/T1-a-OOD9_r01_F.mp4) |
| `T1-a-OOD9_r02` | ✅ success | 225 | 60.0 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-a-ood/T1-a-OOD9_r02_T/T1-a-OOD9_r02_T.mp4) |
| `T1-a-OOD10_r01` | ❌ fail ⚠️ | 880 | 234.8 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-a-ood/T1-a-OOD10_r01_T/T1-a-OOD10_r01_T.mp4) |
| `T1-a-OOD10_r02` | ✅ success | 139 | 36.8 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-a-ood/T1-a-OOD10_r02_T/T1-a-OOD10_r02_T.mp4) |
| `T1-a-OOD11_r01` | ✅ success | 106 | 28.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-a-ood/T1-a-OOD11_r01_T/T1-a-OOD11_r01_T.mp4) |
| `T1-a-OOD11_r02` | ✅ success | 134 | 35.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-a-ood/T1-a-OOD11_r02_T/T1-a-OOD11_r02_T.mp4) |
| `T1-a-OOD12_r01` | ✅ success | 85 | 22.4 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-a-ood/T1-a-OOD12_r01_T/T1-a-OOD12_r01_T.mp4) |
| `T1-a-OOD12_r02` | ✅ success | 252 | 67.3 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-a-ood/T1-a-OOD12_r02_T/T1-a-OOD12_r02_T.mp4) |

</details>

<details>
<summary><b>T1-b</b> — <i>stack the red block on top of the blue block</i> — <b>7/24</b> (29%)</summary>

| rollout | verdict | steps | duration | video |
|---|---|---:|---:|---|
| `T1-b-OOD1_r01` | ❌ fail | 319 | 84.8 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-b-ood/T1-b-OOD1_r01_F/T1-b-OOD1_r01_F.mp4) |
| `T1-b-OOD1_r02` | ✅ success | 461 | 122.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-b-ood/T1-b-OOD1_r02_T/T1-b-OOD1_r02_T.mp4) |
| `T1-b-OOD2_r01` | ❌ fail | 133 | 35.3 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-b-ood/T1-b-OOD2_r01_F/T1-b-OOD2_r01_F.mp4) |
| `T1-b-OOD2_r02` | ❌ fail | 153 | 40.6 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-b-ood/T1-b-OOD2_r02_F/T1-b-OOD2_r02_F.mp4) |
| `T1-b-OOD3_r01` | ❌ fail | 951 | 253.6 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-b-ood/T1-b-OOD3_r01_F/T1-b-OOD3_r01_F.mp4) |
| `T1-b-OOD3_r02` | ❌ fail | 296 | 79.0 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-b-ood/T1-b-OOD3_r02_F/T1-b-OOD3_r02_F.mp4) |
| `T1-b-OOD4_r01` | ❌ fail | 773 | 206.0 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-b-ood/T1-b-OOD4_r01_F/T1-b-OOD4_r01_F.mp4) |
| `T1-b-OOD4_r02` | ❌ fail | 315 | 83.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-b-ood/T1-b-OOD4_r02_F/T1-b-OOD4_r02_F.mp4) |
| `T1-b-OOD5_r01` | ❌ fail | 1078 | 287.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-b-ood/T1-b-OOD5_r01_F/T1-b-OOD5_r01_F.mp4) |
| `T1-b-OOD5_r02` | ❌ fail | 329 | 87.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-b-ood/T1-b-OOD5_r02_F/T1-b-OOD5_r02_F.mp4) |
| `T1-b-OOD6_r01` | ❌ fail | 165 | 44.2 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-b-ood/T1-b-OOD6_r01_F/T1-b-OOD6_r01_F.mp4) |
| `T1-b-OOD6_r02` | ❌ fail | 112 | 29.6 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-b-ood/T1-b-OOD6_r02_F/T1-b-OOD6_r02_F.mp4) |
| `T1-b-OOD7_r01` | ❌ fail | 371 | 99.8 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-b-ood/T1-b-OOD7_r01_F/T1-b-OOD7_r01_F.mp4) |
| `T1-b-OOD7_r02` | ❌ fail | 616 | 165.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-b-ood/T1-b-OOD7_r02_F/T1-b-OOD7_r02_F.mp4) |
| `T1-b-OOD11_r01` | ✅ success | 80 | 21.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-b-ood/T1-b-OOD11_r01_T/T1-b-OOD11_r01_T.mp4) |
| `T1-b-OOD11_r02` | ✅ success | 99 | 26.3 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-b-ood/T1-b-OOD11_r02_T/T1-b-OOD11_r02_T.mp4) |
| `T1-b-OOD12_r01` | ❌ fail | 281 | 75.2 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-b-ood/T1-b-OOD12_r01_F/T1-b-OOD12_r01_F.mp4) |
| `T1-b-OOD12_r02` | ❌ fail | 310 | 83.3 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-b-ood/T1-b-OOD12_r02_F/T1-b-OOD12_r02_F.mp4) |
| `T1-b-OOD13_r01` | ✅ success | 125 | 33.6 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-b-ood/T1-b-OOD13_r01_T/T1-b-OOD13_r01_T.mp4) |
| `T1-b-OOD13_r02` | ✅ success | 94 | 24.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-b-ood/T1-b-OOD13_r02_T/T1-b-OOD13_r02_T.mp4) |
| `T1-b-OOD14_r01` | ✅ success | 93 | 24.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-b-ood/T1-b-OOD14_r01_T/T1-b-OOD14_r01_T.mp4) |
| `T1-b-OOD14_r02` | ✅ success | 132 | 35.3 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-b-ood/T1-b-OOD14_r02_T/T1-b-OOD14_r02_T.mp4) |
| `T1-b-OOD15_r01` | ❌ fail | 997 | 266.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-b-ood/T1-b-OOD15_r01_F/T1-b-OOD15_r01_F.mp4) |
| `T1-b-OOD15_r02` | ❌ fail | 775 | 207.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T1-b-ood/T1-b-OOD15_r02_F/T1-b-OOD15_r02_F.mp4) |

</details>

<details>
<summary><b>T2-a</b> — <i>press the blue button</i> — <b>18/20</b> (90%)</summary>

| rollout | verdict | steps | duration | video |
|---|---|---:|---:|---|
| `T2-a-OOD1_r01` | ✅ success | 144 | 38.8 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-a-ood/T2-a-OOD1_r01_T/T2-a-OOD1_r01_T.mp4) |
| `T2-a-OOD1_r02` | ✅ success | 96 | 25.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-a-ood/T2-a-OOD1_r02_T/T2-a-OOD1_r02_T.mp4) |
| `T2-a-OOD2_r01` | ✅ success | 63 | 16.6 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-a-ood/T2-a-OOD2_r01_T/T2-a-OOD2_r01_T.mp4) |
| `T2-a-OOD2_r02` | ✅ success | 48 | 12.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-a-ood/T2-a-OOD2_r02_T/T2-a-OOD2_r02_T.mp4) |
| `T2-a-OOD3_r01` | ❌ fail | 269 | 72.6 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-a-ood/T2-a-OOD3_r01_F/T2-a-OOD3_r01_F.mp4) |
| `T2-a-OOD3_r02` | ❌ fail | 247 | 66.0 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-a-ood/T2-a-OOD3_r02_F/T2-a-OOD3_r02_F.mp4) |
| `T2-a-OOD4_r01` | ✅ success | 61 | 16.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-a-ood/T2-a-OOD4_r01_T/T2-a-OOD4_r01_T.mp4) |
| `T2-a-OOD4_r02` | ✅ success | 46 | 12.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-a-ood/T2-a-OOD4_r02_T/T2-a-OOD4_r02_T.mp4) |
| `T2-a-OOD5_r01` | ✅ success | 68 | 18.2 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-a-ood/T2-a-OOD5_r01_T/T2-a-OOD5_r01_T.mp4) |
| `T2-a-OOD5_r02` | ✅ success | 64 | 16.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-a-ood/T2-a-OOD5_r02_T/T2-a-OOD5_r02_T.mp4) |
| `T2-a-OOD6_r01` | ✅ success | 56 | 15.0 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-a-ood/T2-a-OOD6_r01_T/T2-a-OOD6_r01_T.mp4) |
| `T2-a-OOD6_r02` | ✅ success | 71 | 18.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-a-ood/T2-a-OOD6_r02_T/T2-a-OOD6_r02_T.mp4) |
| `T2-a-OOD7_r01` | ✅ success | 338 | 90.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-a-ood/T2-a-OOD7_r01_T/T2-a-OOD7_r01_T.mp4) |
| `T2-a-OOD7_r02` | ✅ success | 204 | 54.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-a-ood/T2-a-OOD7_r02_T/T2-a-OOD7_r02_T.mp4) |
| `T2-a-OOD8_r01` | ✅ success | 39 | 10.4 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-a-ood/T2-a-OOD8_r01_T/T2-a-OOD8_r01_T.mp4) |
| `T2-a-OOD8_r02` | ✅ success | 46 | 12.0 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-a-ood/T2-a-OOD8_r02_T/T2-a-OOD8_r02_T.mp4) |
| `T2-a-OOD9_r01` | ✅ success | 70 | 18.8 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-a-ood/T2-a-OOD9_r01_T/T2-a-OOD9_r01_T.mp4) |
| `T2-a-OOD9_r02` | ✅ success | 80 | 21.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-a-ood/T2-a-OOD9_r02_T/T2-a-OOD9_r02_T.mp4) |
| `T2-a-OOD10_r01` | ✅ success | 47 | 12.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-a-ood/T2-a-OOD10_r01_T/T2-a-OOD10_r01_T.mp4) |
| `T2-a-OOD10_r02` | ✅ success | 49 | 12.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-a-ood/T2-a-OOD10_r02_T/T2-a-OOD10_r02_T.mp4) |

</details>

<details>
<summary><b>T2-b</b> — <i>close the lid of the wooden shape sorter box</i> — <b>19/19</b> (100%)</summary>

| rollout | verdict | steps | duration | video |
|---|---|---:|---:|---|
| `T2-b-OOD1_r01` | ✅ success | 366 | 97.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-b-ood/T2-b-OOD1_r01_T/T2-b-OOD1_r01_T.mp4) |
| `T2-b-OOD1_r02` | ✅ success | 340 | 90.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-b-ood/T2-b-OOD1_r02_T/T2-b-OOD1_r02_T.mp4) |
| `T2-b-OOD2_r01` | ✅ success | 918 | 244.6 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-b-ood/T2-b-OOD2_r01_T/T2-b-OOD2_r01_T.mp4) |
| `T2-b-OOD2_r02` | ✅ success | 95 | 25.4 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-b-ood/T2-b-OOD2_r02_T/T2-b-OOD2_r02_T.mp4) |
| `T2-b-OOD3_r01` | ✅ success | 126 | 33.4 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-b-ood/T2-b-OOD3_r01_T/T2-b-OOD3_r01_T.mp4) |
| `T2-b-OOD3_r02` | ✅ success | 1013 | 270.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-b-ood/T2-b-OOD3_r02_T/T2-b-OOD3_r02_T.mp4) |
| `T2-b-OOD4_r01` | ✅ success | 98 | 26.2 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-b-ood/T2-b-OOD4_r01_T/T2-b-OOD4_r01_T.mp4) |
| `T2-b-OOD4_r02` | ✅ success | 130 | 34.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-b-ood/T2-b-OOD4_r02_T/T2-b-OOD4_r02_T.mp4) |
| `T2-b-OOD5_r01` | ✅ success ⏱ | 1200 | 319.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-b-ood/T2-b-OOD5_r01_T/T2-b-OOD5_r01_T.mp4) |
| `T2-b-OOD5_r02` | ✅ success | 728 | 193.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-b-ood/T2-b-OOD5_r02_T/T2-b-OOD5_r02_T.mp4) |
| `T2-b-OOD6_r01` | ✅ success | 68 | 18.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-b-ood/T2-b-OOD6_r01_T/T2-b-OOD6_r01_T.mp4) |
| `T2-b-OOD6_r02` | ✅ success | 588 | 156.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-b-ood/T2-b-OOD6_r02_T/T2-b-OOD6_r02_T.mp4) |
| `T2-b-OOD7_r01` | ✅ success | 833 | 221.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-b-ood/T2-b-OOD7_r01_T/T2-b-OOD7_r01_T.mp4) |
| `T2-b-OOD7_r02` | ✅ success | 866 | 230.8 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-b-ood/T2-b-OOD7_r02_T/T2-b-OOD7_r02_T.mp4) |
| `T2-b-OOD8_r01` | ✅ success | 108 | 28.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-b-ood/T2-b-OOD8_r01_T/T2-b-OOD8_r01_T.mp4) |
| `T2-b-OOD8_r02` | ✅ success | 125 | 33.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-b-ood/T2-b-OOD8_r02_T/T2-b-OOD8_r02_T.mp4) |
| `T2-b-OOD9_r01` | ✅ success | 357 | 95.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-b-ood/T2-b-OOD9_r01_T/T2-b-OOD9_r01_T.mp4) |
| `T2-b-OOD10_r01` | ✅ success | 161 | 42.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-b-ood/T2-b-OOD10_r01_T/T2-b-OOD10_r01_T.mp4) |
| `T2-b-OOD10_r02` | ✅ success | 98 | 25.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T2-b-ood/T2-b-OOD10_r02_T/T2-b-OOD10_r02_T.mp4) |

</details>

<details>
<summary><b>T3-a</b> — <i>align the three colored blocks to the same orientation</i> — <b>8/20</b> (40%)</summary>

| rollout | verdict | steps | duration | video |
|---|---|---:|---:|---|
| `T3-a-OOD1_r01` | ❌ fail | 652 | 174.0 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-a-ood/T3-a-OOD1_r01_F/T3-a-OOD1_r01_F.mp4) |
| `T3-a-OOD1_r02` | ✅ success | 71 | 18.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-a-ood/T3-a-OOD1_r02_T/T3-a-OOD1_r02_T.mp4) |
| `T3-a-OOD2_r01` | ✅ success | 83 | 21.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-a-ood/T3-a-OOD2_r01_T/T3-a-OOD2_r01_T.mp4) |
| `T3-a-OOD2_r02` | ❌ fail ⚠️ | 75 | 20.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-a-ood/T3-a-OOD2_r02_T/T3-a-OOD2_r02_T.mp4) |
| `T3-a-OOD3_r01` | ❌ fail | 145 | 38.4 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-a-ood/T3-a-OOD3_r01_F/T3-a-OOD3_r01_F.mp4) |
| `T3-a-OOD3_r02` | ❌ fail | 254 | 67.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-a-ood/T3-a-OOD3_r02_F/T3-a-OOD3_r02_F.mp4) |
| `T3-a-OOD4_r01` | ❌ fail | 605 | 161.3 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-a-ood/T3-a-OOD4_r01_F/T3-a-OOD4_r01_F.mp4) |
| `T3-a-OOD4_r02` | ❌ fail | 195 | 51.8 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-a-ood/T3-a-OOD4_r02_F/T3-a-OOD4_r02_F.mp4) |
| `T3-a-OOD5_r01` | ❌ fail | 299 | 79.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-a-ood/T3-a-OOD5_r01_F/T3-a-OOD5_r01_F.mp4) |
| `T3-a-OOD5_r02` | ✅ success | 87 | 22.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-a-ood/T3-a-OOD5_r02_T/T3-a-OOD5_r02_T.mp4) |
| `T3-a-OOD6_r01` | ✅ success | 83 | 21.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-a-ood/T3-a-OOD6_r01_T/T3-a-OOD6_r01_T.mp4) |
| `T3-a-OOD6_r02` | ✅ success | 72 | 18.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-a-ood/T3-a-OOD6_r02_T/T3-a-OOD6_r02_T.mp4) |
| `T3-a-OOD7_r01` | ✅ success | 169 | 44.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-a-ood/T3-a-OOD7_r01_T/T3-a-OOD7_r01_T.mp4) |
| `T3-a-OOD7_r02` | ❌ fail | 141 | 37.6 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-a-ood/T3-a-OOD7_r02_F/T3-a-OOD7_r02_F.mp4) |
| `T3-a-OOD8_r01` | ❌ fail | 217 | 57.6 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-a-ood/T3-a-OOD8_r01_F/T3-a-OOD8_r01_F.mp4) |
| `T3-a-OOD8_r02` | ✅ success | 117 | 31.2 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-a-ood/T3-a-OOD8_r02_T/T3-a-OOD8_r02_T.mp4) |
| `T3-a-OOD9_r01` | ❌ fail | 104 | 27.6 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-a-ood/T3-a-OOD9_r01_F/T3-a-OOD9_r01_F.mp4) |
| `T3-a-OOD9_r02` | ❌ fail | 91 | 24.3 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-a-ood/T3-a-OOD9_r02_F/T3-a-OOD9_r02_F.mp4) |
| `T3-a-OOD10_r01` | ✅ success | 95 | 25.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-a-ood/T3-a-OOD10_r01_T/T3-a-OOD10_r01_T.mp4) |
| `T3-a-OOD10_r02` | ❌ fail | 274 | 73.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-a-ood/T3-a-OOD10_r02_F/T3-a-OOD10_r02_F.mp4) |

</details>

<details>
<summary><b>T3-b</b> — <i>rotate the red block so that it is perpendicular to the blue block</i> — <b>11/20</b> (55%)</summary>

| rollout | verdict | steps | duration | video |
|---|---|---:|---:|---|
| `T3-b-OOD1_r01` | ❌ fail | 98 | 25.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-b-ood/T3-b-OOD1_r01_F/T3-b-OOD1_r01_F.mp4) |
| `T3-b-OOD1_r02` | ❌ fail | 186 | 49.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-b-ood/T3-b-OOD1_r02_F/T3-b-OOD1_r02_F.mp4) |
| `T3-b-OOD2_r01` | ✅ success | 65 | 17.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-b-ood/T3-b-OOD2_r01_T/T3-b-OOD2_r01_T.mp4) |
| `T3-b-OOD2_r02` | ✅ success | 71 | 18.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-b-ood/T3-b-OOD2_r02_T/T3-b-OOD2_r02_T.mp4) |
| `T3-b-OOD3_r01` | ✅ success | 558 | 148.6 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-b-ood/T3-b-OOD3_r01_T/T3-b-OOD3_r01_T.mp4) |
| `T3-b-OOD3_r02` | ✅ success | 121 | 32.3 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-b-ood/T3-b-OOD3_r02_T/T3-b-OOD3_r02_T.mp4) |
| `T3-b-OOD4_r01` | ✅ success | 61 | 16.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-b-ood/T3-b-OOD4_r01_T/T3-b-OOD4_r01_T.mp4) |
| `T3-b-OOD4_r02` | ✅ success | 74 | 19.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-b-ood/T3-b-OOD4_r02_T/T3-b-OOD4_r02_T.mp4) |
| `T3-b-OOD5_r01` | ❌ fail | 124 | 32.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-b-ood/T3-b-OOD5_r01_F/T3-b-OOD5_r01_F.mp4) |
| `T3-b-OOD5_r02` | ✅ success | 76 | 20.0 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-b-ood/T3-b-OOD5_r02_T/T3-b-OOD5_r02_T.mp4) |
| `T3-b-OOD6_r01` | ❌ fail | 484 | 129.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-b-ood/T3-b-OOD6_r01_F/T3-b-OOD6_r01_F.mp4) |
| `T3-b-OOD6_r02` | ✅ success | 61 | 16.2 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-b-ood/T3-b-OOD6_r02_T/T3-b-OOD6_r02_T.mp4) |
| `T3-b-OOD7_r01` | ✅ success | 112 | 29.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-b-ood/T3-b-OOD7_r01_T/T3-b-OOD7_r01_T.mp4) |
| `T3-b-OOD7_r02` | ❌ fail | 76 | 20.0 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-b-ood/T3-b-OOD7_r02_F/T3-b-OOD7_r02_F.mp4) |
| `T3-b-OOD8_r01` | ❌ fail ⚠️ | 111 | 29.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-b-ood/T3-b-OOD8_r01_T/T3-b-OOD8_r01_T.mp4) |
| `T3-b-OOD8_r02` | ❌ fail | 97 | 25.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-b-ood/T3-b-OOD8_r02_F/T3-b-OOD8_r02_F.mp4) |
| `T3-b-OOD9_r01` | ❌ fail | 233 | 62.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-b-ood/T3-b-OOD9_r01_F/T3-b-OOD9_r01_F.mp4) |
| `T3-b-OOD9_r02` | ✅ success | 362 | 96.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-b-ood/T3-b-OOD9_r02_T/T3-b-OOD9_r02_T.mp4) |
| `T3-b-OOD10_r01` | ✅ success | 102 | 27.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-b-ood/T3-b-OOD10_r01_T/T3-b-OOD10_r01_T.mp4) |
| `T3-b-OOD10_r02` | ❌ fail | 167 | 44.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T3-b-ood/T3-b-OOD10_r02_F/T3-b-OOD10_r02_F.mp4) |

</details>

<details>
<summary><b>T4-a</b> — <i>insert the orange block into the wooden shape sorter box</i> — <b>9/20</b> (45%)</summary>

| rollout | verdict | steps | duration | video |
|---|---|---:|---:|---|
| `T4-a-OOD1_r01` | ❌ fail | 147 | 38.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-a-ood/T4-a-OOD1_r01_F/T4-a-OOD1_r01_F.mp4) |
| `T4-a-OOD1_r02` | ❌ fail | 101 | 26.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-a-ood/T4-a-OOD1_r02_F/T4-a-OOD1_r02_F.mp4) |
| `T4-a-OOD2_r01` | ❌ fail | 285 | 76.0 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-a-ood/T4-a-OOD2_r01_F/T4-a-OOD2_r01_F.mp4) |
| `T4-a-OOD2_r02` | ✅ success | 173 | 46.0 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-a-ood/T4-a-OOD2_r02_T/T4-a-OOD2_r02_T.mp4) |
| `T4-a-OOD3_r01` | ✅ success | 98 | 26.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-a-ood/T4-a-OOD3_r01_T/T4-a-OOD3_r01_T.mp4) |
| `T4-a-OOD3_r02` | ✅ success | 157 | 41.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-a-ood/T4-a-OOD3_r02_T/T4-a-OOD3_r02_T.mp4) |
| `T4-a-OOD4_r01` | ✅ success | 98 | 25.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-a-ood/T4-a-OOD4_r01_T/T4-a-OOD4_r01_T.mp4) |
| `T4-a-OOD4_r02` | ❌ fail | 129 | 34.4 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-a-ood/T4-a-OOD4_r02_F/T4-a-OOD4_r02_F.mp4) |
| `T4-a-OOD5_r01` | ✅ success | 98 | 26.0 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-a-ood/T4-a-OOD5_r01_T/T4-a-OOD5_r01_T.mp4) |
| `T4-a-OOD5_r02` | ✅ success | 83 | 21.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-a-ood/T4-a-OOD5_r02_T/T4-a-OOD5_r02_T.mp4) |
| `T4-a-OOD6_r01` | ❌ fail | 306 | 81.6 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-a-ood/T4-a-OOD6_r01_F/T4-a-OOD6_r01_F.mp4) |
| `T4-a-OOD6_r02` | ❌ fail | 209 | 55.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-a-ood/T4-a-OOD6_r02_F/T4-a-OOD6_r02_F.mp4) |
| `T4-a-OOD7_r01` | ❌ fail | 104 | 27.8 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-a-ood/T4-a-OOD7_r01_F/T4-a-OOD7_r01_F.mp4) |
| `T4-a-OOD7_r02` | ✅ success | 128 | 34.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-a-ood/T4-a-OOD7_r02_T/T4-a-OOD7_r02_T.mp4) |
| `T4-a-OOD8_r01` | ❌ fail | 551 | 146.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-a-ood/T4-a-OOD8_r01_F/T4-a-OOD8_r01_F.mp4) |
| `T4-a-OOD8_r02` | ❌ fail | 253 | 67.3 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-a-ood/T4-a-OOD8_r02_F/T4-a-OOD8_r02_F.mp4) |
| `T4-a-OOD9_r01` | ✅ success | 77 | 20.3 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-a-ood/T4-a-OOD9_r01_T/T4-a-OOD9_r01_T.mp4) |
| `T4-a-OOD9_r02` | ✅ success | 85 | 22.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-a-ood/T4-a-OOD9_r02_T/T4-a-OOD9_r02_T.mp4) |
| `T4-a-OOD10_r01` | ❌ fail | 135 | 35.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-a-ood/T4-a-OOD10_r01_F/T4-a-OOD10_r01_F.mp4) |
| `T4-a-OOD10_r02` | ❌ fail | 230 | 61.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-a-ood/T4-a-OOD10_r02_F/T4-a-OOD10_r02_F.mp4) |

</details>

<details>
<summary><b>T4-b</b> — <i>insert the book into the black book stand</i> — <b>8/22</b> (36%)</summary>

| rollout | verdict | steps | duration | video |
|---|---|---:|---:|---|
| `T4-b-OOD1_r01` | ✅ success | 106 | 28.0 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-b-ood/T4-b-OOD1_r01_T/T4-b-OOD1_r01_T.mp4) |
| `T4-b-OOD1_r02` | ✅ success | 166 | 44.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-b-ood/T4-b-OOD1_r02_T/T4-b-OOD1_r02_T.mp4) |
| `T4-b-OOD2_r01` | ❌ fail | 102 | 27.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-b-ood/T4-b-OOD2_r01_F/T4-b-OOD2_r01_F.mp4) |
| `T4-b-OOD2_r02` | ✅ success | 96 | 25.4 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-b-ood/T4-b-OOD2_r02_T/T4-b-OOD2_r02_T.mp4) |
| `T4-b-OOD3_r01` | ❌ fail | 213 | 56.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-b-ood/T4-b-OOD3_r01_F/T4-b-OOD3_r01_F.mp4) |
| `T4-b-OOD3_r02` | ❌ fail | 286 | 76.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-b-ood/T4-b-OOD3_r02_F/T4-b-OOD3_r02_F.mp4) |
| `T4-b-OOD3_r03` | ❌ fail | 190 | 50.6 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-b-ood/T4-b-OOD3_r03_F/T4-b-OOD3_r03_F.mp4) |
| `T4-b-OOD3_r04` | ❌ fail | 335 | 89.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-b-ood/T4-b-OOD3_r04_F/T4-b-OOD3_r04_F.mp4) |
| `T4-b-OOD4_r01` | ✅ success | 110 | 29.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-b-ood/T4-b-OOD4_r01_T/T4-b-OOD4_r01_T.mp4) |
| `T4-b-OOD4_r02` | ✅ success | 97 | 25.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-b-ood/T4-b-OOD4_r02_T/T4-b-OOD4_r02_T.mp4) |
| `T4-b-OOD5_r01` | ❌ fail | 233 | 62.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-b-ood/T4-b-OOD5_r01_F/T4-b-OOD5_r01_F.mp4) |
| `T4-b-OOD5_r02` | ✅ success | 287 | 76.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-b-ood/T4-b-OOD5_r02_T/T4-b-OOD5_r02_T.mp4) |
| `T4-b-OOD6_r01` | ❌ fail | 400 | 107.0 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-b-ood/T4-b-OOD6_r01_F/T4-b-OOD6_r01_F.mp4) |
| `T4-b-OOD6_r02` | ❌ fail | 325 | 86.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-b-ood/T4-b-OOD6_r02_F/T4-b-OOD6_r02_F.mp4) |
| `T4-b-OOD7_r01` | ✅ success | 377 | 100.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-b-ood/T4-b-OOD7_r01_T/T4-b-OOD7_r01_T.mp4) |
| `T4-b-OOD7_r02` | ❌ fail | 617 | 165.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-b-ood/T4-b-OOD7_r02_F/T4-b-OOD7_r02_F.mp4) |
| `T4-b-OOD8_r01` | ❌ fail | 410 | 110.0 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-b-ood/T4-b-OOD8_r01_F/T4-b-OOD8_r01_F.mp4) |
| `T4-b-OOD8_r02` | ❌ fail | 168 | 44.8 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-b-ood/T4-b-OOD8_r02_F/T4-b-OOD8_r02_F.mp4) |
| `T4-b-OOD9_r01` | ❌ fail | 933 | 251.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-b-ood/T4-b-OOD9_r01_F/T4-b-OOD9_r01_F.mp4) |
| `T4-b-OOD9_r02` | ❌ fail | 329 | 88.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-b-ood/T4-b-OOD9_r02_F/T4-b-OOD9_r02_F.mp4) |
| `T4-b-OOD10_r01` | ✅ success | 367 | 98.3 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-b-ood/T4-b-OOD10_r01_T/T4-b-OOD10_r01_T.mp4) |
| `T4-b-OOD10_r02` | ❌ fail | 332 | 88.8 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T4-b-ood/T4-b-OOD10_r02_F/T4-b-OOD10_r02_F.mp4) |

</details>

<details>
<summary><b>T5-a</b> — <i>pull the smaller book out of the black book stand</i> — <b>15/21</b> (71%)</summary>

| rollout | verdict | steps | duration | video |
|---|---|---:|---:|---|
| `T5-a-OOD1_r01` | ❌ fail | 125 | 33.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-a-ood/T5-a-OOD1_r01_F/T5-a-OOD1_r01_F.mp4) |
| `T5-a-OOD1_r02` | ✅ success | 57 | 14.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-a-ood/T5-a-OOD1_r02_T/T5-a-OOD1_r02_T.mp4) |
| `T5-a-OOD1_r03` | ✅ success | 56 | 14.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-a-ood/T5-a-OOD1_r03_T/T5-a-OOD1_r03_T.mp4) |
| `T5-a-OOD2_r01` | ✅ success | 105 | 28.0 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-a-ood/T5-a-OOD2_r01_T/T5-a-OOD2_r01_T.mp4) |
| `T5-a-OOD2_r02` | ✅ success | 64 | 17.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-a-ood/T5-a-OOD2_r02_T/T5-a-OOD2_r02_T.mp4) |
| `T5-a-OOD3_r01` | ✅ success | 59 | 15.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-a-ood/T5-a-OOD3_r01_T/T5-a-OOD3_r01_T.mp4) |
| `T5-a-OOD3_r02` | ✅ success | 133 | 35.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-a-ood/T5-a-OOD3_r02_T/T5-a-OOD3_r02_T.mp4) |
| `T5-a-OOD4_r01` | ❌ fail | 103 | 27.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-a-ood/T5-a-OOD4_r01_F/T5-a-OOD4_r01_F.mp4) |
| `T5-a-OOD4_r02` | ❌ fail | 172 | 45.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-a-ood/T5-a-OOD4_r02_F/T5-a-OOD4_r02_F.mp4) |
| `T5-a-OOD5_r01` | ✅ success | 98 | 25.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-a-ood/T5-a-OOD5_r01_T/T5-a-OOD5_r01_T.mp4) |
| `T5-a-OOD5_r02` | ✅ success | 114 | 30.3 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-a-ood/T5-a-OOD5_r02_T/T5-a-OOD5_r02_T.mp4) |
| `T5-a-OOD6_r01` | ✅ success | 93 | 24.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-a-ood/T5-a-OOD6_r01_T/T5-a-OOD6_r01_T.mp4) |
| `T5-a-OOD6_r02` | ✅ success | 94 | 25.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-a-ood/T5-a-OOD6_r02_T/T5-a-OOD6_r02_T.mp4) |
| `T5-a-OOD7_r01` | ✅ success | 86 | 22.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-a-ood/T5-a-OOD7_r01_T/T5-a-OOD7_r01_T.mp4) |
| `T5-a-OOD7_r02` | ❌ fail | 221 | 59.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-a-ood/T5-a-OOD7_r02_F/T5-a-OOD7_r02_F.mp4) |
| `T5-a-OOD8_r01` | ✅ success | 68 | 17.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-a-ood/T5-a-OOD8_r01_T/T5-a-OOD8_r01_T.mp4) |
| `T5-a-OOD8_r02` | ✅ success | 69 | 18.3 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-a-ood/T5-a-OOD8_r02_T/T5-a-OOD8_r02_T.mp4) |
| `T5-a-OOD9_r01` | ✅ success | 85 | 22.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-a-ood/T5-a-OOD9_r01_T/T5-a-OOD9_r01_T.mp4) |
| `T5-a-OOD9_r02` | ✅ success | 63 | 16.8 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-a-ood/T5-a-OOD9_r02_T/T5-a-OOD9_r02_T.mp4) |
| `T5-a-OOD10_r01` | ❌ fail | 284 | 75.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-a-ood/T5-a-OOD10_r01_F/T5-a-OOD10_r01_F.mp4) |
| `T5-a-OOD10_r02` | ❌ fail | 258 | 68.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-a-ood/T5-a-OOD10_r02_F/T5-a-OOD10_r02_F.mp4) |

</details>

<details>
<summary><b>T5-b</b> — <i>pull the small block out from under the large block</i> — <b>7/20</b> (35%)</summary>

| rollout | verdict | steps | duration | video |
|---|---|---:|---:|---|
| `T5-b-OOD1_r01` | ❌ fail | 1067 | 286.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-b-ood/T5-b-OOD1_r01_F/T5-b-OOD1_r01_F.mp4) |
| `T5-b-OOD1_r02` | ✅ success | 363 | 96.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-b-ood/T5-b-OOD1_r02_T/T5-b-OOD1_r02_T.mp4) |
| `T5-b-OOD2_r01` | ✅ success | 1074 | 289.0 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-b-ood/T5-b-OOD2_r01_T/T5-b-OOD2_r01_T.mp4) |
| `T5-b-OOD2_r02` | ✅ success | 565 | 152.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-b-ood/T5-b-OOD2_r02_T/T5-b-OOD2_r02_T.mp4) |
| `T5-b-OOD3_r01` | ❌ fail ⏱ | 1200 | 323.3 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-b-ood/T5-b-OOD3_r01_F/T5-b-OOD3_r01_F.mp4) |
| `T5-b-OOD3_r02` | ✅ success | 833 | 225.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-b-ood/T5-b-OOD3_r02_T/T5-b-OOD3_r02_T.mp4) |
| `T5-b-OOD4_r01` | ❌ fail ⏱ ⚠️ | 1200 | 324.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-b-ood/T5-b-OOD4_r01/T5-b-OOD4_r01.mp4) |
| `T5-b-OOD4_r02` | ❌ fail | 1114 | 302.0 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-b-ood/T5-b-OOD4_r02_F/T5-b-OOD4_r02_F.mp4) |
| `T5-b-OOD5_r01` | ❌ fail | 1005 | 273.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-b-ood/T5-b-OOD5_r01_F/T5-b-OOD5_r01_F.mp4) |
| `T5-b-OOD5_r02` | ✅ success | 760 | 206.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-b-ood/T5-b-OOD5_r02_T/T5-b-OOD5_r02_T.mp4) |
| `T5-b-OOD6_r01` | ✅ success | 345 | 93.2 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-b-ood/T5-b-OOD6_r01_T/T5-b-OOD6_r01_T.mp4) |
| `T5-b-OOD6_r02` | ❌ fail ⏱ | 1200 | 326.2 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-b-ood/T5-b-OOD6_r02_F/T5-b-OOD6_r02_F.mp4) |
| `T5-b-OOD7_r01` | ❌ fail | 1198 | 325.4 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-b-ood/T5-b-OOD7_r01_F/T5-b-OOD7_r01_F.mp4) |
| `T5-b-OOD7_r02` | ❌ fail | 982 | 268.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-b-ood/T5-b-OOD7_r02_F/T5-b-OOD7_r02_F.mp4) |
| `T5-b-OOD8_r01` | ❌ fail | 996 | 271.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-b-ood/T5-b-OOD8_r01_F/T5-b-OOD8_r01_F.mp4) |
| `T5-b-OOD8_r02` | ❌ fail | 1128 | 309.3 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-b-ood/T5-b-OOD8_r02_F/T5-b-OOD8_r02_F.mp4) |
| `T5-b-OOD9_r01` | ✅ success | 476 | 129.8 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-b-ood/T5-b-OOD9_r01_T/T5-b-OOD9_r01_T.mp4) |
| `T5-b-OOD9_r02` | ❌ fail ⏱ | 1200 | 329.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-b-ood/T5-b-OOD9_r02_F/T5-b-OOD9_r02_F.mp4) |
| `T5-b-OOD10_r01` | ❌ fail | 1182 | 324.6 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-b-ood/T5-b-OOD10_r01_F/T5-b-OOD10_r01_F.mp4) |
| `T5-b-OOD10_r02` | ❌ fail | 929 | 253.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k/blob/main/web/T5-b-ood/T5-b-OOD10_r02_F/T5-b-OOD10_r02_F.mp4) |

</details>
<!-- /gen:index -->
