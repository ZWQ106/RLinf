---
license: cc-by-nc-4.0
pretty_name: FR3 OOD rollouts — cotrain-pbc-v2-8000
task_categories:
  - robotics
tags:
  - robotics
  - franka-fr3
  - manipulation
  - pi05
  - openpi
  - out-of-distribution
  - real-robot
size_categories:
  - n<1K
configs:
  - config_name: default
    drop_labels: true
    data_files:
      - split: train
        path:
          - metadata.jsonl
          - web/**/*.mp4
---

# cotrain-pbc-v2-8000 — OOD rollouts (T1-a … T4-b)

OOD (out-of-distribution layout) rollouts of the pbc co-trained π0.5 checkpoint on the
real Franka FR3 bench, recorded through the eval portal (`tasl/dashboards/openpi.py`).

- **ckpt:** `cotrain_pbc_v2/8000 +rtc` — served as `pi05_cotrain_franka_serve`
  (action_horizon 15), co-trained on the center-cropped 250ep-pbc + filtered pbc-v2 sets,
  step 8000, served **with RTC** (real-time chunking, arXiv:2506.07339).
- **Coverage:** <!-- gen:summary -->**205 rollouts / 4.4 GB** over **10 tasks** (T1-a … T5-b), ~2 rollouts per OOD layout, recorded 2026-08-30 … 2026-08-31. **87/201 succeeded (43.3%)**; 4 marked `unsure` and excluded from the rates.<!-- /gen:summary -->
- **Files:** `<task>-ood/<layout>_rNN_<T|F|Q>/<same stem>.{mp4,traj.jsonl,frames.json,json}`
  (one folder per rollout), mirroring `saved_demo/<task>-ood/cotrain-pbc-v2-8000/` on disk
  minus the checkpoint level.
  - `.mp4` — recorded rollout video
  - `.frames.json` — per-frame wall-clock timestamps
  - `.traj.jsonl` — per-control-step record: joint state `q`, gripper, RTC scheduler counters,
    inference latency, and the predicted action chunk
  - `.json` — episode sidecar (task, prompt, layout, ckpt, timings, `steps`, `mark`)
- **Verdict** comes from `mark` in the sidecar json, **not** the filename suffix.
  `success` / `fail` / `unsure` map to the `_T` / `_F` / `_Q` stems. `unsure` rollouts are
  excluded from every success-rate denominator — filter on `not unsure`, never on `not success`.
- `steps` = policy control steps at 15 Hz, capped at **1200** by the eval runner. Rollouts at
  the cap are timeouts (`timeout: true`) and are marked ⏱ in the index below; their step and
  time figures are censored.

## Browsing the rollouts

### Space

**https://huggingface.co/spaces/axisrobotics/fr3-rollout-browser** — pick this dataset from the
checkpoint selector, then filter by task / layout / verdict / timeout, search prompts, and watch
each rollout beside its control trajectory (7 joint positions, gripper, inference latency, RTC
ticks). There is also a per-task success-rate summary and a 6-up grid.

An index of all 165 rollouts with direct video links is at the bottom of this card.

### Two encodings — use `web/` in a browser

<!-- gen:encodings -->
The recorder wrote **MPEG-4 Part 2** (`mp4v`, Simple Profile) with the `moov` atom
*after* `mdat`. No browser decodes `mp4v`, and the trailing `moov` blocks progressive
playback, so the original files will not play in the Hub preview, the dataset viewer,
or any `<video>` element — they need VLC/ffmpeg.

`web/` mirrors the tree with **H.264 High / yuv420p, `+faststart`, 2 s keyframes** at
the same native resolution and frame rate. Same frames, plays everywhere, ~46% of the size.

| | codec | size | plays in a browser |
|---|---|---|---|
| `<task>-ood/…/<stem>.mp4` | `mp4v` (MPEG-4 Part 2) | 4.4 GB | ❌ |
| `web/<task>-ood/…/<stem>.mp4` | `avc1` (H.264 High) | 2.0 GB | ✅ |
<!-- /gen:encodings -->

`metadata.jsonl` points `file_name` at the `web/` copy, so the dataset viewer, `load_dataset`
and the Space all get the playable one; `source_video` keeps the path to the original.

> **Note:** the Hub dataset viewer does not run on *private* datasets unless the **owning
> organization** is on a Team/Enterprise plan. While this repo is private under `TASL-FR3`, use
> the Space, the index below, or `load_dataset` locally. The `configs:` block above is already
> in place, so the viewer switches on by itself if the repo goes public or the org moves to Team.

### Locally

```python
from datasets import load_dataset  # needs: pip install "datasets" torchcodec

ds = load_dataset("TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000", split="train")
ds = ds.filter(lambda r: r["task"] == "T4-a" and not r["success"] and not r["unsure"])
ds[0]["video"], ds[0]["prompt"], ds[0]["steps"], ds[0]["mark"]
```

### `metadata.jsonl` columns

Beyond the raw sidecar fields:

| column | meaning |
|---|---|
| `file_name` | relative path to the **`web/`** mp4 — links the row to its video for the viewer |
| `source_video` | relative path to the original `mp4v` recording |
| `success` | `mark == "success"` |
| `unsure` | `mark == "unsure"` — **excluded from every success-rate denominator**, so filter on `not unsure` rather than treating `not success` as failure |
| `timeout` | `steps >= 1200` (the eval-runner cap): the rollout was cut off, not finished |
| `suffix_stale` | filename `_T`/`_F`/`_Q` suffix disagrees with `mark` — always trust `mark` |
| `task_group`, `variant`, `ood_index`, `rollout` | parsed out of the stem for grouping/filtering |
| `image_mode` | `crop` where the sidecar recorded it, else empty |
| `traj_file`, `frames_file`, `sidecar_file` | relative paths to the other three per-rollout files |

## Stats

`cotrain-pbc-v2-8000_ood_stats.xlsx` has two sheets: `per-task` (aggregates + an `ALL` row)
and `rollouts` (one row per rollout). `metadata.jsonl` carries the same sidecars, one JSON
object per line, so the dataset can be indexed without opening every small file.

`unsure` rollouts are excluded from `n`, `SR %` and the step/time statistics, and counted in
the trailing `unsure` column.

<!-- gen:stats -->
| task | prompt | n | SR % | steps mean | time mean s |
|------|--------|---|------|-----------|-------------|
| T1-a | pick up the blue cup and place it into the red cup | 20 | 65.0 | 324.4 | 86.3 |
| T1-b | stack the red block on top of the blue block | 20 | 20.0 | 529.2 | 141.0 |
| T2-a | press the blue button | 20 | 90.0 | 137.1 | 36.4 |
| T2-b | close the lid of the wooden shape sorter box | 20 | 15.0 | 747.1 | 199.2 |
| T3-a | align the three colored blocks to the same orientation | 20 | 55.0 | 150.1 | 39.8 |
| T3-b | rotate the red block so that it is perpendicular to the blue block | 21 | 33.3 | 196.2 | 52.1 |
| T4-a | insert the orange block into the wooden shape sorter box | 20 | 10.0 | 182.8 | 48.6 |
| T4-b | insert the book into the black book stand | 20 | 30.0 | 294.1 | 78.3 |
| T5-a | pull the smaller book out of the black book stand | 20 | 75.0 | 178.9 | 47.6 |
| T5-b | pull the small block out from under the large block | 20 | 40.0 | 772.4 | 206.4 |
| **ALL** | | **201** | **43.3** | **350.5** | **93.4** |
<!-- /gen:stats -->

### vs `pi05-droid-ft-15k` on the shared tasks

Companion dataset: `TASL-FR3/fr3-ood-rollouts-pi05-droid-ft-15k`
(`pi05_droid_franka_lora_10task_v2/15000 +rtc`, 206 rollouts over 10 tasks).

<!-- gen:compare -->
| task | pi05-droid-ft-15k | cotrain-pbc-v2-8000 | Δ |
|------|-----------------:|-------------------:|---:|
| T1-a | 75.0 % (n=20) | 65.0 % (n=20) | −10.0 |
| T1-b | 29.2 % (n=24) | 20.0 % (n=20) | −9.2 |
| T2-a | 90.0 % (n=20) | 90.0 % (n=20) | ±0.0 |
| T2-b | 100.0 % (n=19) | 15.0 % (n=20) | −85.0 |
| T3-a | 40.0 % (n=20) | 55.0 % (n=20) | +15.0 |
| T3-b | 55.0 % (n=20) | 33.3 % (n=21) | −21.7 |
| T4-a | 45.0 % (n=20) | 10.0 % (n=20) | −35.0 |
| T4-b | 36.4 % (n=22) | 30.0 % (n=20) | −6.4 |
| T5-a | 71.4 % (n=21) | 75.0 % (n=20) | +3.6 |
| T5-b | 35.0 % (n=20) | 40.0 % (n=20) | +5.0 |
| **overlap** | **56.8 % (n=206)** | **43.3 % (n=201)** | **−13.5** |
<!-- /gen:compare -->

Caveat: the two evals were recorded on different days (08-27/28 vs 08-30/31) with the bench
re-set between them, so per-layout scene state is not identical — treat the Δ as a
task-level trend, not a paired comparison.

## Reproducing / republishing

```bash
python tasl/tools/publish_ood_rollouts.py cotrain-pbc-v2-8000 --compare pi05-droid-ft-15k
```

Re-encodes any new rollout into `web/`, rebuilds `metadata.jsonl` and the stats workbook,
regenerates the `<!-- gen:* -->` blocks in this card (coverage, encodings, stats, comparison,
index), and uploads the tree. It is idempotent — already-encoded rollouts are skipped and the
hand-written prose around the generated blocks is left alone.

## Rollout index

<!-- gen:index -->
All 205 rollouts, grouped by task. Each link opens the browser-playable copy under `web/` on the Hub (you must be signed in — the repo is private).

Verdicts come from `mark`; ❓ unsure rollouts are excluded from the success rates above, ⏱ marks a rollout that hit the step cap, ⚠️ one whose filename suffix is stale.

<details>
<summary><b>T1-a</b> — <i>pick up the blue cup and place it into the red cup</i> — <b>13/20</b> (65%)</summary>

| rollout | verdict | steps | duration | video |
|---|---|---:|---:|---|
| `T1-a-OOD1_r01` | ✅ success | 77 | 20.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-a-ood/T1-a-OOD1_r01_T/T1-a-OOD1_r01_T.mp4) |
| `T1-a-OOD1_r02` | ✅ success | 90 | 23.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-a-ood/T1-a-OOD1_r02_T/T1-a-OOD1_r02_T.mp4) |
| `T1-a-OOD2_r01` | ✅ success | 77 | 20.3 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-a-ood/T1-a-OOD2_r01_T/T1-a-OOD2_r01_T.mp4) |
| `T1-a-OOD2_r02` | ✅ success | 74 | 19.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-a-ood/T1-a-OOD2_r02_T/T1-a-OOD2_r02_T.mp4) |
| `T1-a-OOD3_r01` | ✅ success | 136 | 36.0 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-a-ood/T1-a-OOD3_r01_T/T1-a-OOD3_r01_T.mp4) |
| `T1-a-OOD3_r02` | ✅ success | 158 | 41.8 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-a-ood/T1-a-OOD3_r02_T/T1-a-OOD3_r02_T.mp4) |
| `T1-a-OOD6_r01` | ❌ fail | 328 | 87.3 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-a-ood/T1-a-OOD6_r01_F/T1-a-OOD6_r01_F.mp4) |
| `T1-a-OOD6_r02` | ✅ success | 81 | 21.3 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-a-ood/T1-a-OOD6_r02_T/T1-a-OOD6_r02_T.mp4) |
| `T1-a-OOD7_r01` | ❌ fail | 479 | 127.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-a-ood/T1-a-OOD7_r01_F/T1-a-OOD7_r01_F.mp4) |
| `T1-a-OOD7_r02` | ❌ fail | 931 | 248.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-a-ood/T1-a-OOD7_r02_F/T1-a-OOD7_r02_F.mp4) |
| `T1-a-OOD8_r01` | ✅ success | 78 | 20.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-a-ood/T1-a-OOD8_r01_T/T1-a-OOD8_r01_T.mp4) |
| `T1-a-OOD8_r02` | ✅ success | 79 | 21.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-a-ood/T1-a-OOD8_r02_T/T1-a-OOD8_r02_T.mp4) |
| `T1-a-OOD9_r01` | ✅ success | 95 | 25.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-a-ood/T1-a-OOD9_r01_T/T1-a-OOD9_r01_T.mp4) |
| `T1-a-OOD9_r02` | ❌ fail | 1043 | 277.8 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-a-ood/T1-a-OOD9_r02_F/T1-a-OOD9_r02_F.mp4) |
| `T1-a-OOD10_r01` | ✅ success | 436 | 116.0 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-a-ood/T1-a-OOD10_r01_T/T1-a-OOD10_r01_T.mp4) |
| `T1-a-OOD10_r02` | ❌ fail | 1033 | 275.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-a-ood/T1-a-OOD10_r02_F/T1-a-OOD10_r02_F.mp4) |
| `T1-a-OOD11_r01` | ✅ success | 105 | 27.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-a-ood/T1-a-OOD11_r01_T/T1-a-OOD11_r01_T.mp4) |
| `T1-a-OOD11_r02` | ✅ success | 89 | 23.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-a-ood/T1-a-OOD11_r02_T/T1-a-OOD11_r02_T.mp4) |
| `T1-a-OOD12_r01` | ❌ fail | 381 | 101.6 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-a-ood/T1-a-OOD12_r01_F/T1-a-OOD12_r01_F.mp4) |
| `T1-a-OOD12_r02` | ❌ fail | 717 | 191.2 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-a-ood/T1-a-OOD12_r02_F/T1-a-OOD12_r02_F.mp4) |

</details>

<details>
<summary><b>T1-b</b> — <i>stack the red block on top of the blue block</i> — <b>4/20</b> (20%)</summary>

| rollout | verdict | steps | duration | video |
|---|---|---:|---:|---|
| `T1-b-OOD1_r01` | ❌ fail | 868 | 231.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-b-ood/T1-b-OOD1_r01_F/T1-b-OOD1_r01_F.mp4) |
| `T1-b-OOD1_r02` | ❌ fail | 871 | 231.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-b-ood/T1-b-OOD1_r02_F/T1-b-OOD1_r02_F.mp4) |
| `T1-b-OOD2_r01` | ❌ fail | 1117 | 297.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-b-ood/T1-b-OOD2_r01_F/T1-b-OOD2_r01_F.mp4) |
| `T1-b-OOD2_r02` | ❌ fail | 768 | 204.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-b-ood/T1-b-OOD2_r02_F/T1-b-OOD2_r02_F.mp4) |
| `T1-b-OOD3_r01` | ❌ fail | 45 | 12.0 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-b-ood/T1-b-OOD3_r01_F/T1-b-OOD3_r01_F.mp4) |
| `T1-b-OOD3_r02` | ❌ fail | 111 | 29.3 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-b-ood/T1-b-OOD3_r02_F/T1-b-OOD3_r02_F.mp4) |
| `T1-b-OOD4_r01` | ✅ success | 202 | 53.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-b-ood/T1-b-OOD4_r01_T/T1-b-OOD4_r01_T.mp4) |
| `T1-b-OOD4_r02` | ❌ fail | 1131 | 301.4 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-b-ood/T1-b-OOD4_r02_F/T1-b-OOD4_r02_F.mp4) |
| `T1-b-OOD5_r01` | ❌ fail | 1112 | 296.6 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-b-ood/T1-b-OOD5_r01_F/T1-b-OOD5_r01_F.mp4) |
| `T1-b-OOD5_r02` | ❌ fail | 975 | 259.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-b-ood/T1-b-OOD5_r02_F/T1-b-OOD5_r02_F.mp4) |
| `T1-b-OOD6_r01` | ❌ fail | 83 | 22.0 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-b-ood/T1-b-OOD6_r01_F/T1-b-OOD6_r01_F.mp4) |
| `T1-b-OOD6_r02` | ❌ fail | 40 | 10.6 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-b-ood/T1-b-OOD6_r02_F/T1-b-OOD6_r02_F.mp4) |
| `T1-b-OOD7_r01` | ❌ fail | 998 | 266.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-b-ood/T1-b-OOD7_r01_F/T1-b-OOD7_r01_F.mp4) |
| `T1-b-OOD7_r02` | ❌ fail | 366 | 97.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-b-ood/T1-b-OOD7_r02_F/T1-b-OOD7_r02_F.mp4) |
| `T1-b-OOD11_r01` | ✅ success | 79 | 21.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-b-ood/T1-b-OOD11_r01_T/T1-b-OOD11_r01_T.mp4) |
| `T1-b-OOD11_r02` | ✅ success | 116 | 30.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-b-ood/T1-b-OOD11_r02_T/T1-b-OOD11_r02_T.mp4) |
| `T1-b-OOD14_r01` | ✅ success | 409 | 108.8 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-b-ood/T1-b-OOD14_r01_T/T1-b-OOD14_r01_T.mp4) |
| `T1-b-OOD14_r02` | ❌ fail | 475 | 126.4 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-b-ood/T1-b-OOD14_r02_F/T1-b-OOD14_r02_F.mp4) |
| `T1-b-OOD15_r01` | ❌ fail | 281 | 74.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-b-ood/T1-b-OOD15_r01_F/T1-b-OOD15_r01_F.mp4) |
| `T1-b-OOD15_r02` | ❌ fail | 538 | 143.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T1-b-ood/T1-b-OOD15_r02_F/T1-b-OOD15_r02_F.mp4) |

</details>

<details>
<summary><b>T2-a</b> — <i>press the blue button</i> — <b>18/20</b> (90%)</summary>

| rollout | verdict | steps | duration | video |
|---|---|---:|---:|---|
| `T2-a-OOD1_r01` | ✅ success | 79 | 20.8 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-a-ood/T2-a-OOD1_r01_T/T2-a-OOD1_r01_T.mp4) |
| `T2-a-OOD1_r02` | ✅ success | 91 | 24.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-a-ood/T2-a-OOD1_r02_T/T2-a-OOD1_r02_T.mp4) |
| `T2-a-OOD2_r01` | ✅ success | 118 | 31.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-a-ood/T2-a-OOD2_r01_T/T2-a-OOD2_r01_T.mp4) |
| `T2-a-OOD2_r02` | ✅ success | 115 | 30.4 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-a-ood/T2-a-OOD2_r02_T/T2-a-OOD2_r02_T.mp4) |
| `T2-a-OOD3_r01` | ✅ success | 43 | 11.2 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-a-ood/T2-a-OOD3_r01_T/T2-a-OOD3_r01_T.mp4) |
| `T2-a-OOD3_r02` | ✅ success | 54 | 14.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-a-ood/T2-a-OOD3_r02_T/T2-a-OOD3_r02_T.mp4) |
| `T2-a-OOD4_r01` | ✅ success | 54 | 14.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-a-ood/T2-a-OOD4_r01_T/T2-a-OOD4_r01_T.mp4) |
| `T2-a-OOD4_r02` | ✅ success | 76 | 20.3 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-a-ood/T2-a-OOD4_r02_T/T2-a-OOD4_r02_T.mp4) |
| `T2-a-OOD5_r01` | ✅ success | 67 | 17.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-a-ood/T2-a-OOD5_r01_T/T2-a-OOD5_r01_T.mp4) |
| `T2-a-OOD5_r02` | ✅ success | 137 | 36.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-a-ood/T2-a-OOD5_r02_T/T2-a-OOD5_r02_T.mp4) |
| `T2-a-OOD6_r01` | ✅ success | 44 | 11.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-a-ood/T2-a-OOD6_r01_T/T2-a-OOD6_r01_T.mp4) |
| `T2-a-OOD6_r02` | ✅ success | 36 | 9.4 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-a-ood/T2-a-OOD6_r02_T/T2-a-OOD6_r02_T.mp4) |
| `T2-a-OOD7_r01` | ✅ success | 100 | 26.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-a-ood/T2-a-OOD7_r01_T/T2-a-OOD7_r01_T.mp4) |
| `T2-a-OOD7_r02` | ✅ success | 71 | 18.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-a-ood/T2-a-OOD7_r02_T/T2-a-OOD7_r02_T.mp4) |
| `T2-a-OOD8_r01` | ✅ success | 42 | 10.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-a-ood/T2-a-OOD8_r01_T/T2-a-OOD8_r01_T.mp4) |
| `T2-a-OOD8_r02` | ✅ success | 45 | 12.0 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-a-ood/T2-a-OOD8_r02_T/T2-a-OOD8_r02_T.mp4) |
| `T2-a-OOD9_r01` | ❌ fail | 122 | 32.2 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-a-ood/T2-a-OOD9_r01_F/T2-a-OOD9_r01_F.mp4) |
| `T2-a-OOD9_r02` | ❌ fail | 1153 | 307.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-a-ood/T2-a-OOD9_r02_F/T2-a-OOD9_r02_F.mp4) |
| `T2-a-OOD10_r01` | ✅ success | 202 | 53.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-a-ood/T2-a-OOD10_r01_T/T2-a-OOD10_r01_T.mp4) |
| `T2-a-OOD10_r02` | ✅ success | 92 | 24.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-a-ood/T2-a-OOD10_r02_T/T2-a-OOD10_r02_T.mp4) |

</details>

<details>
<summary><b>T2-b</b> — <i>close the lid of the wooden shape sorter box</i> — <b>3/20</b> (15%)</summary>

| rollout | verdict | steps | duration | video |
|---|---|---:|---:|---|
| `T2-b-OOD1_r01` | ❌ fail | 1117 | 297.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-b-ood/T2-b-OOD1_r01_F/T2-b-OOD1_r01_F.mp4) |
| `T2-b-OOD1_r02` | ❌ fail | 1024 | 272.8 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-b-ood/T2-b-OOD1_r02_F/T2-b-OOD1_r02_F.mp4) |
| `T2-b-OOD2_r01` | ❌ fail | 611 | 162.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-b-ood/T2-b-OOD2_r01_F/T2-b-OOD2_r01_F.mp4) |
| `T2-b-OOD2_r02` | ❌ fail | 708 | 188.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-b-ood/T2-b-OOD2_r02_F/T2-b-OOD2_r02_F.mp4) |
| `T2-b-OOD3_r01` | ✅ success | 172 | 45.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-b-ood/T2-b-OOD3_r01_T/T2-b-OOD3_r01_T.mp4) |
| `T2-b-OOD3_r02` | ❌ fail | 750 | 200.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-b-ood/T2-b-OOD3_r02_F/T2-b-OOD3_r02_F.mp4) |
| `T2-b-OOD4_r01` | ❌ fail | 1170 | 312.0 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-b-ood/T2-b-OOD4_r01_F/T2-b-OOD4_r01_F.mp4) |
| `T2-b-OOD4_r02` | ❌ fail | 553 | 147.6 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-b-ood/T2-b-OOD4_r02_F/T2-b-OOD4_r02_F.mp4) |
| `T2-b-OOD5_r01` | ❌ fail | 919 | 244.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-b-ood/T2-b-OOD5_r01_F/T2-b-OOD5_r01_F.mp4) |
| `T2-b-OOD5_r02` | ❌ fail | 503 | 134.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-b-ood/T2-b-OOD5_r02_F/T2-b-OOD5_r02_F.mp4) |
| `T2-b-OOD6_r01` | ❌ fail | 1012 | 271.0 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-b-ood/T2-b-OOD6_r01_F/T2-b-OOD6_r01_F.mp4) |
| `T2-b-OOD6_r02` | ✅ success | 107 | 28.2 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-b-ood/T2-b-OOD6_r02_T/T2-b-OOD6_r02_T.mp4) |
| `T2-b-OOD7_r01` | ❌ fail | 392 | 104.3 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-b-ood/T2-b-OOD7_r01_F/T2-b-OOD7_r01_F.mp4) |
| `T2-b-OOD7_r02` | ❌ fail | 1106 | 294.8 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-b-ood/T2-b-OOD7_r02_F/T2-b-OOD7_r02_F.mp4) |
| `T2-b-OOD8_r01` | ❌ fail | 469 | 124.8 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-b-ood/T2-b-OOD8_r01_F/T2-b-OOD8_r01_F.mp4) |
| `T2-b-OOD8_r02` | ❌ fail | 509 | 135.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-b-ood/T2-b-OOD8_r02_F/T2-b-OOD8_r02_F.mp4) |
| `T2-b-OOD9_r01` | ❌ fail | 913 | 243.2 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-b-ood/T2-b-OOD9_r01_F/T2-b-OOD9_r01_F.mp4) |
| `T2-b-OOD9_r02` | ✅ success | 838 | 223.2 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-b-ood/T2-b-OOD9_r02_T/T2-b-OOD9_r02_T.mp4) |
| `T2-b-OOD10_r01` | ❌ fail | 1001 | 266.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-b-ood/T2-b-OOD10_r01_F/T2-b-OOD10_r01_F.mp4) |
| `T2-b-OOD10_r02` | ❌ fail | 1068 | 284.8 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T2-b-ood/T2-b-OOD10_r02_F/T2-b-OOD10_r02_F.mp4) |

</details>

<details>
<summary><b>T3-a</b> — <i>align the three colored blocks to the same orientation</i> — <b>11/20</b> (55%, 3 unsure)</summary>

| rollout | verdict | steps | duration | video |
|---|---|---:|---:|---|
| `T3-a-OOD1_r01` | ❓ unsure | 81 | 21.6 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-a-ood/T3-a-OOD1_r01_Q/T3-a-OOD1_r01_Q.mp4) |
| `T3-a-OOD1_r02` | ✅ success | 65 | 17.3 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-a-ood/T3-a-OOD1_r02_T/T3-a-OOD1_r02_T.mp4) |
| `T3-a-OOD1_r03` | ✅ success | 73 | 19.2 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-a-ood/T3-a-OOD1_r03_T/T3-a-OOD1_r03_T.mp4) |
| `T3-a-OOD2_r01` | ❓ unsure | 36 | 9.3 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-a-ood/T3-a-OOD2_r01_Q/T3-a-OOD2_r01_Q.mp4) |
| `T3-a-OOD2_r02` | ✅ success | 104 | 27.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-a-ood/T3-a-OOD2_r02_T/T3-a-OOD2_r02_T.mp4) |
| `T3-a-OOD2_r03` | ❌ fail | 142 | 37.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-a-ood/T3-a-OOD2_r03_F/T3-a-OOD2_r03_F.mp4) |
| `T3-a-OOD3_r01` | ❌ fail | 183 | 48.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-a-ood/T3-a-OOD3_r01_F/T3-a-OOD3_r01_F.mp4) |
| `T3-a-OOD3_r02` | ❌ fail | 97 | 25.6 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-a-ood/T3-a-OOD3_r02_F/T3-a-OOD3_r02_F.mp4) |
| `T3-a-OOD4_r01` | ❌ fail | 133 | 35.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-a-ood/T3-a-OOD4_r01_F/T3-a-OOD4_r01_F.mp4) |
| `T3-a-OOD4_r02` | ❌ fail | 173 | 45.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-a-ood/T3-a-OOD4_r02_F/T3-a-OOD4_r02_F.mp4) |
| `T3-a-OOD5_r01` | ✅ success | 125 | 33.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-a-ood/T3-a-OOD5_r01_T/T3-a-OOD5_r01_T.mp4) |
| `T3-a-OOD5_r02` | ✅ success | 97 | 25.8 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-a-ood/T3-a-OOD5_r02_T/T3-a-OOD5_r02_T.mp4) |
| `T3-a-OOD6_r01` | ✅ success | 101 | 26.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-a-ood/T3-a-OOD6_r01_T/T3-a-OOD6_r01_T.mp4) |
| `T3-a-OOD6_r02` | ✅ success | 54 | 14.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-a-ood/T3-a-OOD6_r02_T/T3-a-OOD6_r02_T.mp4) |
| `T3-a-OOD7_r01` | ❓ unsure | 180 | 47.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-a-ood/T3-a-OOD7_r01_Q/T3-a-OOD7_r01_Q.mp4) |
| `T3-a-OOD7_r02` | ❌ fail | 222 | 58.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-a-ood/T3-a-OOD7_r02_F/T3-a-OOD7_r02_F.mp4) |
| `T3-a-OOD7_r03` | ❌ fail | 174 | 46.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-a-ood/T3-a-OOD7_r03_F/T3-a-OOD7_r03_F.mp4) |
| `T3-a-OOD8_r01` | ✅ success | 211 | 56.0 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-a-ood/T3-a-OOD8_r01_T/T3-a-OOD8_r01_T.mp4) |
| `T3-a-OOD8_r02` | ✅ success | 295 | 78.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-a-ood/T3-a-OOD8_r02_T/T3-a-OOD8_r02_T.mp4) |
| `T3-a-OOD9_r01` | ✅ success | 84 | 22.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-a-ood/T3-a-OOD9_r01_T/T3-a-OOD9_r01_T.mp4) |
| `T3-a-OOD9_r02` | ✅ success | 84 | 22.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-a-ood/T3-a-OOD9_r02_T/T3-a-OOD9_r02_T.mp4) |
| `T3-a-OOD10_r01` | ❌ fail | 366 | 97.6 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-a-ood/T3-a-OOD10_r01_F/T3-a-OOD10_r01_F.mp4) |
| `T3-a-OOD10_r02` | ❌ fail | 219 | 58.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-a-ood/T3-a-OOD10_r02_F/T3-a-OOD10_r02_F.mp4) |

</details>

<details>
<summary><b>T3-b</b> — <i>rotate the red block so that it is perpendicular to the blue block</i> — <b>7/21</b> (33%, 1 unsure)</summary>

| rollout | verdict | steps | duration | video |
|---|---|---:|---:|---|
| `T3-b-OOD1_r01` | ❌ fail | 161 | 42.6 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-b-ood/T3-b-OOD1_r01_F/T3-b-OOD1_r01_F.mp4) |
| `T3-b-OOD1_r02` | ❌ fail | 186 | 49.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-b-ood/T3-b-OOD1_r02_F/T3-b-OOD1_r02_F.mp4) |
| `T3-b-OOD1_r03` | ✅ success | 57 | 14.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-b-ood/T3-b-OOD1_r03_T/T3-b-OOD1_r03_T.mp4) |
| `T3-b-OOD2_r01` | ✅ success | 88 | 23.2 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-b-ood/T3-b-OOD2_r01_T/T3-b-OOD2_r01_T.mp4) |
| `T3-b-OOD2_r02` | ✅ success | 77 | 20.2 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-b-ood/T3-b-OOD2_r02_T/T3-b-OOD2_r02_T.mp4) |
| `T3-b-OOD3_r01` | ❌ fail | 104 | 27.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-b-ood/T3-b-OOD3_r01_F/T3-b-OOD3_r01_F.mp4) |
| `T3-b-OOD3_r02` | ❌ fail | 71 | 18.8 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-b-ood/T3-b-OOD3_r02_F/T3-b-OOD3_r02_F.mp4) |
| `T3-b-OOD4_r01` | ❌ fail | 339 | 90.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-b-ood/T3-b-OOD4_r01_F/T3-b-OOD4_r01_F.mp4) |
| `T3-b-OOD4_r02` | ❌ fail | 504 | 134.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-b-ood/T3-b-OOD4_r02_F/T3-b-OOD4_r02_F.mp4) |
| `T3-b-OOD5_r01` | ✅ success | 188 | 50.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-b-ood/T3-b-OOD5_r01_T/T3-b-OOD5_r01_T.mp4) |
| `T3-b-OOD5_r02` | ✅ success | 217 | 57.6 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-b-ood/T3-b-OOD5_r02_T/T3-b-OOD5_r02_T.mp4) |
| `T3-b-OOD6_r01` | ❌ fail | 499 | 133.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-b-ood/T3-b-OOD6_r01_F/T3-b-OOD6_r01_F.mp4) |
| `T3-b-OOD6_r02` | ❌ fail | 229 | 60.8 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-b-ood/T3-b-OOD6_r02_F/T3-b-OOD6_r02_F.mp4) |
| `T3-b-OOD7_r01` | ❓ unsure | 136 | 36.0 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-b-ood/T3-b-OOD7_r01_Q/T3-b-OOD7_r01_Q.mp4) |
| `T3-b-OOD7_r02` | ❌ fail | 258 | 68.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-b-ood/T3-b-OOD7_r02_F/T3-b-OOD7_r02_F.mp4) |
| `T3-b-OOD7_r03` | ❌ fail | 115 | 30.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-b-ood/T3-b-OOD7_r03_F/T3-b-OOD7_r03_F.mp4) |
| `T3-b-OOD8_r01` | ❌ fail | 177 | 46.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-b-ood/T3-b-OOD8_r01_F/T3-b-OOD8_r01_F.mp4) |
| `T3-b-OOD8_r02` | ❌ fail | 193 | 51.3 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-b-ood/T3-b-OOD8_r02_F/T3-b-OOD8_r02_F.mp4) |
| `T3-b-OOD9_r01` | ❌ fail | 119 | 31.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-b-ood/T3-b-OOD9_r01_F/T3-b-OOD9_r01_F.mp4) |
| `T3-b-OOD9_r02` | ❌ fail | 74 | 19.8 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-b-ood/T3-b-OOD9_r02_F/T3-b-OOD9_r02_F.mp4) |
| `T3-b-OOD10_r01` | ✅ success | 165 | 44.0 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-b-ood/T3-b-OOD10_r01_T/T3-b-OOD10_r01_T.mp4) |
| `T3-b-OOD10_r02` | ✅ success | 300 | 79.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T3-b-ood/T3-b-OOD10_r02_T/T3-b-OOD10_r02_T.mp4) |

</details>

<details>
<summary><b>T4-a</b> — <i>insert the orange block into the wooden shape sorter box</i> — <b>2/20</b> (10%)</summary>

| rollout | verdict | steps | duration | video |
|---|---|---:|---:|---|
| `T4-a-OOD1_r01` | ❌ fail | 176 | 46.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-a-ood/T4-a-OOD1_r01_F/T4-a-OOD1_r01_F.mp4) |
| `T4-a-OOD1_r02` | ❌ fail | 66 | 17.6 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-a-ood/T4-a-OOD1_r02_F/T4-a-OOD1_r02_F.mp4) |
| `T4-a-OOD2_r01` | ❌ fail | 142 | 37.8 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-a-ood/T4-a-OOD2_r01_F/T4-a-OOD2_r01_F.mp4) |
| `T4-a-OOD2_r02` | ❌ fail | 129 | 34.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-a-ood/T4-a-OOD2_r02_F/T4-a-OOD2_r02_F.mp4) |
| `T4-a-OOD3_r01` | ❌ fail | 161 | 42.6 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-a-ood/T4-a-OOD3_r01_F/T4-a-OOD3_r01_F.mp4) |
| `T4-a-OOD3_r02` | ❌ fail | 288 | 76.6 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-a-ood/T4-a-OOD3_r02_F/T4-a-OOD3_r02_F.mp4) |
| `T4-a-OOD4_r01` | ❌ fail | 138 | 36.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-a-ood/T4-a-OOD4_r01_F/T4-a-OOD4_r01_F.mp4) |
| `T4-a-OOD4_r02` | ❌ fail | 116 | 30.6 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-a-ood/T4-a-OOD4_r02_F/T4-a-OOD4_r02_F.mp4) |
| `T4-a-OOD5_r01` | ❌ fail | 120 | 32.0 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-a-ood/T4-a-OOD5_r01_F/T4-a-OOD5_r01_F.mp4) |
| `T4-a-OOD5_r02` | ❌ fail | 413 | 110.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-a-ood/T4-a-OOD5_r02_F/T4-a-OOD5_r02_F.mp4) |
| `T4-a-OOD6_r01` | ❌ fail | 208 | 55.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-a-ood/T4-a-OOD6_r01_F/T4-a-OOD6_r01_F.mp4) |
| `T4-a-OOD6_r02` | ❌ fail | 320 | 85.0 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-a-ood/T4-a-OOD6_r02_F/T4-a-OOD6_r02_F.mp4) |
| `T4-a-OOD7_r01` | ❌ fail | 180 | 47.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-a-ood/T4-a-OOD7_r01_F/T4-a-OOD7_r01_F.mp4) |
| `T4-a-OOD7_r02` | ❌ fail | 82 | 21.6 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-a-ood/T4-a-OOD7_r02_F/T4-a-OOD7_r02_F.mp4) |
| `T4-a-OOD8_r01` | ❌ fail | 123 | 32.8 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-a-ood/T4-a-OOD8_r01_F/T4-a-OOD8_r01_F.mp4) |
| `T4-a-OOD8_r02` | ✅ success | 163 | 43.3 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-a-ood/T4-a-OOD8_r02_T/T4-a-OOD8_r02_T.mp4) |
| `T4-a-OOD9_r01` | ❌ fail | 262 | 69.8 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-a-ood/T4-a-OOD9_r01_F/T4-a-OOD9_r01_F.mp4) |
| `T4-a-OOD9_r02` | ✅ success | 110 | 29.2 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-a-ood/T4-a-OOD9_r02_T/T4-a-OOD9_r02_T.mp4) |
| `T4-a-OOD10_r01` | ❌ fail | 203 | 53.8 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-a-ood/T4-a-OOD10_r01_F/T4-a-OOD10_r01_F.mp4) |
| `T4-a-OOD10_r02` | ❌ fail | 257 | 68.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-a-ood/T4-a-OOD10_r02_F/T4-a-OOD10_r02_F.mp4) |

</details>

<details>
<summary><b>T4-b</b> — <i>insert the book into the black book stand</i> — <b>6/20</b> (30%)</summary>

| rollout | verdict | steps | duration | video |
|---|---|---:|---:|---|
| `T4-b-OOD1_r01` | ❌ fail | 218 | 57.8 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-b-ood/T4-b-OOD1_r01_F/T4-b-OOD1_r01_F.mp4) |
| `T4-b-OOD1_r02` | ❌ fail | 230 | 61.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-b-ood/T4-b-OOD1_r02_F/T4-b-OOD1_r02_F.mp4) |
| `T4-b-OOD2_r02` | ❌ fail | 285 | 75.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-b-ood/T4-b-OOD2_r02_F/T4-b-OOD2_r02_F.mp4) |
| `T4-b-OOD2_r03` | ❌ fail | 440 | 117.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-b-ood/T4-b-OOD2_r03_F/T4-b-OOD2_r03_F.mp4) |
| `T4-b-OOD3_r01` | ❌ fail | 427 | 113.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-b-ood/T4-b-OOD3_r01_F/T4-b-OOD3_r01_F.mp4) |
| `T4-b-OOD3_r02` | ✅ success | 129 | 34.4 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-b-ood/T4-b-OOD3_r02_T/T4-b-OOD3_r02_T.mp4) |
| `T4-b-OOD4_r01` | ❌ fail | 454 | 121.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-b-ood/T4-b-OOD4_r01_F/T4-b-OOD4_r01_F.mp4) |
| `T4-b-OOD4_r02` | ❌ fail | 523 | 139.3 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-b-ood/T4-b-OOD4_r02_F/T4-b-OOD4_r02_F.mp4) |
| `T4-b-OOD5_r01` | ✅ success | 216 | 57.3 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-b-ood/T4-b-OOD5_r01_T/T4-b-OOD5_r01_T.mp4) |
| `T4-b-OOD5_r02` | ❌ fail | 597 | 159.3 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-b-ood/T4-b-OOD5_r02_F/T4-b-OOD5_r02_F.mp4) |
| `T4-b-OOD6_r01` | ✅ success | 218 | 57.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-b-ood/T4-b-OOD6_r01_T/T4-b-OOD6_r01_T.mp4) |
| `T4-b-OOD6_r02` | ❌ fail | 158 | 41.8 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-b-ood/T4-b-OOD6_r02_F/T4-b-OOD6_r02_F.mp4) |
| `T4-b-OOD7_r01` | ❌ fail | 288 | 76.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-b-ood/T4-b-OOD7_r01_F/T4-b-OOD7_r01_F.mp4) |
| `T4-b-OOD7_r02` | ❌ fail | 269 | 71.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-b-ood/T4-b-OOD7_r02_F/T4-b-OOD7_r02_F.mp4) |
| `T4-b-OOD8_r01` | ❌ fail | 413 | 110.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-b-ood/T4-b-OOD8_r01_F/T4-b-OOD8_r01_F.mp4) |
| `T4-b-OOD8_r02` | ❌ fail | 234 | 62.4 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-b-ood/T4-b-OOD8_r02_F/T4-b-OOD8_r02_F.mp4) |
| `T4-b-OOD9_r01` | ✅ success | 62 | 16.2 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-b-ood/T4-b-OOD9_r01_T/T4-b-OOD9_r01_T.mp4) |
| `T4-b-OOD9_r02` | ✅ success | 98 | 25.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-b-ood/T4-b-OOD9_r02_T/T4-b-OOD9_r02_T.mp4) |
| `T4-b-OOD10_r01` | ✅ success | 96 | 25.3 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-b-ood/T4-b-OOD10_r01_T/T4-b-OOD10_r01_T.mp4) |
| `T4-b-OOD10_r02` | ❌ fail | 527 | 140.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T4-b-ood/T4-b-OOD10_r02_F/T4-b-OOD10_r02_F.mp4) |

</details>

<details>
<summary><b>T5-a</b> — <i>pull the smaller book out of the black book stand</i> — <b>15/20</b> (75%)</summary>

| rollout | verdict | steps | duration | video |
|---|---|---:|---:|---|
| `T5-a-OOD1_r01` | ✅ success | 111 | 29.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-a-ood/T5-a-OOD1_r01_T/T5-a-OOD1_r01_T.mp4) |
| `T5-a-OOD1_r02` | ✅ success | 116 | 30.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-a-ood/T5-a-OOD1_r02_T/T5-a-OOD1_r02_T.mp4) |
| `T5-a-OOD2_r01` | ❌ fail | 114 | 30.4 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-a-ood/T5-a-OOD2_r01_F/T5-a-OOD2_r01_F.mp4) |
| `T5-a-OOD2_r02` | ✅ success | 80 | 21.3 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-a-ood/T5-a-OOD2_r02_T/T5-a-OOD2_r02_T.mp4) |
| `T5-a-OOD3_r01` | ✅ success | 105 | 27.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-a-ood/T5-a-OOD3_r01_T/T5-a-OOD3_r01_T.mp4) |
| `T5-a-OOD3_r02` | ✅ success | 73 | 19.2 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-a-ood/T5-a-OOD3_r02_T/T5-a-OOD3_r02_T.mp4) |
| `T5-a-OOD4_r01` | ✅ success | 244 | 65.0 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-a-ood/T5-a-OOD4_r01_T/T5-a-OOD4_r01_T.mp4) |
| `T5-a-OOD4_r02` | ❌ fail | 267 | 70.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-a-ood/T5-a-OOD4_r02_F/T5-a-OOD4_r02_F.mp4) |
| `T5-a-OOD5_r01` | ❌ fail | 155 | 41.0 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-a-ood/T5-a-OOD5_r01_F/T5-a-OOD5_r01_F.mp4) |
| `T5-a-OOD5_r02` | ✅ success | 100 | 26.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-a-ood/T5-a-OOD5_r02_T/T5-a-OOD5_r02_T.mp4) |
| `T5-a-OOD6_r01` | ✅ success | 158 | 42.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-a-ood/T5-a-OOD6_r01_T/T5-a-OOD6_r01_T.mp4) |
| `T5-a-OOD6_r02` | ✅ success | 123 | 32.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-a-ood/T5-a-OOD6_r02_T/T5-a-OOD6_r02_T.mp4) |
| `T5-a-OOD7_r01` | ✅ success | 87 | 22.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-a-ood/T5-a-OOD7_r01_T/T5-a-OOD7_r01_T.mp4) |
| `T5-a-OOD7_r02` | ✅ success | 117 | 31.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-a-ood/T5-a-OOD7_r02_T/T5-a-OOD7_r02_T.mp4) |
| `T5-a-OOD8_r01` | ✅ success | 445 | 118.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-a-ood/T5-a-OOD8_r01_T/T5-a-OOD8_r01_T.mp4) |
| `T5-a-OOD8_r02` | ✅ success | 149 | 39.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-a-ood/T5-a-OOD8_r02_T/T5-a-OOD8_r02_T.mp4) |
| `T5-a-OOD9_r01` | ❌ fail | 376 | 100.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-a-ood/T5-a-OOD9_r01_F/T5-a-OOD9_r01_F.mp4) |
| `T5-a-OOD9_r02` | ❌ fail | 368 | 97.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-a-ood/T5-a-OOD9_r02_F/T5-a-OOD9_r02_F.mp4) |
| `T5-a-OOD10_r01` | ✅ success | 69 | 18.2 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-a-ood/T5-a-OOD10_r01_T/T5-a-OOD10_r01_T.mp4) |
| `T5-a-OOD10_r02` | ✅ success | 321 | 85.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-a-ood/T5-a-OOD10_r02_T/T5-a-OOD10_r02_T.mp4) |

</details>

<details>
<summary><b>T5-b</b> — <i>pull the small block out from under the large block</i> — <b>8/20</b> (40%)</summary>

| rollout | verdict | steps | duration | video |
|---|---|---:|---:|---|
| `T5-b-OOD1_r01` | ✅ success | 489 | 130.4 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-b-ood/T5-b-OOD1_r01_T/T5-b-OOD1_r01_T.mp4) |
| `T5-b-OOD1_r02` | ✅ success | 206 | 54.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-b-ood/T5-b-OOD1_r02_T/T5-b-OOD1_r02_T.mp4) |
| `T5-b-OOD2_r01` | ❌ fail | 521 | 138.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-b-ood/T5-b-OOD2_r01_F/T5-b-OOD2_r01_F.mp4) |
| `T5-b-OOD2_r02` | ✅ success | 384 | 102.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-b-ood/T5-b-OOD2_r02_T/T5-b-OOD2_r02_T.mp4) |
| `T5-b-OOD3_r01` | ✅ success | 763 | 203.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-b-ood/T5-b-OOD3_r01_T/T5-b-OOD3_r01_T.mp4) |
| `T5-b-OOD3_r02` | ❌ fail ⏱ | 1200 | 320.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-b-ood/T5-b-OOD3_r02_F/T5-b-OOD3_r02_F.mp4) |
| `T5-b-OOD4_r01` | ❌ fail | 1190 | 318.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-b-ood/T5-b-OOD4_r01_F/T5-b-OOD4_r01_F.mp4) |
| `T5-b-OOD4_r02` | ❌ fail | 1012 | 270.7 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-b-ood/T5-b-OOD4_r02_F/T5-b-OOD4_r02_F.mp4) |
| `T5-b-OOD5_r01` | ✅ success | 1156 | 309.1 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-b-ood/T5-b-OOD5_r01_T/T5-b-OOD5_r01_T.mp4) |
| `T5-b-OOD5_r02` | ✅ success | 458 | 122.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-b-ood/T5-b-OOD5_r02_T/T5-b-OOD5_r02_T.mp4) |
| `T5-b-OOD6_r01` | ❌ fail | 1185 | 318.3 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-b-ood/T5-b-OOD6_r01_F/T5-b-OOD6_r01_F.mp4) |
| `T5-b-OOD6_r02` | ❌ fail | 1138 | 305.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-b-ood/T5-b-OOD6_r02_F/T5-b-OOD6_r02_F.mp4) |
| `T5-b-OOD7_r01` | ❌ fail | 537 | 143.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-b-ood/T5-b-OOD7_r01_F/T5-b-OOD7_r01_F.mp4) |
| `T5-b-OOD7_r02` | ❌ fail | 1048 | 279.3 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-b-ood/T5-b-OOD7_r02_F/T5-b-OOD7_r02_F.mp4) |
| `T5-b-OOD8_r01` | ❌ fail | 880 | 234.5 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-b-ood/T5-b-OOD8_r01_F/T5-b-OOD8_r01_F.mp4) |
| `T5-b-OOD8_r02` | ❌ fail | 1004 | 268.2 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-b-ood/T5-b-OOD8_r02_F/T5-b-OOD8_r02_F.mp4) |
| `T5-b-OOD9_r01` | ❌ fail | 1086 | 289.6 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-b-ood/T5-b-OOD9_r01_F/T5-b-OOD9_r01_F.mp4) |
| `T5-b-OOD9_r02` | ❌ fail | 923 | 246.2 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-b-ood/T5-b-OOD9_r02_F/T5-b-OOD9_r02_F.mp4) |
| `T5-b-OOD10_r01` | ✅ success | 136 | 36.2 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-b-ood/T5-b-OOD10_r01_T/T5-b-OOD10_r01_T.mp4) |
| `T5-b-OOD10_r02` | ✅ success | 131 | 34.9 s | [▶ play](https://huggingface.co/datasets/TASL-FR3/fr3-ood-rollouts-cotrain-pbc-v2-8000/blob/main/web/T5-b-ood/T5-b-OOD10_r02_T/T5-b-OOD10_r02_T.mp4) |

</details>
<!-- /gen:index -->
