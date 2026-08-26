#!/usr/bin/env python3
"""Rebuild a padded-224 LeRobot dataset as a CENTER-CROP-224 one from the SVOs.

Runs INSIDE rlinf-eval (pyzed + pyarrow + lerobot):

    docker exec -d rlinf-eval sh -c '/opt/venv/openpi/bin/python \
        /workspace/rlinf/tasl/tools/make_centercrop_dataset.py \
        --src /workspace/rlinf/outputs/merged/tasl_fr3_10task_250ep_src \
        --out /workspace/rlinf/outputs/merged/tasl_fr3_10task_250ep_cc > /workspace/rlinf/outputs/merged/build_cc.log 2>&1'

What changes: ONLY the pixel bytes of `image` / `extra_view_image`. Every
row, every other column, the struct `path` strings, the parquet schema (incl.
the `huggingface` metadata), meta/info.json, tasks.jsonl and episodes.jsonl
are carried over verbatim. episodes_stats.jsonl is copied with just the two
image entries recomputed through lerobot's own compute_episode_stats().

Where the new pixels come from: the per-episode SVO recordings (1280x720
H264), center-cropped to 720x720 and cv2.resize'd to 224 — exactly
franka_env._crop_frame's "crop" branch. The padded parquet image cannot be
un-padded (its content is only 224x126), so the full-res source is required.

Which SVO frame is a given row: NOT a fixed index offset. The env pops frames
from the camera thread's queue, so a row usually shows SVO frame i-2 but the
lag drifts on slow steps. Each row is therefore matched by image similarity:
the SVO frames are padded/resized exactly like the env did and the one with
the lowest MSE against the row's stored image (within ±WINDOW frames) wins.
Rows recorded before the SVO started (the first 1-2, arm static) fall back to
the nearest available frame and are counted in the report.

Resumable: per-episode results land in <out>_work/; re-running skips them.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

DATASET_ROOT = "/workspace/rlinf/datasets"
# merged episode order (verified 2026-08-25 by per-episode length + byte-identical rows)
SOURCE_ORDER = ["T1-a", "T1-b", "T2-a-10ep", "T2-a-15ep", "T2-b", "T3-a-25ep",
                "T3-b", "T4-a-25ep", "T4-b", "T5-a-25ep", "T5-b"]
COL_CAMERA = {"image": "wrist_1", "extra_view_image": "wrist_2"}
WINDOW = 4
SIZE = 224


def pad224(bgr):
    h, w = bgr.shape[:2]
    side = max(h, w)
    sq = np.zeros((side, side, 3), np.uint8)
    y, x = (side - h) // 2, (side - w) // 2
    sq[y:y + h, x:x + w] = bgr
    return cv2.resize(sq, (SIZE, SIZE))[:, :, ::-1]          # -> RGB


def crop224(bgr):
    h, w = bgr.shape[:2]
    c = min(h, w)
    y, x = (h - c) // 2, (w - c) // 2
    return cv2.resize(bgr[y:y + c, x:x + c], (SIZE, SIZE))[:, :, ::-1]   # -> RGB


def svo_pad_crop(path):
    """Decode every frame once; keep only the two 224 derivatives."""
    ip = sl.InitParameters()
    ip.set_from_svo_file(path)
    ip.depth_mode = sl.DEPTH_MODE.NONE
    ip.svo_real_time_mode = False
    ip.sdk_verbose = 0
    cam = sl.Camera()
    if cam.open(ip) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"cannot open {path}")
    m = sl.Mat()
    pads, crops = [], []
    try:
        while cam.grab(sl.RuntimeParameters()) == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(m, sl.VIEW.LEFT)
            bgr = m.get_data()[:, :, :3]
            pads.append(pad224(bgr))
            crops.append(crop224(bgr))
    finally:
        cam.close()
    return np.stack(pads), np.stack(crops)


def png_bytes(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def decode_png(b):
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))


def build_mapping(src_meta):
    merged = [json.loads(l) for l in open(f"{src_meta}/episodes.jsonl")]
    mapping, i = [], 0
    for ds in SOURCE_ORDER:
        for e, line in enumerate(open(f"{DATASET_ROOT}/{ds}/meta/episodes.jsonl")):
            L = json.loads(line)["length"]
            if merged[i]["length"] != L:
                raise SystemExit(f"length mismatch at merged ep {i}: {ds} ep{e} {L} vs {merged[i]['length']}")
            mapping.append((i, ds, e, L))
            i += 1
    if i != len(merged):
        raise SystemExit(f"mapped {i} episodes but merged has {len(merged)}")
    return mapping


def lerobot_image_stats(png_list, key, features):
    """Exactly lerobot 0.1.0 compute_episode_stats for one image feature,
    with its file loader swapped for our in-memory PNGs."""
    import lerobot.common.datasets.compute_stats as cs
    store = {f"mem://{k}": b for k, b in enumerate(png_list)}
    orig = cs.load_image_as_numpy

    def _load(path, dtype=np.float32, channel_first=True):
        arr = decode_png(store[path]).astype(dtype)
        return np.transpose(arr, (2, 0, 1)) if channel_first else arr
    cs.load_image_as_numpy = _load
    try:
        st = cs.compute_episode_stats({key: list(store.keys())}, features)[key]
    finally:
        cs.load_image_as_numpy = orig
    return {k: (np.asarray(v).tolist()) for k, v in st.items()}


def process_episode(mi, ds, ep, src_dir, out_dir, work_dir, features):
    t0 = time.time()
    rel = f"data/chunk-{mi // 1000:03d}/episode_{mi:06d}.parquet"
    table = pq.read_table(f"{src_dir}/{rel}")
    n = table.num_rows
    svo_index = json.load(open(f"{DATASET_ROOT}/{ds}_svo/svo_index.json"))[str(ep)]
    report = {"merged_ep": mi, "dataset": ds, "episode": ep, "rows": n, "cols": {}}
    new_cols = {}
    for col, cam in COL_CAMERA.items():
        fn = next(x for x in svo_index if x.endswith(f"_{cam}.svo2"))
        pads, crops = svo_pad_crop(f"{DATASET_ROOT}/{ds}_svo/{fn}")
        m = len(pads)
        stored = table.column(col).to_pylist()
        paths = [r["path"] for r in stored]
        pads_f = pads.astype(np.float32)
        out_png, offs, mses = [], [], []
        for i, r in enumerate(stored):
            ref = decode_png(r["bytes"]).astype(np.float32)
            lo, hi = max(0, i - WINDOW), min(m - 1, i + WINDOW)
            if lo > hi:                      # row past the end of the SVO
                j = m - 1
                mse = float(np.mean((ref - pads_f[j]) ** 2))
            else:
                d = np.mean((pads_f[lo:hi + 1] - ref[None]) ** 2, axis=(1, 2, 3))
                k = int(np.argmin(d))
                j, mse = lo + k, float(d[k])
            offs.append(j - i)
            mses.append(mse)
            out_png.append(png_bytes(crops[j]))
        # Rows whose "true" frame predates / postdates the SVO (env queue
        # lag: typical offset is the episode's median) — they were matched
        # to the nearest existing frame, which is fine while the arm is still.
        med_off = int(np.median(offs))
        fallback = sum(1 for i in range(n) if not (0 <= i + med_off < m))
        hist = {}
        for o in offs:
            hist[str(o)] = hist.get(str(o), 0) + 1
        report["cols"][col] = {"svo": fn, "svo_frames": m, "offset_hist": hist,
                               "mse_median": float(np.median(mses)),
                               "mse_p95": float(np.percentile(mses, 95)),
                               "mse_max": float(np.max(mses)),
                               "median_offset": med_off,
                               "rows_outside_svo": fallback}
        report["cols"][col]["stats"] = lerobot_image_stats(out_png, col, features)
        new_cols[col] = pa.StructArray.from_arrays(
            [pa.array(out_png, pa.binary()), pa.array(paths, pa.string())],
            names=["bytes", "path"])
    # swap the two image columns, keep everything else + schema metadata
    for col, arr in new_cols.items():
        idx = table.schema.get_field_index(col)
        field = table.schema.field(idx)
        assert arr.type == field.type, (arr.type, field.type)
        table = table.set_column(idx, field, arr)
    table = table.replace_schema_metadata(pq.read_schema(f"{src_dir}/{rel}").metadata)
    os.makedirs(os.path.dirname(f"{out_dir}/{rel}"), exist_ok=True)
    tmp = f"{out_dir}/{rel}.part"
    pq.write_table(table, tmp, compression="snappy")
    os.replace(tmp, f"{out_dir}/{rel}")
    report["seconds"] = round(time.time() - t0, 1)
    json.dump(report, open(f"{work_dir}/ep_{mi:06d}.json", "w"))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", nargs="*", type=int, help="subset of merged episode ids")
    args = ap.parse_args()
    global cv2, sl
    import cv2
    import pyzed.sl as sl

    src, out = args.src.rstrip("/"), args.out.rstrip("/")
    work = out + "_work"
    os.makedirs(work, exist_ok=True)
    os.makedirs(f"{out}/meta", exist_ok=True)
    info = json.load(open(f"{src}/meta/info.json"))
    features = info["features"]
    mapping = build_mapping(f"{src}/meta")
    todo = [m for m in mapping if not args.episodes or m[0] in set(args.episodes)]
    t_all = time.time()
    for k, (mi, ds, ep, L) in enumerate(todo):
        if os.path.exists(f"{work}/ep_{mi:06d}.json") and os.path.exists(
                f"{out}/data/chunk-{mi // 1000:03d}/episode_{mi:06d}.parquet"):
            continue
        r = process_episode(mi, ds, ep, src, out, work, features)
        c = r["cols"]["image"]
        print(f"[{k + 1}/{len(todo)}] ep{mi:03d} {ds} ep{ep:02d} rows={L} "
              f"svo={c['svo_frames']} off={c['offset_hist']} mse_med={c['mse_median']:.1f} "
              f"p95={c['mse_p95']:.1f} outside={c['rows_outside_svo']} {r['seconds']}s "
              f"(total {time.time() - t_all:.0f}s)", flush=True)

    if args.episodes:
        print("subset done; run without --episodes to assemble meta", flush=True)
        return
    # ---- meta: verbatim copies + stats with the two image entries swapped ----
    import shutil
    for name in ("info.json", "tasks.jsonl", "episodes.jsonl"):
        shutil.copyfile(f"{src}/meta/{name}", f"{out}/meta/{name}")
    reports = {}
    with open(f"{out}/meta/episodes_stats.jsonl", "w") as f_out:
        for line in open(f"{src}/meta/episodes_stats.jsonl"):
            rec = json.loads(line)
            mi = rec["episode_index"]
            rep = json.load(open(f"{work}/ep_{mi:06d}.json"))
            for col in COL_CAMERA:
                rec["stats"][col] = rep["cols"][col].pop("stats")
            reports[mi] = rep
            f_out.write(json.dumps(rec) + "\n")
    json.dump({"src": src, "out": out, "window": WINDOW, "size": SIZE,
               "episodes": [reports[k] for k in sorted(reports)]},
              open(out + "_build_report.json", "w"), indent=1)
    print(f"DONE {len(reports)} episodes in {time.time() - t_all:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
