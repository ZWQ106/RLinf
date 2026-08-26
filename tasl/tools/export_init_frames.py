#!/usr/bin/env python3
"""Dump every episode's initial frame (both cameras) for every registered task.

Runs INSIDE the rlinf-eval container (needs pyzed + PIL):

    docker exec -i rlinf-eval /opt/venv/openpi/bin/python \
        /workspace/rlinf/tasl/tools/export_init_frames.py [--out /workspace/rlinf/saved_demo] [--tasks T1-a T5-b]

For each task in tasks_store.json and each dataset attached to it, decodes SVO
frame 0 of every episode (the bench exactly as recording started) and writes

    <out>/<task>/init_layouts/<dataset>_ep<NN>.exterior.jpg   (ZED 2i agent view, camera wrist_1)
    <out>/<task>/init_layouts/_sheet.jpg                      labelled contact sheet

The wrist (eye-in-hand) view is NOT exported by default — a layout is the
scene, and the exterior camera is what shows it. `--with-wrist` adds
`<dataset>_ep<NN>.wrist.jpg` (ZED Mini, camera wrist_2) if needed.
    <out>/<task>/init_layouts/index.json                      provenance per frame

These are candidates: the operator picks the ones that represent a layout,
then `layout_from_dataset.py <dataset> --episode N --layout-id …` promotes the
pick into a real portal layout. Frames already on disk are skipped (re-runs
are cheap).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO = "/workspace/rlinf"
DATASET_ROOT = f"{REPO}/datasets"
VIEW_CAMERA = {"exterior": "wrist_1", "wrist": "wrist_2"}


def decode_frame0(sl, path: str):
    ip = sl.InitParameters()
    ip.set_from_svo_file(path)
    ip.depth_mode = sl.DEPTH_MODE.NONE
    ip.svo_real_time_mode = False
    ip.sdk_verbose = 0
    cam = sl.Camera()
    if cam.open(ip) != sl.ERROR_CODE.SUCCESS:
        return None, "open failed"
    try:
        n = cam.get_svo_number_of_frames()
        if cam.grab(sl.RuntimeParameters()) != sl.ERROR_CODE.SUCCESS:
            return None, "grab failed"
        m = sl.Mat()
        cam.retrieve_image(m, sl.VIEW.LEFT)
        return m.get_data()[:, :, 2::-1].copy(), n   # BGRA -> RGB
    finally:
        cam.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=f"{REPO}/saved_demo")
    ap.add_argument("--tasks", nargs="*", help="task ids (default: all)")
    ap.add_argument("--tasks-store", default=f"{REPO}/tasl/tasks_store.json")
    ap.add_argument("--with-wrist", action="store_true",
                    help="also export the wrist (eye-in-hand) frame")
    args = ap.parse_args()
    views = {v: c for v, c in VIEW_CAMERA.items()
             if v == "exterior" or args.with_wrist}

    import pyzed.sl as sl
    from PIL import Image, ImageDraw

    tasks = json.load(open(args.tasks_store))
    if args.tasks:
        tasks = [t for t in tasks if t["id"] in set(args.tasks)]
    t_start = time.time()
    for t in tasks:
        tid = t["id"]
        out_dir = os.path.join(args.out, tid, "init_layouts")
        os.makedirs(out_dir, exist_ok=True)
        index, tiles = [], []
        for ds in t.get("datasets", []):
            idx_path = f"{DATASET_ROOT}/{ds}_svo/svo_index.json"
            if not os.path.isfile(idx_path):
                print(f"[{tid}] {ds}: no SVO index, skipped", flush=True)
                continue
            svo_index = json.load(open(idx_path))
            for ep in sorted(svo_index, key=int):
                names = svo_index[ep]
                stem = f"{ds}_ep{int(ep):02d}"
                entry = {"task": tid, "dataset": ds, "episode": int(ep), "files": {}}
                imgs = {}
                for view, cam in views.items():
                    fn = next((n for n in names if n.endswith(f"_{cam}.svo2")), None)
                    if fn is None:
                        continue
                    dst = os.path.join(out_dir, f"{stem}.{view}.jpg")
                    if os.path.isfile(dst):
                        imgs[view] = Image.open(dst).convert("RGB")
                    else:
                        arr, n = decode_frame0(sl, f"{DATASET_ROOT}/{ds}_svo/{fn}")
                        if arr is None:
                            print(f"[{tid}] {stem}.{view}: {n}", flush=True)
                            continue
                        imgs[view] = Image.fromarray(arr)
                        imgs[view].save(dst, "JPEG", quality=90)
                        entry["n_frames"] = n
                    entry["files"][view] = os.path.basename(dst)
                    entry.setdefault("svo", {})[view] = f"{ds}_svo/{fn}"
                index.append(entry)
                # contact-sheet tile: the exported views side by side,
                # downscaled and labelled with the episode stem.
                tw, th = 400, 225
                tile = Image.new("RGB", ((tw + 4) * len(views) - 4, th + 18), (255, 255, 255))
                for i, view in enumerate(views):
                    if view in imgs:
                        tile.paste(imgs[view].resize((tw, th)), (i * (tw + 4), 18))
                ImageDraw.Draw(tile).text((4, 3), f"{stem}", fill=(0, 0, 0))
                tiles.append(tile)
            print(f"[{tid}] {ds}: {len(svo_index)} episodes done "
                  f"({time.time() - t_start:.0f}s)", flush=True)
        if tiles:
            cols = 5 if len(views) == 1 else 3
            rows = (len(tiles) + cols - 1) // cols
            w, h = tiles[0].size
            sheet = Image.new("RGB", (cols * w + (cols - 1) * 6, rows * h + (rows - 1) * 6), (60, 60, 60))
            for i, tl in enumerate(tiles):
                sheet.paste(tl, ((i % cols) * (w + 6), (i // cols) * (h + 6)))
            sheet.save(os.path.join(out_dir, "_sheet.jpg"), "JPEG", quality=85)
        json.dump(index, open(os.path.join(out_dir, "index.json"), "w"), indent=1)
        print(f"[{tid}] wrote {len(index)} episodes -> {out_dir}", flush=True)
    print(f"all done in {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
