#!/usr/bin/env python3
"""Derive a collection layout from a recorded dataset episode.

Why: in-domain eval needs the bench arranged the way the training data was
recorded, but several datasets were collected without registering a layout
(or under a borrowed one — every ``T*-b`` set was gated on ``T1-b-1``). The
scene IS on disk though: each episode's first SVO frame shows both cameras at
the moment recording began. This tool turns that frame into a normal layout
(``<id>.json`` + ``<id>.exterior.jpg`` + ``<id>.wrist.jpg`` under the layout
dir, no markers, 3x3 grid) and links it to a task in ``tasks_store.json`` so
both portals can pick it straight away.

Runs on the HOST (pure stdlib) and shells into the ``rlinf-eval`` container
for the decode, because only the container has pyzed/PIL and the ZED SDK.
The SVO frames are full 1280x720 camera views — the same framing the live
preview shows — so the ghost lines up pixel-for-pixel; the 224x224 images in
the parquet are cropped/padded and are NOT used.

Camera -> view mapping mirrors ``collect.py``'s ``LiveCamSource.NAME_MAP``:
``wrist_1`` (ZED 2i, exterior) -> ``exterior``; ``wrist_2`` (ZED Mini) ->
``wrist``.

Usage::

    # layout T1-b-L1 from episode 0, frame 0, and make it T1-b's default
    python3 tasl/tools/layout_from_dataset.py T1-b --layout-id T1-b-L1 --task T1-b

    # a second layout for the same task (episode 16), kept as non-default
    python3 tasl/tools/layout_from_dataset.py T1-a --episode 16 \
        --layout-id T1-a-L2 --task T1-a --no-default

    # inspect the contact sheet first if unsure which episode is canonical:
    python3 tasl/tools/layout_from_dataset.py T5-b --sheet /tmp/T5-b.jpg
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys

_TASL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_TASL_DIR, "dashboards"))
from layout_store import LayoutError, LayoutStore, is_usable_snapshot  # noqa: E402
from task_store import TaskStore  # noqa: E402

# Same conventions as the dashboards: paths anchored on the checkout, not
# $HOME, so they survive a sudo re-exec.
_HOME_DIR = os.path.dirname(os.path.dirname(_TASL_DIR))  # /home/franka_desktop
DATA_DIR_HOST = os.environ.get("RLINF_DATA_DIR",
                               os.path.join(_HOME_DIR, "rlinf_data"))
LAYOUT_DIR = os.environ.get("RLINF_LAYOUT_DIR",
                            os.path.join(DATA_DIR_HOST, "layouts"))
CONTAINER = os.environ.get("RLINF_CONTAINER", "rlinf-eval")
CONTAINER_PY = "/opt/venv/openpi/bin/python"
CONTAINER_DATASET_ROOT = "/workspace/rlinf/datasets"

VIEW_CAMERA = {"exterior": "wrist_1", "wrist": "wrist_2"}
SENTINEL = "@@LAYOUT_FRAMES@@"

# Runs inside the container: decode one frame per view from the SVOs and hand
# the JPEGs back on stdout. Everything before the sentinel line is SDK chatter.
_EXTRACTOR = r'''
import base64, io, json, sys
req = json.loads(sys.argv[1])
import pyzed.sl as sl
from PIL import Image
out = {"frames": {}, "info": {}}
for view, path in req["files"].items():
    ip = sl.InitParameters()
    ip.set_from_svo_file(path)
    ip.depth_mode = sl.DEPTH_MODE.NONE
    ip.svo_real_time_mode = False
    ip.sdk_verbose = 0
    cam = sl.Camera()
    st = cam.open(ip)
    if st != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(f"{view}: cannot open {path}: {st}")
    n = cam.get_svo_number_of_frames()
    frame = req["frame"] if req["frame"] >= 0 else n + req["frame"]
    if not 0 <= frame < n:
        raise SystemExit(f"{view}: frame {req['frame']} out of range (0..{n-1})")
    if frame:
        cam.set_svo_position(frame)
    if cam.grab(sl.RuntimeParameters()) != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(f"{view}: grab failed at frame {frame}")
    m = sl.Mat()
    cam.retrieve_image(m, sl.VIEW.LEFT)
    a = m.get_data()                       # BGRA
    img = Image.fromarray(a[:, :, 2::-1])  # -> RGB
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    out["frames"][view] = base64.b64encode(buf.getvalue()).decode()
    out["info"][view] = {"file": path, "frame": frame, "n_frames": n,
                         "size": [img.width, img.height]}
    cam.close()
print(SENTINEL + json.dumps(out))
'''.replace("SENTINEL", repr(SENTINEL))

# Contact sheet: first parquet frame of every episode, both views, so the
# operator can see whether the dataset really has ONE arrangement before
# blessing episode 0 as the layout.
_SHEET = r'''
import glob, io, json, os, sys
req = json.loads(sys.argv[1])
import pyarrow.parquet as pq
from PIL import Image, ImageDraw
files = sorted(glob.glob(os.path.join(req["dataset_dir"], "data", "*", "episode_*.parquet")))
tiles = []
for f in files:
    row = pq.read_table(f, columns=["image", "extra_view_image"]).slice(0, 1).to_pylist()[0]
    a = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
    b = Image.open(io.BytesIO(row["extra_view_image"]["bytes"])).convert("RGB")
    t = Image.new("RGB", (a.width + b.width, a.height + 14), (255, 255, 255))
    t.paste(a, (0, 14)); t.paste(b, (a.width, 14))
    ImageDraw.Draw(t).text((2, 1), "ep" + os.path.basename(f)[8:14].lstrip("0").rjust(1, "0"), fill=(0, 0, 0))
    tiles.append(t)
if not tiles:
    raise SystemExit("no parquet episodes under " + req["dataset_dir"])
cols = 5; rows = (len(tiles) + cols - 1) // cols; w, h = tiles[0].size
sheet = Image.new("RGB", (cols * w, rows * h), (200, 200, 200))
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % cols) * w, (i // cols) * h))
sheet.save(req["out"], quality=85)
print(SENTINEL + json.dumps({"episodes": len(tiles), "size": sheet.size}))
'''.replace("SENTINEL", repr(SENTINEL))


def _die(msg: str, code: int = 2) -> "NoReturn":  # type: ignore[name-defined]
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def _run_in_container(container: str, script: str, req: dict) -> dict:
    cmd = ["docker", "exec", "-i", container, CONTAINER_PY, "-", json.dumps(req)]
    proc = subprocess.run(cmd, input=script, text=True, capture_output=True)
    payload = None
    for line in proc.stdout.splitlines():
        if line.startswith(SENTINEL):
            payload = line[len(SENTINEL):]
    if proc.returncode != 0 or payload is None:
        tail = (proc.stderr.strip().splitlines() or proc.stdout.strip().splitlines() or ["?"])[-3:]
        _die(f"container step failed (rc={proc.returncode}): " + " | ".join(tail))
    return json.loads(payload)


def _svo_files(dataset: str, episode: int) -> dict:
    """{view: container path of the SVO} for one episode, via svo_index.json."""
    idx_path = os.path.join(DATA_DIR_HOST, "datasets", f"{dataset}_svo", "svo_index.json")
    try:
        with open(idx_path, "r", encoding="utf-8") as f:
            index = json.load(f)
    except OSError as e:
        _die(f"no SVO index for dataset '{dataset}' ({e}); this tool needs the "
             f"<dataset>_svo recordings — the parquet images are cropped 224x224")
    names = index.get(str(episode))
    if not names:
        _die(f"episode {episode} not in {idx_path} (has {len(index)} episodes)")
    files = {}
    for view, cam in VIEW_CAMERA.items():
        match = [n for n in names if n.endswith(f"_{cam}.svo2")]
        if not match:
            _die(f"episode {episode}: no '{cam}' SVO for view '{view}' in {names}")
        files[view] = f"{CONTAINER_DATASET_ROOT}/{dataset}_svo/{match[0]}"
    return files


def _dataset_prompt(dataset: str) -> str:
    path = os.path.join(DATA_DIR_HOST, "datasets", dataset, "meta", "tasks.jsonl")
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    return str(json.loads(line).get("task", ""))
    except (OSError, ValueError):
        pass
    return ""


def cmd_sheet(args) -> None:
    out_ctr = f"/workspace/rlinf/outputs/layout_extract/sheet_{args.dataset}.jpg"
    out_host = os.path.join(DATA_DIR_HOST, "outputs", "layout_extract", f"sheet_{args.dataset}.jpg")
    res = _run_in_container(args.container, _SHEET, {
        "dataset_dir": f"{CONTAINER_DATASET_ROOT}/{args.dataset}", "out": out_ctr})
    if args.sheet != out_host:
        import shutil
        os.makedirs(os.path.dirname(os.path.abspath(args.sheet)) or ".", exist_ok=True)
        shutil.copyfile(out_host, args.sheet)
    print(f"contact sheet: {args.sheet}  ({res['episodes']} episodes, "
          f"{res['size'][0]}x{res['size'][1]}; per tile: exterior | wrist)")


def cmd_extract(args) -> None:
    layout_id = args.layout_id or f"{args.dataset}-L1"
    store = LayoutStore(args.layout_dir)
    if store.exists(layout_id) and not args.force:
        _die(f"layout '{layout_id}' already exists in {args.layout_dir} (use --force)")

    files = _svo_files(args.dataset, args.episode)
    res = _run_in_container(args.container, _EXTRACTOR,
                            {"files": files, "frame": args.frame})
    snapshots = {}
    for view, b64 in res["frames"].items():
        buf = base64.b64decode(b64)
        if not is_usable_snapshot(buf):
            _die(f"{view}: decoded frame is not a usable snapshot")
        snapshots[view] = buf

    prompt = _dataset_prompt(args.dataset)
    note = args.note or (
        f"from dataset {args.dataset} ep{args.episode} frame{args.frame}"
        + (f' — "{prompt}"' if prompt else ""))
    try:
        lay = store.save(layout_id, {}, note=note, snapshots=snapshots)
    except LayoutError as e:
        _die(str(e))

    print(f"layout '{layout_id}' -> {store.json_path(layout_id)}")
    for view in ("exterior", "wrist"):
        info = res["info"][view]
        print(f"  {view:8s} <- {os.path.basename(info['file'])} "
              f"frame {info['frame']}/{info['n_frames']} "
              f"({info['size'][0]}x{info['size'][1]})")
    print(f"  note: {lay['note']}")

    if args.task:
        tasks = TaskStore()
        rec = tasks.get(args.task)
        if rec is None:
            _die(f"task '{args.task}' not in tasks_store.json "
                 f"(have: {', '.join(t['id'] for t in tasks.list())})")
        prev_default = rec.get("layout", "")
        tasks.add_layout(args.task, layout_id)
        if args.no_default and prev_default and prev_default != layout_id:
            tasks.update(args.task, {"layout": prev_default})
        rec = tasks.get(args.task)
        print(f"task '{args.task}': layouts={rec['layouts']} default={rec['layout']}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage::", 1)[1] if "Usage::" in __doc__ else None)
    p.add_argument("dataset", help="dataset name under rlinf_data/datasets (e.g. T1-b)")
    p.add_argument("--episode", type=int, default=0, help="episode index (default 0)")
    p.add_argument("--frame", type=int, default=0,
                   help="SVO frame within the episode; negative counts from the end (default 0)")
    p.add_argument("--layout-id", help="layout id to write (default <dataset>-L1)")
    p.add_argument("--task", help="task id in tasks_store.json to link the layout to")
    p.add_argument("--no-default", action="store_true",
                   help="with --task: append to the task's layouts but keep its current default")
    p.add_argument("--note", help="override the layout note (default records dataset/episode/frame)")
    p.add_argument("--force", action="store_true", help="overwrite an existing layout id")
    p.add_argument("--layout-dir", default=LAYOUT_DIR)
    p.add_argument("--container", default=CONTAINER)
    p.add_argument("--sheet", metavar="OUT.jpg",
                   help="instead of extracting, write a contact sheet of every episode's first frame")
    args = p.parse_args(argv)
    if args.sheet:
        cmd_sheet(args)
    else:
        cmd_extract(args)


if __name__ == "__main__":
    main()
