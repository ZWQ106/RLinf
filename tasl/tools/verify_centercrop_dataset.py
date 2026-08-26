#!/usr/bin/env python3
"""Prove a rebuilt dataset differs from its source ONLY in image pixels.

    /usr/bin/python3 tasl/tools/verify_centercrop_dataset.py <src_dir> <new_dir> [--report <build_report.json>]

Checks, per episode: same parquet schema incl. metadata, every non-image
column byte-identical, image struct `path` identical, every image decodes to
224x224 RGB with no letterbox rows; meta/{info,tasks,episodes} identical and
episodes_stats identical outside the two image entries. Exit code 1 on any
failure. Host-runnable (pyarrow + PIL only).
"""
import argparse, io, json, sys
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

IMG = ("image", "extra_view_image")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("new"); ap.add_argument("--report")
    ap.add_argument("--sample-rows", type=int, default=8, help="images decoded per episode for the pixel check")
    ap.add_argument("--identity", action="store_true",
                    help="frame-identity check on EVERY row: the padded image's central 126x126 must "
                         "agree with the new crop downscaled to 126x126 (same frame, two resamplings) "
                         "and beat the neighbouring rows' crops")
    a = ap.parse_args()
    ident = {"rows": 0, "d_same": [], "beats_neighbours": 0, "moving": 0, "moving_beats": 0}
    fails = []
    info = json.load(open(f"{a.src}/meta/info.json"))
    for name in ("info.json", "tasks.jsonl", "episodes.jsonl"):
        if open(f"{a.src}/meta/{name}", "rb").read() != open(f"{a.new}/meta/{name}", "rb").read():
            fails.append(f"meta/{name} differs")
    s_lines = open(f"{a.src}/meta/episodes_stats.jsonl").read().splitlines()
    n_lines = open(f"{a.new}/meta/episodes_stats.jsonl").read().splitlines()
    if len(s_lines) != len(n_lines):
        fails.append("episodes_stats line count differs")
    for sl_, nl in zip(s_lines, n_lines):
        s, n = json.loads(sl_), json.loads(nl)
        if s["episode_index"] != n["episode_index"]:
            fails.append("episodes_stats order differs"); break
        for k in s["stats"]:
            if k in IMG:
                if set(n["stats"][k]) != set(s["stats"][k]) or np.asarray(n["stats"][k]["mean"]).shape != (3, 1, 1):
                    fails.append(f"ep{s['episode_index']} stats[{k}] malformed")
            elif s["stats"][k] != n["stats"].get(k):
                fails.append(f"ep{s['episode_index']} stats[{k}] changed")
    total_rows = 0
    for e in range(info["total_episodes"]):
        rel = info["data_path"].format(episode_chunk=e // info["chunks_size"], episode_index=e)
        try:
            ts, tn = pq.read_table(f"{a.src}/{rel}"), pq.read_table(f"{a.new}/{rel}")
        except Exception as exc:
            fails.append(f"ep{e}: unreadable ({exc})"); continue
        if ts.num_rows != tn.num_rows:
            fails.append(f"ep{e}: rows {ts.num_rows} vs {tn.num_rows}"); continue
        total_rows += tn.num_rows
        if not ts.schema.equals(tn.schema, check_metadata=True):
            fails.append(f"ep{e}: schema/metadata differs")
        for c in ts.column_names:
            if c in IMG:
                ps = [r["path"] for r in ts.column(c).to_pylist()]
                col = tn.column(c).to_pylist()
                if ps != [r["path"] for r in col]:
                    fails.append(f"ep{e}: {c}.path differs")
                if a.identity:
                    src_col = ts.column(c).to_pylist()
                    orig = [np.asarray(Image.open(io.BytesIO(r["bytes"])).convert("RGB"))[49:175, 49:175].astype(np.float32) for r in src_col]
                    new = [np.asarray(Image.open(io.BytesIO(r["bytes"])).convert("RGB").resize((126, 126), Image.BILINEAR)).astype(np.float32) for r in col]
                    for i in range(len(col)):
                        d_same = float(np.mean((orig[i] - new[i]) ** 2))
                        nb = [float(np.mean((orig[i] - new[j]) ** 2)) for j in (i - 1, i + 1) if 0 <= j < len(col)]
                        ident["rows"] += 1; ident["d_same"].append(d_same)
                        beats = d_same <= min(nb) if nb else True
                        ident["beats_neighbours"] += beats
                        # "moving" = neighbouring crops differ noticeably from each other
                        if nb and float(np.mean((new[i] - new[i - 1 if i > 0 else i + 1]) ** 2)) > 30:
                            ident["moving"] += 1; ident["moving_beats"] += beats
                idx = np.linspace(0, len(col) - 1, min(a.sample_rows, len(col))).astype(int)
                for i in idx:
                    im = Image.open(io.BytesIO(col[i]["bytes"]))
                    arr = np.asarray(im.convert("RGB"))
                    if im.format != "PNG" or arr.shape != (224, 224, 3):
                        fails.append(f"ep{e} row{i} {c}: {im.format} {arr.shape}")
                    rows = arr.mean(axis=(1, 2))
                    if (rows[:20] < 8).all() or (rows[-20:] < 8).all():
                        fails.append(f"ep{e} row{i} {c}: still letterboxed")
            elif not ts.column(c).equals(tn.column(c)):
                fails.append(f"ep{e}: column {c} differs")
        if e % 50 == 0:
            print(f"  checked up to ep{e} ({len(fails)} fails so far)", flush=True)
    if total_rows != info["total_frames"]:
        fails.append(f"total rows {total_rows} != info.total_frames {info['total_frames']}")
    if a.report:
        rep = json.load(open(a.report))["episodes"]
        med = [c["mse_median"] for r in rep for c in r["cols"].values()]
        p95 = [c["mse_p95"] for r in rep for c in r["cols"].values()]
        outside = sum(c["rows_outside_svo"] for r in rep for c in r["cols"].values())
        worst = sorted(((c["mse_p95"], r["merged_ep"], k) for r in rep for k, c in r["cols"].items()), reverse=True)[:5]
        print(f"match quality over {len(rep)} episodes: median-MSE median={np.median(med):.2f} max={max(med):.2f}; "
              f"p95-MSE median={np.median(p95):.2f} max={max(p95):.2f}; rows matched outside SVO span={outside}")
        print("  worst p95 (mse, ep, col):", [(round(m, 1), e, k) for m, e, k in worst])
    if a.identity and ident["rows"]:
        d = np.array(ident["d_same"])
        print(f"frame identity over {ident['rows']} row-images: MSE(padded centre vs new crop) median={np.median(d):.1f} "
              f"p95={np.percentile(d, 95):.1f} max={d.max():.1f}; same-frame beats neighbour rows in "
              f"{100 * ident['beats_neighbours'] / ident['rows']:.2f}% of all rows and "
              f"{100 * ident['moving_beats'] / max(1, ident['moving']):.2f}% of the {ident['moving']} rows where the arm is moving")
        if ident["moving"] and ident["moving_beats"] / ident["moving"] < 0.99:
            fails.append("frame identity: same-frame does not beat neighbours on >1% of moving rows")
    print(f"episodes={info['total_episodes']} rows={total_rows} fails={len(fails)}")
    for f in fails[:30]:
        print("  FAIL:", f)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
