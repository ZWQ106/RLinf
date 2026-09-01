#!/usr/bin/env python3
"""Publish one eval checkpoint's OOD rollouts to the Hub in the house format.

The recorder writes MPEG-4 Part 2 (``mp4v``) with a trailing ``moov`` atom: no
browser decodes it and the trailing ``moov`` blocks progressive playback, so the
raw files are useless in the Hub preview, the dataset viewer and any ``<video>``
element.  This tool mirrors every rollout into ``web/`` as H.264 High +
``+faststart`` (the encode the ``cotrain-pbc-v2-8000`` card established), builds
the enriched ``metadata.jsonl`` whose ``file_name`` points at the playable copy,
refreshes the stats workbook, regenerates the README's per-task index, and
uploads the tree.

Everything is idempotent: re-run it after recording more rollouts and only the
new episodes are encoded and only the generated README blocks are rewritten.

Usage:
    # refresh stats + index and upload the new rollouts
    python tasl/tools/publish_ood_rollouts.py cotrain-pbc-v2-8000

    # dry run: encode + render locally, upload nothing
    python tasl/tools/publish_ood_rollouts.py pi05-droid-ft-15k --no-upload

    # just re-check what is already on the Hub
    python tasl/tools/publish_ood_rollouts.py pi05-droid-ft-15k --verify-only
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ood_stats  # noqa: E402  (sibling tool, reused for the workbook)

SAVED_DEMO = ood_stats.SAVED_DEMO
REPO_ORG = "TASL-FR3"
SPACE_URL = "https://huggingface.co/spaces/axisrobotics/fr3-rollout-browser"

# x264 settings recovered from the SEI of the cotrain-pbc-v2-8000 web/ files, so
# re-encodes stay bit-comparable with what is already published:
#   High profile / yuv420p / crf=26 / keyint=30 (2 s @ 15 fps) / preset medium
FFMPEG_ARGS = [
    "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p",
    "-crf", "26", "-preset", "medium", "-g", "30",
    "-movflags", "+faststart", "-an",
]

MARK_SUFFIX = {"success": "T", "fail": "F", "unsure": "Q"}
VERDICT_ICON = {"success": "✅ success", "fail": "❌ fail", "unsure": "❓ unsure"}
STEM_RE = re.compile(r"^(?P<layout>.+?)_(?P<rollout>r\d+)(?:_(?P<suffix>[TFQ]))?$")

# ---------------------------------------------------------------- records ----

def build_records(ckpt_dir: str, root: str, step_cap: int) -> list[dict]:
    """Enriched, Hub-ready metadata rows — one per rollout, sorted by path."""
    out = []
    for rec in ood_stats.load_sidecars(ckpt_dir, root):
        rel = rec.pop("_relpath")                      # <task>-ood/<ckpt>/<stem>/<stem>.json
        parts = rel.split(os.sep)
        repo_rel = os.path.join(parts[0], *parts[2:])  # drop the ckpt level
        base = repo_rel[: -len(".json")]
        stem = rec["demo"]["stem"]

        m = STEM_RE.match(stem)
        layout = m.group("layout") if m else rec.get("layout", "")
        rollout = m.group("rollout") if m else ""
        suffix = (m.group("suffix") or "") if m else ""

        task = rec.get("task", "")
        tm = re.match(r"^(T\d+)-([ab])$", task)
        om = re.search(r"OOD(\d+)$", rec.get("layout", layout) or "")
        mark = rec.get("mark", "")

        out.append({
            "file_name": f"web/{base}.mp4",
            "source_video": f"{base}.mp4",
            "stem": stem,
            "task": task,
            "task_group": tm.group(1) if tm else "",
            "variant": tm.group(2) if tm else "",
            "layout": rec.get("layout", layout),
            "ood_index": int(om.group(1)) if om else 0,
            "rollout": rollout,
            "prompt": rec.get("prompt", ""),
            "mark": mark,
            "success": mark == "success",
            "unsure": mark == "unsure",
            "timeout": bool(rec.get("steps", 0) >= step_cap),
            "suffix_stale": suffix != MARK_SUFFIX.get(mark, ""),
            "steps": rec.get("steps", 0),
            "traj_steps": rec.get("traj_steps", 0),
            "frames": rec.get("frames", 0),
            "duration_s": rec.get("duration_s", 0.0),
            "fps": rec.get("fps", 0.0),
            "image_mode": rec.get("image_mode", ""),
            "abort": rec.get("abort", ""),
            "note": rec.get("note", ""),
            "ep_id": rec.get("ep_id", ""),
            "ckpt": rec.get("ckpt", ""),
            "start_time": rec.get("start_time", ""),
            "end_time": rec.get("end_time", ""),
            "t0": rec.get("t0", 0.0),
            "traj_file": f"{base}.traj.jsonl",
            "frames_file": f"{base}.frames.json",
            "sidecar_file": f"{base}.json",
            "source_dir": rec["demo"]["dir"],
        })
    return out


def sort_key(r: dict) -> tuple:
    return (r["task"], r["ood_index"], r["rollout"])


# ---------------------------------------------------------------- encoding ---

def probe_ok(path: str) -> bool:
    """True when *path* is H.264 with the moov atom ahead of mdat."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(262144)
    except OSError:
        return False
    order, off = [], 0
    while off + 8 <= len(head) and len(order) < 4:
        size = struct.unpack(">I", head[off:off + 4])[0]
        order.append(head[off + 4:off + 8].decode("latin1", "replace"))
        if size == 1:
            size = struct.unpack(">Q", head[off + 8:off + 16])[0]
        if size < 8:
            break
        off += size
    faststart = len(order) >= 2 and order[0] == "ftyp" and order[1] == "moov"
    return faststart and b"avc1" in head and b"mp4v" not in head


def encode_one(job: tuple[str, str]) -> tuple[str, str | None]:
    src, dst = job
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src) and probe_ok(dst):
        return dst, None                                    # already good
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part.mp4"
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", src, *FFMPEG_ARGS, tmp]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        return dst, (exc.stderr or "").strip()[:400] or "ffmpeg failed"
    if not probe_ok(tmp):
        os.remove(tmp)
        return dst, "output is not H.264 + faststart"
    os.replace(tmp, dst)
    return dst, None


def encode_web(records: list[dict], root: str, ckpt_dir: str, stage: str,
               jobs: int, skip: set[str] | None = None) -> None:
    skip = skip or set()
    todo = []
    for r in records:
        if r["file_name"] in skip:
            continue                       # the Hub already has a correct copy
        parts = r["source_video"].split("/")
        src = os.path.join(root, parts[0], ckpt_dir, *parts[1:])
        todo.append((src, os.path.join(stage, r["file_name"])))

    if not todo:
        print("  nothing to encode — every web video is already published")
        return

    missing = [s for s, _ in todo if not os.path.exists(s)]
    if missing:
        raise SystemExit(f"{len(missing)} source video(s) missing, first: {missing[0]}")

    done = failed = 0
    with cf.ThreadPoolExecutor(jobs) as ex:
        for dst, err in ex.map(encode_one, todo):
            done += 1
            if err:
                failed += 1
                print(f"  !! {os.path.relpath(dst, stage)}: {err}", file=sys.stderr)
            if done % 20 == 0 or done == len(todo):
                print(f"  encoded {done}/{len(todo)}")
    if failed:
        raise SystemExit(f"{failed} encode(s) failed")


def remote_state(repo_id: str, jobs: int) -> tuple[dict[str, int], set[str]]:
    """``({path: size_in_bytes}, {web paths that already decode as H.264+faststart})``.

    Files the Hub already holds in the right shape are left completely alone: not
    re-encoded, not re-staged, not re-uploaded. Re-encoding is not reproducible at
    the byte level — x264 output depends on the thread count and the library build
    — so uploading a local re-encode of an already-correct video would replace a
    good file with a different-but-equivalent one and grow the repo's LFS history
    for nothing.
    """
    from huggingface_hub import HfApi, get_token
    api, token = HfApi(), get_token()
    try:
        tree = list(api.list_repo_tree(repo_id, repo_type="dataset",
                                       recursive=True, expand=True))
    except Exception as exc:                       # repo does not exist yet
        print(f"  no remote state ({type(exc).__name__}) — publishing everything")
        return {}, set()

    sizes = {e.path: e.size for e in tree if getattr(e, "size", None) is not None}
    web = [p for p in sizes if p.startswith("web/") and p.endswith(".mp4")]

    def ok(path: str) -> tuple[str, bool]:
        url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{path}"
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {token}", "Range": "bytes=0-262143"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                head = resp.read()
        except Exception:
            return path, False
        order, off = [], 0
        while off + 8 <= len(head) and len(order) < 4:
            size = struct.unpack(">I", head[off:off + 4])[0]
            order.append(head[off + 4:off + 8].decode("latin1", "replace"))
            if size == 1:
                size = struct.unpack(">Q", head[off + 8:off + 16])[0]
            if size < 8:
                break
            off += size
        return path, (len(order) >= 2 and order[0] == "ftyp" and order[1] == "moov"
                      and b"avc1" in head and b"mp4v" not in head)

    good = set()
    if web:
        with cf.ThreadPoolExecutor(jobs) as ex:
            good = {p for p, is_ok in ex.map(ok, web) if is_ok}
    return sizes, good


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def expected_paths(records: list[dict], extra: list[str]) -> set[str]:
    """Every repo-relative path this checkpoint should publish."""
    out = set(extra)
    for r in records:
        out.update((r["file_name"], r["source_video"], r["traj_file"],
                    r["frames_file"], r["sidecar_file"]))
    return out


def prune_stage(stage: str, keep: set[str], root: str) -> list[str]:
    """Drop staged files that no longer correspond to a rollout on disk.

    The eval portal renames an episode's folder when its verdict is changed after
    export, so without this a re-marked rollout would be published twice: once
    under its old stem and once under the new one.

    This deletes files, so it refuses to run anywhere near the recordings: the
    staging directory must not be, contain, or live inside ``root``.
    """
    stage_abs, root_abs = os.path.realpath(stage), os.path.realpath(root)
    if (stage_abs == root_abs
            or stage_abs.startswith(root_abs + os.sep)
            or root_abs.startswith(stage_abs + os.sep)):
        raise SystemExit(
            f"refusing to prune {stage_abs}: it overlaps the recordings at {root_abs}. "
            "Point --stage at a scratch directory.")
    dropped = []
    for dirpath, dirnames, filenames in os.walk(stage, topdown=True):
        # huggingface_hub keeps its upload bookkeeping in .cache/ inside the
        # folder; pruning it would throw away resumability for no gain.
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, stage)
            if rel.split(os.sep)[0].startswith("."):
                continue
            if rel not in keep:
                os.remove(full)
                dropped.append(rel)
    for dirpath, dirnames, filenames in os.walk(stage, topdown=False):
        if dirpath == stage or os.path.basename(dirpath).startswith("."):
            continue
        if not os.listdir(dirpath):
            os.rmdir(dirpath)
    return dropped


# ------------------------------------------------------------------ README ---

def _tasks(records: list[dict]) -> list[tuple[str, list[dict]]]:
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r["task"], []).append(r)
    return [(t, sorted(g, key=sort_key)) for t, g in sorted(groups.items())]


def render_stats(records: list[dict]) -> str:
    lines = ["| task | prompt | n | SR % | steps mean | time mean s |",
             "|------|--------|---|------|-----------|-------------|"]
    tot_scored: list[dict] = []
    for task, group in _tasks(records):
        scored = [r for r in group if not r["unsure"]]
        tot_scored += scored
        if not scored:
            continue
        succ = sum(r["success"] for r in scored)
        lines.append(
            f"| {task} | {group[0]['prompt']} | {len(scored)} | "
            f"{100 * succ / len(scored):.1f} | "
            f"{sum(r['steps'] for r in scored) / len(scored):.1f} | "
            f"{sum(r['duration_s'] for r in scored) / len(scored):.1f} |")
    if tot_scored:
        succ = sum(r["success"] for r in tot_scored)
        lines.append(
            f"| **ALL** | | **{len(tot_scored)}** | "
            f"**{100 * succ / len(tot_scored):.1f}** | "
            f"**{sum(r['steps'] for r in tot_scored) / len(tot_scored):.1f}** | "
            f"**{sum(r['duration_s'] for r in tot_scored) / len(tot_scored):.1f}** |")
    return "\n".join(lines)


def render_index(records: list[dict], repo_id: str) -> str:
    out = [
        f"All {len(records)} rollouts, grouped by task. Each link opens the "
        "browser-playable copy under `web/` on the Hub (you must be signed in — "
        "the repo is private).",
        "",
        "Verdicts come from `mark`, never the filename suffix. ⏱ marks a rollout that "
        "hit the step cap (a capped run is a failure, not a separate outcome), ⚠️ one "
        "whose filename suffix is stale. Runs the operator stopped and abandoned "
        "(`unsure`) are not evaluation data points and are not listed.",
        "",
    ]
    for task, group in _tasks(records):
        scored = [r for r in group if not r["unsure"]]
        succ = sum(r["success"] for r in scored)
        rate = f"{100 * succ / len(scored):.0f}%" if scored else "n/a"
        unsure = len(group) - len(scored)
        if unsure:
            rate += f", {unsure} unsure"
        out += [
            "<details>",
            f"<summary><b>{task}</b> — <i>{group[0]['prompt']}</i> — "
            f"<b>{succ}/{len(scored)}</b> ({rate})</summary>",
            "",
            "| rollout | verdict | steps | duration | video |",
            "|---|---|---:|---:|---|",
        ]
        for r in group:
            verdict = VERDICT_ICON.get(r["mark"], r["mark"])
            if r["timeout"]:
                verdict += " ⏱"
            if r["suffix_stale"]:
                verdict += " ⚠️"
            label = f"{r['layout']}_{r['rollout']}"
            url = f"https://huggingface.co/datasets/{repo_id}/blob/main/{r['file_name']}"
            out.append(f"| `{label}` | {verdict} | {r['steps']} | "
                       f"{r['duration_s']:.1f} s | [▶ play]({url}) |")
        out += ["", "</details>", ""]
    return "\n".join(out).rstrip()


def _delta(d: float) -> str:
    if abs(d) < 0.05:
        return "±0.0"
    return f"{'+' if d > 0 else '−'}{abs(d):.1f}"


def render_compare(records: list[dict], other: list[dict],
                   this_name: str, other_name: str) -> str:
    """Per-task SR of this checkpoint against another, over the shared tasks."""
    def by_task(rs: list[dict]) -> dict[str, list[dict]]:
        d: dict[str, list[dict]] = {}
        for r in rs:
            if not r["unsure"]:
                d.setdefault(r["task"], []).append(r)
        return d

    def sr(rs: list[dict]) -> float:
        return 100 * sum(r["success"] for r in rs) / len(rs)

    a, b = by_task(other), by_task(records)
    shared = sorted(set(a) & set(b))
    lines = [f"| task | {other_name} | {this_name} | \u0394 |",
             "|------|" + "-" * max(6, len(other_name)) + ":|"
             + "-" * max(6, len(this_name)) + ":|---:|"]
    tot_a: list[dict] = []
    tot_b: list[dict] = []
    for task in shared:
        ga, gb = a[task], b[task]
        tot_a += ga
        tot_b += gb
        lines.append(f"| {task} | {sr(ga):.1f} % (n={len(ga)}) | "
                     f"{sr(gb):.1f} % (n={len(gb)}) | {_delta(sr(gb) - sr(ga))} |")
    if tot_a and tot_b:
        lines.append(f"| **overlap** | **{sr(tot_a):.1f} % (n={len(tot_a)})** | "
                     f"**{sr(tot_b):.1f} % (n={len(tot_b)})** | "
                     f"**{_delta(sr(tot_b) - sr(tot_a))}** |")
    return "\n".join(lines)


def _gb(paths: list[str]) -> float:
    return sum(os.path.getsize(p) for p in paths if os.path.exists(p)) / 1e9


def render_summary(records: list[dict], src_gb: float) -> str:
    tasks = {r["task"] for r in records}
    scored = [r for r in records if not r["unsure"]]
    succ = sum(r["success"] for r in scored)
    unsure = len(records) - len(scored)
    dates = sorted({r["start_time"][:10] for r in records if r["start_time"]})
    span = dates[0] if len(dates) < 2 else f"{dates[0]} … {dates[-1]}"
    line = (f"**{len(records)} rollouts / {src_gb:.1f} GB** over **{len(tasks)} tasks** "
            f"({min(tasks)} … {max(tasks)}), ~2 rollouts per OOD layout, recorded {span}. "
            f"**{succ}/{len(scored)} succeeded ({100 * succ / len(scored):.1f}%)**")
    if unsure:
        line += f"; {unsure} marked `unsure` and excluded from the rates"
    return line + "."


def render_encodings(src_gb: float, web_gb: float) -> str:
    pct = 100 * web_gb / src_gb if src_gb else 0
    return "\n".join([
        "The recorder wrote **MPEG-4 Part 2** (`mp4v`, Simple Profile) with the `moov` atom",
        "*after* `mdat`. No browser decodes `mp4v`, and the trailing `moov` blocks progressive",
        "playback, so the original files will not play in the Hub preview, the dataset viewer,",
        "or any `<video>` element — they need VLC/ffmpeg.",
        "",
        "`web/` mirrors the tree with **H.264 High / yuv420p, `+faststart`, 2 s keyframes** at",
        f"the same native resolution and frame rate. Same frames, plays everywhere, ~{pct:.0f}% of the size.",
        "",
        "| | codec | size | plays in a browser |",
        "|---|---|---|---|",
        f"| `<task>-ood/…/<stem>.mp4` | `mp4v` (MPEG-4 Part 2) | {src_gb:.1f} GB | ❌ |",
        f"| `web/<task>-ood/…/<stem>.mp4` | `avc1` (H.264 High) | {web_gb:.1f} GB | ✅ |",
    ])


def scaffold(ckpt_dir: str, repo_id: str, license_id: str) -> str:
    """Full card for a checkpoint that has no README yet, markers included."""
    body = "\n".join([
        "---",
        f"license: {license_id}",
        f"pretty_name: FR3 OOD rollouts — {ckpt_dir}",
        "task_categories:",
        "  - robotics",
        "tags:",
        "  - robotics",
        "  - franka-fr3",
        "  - manipulation",
        "  - pi05",
        "  - openpi",
        "  - out-of-distribution",
        "  - real-robot",
        "size_categories:",
        "  - n<1K",
        "configs:",
        "  - config_name: default",
        "    drop_labels: true",
        "    data_files:",
        "      - split: train",
        "        path:",
        "          - metadata.jsonl",
        "          - web/**/*.mp4",
        "---",
        "",
        f"# {ckpt_dir} — OOD rollouts",
        "",
        "OOD (out-of-distribution layout) rollouts on the real Franka FR3 bench, recorded",
        "through the eval portal (`tasl/dashboards/openpi.py`).",
        "",
        "- **ckpt:** _TODO: describe the checkpoint and serving config._",
        "- **Coverage:** <!-- gen:summary --><!-- /gen:summary -->",
        "- **Files:** `<task>-ood/<layout>_rNN_<T|F|Q>/<same stem>.{mp4,traj.jsonl,frames.json,json}`",
        "  (one folder per rollout), mirroring `saved_demo/<task>-ood/" + ckpt_dir + "/` on disk",
        "  minus the checkpoint level.",
        "  - `.mp4` — recorded rollout video",
        "  - `.frames.json` — per-frame wall-clock timestamps",
        "  - `.traj.jsonl` — per-control-step record: joint state `q`, gripper, RTC scheduler",
        "    counters, inference latency, and the predicted action chunk",
        "  - `.json` — episode sidecar (task, prompt, layout, ckpt, timings, `steps`, `mark`)",
        "- **Verdict** comes from `mark` in the sidecar json, **not** the filename suffix.",
        "",
        "## Browsing the rollouts",
        "",
        "### Space",
        "",
        f"**{SPACE_URL}** — pick this dataset from the checkpoint selector, then filter by",
        "task / layout / verdict / timeout, search prompts, and watch each rollout beside its",
        "control trajectory.",
        "",
        "An index of every rollout with direct video links is at the bottom of this card.",
        "",
        "### Two encodings — use `web/` in a browser",
        "",
        "<!-- gen:encodings -->",
        "<!-- /gen:encodings -->",
        "",
        "`metadata.jsonl` points `file_name` at the `web/` copy, so the dataset viewer,",
        "`load_dataset` and the Space all get the playable one; `source_video` keeps the path",
        "to the original.",
        "",
        "### Locally",
        "",
        "```python",
        "from datasets import load_dataset",
        f'ds = load_dataset("{repo_id}", split="train")',
        "```",
        "",
        "## Stats",
        "",
        "`unsure` rollouts are excluded from `n` and `SR %`.",
        "",
        "<!-- gen:stats -->",
        "<!-- /gen:stats -->",
        "",
        "### vs another checkpoint",
        "",
        "Filled in when the tool is run with `--compare <other-ckpt-dir>`; the two evals are",
        "usually recorded on different days with the bench re-set between them, so read the",
        "\u0394 as a task-level trend, not a paired comparison.",
        "",
        "<!-- gen:compare -->",
        "<!-- /gen:compare -->",
        "",
        "## Reproducing",
        "",
        "```bash",
        f"python tasl/tools/publish_ood_rollouts.py {ckpt_dir}",
        "```",
        "",
        "## Rollout index",
        "",
        "<!-- gen:index -->",
        "<!-- /gen:index -->",
        "",
    ])
    return body


def splice(text: str, name: str, payload: str) -> str:
    """Replace the ``<!-- gen:name -->…<!-- /gen:name -->`` block."""
    open_t, close_t = f"<!-- gen:{name} -->", f"<!-- /gen:{name} -->"
    pattern = re.compile(re.escape(open_t) + r".*?" + re.escape(close_t), re.S)
    if not pattern.search(text):
        return text
    # A single-line payload (the coverage sentence) sits inline so it reads as part
    # of the surrounding bullet; multi-line payloads get their own block.
    repl = (f"{open_t}{payload}{close_t}" if "\n" not in payload
            else f"{open_t}\n{payload}\n{close_t}")
    # A lambda replacement keeps backslashes in the payload literal.
    return pattern.sub(lambda _: repl, text, count=1)


# ------------------------------------------------------------------ upload ---

def verify_remote(repo_id: str, jobs: int,
                  expected: set[str] | None = None) -> int:
    from huggingface_hub import HfApi, get_token
    api, token = HfApi(), get_token()
    files = api.list_repo_files(repo_id, repo_type="dataset")
    web = sorted(f for f in files if f.startswith("web/") and f.endswith(".mp4"))
    src = {f for f in files if not f.startswith("web/") and f.endswith(".mp4")}
    print(f"  remote: {len(src)} source mp4, {len(web)} web mp4")
    web_set = set(web)
    gaps = [s for s in src if f"web/{s}" not in web_set]
    orphans: list[str] = []
    if expected is not None:
        orphans = sorted(f for f in files
                         if f not in expected and not f.startswith("."))
        if orphans:
            print(f"  !! {len(orphans)} file(s) on the Hub no longer exist on disk "
                  f"(renamed or deleted rollouts), e.g. {orphans[0]}")
            print("     delete them with `hf repo file delete` — the upload never removes files")
    if gaps:
        print(f"  !! {len(gaps)} source video(s) have no web/ copy, first: {gaps[0]}")

    def check(path: str) -> tuple[str, bool]:
        url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{path}"
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {token}", "Range": "bytes=0-262143"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            head = resp.read()
        order, off = [], 0
        while off + 8 <= len(head) and len(order) < 4:
            size = struct.unpack(">I", head[off:off + 4])[0]
            order.append(head[off + 4:off + 8].decode("latin1", "replace"))
            if size == 1:
                size = struct.unpack(">Q", head[off + 8:off + 16])[0]
            if size < 8:
                break
            off += size
        ok = (len(order) >= 2 and order[0] == "ftyp" and order[1] == "moov"
              and b"avc1" in head and b"mp4v" not in head)
        return path, ok
    bad = []
    with cf.ThreadPoolExecutor(jobs) as ex:
        for path, ok in ex.map(check, web):
            if not ok:
                bad.append(path)
    for p in bad[:10]:
        print(f"  !! not H.264+faststart: {p}")
    print(f"  {len(web) - len(bad)}/{len(web)} web videos are H.264 High + faststart")
    return len(bad) + len(gaps) + len(orphans)


# -------------------------------------------------------------------- main ---

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt_dir", help="checkpoint folder name, e.g. cotrain-pbc-v2-8000")
    ap.add_argument("--root", default=SAVED_DEMO)
    ap.add_argument("--repo", help=f"default {REPO_ORG}/fr3-ood-rollouts-<ckpt_dir>")
    ap.add_argument("--stage", help="staging dir (default /tmp/hf_pub_<ckpt_dir>)")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--step-cap", type=int, default=1200,
                    help="eval-runner step cap; rollouts at the cap are timeouts")
    ap.add_argument("--license", default="cc-by-nc-4.0")
    ap.add_argument("--compare", metavar="CKPT_DIR",
                    help="fill the <!-- gen:compare --> block with a per-task SR "
                         "table against this other checkpoint on disk")
    ap.add_argument("--public", action="store_true",
                    help="create the repo public (no effect on an existing repo — "
                         "flip visibility in the Hub settings)")
    ap.add_argument("--no-upload", action="store_true", help="build locally, upload nothing")
    ap.add_argument("--verify-only", action="store_true", help="only re-check the Hub copy")
    ap.add_argument("--no-reuse", action="store_true",
                    help="re-encode and re-upload every web video even if the Hub "
                         "already holds a correct one (rewrites history; rarely wanted)")
    args = ap.parse_args()

    ckpt, root = args.ckpt_dir, args.root
    repo_id = args.repo or f"{REPO_ORG}/fr3-ood-rollouts-{ckpt}"
    stage = args.stage or f"/tmp/hf_pub_{ckpt}"

    if args.verify_only:
        print(f"[verify] {repo_id}")
        raise SystemExit(1 if verify_remote(repo_id, args.jobs) else 0)

    print(f"[scan] {root}/T?-?-ood/{ckpt}")
    records = sorted(build_records(ckpt, root, args.step_cap), key=sort_key)
    if not records:
        raise SystemExit(f"no rollouts found for {ckpt} under {root}")
    tasks = sorted({r["task"] for r in records})
    print(f"  {len(records)} rollouts over {len(tasks)} tasks: {', '.join(tasks)}")

    # `unsure` marks a run the operator stopped and abandoned — the policy never
    # got a verdict, so it is not an evaluation data point. A run that hit the
    # step cap is a *failure*, not an unsure, so nothing legitimate is lost here.
    # The files stay in the repo (and out of the orphan check); only the rows go.
    published = [r for r in records if not r["unsure"]]
    dropped_unsure = [r["stem"] for r in records if r["unsure"]]
    if dropped_unsure:
        print(f"  excluding {len(dropped_unsure)} abandoned (unsure) rollout(s) from "
              f"metadata/stats/index, files retained: {', '.join(dropped_unsure)}")
    mismarked = [r["stem"] for r in records if r["timeout"] and r["mark"] != "fail"]
    if mismarked:
        print(f"  !! {len(mismarked)} rollout(s) hit the {args.step_cap}-step cap but are "
              f"not marked fail: {', '.join(mismarked)}", file=sys.stderr)
        print("     a capped rollout never finished the task — re-mark it in the portal",
              file=sys.stderr)

    for flag in ("timeout", "suffix_stale", "unsure"):
        hits = [r["stem"] for r in records if r[flag]]
        if hits:
            print(f"  {flag}: {len(hits)} ({', '.join(hits[:4])}"
                  f"{', …' if len(hits) > 4 else ''})")

    reuse: set[str] = set()
    remote_sizes: dict[str, int] = {}
    if not args.no_reuse:
        print(f"[remote] what does {repo_id} already hold?")
        remote_sizes, remote_web_ok = remote_state(repo_id, args.jobs)
        reuse = {r["file_name"] for r in records if r["file_name"] in remote_web_ok}
        print(f"  {len(reuse)}/{len(records)} web videos already published and "
              f"correctly encoded — left untouched")

    print(f"[encode] -> {stage}/web  (crf 26, {args.jobs} jobs)")
    os.makedirs(stage, exist_ok=True)
    encode_web(records, root, ckpt, stage, args.jobs, skip=reuse)

    print("[assemble] staging tree")
    src_paths, web_paths = [], []
    for r in records:
        parts = r["source_video"].split("/")
        src_dir = os.path.join(root, parts[0], ckpt, *parts[1:-1])
        dst_dir = os.path.join(stage, *parts[:-1])
        os.makedirs(dst_dir, exist_ok=True)
        for key in ("source_video", "traj_file", "frames_file", "sidecar_file"):
            name = os.path.basename(r[key])
            src, link = os.path.join(src_dir, name), os.path.join(dst_dir, name)
            if os.path.islink(link) or os.path.exists(link):
                os.remove(link)
            os.symlink(src, link)
        src_paths.append(os.path.join(src_dir, os.path.basename(r["source_video"])))
        web_paths.append(os.path.join(stage, r["file_name"]))

    meta_path = os.path.join(stage, "metadata.jsonl")
    with open(meta_path, "w") as fh:
        for r in published:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    xlsx_name = f"{ckpt}_ood_stats.xlsx"
    # The workbook is part of the published stats, so it gets the same rows the
    # dataset does. (`ood_stats.py` run on its own still reports the abandoned
    # runs, which is what you want when auditing the raw recordings.)
    sidecars = [r for r in ood_stats.load_sidecars(ckpt, root)
                if r.get("mark") != "unsure"]
    ood_stats.write_xlsx(os.path.join(root, xlsx_name), [
        ("per-task", ood_stats.PER_TASK_COLS, ood_stats.per_task_rows(sidecars)),
        ("rollouts", ood_stats.ROLLOUT_COLS, ood_stats.rollout_rows(sidecars)),
    ])
    shutil.copy(os.path.join(root, xlsx_name), os.path.join(stage, xlsx_name))
    print(f"  metadata.jsonl ({len(published)} rows), {xlsx_name}")

    other_records: list[dict] = []
    if args.compare:
        other_records = [r for r in sorted(build_records(args.compare, root,
                                                            args.step_cap), key=sort_key)
                         if not r["unsure"]]
        if not other_records:
            print(f"  !! --compare {args.compare}: no rollouts on disk, "
                  f"comparison table will be empty", file=sys.stderr)

    print("[readme] regenerating blocks")
    card = os.path.join(root, f"README_{ckpt}_ood.md")
    text = open(card).read() if os.path.exists(card) else scaffold(ckpt, repo_id, args.license)
    changed: list[str] = []
    src_gb = _gb(src_paths)
    # Count each web video once: locally for the ones this run staged, and from
    # the remote tree for the ones already published (a stale local copy of a
    # reused video may still be sitting in the stage until prune_stage runs).
    web_gb = (_gb([p for r, p in zip(records, web_paths)
                   if r["file_name"] not in reuse])
              + sum(remote_sizes.get(p, 0) for p in reuse) / 1e9)
    blocks = [
        ("summary", render_summary(published, src_gb)),
        ("encodings", render_encodings(src_gb, web_gb)),
        ("stats", render_stats(published)),
        ("index", render_index(published, repo_id)),
    ]
    if args.compare:
        blocks.append(
            ("compare", render_compare(published, other_records, ckpt, args.compare)))
    for name, payload in blocks:
        # "unchanged" is the normal case on a re-run, so test for the marker
        # itself rather than inferring its absence from the text not moving.
        if f"<!-- gen:{name} -->" not in text:
            print(f"  !! no <!-- gen:{name} --> marker in {os.path.basename(card)}"
                  f" — block not updated", file=sys.stderr)
            continue
        before = text
        text = splice(text, name, payload)
        if text != before:
            changed.append(name)
    print(f"  updated: {', '.join(changed) if changed else 'nothing (already current)'}")
    with open(card, "w") as fh:
        fh.write(text)
    shutil.copy(card, os.path.join(stage, "README.md"))
    print(f"  {card}  ({src_gb:.2f} GB source -> {web_gb:.2f} GB web)")

    keep = expected_paths(records, ["README.md", "metadata.jsonl", xlsx_name]) - reuse
    dropped = prune_stage(stage, keep, root)
    if dropped:
        print(f"  pruned {len(dropped)} stale staged file(s), e.g. {dropped[0]}")

    if args.no_upload:
        print(f"[done] staged at {stage} (upload skipped)")
        return

    print(f"[upload] {repo_id}")
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=not args.public, exist_ok=True)
    # upload_folder is the resumable, multi-commit path in huggingface_hub >= 1.x
    # (it superseded upload_large_folder); it follows the symlinked originals and
    # skips blobs the Hub already has.
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=stage)

    print("[verify]")
    raise SystemExit(1 if verify_remote(repo_id, args.jobs, keep | reuse) else 0)


if __name__ == "__main__":
    main()
