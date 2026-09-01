#!/usr/bin/env python3
"""Build OOD rollout stats (xlsx + metadata.jsonl) for one eval checkpoint.

Reads ``saved_demo/<task>-ood/<ckpt_dir>/<stem>/<stem>.json`` sidecars and writes
a two-sheet workbook (``per-task`` / ``rollouts``) plus a ``metadata.jsonl`` with
one sidecar per line.

The verdict is taken from the sidecar ``mark`` field, never from the filename
suffix (a re-marked episode can keep a stale ``_T``/``_F``/``_Q`` stem).
``unsure`` rollouts are excluded from ``n``/SR/step stats and reported in a
trailing ``unsure`` column.

Usage:
    python tasl/tools/ood_stats.py cotrain-pbc-v2-8000
    python tasl/tools/ood_stats.py pi05-droid-ft-15k --out /tmp/check.xlsx
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import zipfile
from typing import Any, Sequence

SAVED_DEMO = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "saved_demo")

PER_TASK_COLS = [
    "task", "prompt", "n", "success", "fail", "SR %",
    "steps mean", "steps min", "steps min rollout",
    "steps max", "steps max rollout",
    "time mean s", "time min s", "time max s",
    "succ steps mean", "fail steps mean", "unsure",
]
ROLLOUT_COLS = [
    "task", "layout", "rollout", "mark", "steps", "duration_s",
    "start_time", "ckpt", "ep_id", "file",
]


def load_sidecars(ckpt_dir: str, root: str = SAVED_DEMO) -> list[dict[str, Any]]:
    """Return every rollout sidecar for ``ckpt_dir``, sorted by path."""
    pattern = os.path.join(root, "T?-?-ood", ckpt_dir, "*", "*.json")
    out = []
    for path in sorted(glob.glob(pattern)):
        if path.endswith(".frames.json"):
            continue
        with open(path) as fh:
            rec = json.load(fh)
        rec["_relpath"] = os.path.relpath(path, root)
        out.append(rec)
    return out


def _mean(xs: Sequence[float]) -> str | float:
    return round(sum(xs) / len(xs), 1) if xs else ""


def per_task_rows(recs: list[dict[str, Any]]) -> list[list[Any]]:
    """One row per task plus a trailing ``ALL`` row."""
    tasks: dict[str, list[dict[str, Any]]] = {}
    for r in recs:
        tasks.setdefault(r["task"], []).append(r)

    def row(name: str, group: list[dict[str, Any]], prompt: str) -> list[Any]:
        scored = [r for r in group if r.get("mark") in ("success", "fail")]
        unsure = len(group) - len(scored)
        succ = [r for r in scored if r["mark"] == "success"]
        fail = [r for r in scored if r["mark"] == "fail"]
        steps = [r["steps"] for r in scored]
        times = [r["duration_s"] for r in scored]
        if not scored:
            return [name, prompt, 0, 0, 0, "", "", "", "", "", "", "", "", "", "", "", unsure]
        lo = min(scored, key=lambda r: r["steps"])
        hi = max(scored, key=lambda r: r["steps"])
        named = name != "ALL"
        return [
            name, prompt, len(scored), len(succ), len(fail),
            round(100 * len(succ) / len(scored), 1),
            _mean(steps), min(steps), lo["demo"]["stem"] if named else "",
            max(steps), hi["demo"]["stem"] if named else "",
            _mean(times), round(min(times), 1), round(max(times), 1),
            _mean([r["steps"] for r in succ]), _mean([r["steps"] for r in fail]),
            unsure,
        ]

    rows = [row(t, g, g[0].get("prompt", "")) for t, g in sorted(tasks.items())]
    rows.append(row("ALL", recs, ""))
    return rows


def rollout_rows(recs: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for r in recs:
        stem = r["demo"]["stem"]
        rows.append([
            r["task"], r.get("layout", ""), stem, r.get("mark", ""),
            r["steps"], r["duration_s"], r.get("start_time", ""),
            r.get("ckpt", ""), r.get("ep_id", ""),
            r["_relpath"][: -len(".json")] + ".mp4",
        ])
    return rows


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def _sheet_xml(header: list[str], rows: list[list[Any]]) -> str:
    def col(i: int) -> str:
        name = ""
        i += 1
        while i:
            i, rem = divmod(i - 1, 26)
            name = chr(65 + rem) + name
        return name

    out = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
           '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
           "<sheetData>"]
    for r_i, values in enumerate([header] + rows, start=1):
        out.append(f'<row r="{r_i}">')
        for c_i, v in enumerate(values):
            ref = f"{col(c_i)}{r_i}"
            if v == "" or v is None:
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out.append(f'<c r="{ref}"><v>{v}</v></c>')
            else:
                out.append(f'<c r="{ref}" t="inlineStr"><is><t>{_esc(v)}</t></is></c>')
        out.append("</row>")
    out.append("</sheetData></worksheet>")
    return "".join(out)


def write_xlsx(path: str, sheets: list[tuple[str, list[str], list[list[Any]]]]) -> None:
    """Minimal xlsx writer (inline strings, no styles) so no openpyxl is needed."""
    ns_r = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                   '<Default Extension="xml" ContentType="application/xml"/>'
                   '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                   + "".join(
                       f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
                       'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                       for i in range(1, len(sheets) + 1))
                   + "</Types>")
        z.writestr("_rels/.rels",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   f'<Relationship Id="rId1" Type="{ns_r}/officeDocument" Target="xl/workbook.xml"/>'
                   "</Relationships>")
        z.writestr("xl/workbook.xml",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                   f'xmlns:r="{ns_r}"><sheets>'
                   + "".join(f'<sheet name="{_esc(n)}" sheetId="{i}" r:id="rId{i}"/>'
                             for i, (n, _, _) in enumerate(sheets, 1))
                   + "</sheets></workbook>")
        z.writestr("xl/_rels/workbook.xml.rels",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   + "".join(f'<Relationship Id="rId{i}" Type="{ns_r}/worksheet" '
                             f'Target="worksheets/sheet{i}.xml"/>'
                             for i in range(1, len(sheets) + 1))
                   + "</Relationships>")
        for i, (_, header, rows) in enumerate(sheets, 1):
            z.writestr(f"xl/worksheets/sheet{i}.xml", _sheet_xml(header, rows))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpt_dir", help="checkpoint folder name, e.g. cotrain-pbc-v2-8000")
    ap.add_argument("--root", default=SAVED_DEMO)
    ap.add_argument("--out", help="xlsx path (default <root>/<ckpt_dir>_ood_stats.xlsx)")
    ap.add_argument("--metadata", help="metadata.jsonl path (default: next to --out)")
    args = ap.parse_args()

    recs = load_sidecars(args.ckpt_dir, args.root)
    if not recs:
        raise SystemExit(f"no rollouts found for {args.ckpt_dir} under {args.root}")

    out = args.out or os.path.join(args.root, f"{args.ckpt_dir}_ood_stats.xlsx")
    write_xlsx(out, [
        ("per-task", PER_TASK_COLS, per_task_rows(recs)),
        ("rollouts", ROLLOUT_COLS, rollout_rows(recs)),
    ])

    meta = args.metadata or os.path.join(os.path.dirname(out), "metadata.jsonl")
    with open(meta, "w") as fh:
        for r in recs:
            fh.write(json.dumps({k: v for k, v in r.items() if k != "_relpath"},
                                ensure_ascii=False) + "\n")

    print(f"{len(recs)} rollouts -> {out}")
    print(f"metadata -> {meta}")


if __name__ == "__main__":
    main()
