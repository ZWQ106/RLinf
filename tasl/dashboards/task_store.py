"""Task store — the bench's task registry, shared by BOTH dashboards.

A *task* is the unit both portals now key on: one language prompt + one
layout stencil (placement mask) + the datasets collected for it. The eval
dashboard (openpi.py :8003) and the collection dashboard (collect.py :8004)
read and write the same ``tasl/tasks_store.json``, so a task defined at
collection time is immediately selectable for eval and vice versa.

Contract with the UIs: selecting an existing task locks the prompt and the
layout — free-form prompt entry and mask capture only happen when creating a
NEW task. That keeps every episode (collected or eval'd) attributable to a
registered task instead of an ad-hoc string.

Pure stdlib so it imports cleanly on the dev laptop (no flask/pyzed),
matching the layout_store.py convention.
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import re
import threading
import time

_log = logging.getLogger("task_store")

_TASL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS_PATH = os.path.join(_TASL_DIR, "tasks_store.json")
DATASET_ROOTS = [
    "/home/franka_desktop/lerobot_home",
    "/home/franka_desktop/rlinf_data/datasets",
]


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s or "task"


def _norm_layouts(rec: dict) -> dict:
    """Canonicalize a task's layout fields, in place.

    A task owns MANY layouts (`layouts`, in first-used order); `layout` is
    the most recently used one — the default the UIs arm/ghost on select.
    Legacy records carried a single `layout` string; fold it into the list.
    """
    lays = rec.get("layouts")
    if not isinstance(lays, list):
        lays = []
    lays = [s for s in (str(x).strip() for x in lays) if s]
    solo = str(rec.get("layout", "")).strip()
    if solo and solo not in lays:
        lays.append(solo)
    rec["layouts"] = lays
    rec["layout"] = solo or (lays[-1] if lays else "")
    return rec


def _discover_dataset_tasks() -> dict:
    """Scan local LeRobot dataset dirs for meta/tasks.jsonl →
    {dataset_name: [prompt, ...]}."""
    found: dict = {}
    for root in DATASET_ROOTS:
        root_p = pathlib.Path(root)
        if not root_p.exists():
            continue
        for f in root_p.rglob("tasks.jsonl"):
            if "/meta/" not in str(f):
                continue
            ds_dir = f.parent.parent
            parent = ds_dir.parent
            if str(parent).rstrip("/") == str(root_p).rstrip("/"):
                name = ds_dir.name
            else:
                name = f"{parent.name}/{ds_dir.name}"
            try:
                prompts = [json.loads(l).get("task", "")
                           for l in f.read_text().splitlines() if l.strip()]
            except Exception:
                continue
            prompts = [p for p in prompts if p]
            if prompts:
                seen = found.setdefault(name, [])
                for p in prompts:
                    if p not in seen:
                        seen.append(p)
    return found


class TaskStore:
    """JSON-backed CRUD store for bench tasks (tasks_store.json).

    Both dashboards hold their own instance over the same file, so every
    read goes through a freshness check: if the file changed on disk since
    we last touched it (the other portal wrote), reload before answering.
    """

    def __init__(self, path: str = TASKS_PATH):
        self.path = pathlib.Path(path)
        self._lock = threading.Lock()
        self._mtime = 0.0
        self.tasks: list = self._load()
        if not self.tasks:
            self._seed()
            self._save()

    def _disk_mtime(self) -> float:
        try:
            return self.path.stat().st_mtime
        except OSError:
            return 0.0

    def _load(self) -> list:
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text())
                self._mtime = self._disk_mtime()
                if isinstance(data, list):
                    return [_norm_layouts(t) for t in data
                            if isinstance(t, dict)]
        except Exception as exc:
            _log.warning(f"tasks store unreadable, starting fresh: {exc}")
        return []

    def _refresh(self) -> None:
        """Reload if the other dashboard wrote the file since our last I/O."""
        if self._disk_mtime() != self._mtime:
            fresh = self._load()
            if fresh:
                self.tasks = fresh

    def _seed(self) -> None:
        """First run: derive tasks from the datasets' own tasks.jsonl."""
        for ds, prompts in _discover_dataset_tasks().items():
            for p in prompts:
                tid = slugify(p)
                for t in self.tasks:
                    if t["id"] == tid:
                        if ds not in t["datasets"]:
                            t["datasets"].append(ds)
                        break
                else:
                    now = time.strftime("%Y-%m-%d %H:%M")
                    self.tasks.append(_norm_layouts({
                        "id": tid, "prompt": p, "layout": "",
                        "datasets": [ds], "created": now, "updated": now,
                    }))

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.tasks, ensure_ascii=False, indent=2))
        tmp.replace(self.path)
        self._mtime = self._disk_mtime()

    def list(self) -> list:
        with self._lock:
            self._refresh()
            return list(self.tasks)

    def get(self, tid: str) -> dict | None:
        with self._lock:
            self._refresh()
            for t in self.tasks:
                if t["id"] == tid:
                    return dict(t)
        return None

    def create(self, rec: dict) -> str:
        tid = str(rec.get("id") or slugify(str(rec.get("prompt", "")))).strip()
        if not tid:
            return "task id required"
        with self._lock:
            self._refresh()
            if any(t["id"] == tid for t in self.tasks):
                return f"task {tid} already exists — use update"
            now = time.strftime("%Y-%m-%d %H:%M")
            self.tasks.append(_norm_layouts({
                "id": tid,
                "prompt": str(rec.get("prompt", "")).strip(),
                "layout": str(rec.get("layout", "")).strip(),
                "layouts": rec.get("layouts") or [],
                "datasets": [s.strip() for s in rec.get("datasets", [])
                             if s.strip()],
                "created": now, "updated": now,
            }))
            self._save()
        return "created"

    def update(self, tid: str, rec: dict) -> str:
        with self._lock:
            self._refresh()
            for t in self.tasks:
                if t["id"] == tid:
                    if "prompt" in rec:
                        t["prompt"] = str(rec["prompt"]).strip()
                    if "layouts" in rec:
                        t["layouts"] = rec["layouts"]
                    if "layout" in rec:
                        t["layout"] = str(rec["layout"]).strip()
                    if "datasets" in rec:
                        t["datasets"] = [s.strip()
                                         for s in rec["datasets"] if s.strip()]
                    _norm_layouts(t)
                    t["updated"] = time.strftime("%Y-%m-%d %H:%M")
                    self._save()
                    return "updated"
        return f"task {tid} not found"

    def add_layout(self, tid: str, layout_id: str) -> str:
        """Record that task `tid` used layout `layout_id` (idempotent).

        Appends to the task's `layouts` and marks it as the most recent
        (`layout`), which is what the UIs arm/ghost by default next time.
        """
        layout_id = (layout_id or "").strip()
        if not layout_id:
            return "layout id required"
        with self._lock:
            self._refresh()
            for t in self.tasks:
                if t["id"] == tid:
                    lays = t.get("layouts") or []
                    changed = layout_id not in lays or t.get("layout") != layout_id
                    if layout_id not in lays:
                        lays.append(layout_id)
                    t["layouts"] = lays
                    t["layout"] = layout_id
                    if changed:
                        t["updated"] = time.strftime("%Y-%m-%d %H:%M")
                        self._save()
                    return "updated"
        return f"task {tid} not found"

    def add_dataset(self, tid: str, dataset: str) -> str:
        """Record that `dataset` was collected under task `tid` (idempotent)."""
        dataset = (dataset or "").strip()
        if not dataset:
            return "dataset name required"
        with self._lock:
            self._refresh()
            for t in self.tasks:
                if t["id"] == tid:
                    if dataset not in t["datasets"]:
                        t["datasets"].append(dataset)
                        t["updated"] = time.strftime("%Y-%m-%d %H:%M")
                        self._save()
                    return "updated"
        return f"task {tid} not found"

    def delete(self, tid: str) -> str:
        with self._lock:
            self._refresh()
            before = len(self.tasks)
            self.tasks = [t for t in self.tasks if t["id"] != tid]
            if len(self.tasks) == before:
                return f"task {tid} not found"
            self._save()
        return "deleted"
