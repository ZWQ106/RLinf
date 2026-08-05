"""Layout store — reproducible scene + pose setup for FR3 data collection.

A *layout* is a stencil pinned to the camera images, describing how the bench
must look before an episode is recorded. It is defined by BOTH cameras
together — one layout, two views:

  * ``exterior`` (ZED 2i)  — where the objects go. This is the scene layout.
  * ``wrist``    (ZED Mini, eye-in-hand) — what the gripper should be looking
    at when the episode starts. Because this camera moves WITH the arm, its
    stencil pins the starting POSE rather than the object placement, which is
    what the release phase of the two-stage start gate is for.

Each view carries:
  * a grid (default 3x3, the 九宫格) drawn over the live preview,
  * labelled markers in **normalized [0,1] image coordinates** — continuous,
    never snapped to the grid, so an object may sit anywhere inside a cell;
    normalized so the stencil survives a resolution or preview-size change,
  * a reference JPEG snapshot, replayed as a semi-transparent ghost during
    layout-prepare so the operator can align the real bench against it.

Reproducibility is the whole point: the layout id + a content hash over BOTH
views are recorded into the dataset (see `dataset_layout_entry`), so a dataset
always says which physical arrangement produced it.

Pure stdlib and no I/O beyond the layout directory — this module must import
cleanly on the dev laptop (no flask/pyzed) so it stays unit-testable, matching
the convention used by the log parser in `collect.py`.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from typing import Optional

# Layout ids double as filenames — keep them shell- and path-safe, same
# character class the dashboard already enforces for dataset names.
_LAYOUT_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

# The two bench cameras, in display order. A layout always carries both keys;
# a view with no markers is legitimate (the grid alone is still a reference).
VIEWS = ("exterior", "wrist")

MAX_OBJECTS = 32
MAX_LABEL_LEN = 40
MIN_GRID, MAX_GRID = 1, 12
DEFAULT_GRID = {"rows": 3, "cols": 3}

SCHEMA_VERSION = 2


class LayoutError(ValueError):
    """Invalid layout id or payload. Carries an operator-readable message."""


def valid_layout_id(name: str) -> bool:
    return bool(_LAYOUT_ID_RE.match(name or ""))


def valid_view(view: str) -> bool:
    return view in VIEWS


def _require_id(layout_id: str) -> str:
    if not valid_layout_id(layout_id):
        raise LayoutError(
            "invalid layout id (allowed: letters, digits, . _ -, max 64)")
    return layout_id


def _require_view(view: str) -> str:
    if not valid_view(view):
        raise LayoutError(f"unknown view '{view}' (expected one of {list(VIEWS)})")
    return view


def _clamp01(v: object, field: str) -> float:
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise LayoutError(f"{field} must be a number in [0,1]") from None
    if f != f:  # NaN
        raise LayoutError(f"{field} must be a number in [0,1]")
    return min(1.0, max(0.0, f))


def normalize_grid(raw: Optional[dict]) -> dict:
    """Validate/clamp a grid spec. Missing or partial input falls back to 3x3."""
    raw = raw or {}
    out = {}
    for key in ("rows", "cols"):
        val = raw.get(key, DEFAULT_GRID[key])
        try:
            n = int(val)
        except (TypeError, ValueError):
            raise LayoutError(f"grid.{key} must be an integer") from None
        if not (MIN_GRID <= n <= MAX_GRID):
            raise LayoutError(
                f"grid.{key} must be between {MIN_GRID} and {MAX_GRID}")
        out[key] = n
    return out


def normalize_objects(raw: object, where: str = "objects") -> list[dict]:
    """Validate a marker list into canonical [{label, x, y}] form.

    Coordinates are clamped rather than rejected: the UI nudges markers with
    the arrow keys and dragging one off the edge should pin it to the border,
    not throw away the edit. Positions stay continuous — nothing snaps to the
    grid, which is a visual reference only.
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise LayoutError(f"{where} must be a list")
    if len(raw) > MAX_OBJECTS:
        raise LayoutError(f"too many objects in {where} (max {MAX_OBJECTS})")
    out: list[dict] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise LayoutError(f"{where}[{i}] must be an object")
        label = str(item.get("label", "")).strip()
        if not label:
            raise LayoutError(f"{where}[{i}].label is required")
        if len(label) > MAX_LABEL_LEN:
            raise LayoutError(
                f"{where}[{i}].label too long (max {MAX_LABEL_LEN})")
        out.append({
            "label": label,
            "x": round(_clamp01(item.get("x"), f"{where}[{i}].x"), 5),
            "y": round(_clamp01(item.get("y"), f"{where}[{i}].y"), 5),
        })
    return out


def normalize_views(raw: object) -> dict:
    """Canonical {view: {grid, objects}} for every known view.

    Accepts the flat single-view shape ({grid, objects}) as the exterior view
    so a hand-written or older layout still loads.
    """
    raw = raw or {}
    if not isinstance(raw, dict):
        raise LayoutError("views must be an object")
    flat = "grid" in raw or "objects" in raw
    out = {}
    for view in VIEWS:
        if flat:
            spec = raw if view == VIEWS[0] else {}
        else:
            spec = raw.get(view) or {}
        if not isinstance(spec, dict):
            raise LayoutError(f"views.{view} must be an object")
        out[view] = {
            "grid": normalize_grid(spec.get("grid")),
            "objects": normalize_objects(spec.get("objects"), f"views.{view}.objects"),
        }
    return out


def layout_hash(views: dict) -> str:
    """Content hash over the whole stencil — BOTH views.

    A layout is defined by the two cameras jointly, so moving a marker in
    either one is a different layout. Deliberately excludes snapshots and
    timestamps: the same markers in the same places are the same layout even
    if re-photographed under different lighting.
    """
    canon = json.dumps({v: views[v] for v in VIEWS},
                       sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


# A reference snapshot must be an actual view of the bench. The live-camera
# source hands back an 8x8 placeholder whenever a feed is stale or absent, and
# silently storing that would give the operator a grey square to align against
# — worse than having no ghost at all, because the UI would claim one exists.
MIN_SNAPSHOT_W, MIN_SNAPSHOT_H = 160, 120


def jpeg_dimensions(buf: Optional[bytes]) -> Optional[tuple]:
    """(width, height) from a JPEG's SOF segment, or None if unparseable."""
    if not buf or not buf.startswith(b"\xff\xd8"):
        return None
    i, n = 2, len(buf)
    while i + 3 < n:
        if buf[i] != 0xFF:
            i += 1
            continue
        marker = buf[i + 1]
        if marker == 0xFF:            # fill byte
            i += 1
            continue
        if marker == 0x01 or 0xD0 <= marker <= 0xD8:   # standalone markers
            i += 2
            continue
        if marker == 0xD9:            # EOI before any SOF
            return None
        seg = int.from_bytes(buf[i + 2:i + 4], "big")
        if seg < 2:
            return None
        # SOF0..SOF15 carry the frame size; C4/C8/CC are DHT/JPG/DAC.
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if i + 9 > n:
                return None
            return (int.from_bytes(buf[i + 7:i + 9], "big"),
                    int.from_bytes(buf[i + 5:i + 7], "big"))
        i += 2 + seg
    return None


def is_usable_snapshot(buf: Optional[bytes]) -> bool:
    dims = jpeg_dimensions(buf)
    return bool(dims and dims[0] >= MIN_SNAPSHOT_W and dims[1] >= MIN_SNAPSHOT_H)


def _atomic_write(path: str, data: bytes) -> None:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


class LayoutStore:
    """CRUD over a directory of `<id>.json` (+ `<id>.<view>.jpg`) layouts."""

    def __init__(self, root: str):
        self.root = root

    # -- paths ---------------------------------------------------------
    def json_path(self, layout_id: str) -> str:
        return os.path.join(self.root, _require_id(layout_id) + ".json")

    def snapshot_path(self, layout_id: str, view: str) -> str:
        return os.path.join(
            self.root, f"{_require_id(layout_id)}.{_require_view(view)}.jpg")

    def _read_snapshot_paths(self, layout_id: str, view: str) -> list:
        """Where a snapshot may live, newest naming first.

        Before the layout gained a second view there was a single `<id>.jpg`
        holding the exterior shot. Fall back to it on read so a layout authored
        then keeps its ghost; the next save writes the current name.
        """
        paths = [self.snapshot_path(layout_id, view)]
        if view == VIEWS[0]:
            paths.append(os.path.join(self.root, f"{layout_id}.jpg"))
        return paths

    def exists(self, layout_id: str) -> bool:
        return os.path.isfile(self.json_path(layout_id))

    # -- read ----------------------------------------------------------
    def load(self, layout_id: str) -> dict:
        path = self.json_path(layout_id)
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except FileNotFoundError:
            raise LayoutError(f"layout '{layout_id}' not found") from None
        except (OSError, json.JSONDecodeError) as e:
            raise LayoutError(f"layout '{layout_id}' unreadable: {e}") from None
        if not isinstance(raw, dict):
            raise LayoutError(f"layout '{layout_id}' is not an object")
        views = normalize_views(raw.get("views") or raw)
        return {
            "id": layout_id,
            "schema_version": raw.get("schema_version", SCHEMA_VERSION),
            "views": views,
            "created_at": raw.get("created_at"),
            "updated_at": raw.get("updated_at"),
            "note": str(raw.get("note", ""))[:200],
            "hash": layout_hash(views),
            "has_snapshot": {
                v: any(os.path.isfile(q)
                       for q in self._read_snapshot_paths(layout_id, v))
                for v in VIEWS
            },
        }

    def list(self) -> list[dict]:
        """Summaries of every readable layout, newest-updated first.

        Unreadable files are surfaced with an `error` rather than dropped —
        a layout silently vanishing from the picker is worse than a bad row.
        """
        try:
            names = os.listdir(self.root)
        except OSError:
            return []
        out: list[dict] = []
        for fn in names:
            if not fn.endswith(".json") or fn.startswith("."):
                continue
            layout_id = fn[:-len(".json")]
            if not valid_layout_id(layout_id):
                continue
            try:
                lay = self.load(layout_id)
            except LayoutError as e:
                out.append({"id": layout_id, "error": str(e)})
                continue
            out.append({
                "id": layout_id,
                "n_objects": {v: len(lay["views"][v]["objects"]) for v in VIEWS},
                "total_objects": sum(
                    len(lay["views"][v]["objects"]) for v in VIEWS),
                "hash": lay["hash"],
                "created_at": lay["created_at"],
                "updated_at": lay["updated_at"],
                "has_snapshot": lay["has_snapshot"],
            })
        out.sort(key=lambda d: (d.get("updated_at") or d.get("created_at") or "",
                                d["id"]), reverse=True)
        return out

    def snapshot_bytes(self, layout_id: str, view: str) -> Optional[bytes]:
        for path in self._read_snapshot_paths(layout_id, view):
            try:
                with open(path, "rb") as f:
                    return f.read()
            except OSError:
                continue
        return None

    # -- write ---------------------------------------------------------
    def save(self, layout_id: str, views: object, note: str = "",
             snapshots: Optional[dict] = None) -> dict:
        """Create or overwrite a layout.

        `snapshots` maps view -> JPEG bytes; a view left out keeps whatever
        snapshot it already had, so re-nudging markers never drops a ghost.
        """
        _require_id(layout_id)
        norm = normalize_views(views)
        snaps = {}
        for view, buf in (snapshots or {}).items():
            _require_view(view)
            if buf is None:
                continue
            if not is_usable_snapshot(buf):
                raise LayoutError(
                    f"{view} 的参考快照不是一张可用的相机画面（可能是占位帧）— 未保存")
            snaps[view] = buf
        prev_created = None
        if self.exists(layout_id):
            try:
                prev_created = self.load(layout_id).get("created_at")
            except LayoutError:
                prev_created = None
        now = _iso_now()
        doc = {
            "schema_version": SCHEMA_VERSION,
            "id": layout_id,
            "views": norm,
            "note": str(note or "")[:200],
            "created_at": prev_created or now,
            "updated_at": now,
            "hash": layout_hash(norm),
        }
        _atomic_write(self.json_path(layout_id),
                      json.dumps(doc, indent=2, ensure_ascii=False).encode("utf-8"))
        for view, buf in snaps.items():
            _atomic_write(self.snapshot_path(layout_id, view), buf)
        return self.load(layout_id)

    def delete(self, layout_id: str) -> str:
        _require_id(layout_id)
        if not self.exists(layout_id):
            raise LayoutError(f"layout '{layout_id}' not found")
        os.unlink(self.json_path(layout_id))
        for view in VIEWS:
            for path in self._read_snapshot_paths(layout_id, view):
                try:
                    os.unlink(path)
                except OSError:
                    pass
        return f"deleted layout '{layout_id}'"


# ─────────────────────────────────────────────────────────────────────
# Dataset metadata — what gets written into datasets/<name>/meta/layout.json
# ─────────────────────────────────────────────────────────────────────
def dataset_layout_entry(layout: dict, task: str, num_episodes: int,
                         applied_at: Optional[str] = None) -> dict:
    """One collection run's worth of layout provenance (both views)."""
    return {
        "applied_at": applied_at or _iso_now(),
        "layout_id": layout["id"],
        "layout_hash": layout["hash"],
        "views": layout["views"],
        "task": task,
        "requested_episodes": num_episodes,
    }


def merge_dataset_layout(existing: Optional[dict], entry: dict) -> dict:
    """Append a run entry to a dataset's layout metadata.

    Datasets are collected across several runs (Start pressed more than once),
    so this accumulates a `runs` list instead of overwriting. `layout_id` at the
    top level reflects the most recent run; `layout_ids` lists every distinct
    layout the dataset has ever been collected under, which is the flag you
    want when a dataset accidentally mixes arrangements.
    """
    doc = dict(existing or {})
    runs = list(doc.get("runs") or [])
    runs.append(entry)
    ids: list[str] = []
    for r in runs:
        rid = r.get("layout_id")
        if rid and rid not in ids:
            ids.append(rid)
    doc.update({
        "schema_version": SCHEMA_VERSION,
        "layout_id": entry["layout_id"],
        "layout_hash": entry["layout_hash"],
        "applied_at": entry["applied_at"],
        "layout_ids": ids,
        "mixed_layouts": len(ids) > 1,
        "runs": runs,
    })
    return doc


def write_dataset_layout(dataset_dir: str, entry: dict) -> str:
    """Merge `entry` into `<dataset_dir>/meta/layout.json`. Returns the path.

    Caller must ensure `dataset_dir` already exists — writing into a dataset
    directory before LeRobot creates it would make `LeRobotDataset.create`
    trip over a non-empty target.
    """
    meta_dir = os.path.join(dataset_dir, "meta")
    path = os.path.join(meta_dir, "layout.json")
    existing = None
    try:
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except (OSError, json.JSONDecodeError):
        existing = None
    doc = merge_dataset_layout(existing if isinstance(existing, dict) else None,
                               entry)
    _atomic_write(path, json.dumps(doc, indent=2,
                                   ensure_ascii=False).encode("utf-8"))
    return path
