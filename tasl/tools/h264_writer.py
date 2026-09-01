#!/usr/bin/env python3
"""The house rollout-video format, in one place.

Every rollout video the bench produces — eval portal, data collection, and the
`web/` copies `publish_ood_rollouts.py` uploads — must be **H.264 High / yuv420p
with the `moov` atom ahead of `mdat`**. Anything else is unplayable where it
matters: no browser decodes MPEG-4 Part 2 (`mp4v`, which is what
`cv2.VideoWriter` writes by default), and a trailing `moov` blocks progressive
playback even for a stream the browser *can* decode. Videos in that shape render
as a blank pane in the Hub preview, the dataset viewer, and the rollout-browser
Space.

`H264Writer` is a drop-in for the `cv2.VideoWriter` API actually used by the
dashboards (`isOpened` / `write` / `release`), backed by an ffmpeg subprocess.
Encoding happens out of process, so it does not hold the GIL against the
recorder thread; measured on the FR3 desktop, `preset medium` encodes 2560x720
@15 fps at ~21x realtime, so it keeps up with live recording comfortably.

If ffmpeg is unavailable the writer falls back to `cv2.VideoWriter` and says so:
a rollout is expensive to re-run, so a wrong-format recording beats no recording.
`publish_ood_rollouts.py` re-encodes anything that slips through.
"""

from __future__ import annotations

import logging
import shutil
import subprocess

_log = logging.getLogger(__name__)

#: Encoder settings, recovered from the x264 SEI of the first correctly-published
#: dataset (`cotrain-pbc-v2-8000`) so every later checkpoint matches it exactly:
#: High profile, CRF 26, 2 s keyframes at 15 fps, faststart, no audio.
X264_ARGS = [
    "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p",
    "-crf", "26", "-preset", "medium", "-g", "30",
    "-movflags", "+faststart", "-an",
]

#: libx264 with yuv420p needs even dimensions; the wrist tile is resized by
#: aspect ratio and can land on an odd width, so trim at most one pixel.
EVEN_DIMS = ["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"]


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


class H264Writer:
    """Write BGR frames to a browser-playable H.264 mp4.

    Mirrors the slice of the ``cv2.VideoWriter`` API the dashboards use, so it
    drops into an existing recorder without touching the frame loop.
    """

    def __init__(self, path: str, fps: float, size: tuple[int, int]):
        """``size`` is ``(width, height)``, matching ``cv2.VideoWriter``."""
        self.path = str(path)
        self._proc: subprocess.Popen | None = None
        self._fallback = None
        width, height = int(size[0]), int(size[1])

        if ffmpeg_available():
            cmd = [
                "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-",
                *EVEN_DIMS, *X264_ARGS, self.path,
            ]
            try:
                self._proc = subprocess.Popen(
                    cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE)
            except Exception as exc:
                _log.error("ffmpeg spawn failed (%s) — falling back to cv2 mp4v", exc)
                self._proc = None
        else:
            _log.error("ffmpeg not on PATH — falling back to cv2 mp4v; "
                       "the recording will NOT be browser-playable")

        if self._proc is None:
            import cv2
            self._fallback = cv2.VideoWriter(
                self.path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    def isOpened(self) -> bool:                              # noqa: N802 (cv2 API)
        if self._proc is not None:
            return self._proc.poll() is None
        return bool(self._fallback is not None and self._fallback.isOpened())

    def write(self, frame) -> None:
        if self._fallback is not None:
            self._fallback.write(frame)
            return
        if self._proc is None or self._proc.stdin is None:
            return
        import numpy as np
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, ValueError):
            err = b""
            if self._proc.stderr is not None:
                try:
                    err = self._proc.stderr.read() or b""
                except Exception:
                    pass
            _log.error("ffmpeg died mid-recording: %s",
                       err.decode("utf-8", "replace").strip()[:400])
            self._proc = None

    def release(self) -> None:
        if self._fallback is not None:
            try:
                self._fallback.release()
            finally:
                self._fallback = None
            return
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass
        try:
            # faststart rewrites the file to move moov to the front, so give the
            # muxer room to finish rather than truncating a good recording.
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            _log.error("ffmpeg did not finish in 120s — killing; %s may be truncated",
                       self.path)
            proc.kill()
        if proc.returncode not in (0, None):
            err = b""
            if proc.stderr is not None:
                try:
                    err = proc.stderr.read() or b""
                except Exception:
                    pass
            _log.error("ffmpeg exited %s: %s", proc.returncode,
                       err.decode("utf-8", "replace").strip()[:400])
