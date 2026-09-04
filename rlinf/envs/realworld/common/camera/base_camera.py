# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CameraInfo:
    """Descriptor for a single camera device."""

    name: str
    serial_number: str
    camera_type: str = "realsense"
    resolution: tuple[int, int] = (640, 480)
    fps: int = 15
    enable_depth: bool = False
    crop_region: Optional[tuple[float, float, float, float]] = None
    # Optional live view: every captured frame is JPEG-encoded and published
    # on this ZeroMQ PUB address (multipart: camera name, jpeg bytes) so the
    # operator's (delayed) display can show the camera the env owns.
    stream_addr: Optional[str] = None
    stream_jpeg_quality: int = 75


class BaseCamera(ABC):
    """Abstract base class for threaded camera capture.

    Subclasses must implement ``_read_frame`` (hardware-specific frame
    acquisition) and ``_close_device`` (hardware-specific cleanup).
    The threading, queue management, and public API (``open``, ``close``,
    ``get_frame``) are handled here.
    """

    def __init__(self, camera_info: CameraInfo):
        self._camera_info = camera_info
        self._frame_queue: queue.Queue = queue.Queue()
        self._frame_capturing_thread = threading.Thread(
            target=self._capture_frames, daemon=True
        )
        self._frame_capturing_start = False
        self._stream_pub = None

    @property
    def name(self) -> str:
        return self._camera_info.name

    def open(self):
        """Start the background frame-capturing thread."""
        if self._camera_info.stream_addr:
            import zmq

            ctx = zmq.Context.instance()
            self._stream_pub = ctx.socket(zmq.PUB)
            self._stream_pub.setsockopt(zmq.SNDHWM, 2)
            self._stream_pub.setsockopt(zmq.LINGER, 0)
            # Several cameras share one address: bind once, others connect.
            try:
                self._stream_pub.bind(self._camera_info.stream_addr)
            except zmq.ZMQError:
                self._stream_pub.connect(self._camera_info.stream_addr)
        self._frame_capturing_start = True
        self._frame_capturing_thread.start()

    def close(self):
        """Stop the capture thread and release hardware resources."""
        self._frame_capturing_start = False
        if self._frame_capturing_thread.is_alive():
            self._frame_capturing_thread.join()
        if self._stream_pub is not None:
            self._stream_pub.close()
            self._stream_pub = None
        self._close_device()

    def get_frame(self, timeout: int = 5) -> np.ndarray:
        """Return the most recent frame (blocks up to *timeout* seconds).

        Args:
            timeout: Maximum seconds to wait for a frame.
        """
        assert self._frame_capturing_start, (
            "Frame capturing is not started. Call open() first."
        )
        return self._frame_queue.get(timeout=timeout)

    def start_recording(self, svo_path: str) -> None:
        """Begin recording the raw camera stream to a file. No-op by default;
        cameras that support it (ZED SVO) override this."""
        return None

    def stop_recording(self) -> None:
        """Stop recording started by :meth:`start_recording`. No-op by default."""
        return None

    # ── internal ──────────────────────────────────────────────────────

    # Cameras whose _read_frame blocks until the next frame (ZED grab) must
    # not sleep a frame period first: that adds up to 1/fps of staleness.
    BLOCKING_READ = False

    def _capture_frames(self):
        while self._frame_capturing_start:
            if not self.BLOCKING_READ:
                time.sleep(1 / self._camera_info.fps)
            has_frame, frame = self._read_frame()
            if not has_frame:
                break
            if not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self._frame_queue.put(frame)
            if self._stream_pub is not None:
                self._publish_frame(frame)

    def _publish_frame(self, frame: np.ndarray) -> None:
        import cv2
        import zmq

        ok, jpg = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._camera_info.stream_jpeg_quality]
        )
        if not ok:
            return
        try:
            # third part: capture wall time (s) so viewers can report frame age
            self._stream_pub.send_multipart(
                [self._camera_info.name.encode(), jpg.tobytes(), repr(time.time()).encode()],
                zmq.NOBLOCK,
            )
        except zmq.Again:
            pass

    @abstractmethod
    def _read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read a single frame from the camera hardware.

        Returns:
            ``(success, frame)`` where *frame* is a BGR ``uint8`` numpy array,
            or ``(False, None)`` on failure.
        """
        raise NotImplementedError

    @abstractmethod
    def _close_device(self) -> None:
        """Release hardware-specific resources (pipeline, SDK handle, …)."""
        raise NotImplementedError
