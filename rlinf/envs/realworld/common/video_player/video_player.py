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

import os
import queue
import threading
import warnings

import cv2
import numpy as np


class VideoPlayer:
    def __init__(self, enable: bool = True, live_dir: str | None = None):
        # Bounded queue + drop-newest put so the control thread never blocks.
        self.queue = queue.Queue(maxsize=2)
        self.is_running = False
        self._live_dir = (
            live_dir
            or os.environ.get("RLINF_LIVE_CAM_DIR")
            or os.path.join(os.getcwd(), "outputs", "live_cam")
        )
        self._show = os.environ.get("DISPLAY") is not None
        if not enable:
            return
        os.makedirs(self._live_dir, exist_ok=True)
        self._run_thread = threading.Thread(target=self._play, daemon=True)
        self._run_thread.start()

    def put_frame(self, frame):
        # Called on the 15 Hz control thread — MUST NOT block.
        if not self.is_running:
            return
        try:
            self.queue.put_nowait(frame)
        except queue.Full:
            pass  # drop-newest: latency over backpressure

    def stop(self):
        # Signal the consumer thread to exit; best-effort, never blocks.
        self.is_running = False
        try:
            self.queue.put_nowait(None)  # sentinel handled by _play
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(None)
            except (queue.Empty, queue.Full):
                pass

    def _play(self):
        # Runs even when headless (no DISPLAY): the file-sink is the output.
        self.is_running = True
        enc = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        while True:
            img_array = self.queue.get()
            if img_array is None:  # sentinel to exit
                break
            try:
                for k, v in img_array.items():
                    if "full" in k or not hasattr(v, "shape") or getattr(v, "ndim", 0) != 3:
                        continue
                    ok, buf = cv2.imencode(".jpg", v, enc)  # frames are BGR (ZED native) -> encode as-is
                    if not ok:
                        continue
                    dst = os.path.join(self._live_dir, f"{k}.jpg")
                    tmp = dst + ".tmp"
                    with open(tmp, "wb") as f:
                        f.write(buf.tobytes())
                    os.replace(tmp, dst)  # atomic swap; reader never sees a partial file
                if self._show:
                    frame = np.concatenate(
                        [v for k, v in img_array.items() if "full" not in k], axis=0
                    )
                    cv2.imshow("Cameras", frame)
                    cv2.waitKey(1)
            except Exception as e:  # a bad frame must never kill the player thread
                warnings.warn(f"VideoPlayer frame error (kept running): {e!r}")
