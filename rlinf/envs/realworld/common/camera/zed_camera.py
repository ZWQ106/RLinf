# Copyright 2025 The RLinf Authors.
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

"""ZED camera capture using the Stereolabs ZED SDK (pyzed).

Requires the ZED SDK to be installed system-wide (not pip-installable);
the ``pyzed`` Python bindings are bundled with the SDK.
"""

import threading
import time
from typing import Optional

import numpy as np

from rlinf.utils.logging import get_logger

from .base_camera import BaseCamera, CameraInfo

_logger = get_logger()


class ZEDCamera(BaseCamera):
    """Camera capture for Stereolabs ZED cameras.

    The interface is identical to :class:`RealSenseCamera`: ``open``,
    ``close``, ``get_frame`` return BGR ``uint8`` numpy arrays.

    ZED cameras output BGRA by default; this class strips the alpha channel
    to produce BGR, consistent with the RealSense pipeline.
    """
    BLOCKING_READ = True  # sl.Camera.grab() waits for the next frame


    def __init__(self, camera_info: CameraInfo):
        import pyzed.sl as sl

        super().__init__(camera_info)
        self._sl = sl

        self._camera = sl.Camera()

        init_params = sl.InitParameters()
        init_params.set_from_serial_number(int(camera_info.serial_number))
        init_params.camera_resolution = self._find_closest_resolution(
            camera_info.resolution
        )
        init_params.camera_fps = camera_info.fps

        if camera_info.enable_depth:
            init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        else:
            init_params.depth_mode = sl.DEPTH_MODE.NONE

        status = self._open_with_retry(
            self._camera, init_params, sl, camera_info.serial_number
        )
        if status == sl.ERROR_CODE.SUCCESS:
            pass
        elif "CALIBRATION" in str(status):
            _logger.warning(
                "ZED camera (serial=%s) opened with warning: %s. "
                "Run ZED Calibration tool to resolve.",
                camera_info.serial_number,
                status,
            )
        else:
            raise RuntimeError(
                f"Failed to open ZED camera "
                f"(serial={camera_info.serial_number}) after retries: {status}"
            )

        self._image = sl.Mat()
        self._depth = sl.Mat() if camera_info.enable_depth else None
        self._runtime_params = sl.RuntimeParameters()
        self._recording = False
        # Serializes camera SDK calls: the background grab thread vs.
        # enable/disable_recording on the control thread. disable_recording
        # racing an in-flight grab() corrupts the SVO trailer ("INVALID SVO
        # FILE"), which only shows up on long episodes (more grabs = more race).
        self._cam_lock = threading.Lock()

    def _read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        # Hold _cam_lock across grab + retrieve so a concurrent
        # enable/disable_recording can't corrupt the in-flight SVO write.
        with self._cam_lock:
            if self._camera.grab(self._runtime_params) != self._sl.ERROR_CODE.SUCCESS:
                return False, None
            self._camera.retrieve_image(self._image, self._sl.VIEW.LEFT)
            # BGRA → BGR (strip alpha to match RealSense output)
            frame = self._image.get_data()[:, :, :3].copy()
            if self._depth is not None:
                self._camera.retrieve_measure(self._depth, self._sl.MEASURE.DEPTH)
                depth_data = self._depth.get_data()
                depth = np.expand_dims(depth_data, axis=2)
                return True, np.concatenate((frame, depth), axis=-1)
        return True, frame

    def _close_device(self) -> None:
        self._camera.close()

    def start_recording(self, svo_path: str) -> None:
        """Record the raw HD camera stream to an SVO2 file (H264). Subsequent
        background grab() calls write frames automatically until stop_recording."""
        rp = self._sl.RecordingParameters(
            svo_path, self._sl.SVO_COMPRESSION_MODE.H264
        )
        with self._cam_lock:
            status = self._camera.enable_recording(rp)
        if status != self._sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(
                f"ZED enable_recording failed for {svo_path}: {status}"
            )
        self._recording = True

    def stop_recording(self) -> None:
        if getattr(self, "_recording", False):
            # Lock so disable_recording finalizes the SVO trailer without a
            # concurrent grab() corrupting it.
            with self._cam_lock:
                self._camera.disable_recording()
            self._recording = False

    @staticmethod
    def _open_with_retry(camera, init_params, sl, serial, attempts=3, settle_s=2.0):
        """Open the ZED, retrying on transient failures.

        Opening right after another process (e.g. the idle dashboard preview)
        releases the camera can hit a transient "CAMERA MOTION SENSORS NOT
        DETECTED" / "opening timeout reached" until the USB device is fully
        reacquirable — more likely at HD1080. Retry a few times with a settle
        delay; a CALIBRATION warning is treated as success (handled by caller)."""
        status = None
        for attempt in range(1, attempts + 1):
            status = camera.open(init_params)
            if status == sl.ERROR_CODE.SUCCESS or "CALIBRATION" in str(status):
                return status
            if attempt < attempts:
                _logger.warning(
                    "ZED open attempt %d/%d (serial=%s) failed: %s; retry in %.0fs",
                    attempt, attempts, serial, status, settle_s,
                )
                camera.close()
                time.sleep(settle_s)
        return status

    _resolution_map: Optional[dict] = None

    @staticmethod
    def _build_resolution_map() -> dict:
        """Lazily build and cache the ZED resolution lookup table."""
        if ZEDCamera._resolution_map is None:
            import pyzed.sl as sl

            ZEDCamera._resolution_map = {
                (2208, 1242): sl.RESOLUTION.HD2K,
                (1920, 1080): sl.RESOLUTION.HD1080,
                (1280, 720): sl.RESOLUTION.HD720,
                (672, 376): sl.RESOLUTION.VGA,
            }
        return ZEDCamera._resolution_map

    @staticmethod
    def _find_closest_resolution(target: tuple[int, int]):
        """Map the requested ``(width, height)`` to the nearest ZED preset."""
        resolution_map = ZEDCamera._build_resolution_map()
        best = None
        best_dist = float("inf")
        for (w, h), res_enum in resolution_map.items():
            dist = abs(w - target[0]) + abs(h - target[1])
            if dist < best_dist:
                best_dist = dist
                best = res_enum
        return best

    @staticmethod
    def get_device_serial_numbers() -> list[str]:
        """Return serial numbers of all connected ZED cameras."""
        try:
            import pyzed.sl as sl
        except ImportError:
            return []
        devices = sl.Camera.get_device_list()
        return [str(dev.serial_number) for dev in devices]
