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

import numbers
import os
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Optional

import gymnasium as gym
import imageio
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from rlinf.envs.utils import put_info_on_image, tile_images


class RecordVideo(gym.Wrapper):
    """
    A general video recording wrapper that owns the recording logic.

    ``RecordVideo`` centralizes frame collection and MP4 writing for both regular
    stepping and chunked stepping APIs. Frames are buffered in memory and flushed
    asynchronously to avoid blocking environment interaction.

    The wrapper supports multiple observation image layouts (single frame, batched
    frames, and temporal batches). For ``chunk_step()``, it correctly handles the
    terminal-to-reset transition by recording terminal observations (for the last
    step in the chunk) and then appending the corresponding reset observations.

    When ``video_cfg.info_on_video`` is enabled, per-frame text metadata is drawn
    through ``put_info_on_image()``. The overlay always includes reward and
    termination when available, and can include extra fields from environment
    ``info`` via ``video_cfg.extra_info_on_video``. Nested keys are supported with
    dot notation, for example
    ``["env_id", "episode.success_once", "episode.episode_len"]``.

    Args:
        env: Wrapped environment. It must expose a ``seed`` attribute and may
            optionally provide ``num_envs`` and metadata for FPS inference.
        video_cfg: Video configuration object/dict. Common fields:
            ``video_base_dir`` (output directory root),
            ``fps`` (optional FPS override),
            ``info_on_video`` (whether to render overlay text),
            ``extra_info_on_video`` (list of ``info`` keys to render).
        fps: Explicit FPS override. If ``None``, FPS is resolved from
            ``video_cfg.fps``, environment config/metadata, then fallback ``30``.
    """

    def __init__(self, env: gym.Env, video_cfg, fps: Optional[int] = None):
        """Initialize the wrapper and set FPS/config."""
        if isinstance(env, gym.Env):
            super().__init__(env)
        else:
            self.env = env

        if not hasattr(env, "seed"):
            raise AttributeError("Environment must have 'seed' attribute")

        self.video_cfg = video_cfg
        self.render_images: list[np.ndarray] = []
        self.video_cnt = 0
        self._num_envs = getattr(env, "num_envs", 1)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._save_futures: list[Future] = []

        if fps is not None:
            self._fps = fps
        else:
            self._fps = self._get_fps_from_env(env)

    @property
    def is_start(self):
        return getattr(self.env, "is_start")

    @is_start.setter
    def is_start(self, value):
        setattr(self.env, "is_start", value)

    def _get_fps_from_env(self, env: gym.Env) -> int:
        """Resolve FPS from config/env metadata with fallback."""
        if hasattr(self.video_cfg, "fps") and self.video_cfg.fps is not None:
            return int(self.video_cfg.fps)
        if hasattr(env, "cfg") and hasattr(env.cfg, "init_params"):
            if hasattr(env.cfg.init_params, "sim_config"):
                if hasattr(env.cfg.init_params.sim_config, "control_freq"):
                    return int(env.cfg.init_params.sim_config.control_freq)
        metadata = getattr(env, "metadata", None)
        if isinstance(metadata, dict) and "render_fps" in metadata:
            return int(metadata["render_fps"])
        return 30

    def _to_numpy(self, value: Any) -> np.ndarray:
        """Convert tensors/arrays to numpy."""
        if torch is not None and isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value
        return np.array(value)

    def _combine_multiview(self, obs: Any) -> Optional[np.ndarray]:
        """Tile main_images + extra_view_images into one wide frame per env.

        RealWorldEnv._wrap_obs emits main_images (E, H, W, 3) and, when there is
        more than one camera, extra_view_images (E, K, H, W, 3). Concatenate them
        along width so the recorded frame shows [main | extra_0 | extra_1 ...].
        Returns None when the obs is not this multi-camera layout (single camera
        or a different obs shape) so the caller falls back to its normal keys.
        ``video_cfg.frames_bgr`` converts BGR->RGB (the real env stores BGR).
        """
        if not isinstance(obs, dict):
            return None
        main = obs.get("main_images")
        extra = obs.get("extra_view_images")
        if main is None or extra is None:
            return None
        main = self._to_numpy(main)
        extra = self._to_numpy(extra)
        if main.ndim == 3:  # (H, W, 3) -> add env axis
            main = main[None]
        if extra.ndim == 4:  # (K, H, W, 3) -> add env axis
            extra = extra[None]
        if main.ndim != 4 or extra.ndim != 5:
            return None  # unexpected layout; let the normal path handle it
        views = [main] + [extra[:, k] for k in range(extra.shape[1])]  # each (E,H,W,3)
        try:
            combined = np.concatenate(views, axis=2)  # tile along width
        except ValueError:
            return None  # mismatched sizes; don't crash recording
        if combined.dtype != np.uint8:
            combined = combined.astype(np.uint8)
        if self.video_cfg.get("frames_bgr", False) and combined.shape[-1] == 3:
            combined = combined[..., ::-1]
        return np.ascontiguousarray(combined)

    def _maybe_upscale(self, img: np.ndarray) -> np.ndarray:
        """Resize a frame up to ``video_cfg.video_height`` px tall (aspect
        preserved), for a larger, more viewable MP4. No-op when unset or the
        frame is already at least that tall. Source frames are small policy-view
        crops, so this is an upscale for viewing, not added detail."""
        target_h = self.video_cfg.get("video_height", None)
        if not target_h:
            return img
        if not isinstance(img, np.ndarray) or img.ndim < 2:
            return img
        h, w = img.shape[:2]
        target_h = int(target_h)
        if h <= 0 or h >= target_h:
            return img
        new_w = max(1, int(round(w * target_h / h)))
        try:
            import cv2

            return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
        except Exception:
            # Integer nearest-neighbour fallback if cv2 is unavailable.
            fy = max(1, target_h // h)
            fx = max(1, new_w // w)
            return np.repeat(np.repeat(img, fy, axis=0), fx, axis=1)

    def _get_image_from_dict(self, obs: dict) -> Optional[Any]:
        """Pick the best image field from an observation dict."""
        if hasattr(self.env, "capture_image"):
            return self.env.capture_image()
        # Real-robot multi-camera obs (RealWorldEnv._wrap_obs) splits cameras
        # into main_images (the policy's main view) + extra_view_images (the
        # rest). Tile ALL of them so the video shows every camera (e.g.
        # exterior | wrist), not just the main one. Falls through to the
        # single-image keys below when there is only one camera.
        multiview = self._combine_multiview(obs)
        if multiview is not None:
            return multiview
        for key in ("main_images", "images", "rgb", "full_image", "main_image"):
            if key in obs and obs[key] is not None:
                return obs[key]
        # Real-robot envs (e.g. Franka) expose per-camera images under a
        # ``frames`` dict {camera_name: HxWx3} rather than one of the keys
        # above. Tile every camera into a single panel so the recorded video
        # shows ALL views (e.g. exterior | wrist), not just the policy's main
        # image. ``video_cfg.frames_bgr`` converts BGR->RGB (the Franka env
        # stores BGR) so colors are correct in the MP4. Sorted names give a
        # stable left-to-right order (wrist_1 then wrist_2).
        frames = obs.get("frames") if isinstance(obs, dict) else None
        if isinstance(frames, dict) and frames:
            bgr = bool(self.video_cfg.get("frames_bgr", False))
            imgs: list[np.ndarray] = []
            for name in sorted(frames):
                img = frames[name]
                if img is None:
                    continue
                img = self._to_numpy(img)
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                if bgr and img.ndim == 3 and img.shape[-1] == 3:
                    img = img[..., ::-1]
                imgs.append(np.ascontiguousarray(img))
            if imgs:
                return imgs[0] if len(imgs) == 1 else tile_images(imgs, nrows=1)
        return None

    def _extract_frame_batches(self, obs: Any) -> list[list[np.ndarray]]:
        """Extract a list of per-step image batches from obs."""
        if obs is None:
            return []

        if isinstance(obs, dict):
            image_src = self._get_image_from_dict(obs)
            if image_src is None:
                return []
            return self._split_image_source(image_src)

        if isinstance(obs, (list, tuple)):
            if len(obs) == 0:
                return []
            if isinstance(obs[0], dict):
                frames = []
                for item in obs:
                    image_src = self._get_image_from_dict(item)
                    if image_src is None:
                        continue
                    batches = self._split_image_source(image_src)
                    if batches:
                        frames.append(batches[0])
                return frames
            images = []
            for item in obs:
                img = self._to_numpy(item)
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                images.append(img)
            return [images] if images else []

        if torch is not None and isinstance(obs, torch.Tensor):
            return self._split_image_source(obs)
        if isinstance(obs, np.ndarray):
            return self._split_image_source(obs)
        return []

    def _split_image_source(self, image_src: Any) -> list[list[np.ndarray]]:
        """Normalize common image tensor layouts into frame batches."""
        img = self._to_numpy(image_src)

        if img.ndim == 3:
            if img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
                img = np.transpose(img, (1, 2, 0))
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            return [[img]]

        if img.ndim == 4:
            if img.shape[1] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
                img = np.transpose(img, (0, 2, 3, 1))
            images = []
            for i in range(img.shape[0]):
                single = img[i]
                if single.dtype != np.uint8:
                    single = single.astype(np.uint8)
                images.append(single)
            return [images]

        if img.ndim == 5:
            if img.shape[2] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
                img = np.transpose(img, (0, 1, 3, 4, 2))
            frames = []
            for t in range(img.shape[1]):
                images = []
                for i in range(img.shape[0]):
                    single = img[i, t]
                    if single.dtype != np.uint8:
                        single = single.astype(np.uint8)
                    images.append(single)
                frames.append(images)
            return frames

        return []

    def _value_for_env(self, value: Any, env_id: int):
        """Select a scalar/value for a specific env from batched inputs."""
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            if value.shape == ():
                return value.item()
            if value.size == 1:
                return value.reshape(-1)[0].item()
            if value.shape[0] > env_id:
                return value[env_id]
            return value.reshape(-1)[0]
        if isinstance(value, (list, tuple)):
            if len(value) > env_id:
                return value[env_id]
            if len(value) > 0:
                return value[0]
        return value

    def _get_task_description(self, obs: Any, env_id: int):
        """Get task description from obs or env attribute."""
        if isinstance(obs, dict) and "task_descriptions" in obs:
            task_desc = obs["task_descriptions"]
            if isinstance(task_desc, (list, tuple)) and len(task_desc) > env_id:
                return task_desc[env_id]
            return task_desc[0] if isinstance(task_desc, (list, tuple)) else task_desc
        if hasattr(self.env, "task_descriptions"):
            task_desc = self.env.task_descriptions
            if isinstance(task_desc, (list, tuple)) and len(task_desc) > env_id:
                return task_desc[env_id]
            return task_desc[0] if isinstance(task_desc, (list, tuple)) else task_desc
        return None

    def _get_video_info_keys(self) -> list[str]:
        """Get configured info keys to overlay on video frames."""
        if hasattr(self.video_cfg, "extra_info_on_video"):
            keys = getattr(self.video_cfg, "extra_info_on_video")
        else:
            keys = None

        if keys:
            if isinstance(keys, str):
                return [keys]
            return list(keys)
        return []

    def _lookup_info_value(self, info: Any, key: str) -> Any:
        """Read a key from info, supporting dotted access for nested dicts."""
        if not isinstance(info, dict):
            return None
        if key in info:
            return info[key]

        value = info
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]
        return value

    def _build_info_item(
        self,
        infos: Optional[Any],
        rewards: Optional[Any],
        terminations: Optional[Any],
        env_id: int,
        time_idx: Optional[int] = None,
    ) -> dict:
        """Build a per-env info dict for overlay."""
        info_item: dict[str, Any] = {}

        if rewards is not None:
            value = self._value_for_env(rewards, env_id)
            if time_idx is not None and isinstance(value, (np.ndarray, list, tuple)):
                if len(value) > time_idx:
                    value = value[time_idx]
            info_item["reward"] = float(value) if value is not None else value

        if terminations is not None:
            value = self._value_for_env(terminations, env_id)
            if time_idx is not None and isinstance(value, (np.ndarray, list, tuple)):
                if len(value) > time_idx:
                    value = value[time_idx]
            info_item["termination"] = bool(value) if value is not None else value

        if infos is not None:
            for key in self._get_video_info_keys():
                value = self._lookup_info_value(infos, key)
                if value is None:
                    continue
                value = self._value_for_env(value, env_id)
                if isinstance(value, np.ndarray):
                    if value.shape == ():
                        value = value.item()
                    elif value.size == 1:
                        value = value.reshape(-1)[0].item()
                elif isinstance(value, numbers.Number):
                    pass
                else:
                    warnings.warn(f"Unsupported value type {type(value)} for key {key}")
                    continue
                info_item[key] = value

        return info_item

    def _append_frame(
        self,
        images: list[np.ndarray],
        infos: Optional[Any],
        rewards: Optional[Any],
        terminations: Optional[Any],
        time_idx: Optional[int] = None,
    ) -> None:
        """Overlay info (optional) and append a tiled frame."""
        if not images:
            return
        # Upscale BEFORE the info overlay so the reward/termination text is
        # drawn crisp at the final resolution (the source frames are the small
        # policy-view crops, e.g. 128²).
        images = [self._maybe_upscale(img) for img in images]
        if self.video_cfg.get("info_on_video", True):
            images = [
                put_info_on_image(
                    img,
                    self._build_info_item(
                        infos, rewards, terminations, env_id, time_idx
                    ),
                )
                for env_id, img in enumerate(images)
            ]
        if len(images) > 1:
            nrows = int(np.sqrt(len(images)))
            full_image = tile_images(images, nrows=nrows)
            self.render_images.append(full_image)
        else:
            self.render_images.append(images[0])

    def add_new_frames(
        self,
        obs: Any,
        infos: Optional[Any] = None,
        rewards: Optional[Any] = None,
        terminations: Optional[Any] = None,
    ):
        """Extract frames from obs and append to the buffer."""
        frames = self._extract_frame_batches(obs)
        if not frames:
            warnings.warn(
                f"Failed to extract images from obs, obs type: {type(obs)}, obs keys: "
                f"{list(obs.keys()) if isinstance(obs, dict) else 'N/A'}"
            )
            return

        if isinstance(infos, (list, tuple)):
            for time_idx, images in enumerate(frames):
                step_info = infos[time_idx] if len(infos) > time_idx else None
                self._append_frame(images, step_info, rewards, terminations, time_idx)
            return

        for time_idx, images in enumerate(frames):
            self._append_frame(images, infos, rewards, terminations, time_idx)

    def reset(self, *args, **kwargs):
        """Reset env and record the initial frame."""
        obs, info = self.env.reset(*args, **kwargs)
        self.add_new_frames(obs, info)
        return obs, info

    def step(self, action):
        """Step env and record the resulting frame."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        terminations = (
            info.get("terminations", terminated)
            if isinstance(info, dict)
            else terminated
        )
        self.add_new_frames(obs, info, reward, terminations)
        return obs, reward, terminated, truncated, info

    def record_video_in_result(self, result) -> None:
        """Record video frames from a chunk_step / async_chunk_step result tuple."""
        if isinstance(result, tuple) and len(result) >= 5:
            obs_list, rewards, terminations, _truncations, infos_list = result[:5]

            # Some envs may skip intermediate observations for performance and return
            # None entries. Filter them out for video collection.
            if isinstance(obs_list, (list, tuple)):
                valid_indices = [i for i, obs in enumerate(obs_list) if obs is not None]
                if len(valid_indices) == 0:
                    return
                if len(valid_indices) != len(obs_list):
                    obs_list = [obs_list[i] for i in valid_indices]
                    if isinstance(infos_list, (list, tuple)):
                        infos_list = [infos_list[i] for i in valid_indices]
                    if (
                        torch is not None
                        and isinstance(rewards, torch.Tensor)
                        and rewards.ndim == 2
                    ):
                        rewards = rewards[:, valid_indices]
                    if (
                        torch is not None
                        and isinstance(terminations, torch.Tensor)
                        and terminations.ndim == 2
                    ):
                        terminations = terminations[:, valid_indices]

            final_obs = None
            last_info = None
            if isinstance(infos_list, (list, tuple)) and len(infos_list) > 0:
                last_info = infos_list[-1]
                if isinstance(last_info, dict):
                    if last_info.get("final_obs") is not None:
                        final_obs = last_info["final_obs"]
                    elif last_info.get("final_observation") is not None:
                        final_obs = last_info["final_observation"]

            if (
                final_obs is not None
                and isinstance(obs_list, (list, tuple))
                and len(obs_list) > 0
            ):
                reset_obs = obs_list[-1]
                obs_main = list(obs_list)
                obs_main[-1] = final_obs
                infos_main = (
                    list(infos_list)
                    if isinstance(infos_list, (list, tuple))
                    else infos_list
                )
                self.add_new_frames(obs_main, infos_main, rewards, terminations)
                self.add_new_frames(reset_obs, None)
            else:
                self.add_new_frames(obs_list, infos_list, rewards, terminations)

    def chunk_step(self, *args, **kwargs):
        """Step a chunk and record all frames from the chunk."""
        result = self.env.chunk_step(*args, **kwargs)
        self.record_video_in_result(result)
        return result

    def flush_video(self, video_sub_dir: Optional[str] = None):
        """Write buffered frames to a fully-finalized MP4 file.

        Encodes on the background executor but BLOCKS until the write completes,
        so the file's moov atom is written before we return. flush_video is
        called at episode/rollout boundaries, so this added wait does not stall
        env stepping — and it guarantees a playable file even if the process is
        killed immediately afterwards (e.g. dashboard "Stop server" -> pkill -9,
        which previously left an unfinalized, unplayable MP4)."""
        if not self.render_images:
            return

        output_dir = os.path.join(
            self.video_cfg.video_base_dir, f"seed_{self.env.seed}"
        )
        if video_sub_dir is not None:
            output_dir = os.path.join(output_dir, f"{video_sub_dir}")

        os.makedirs(output_dir, exist_ok=True)
        mp4_path = os.path.join(output_dir, f"{self.video_cnt}.mp4")
        frames = list(self.render_images)
        self.render_images = []
        self.video_cnt += 1
        future = self._submit_save(frames, mp4_path)
        try:
            future.result()  # block until the MP4 is fully written + finalized
        except Exception as exc:
            warnings.warn(f"Video save did not finalize {mp4_path}: {exc}")

    def _submit_save(self, frames: list[np.ndarray], mp4_path: str) -> "Future":
        """Submit a background job to save the video; return its Future."""
        self._prune_futures()
        future = self._executor.submit(self._save_video, frames, mp4_path)
        self._save_futures.append(future)
        return future

    def _normalize_frames(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Coerce all frames to one constant, even (H, W), 3-channel uint8.

        H.264/yuv420p require every frame to share the same dimensions and even
        width/height. A stray differently-sized frame (e.g. a step where only
        the main camera was present) otherwise yields a broken/unplayable MP4.
        Target size is the first frame's, trimmed to even."""
        if not frames:
            return frames
        h0, w0 = frames[0].shape[:2]
        h0 -= h0 % 2
        w0 -= w0 % 2
        out: list[np.ndarray] = []
        for f in frames:
            f = np.asarray(f)
            if f.ndim == 2:
                f = np.stack([f] * 3, axis=-1)
            if f.ndim == 3 and f.shape[-1] == 4:
                f = f[..., :3]
            if f.shape[0] != h0 or f.shape[1] != w0:
                try:
                    import cv2

                    f = cv2.resize(f, (w0, h0), interpolation=cv2.INTER_LINEAR)
                except Exception:
                    f = f[:h0, :w0]
            if f.dtype != np.uint8:
                f = f.astype(np.uint8)
            out.append(np.ascontiguousarray(f))
        return out

    def _save_video(self, frames: list[np.ndarray], mp4_path: str) -> None:
        """Save frames to disk (runs in background) as H.264 (libx264)."""
        frames = self._normalize_frames(frames)
        if not frames:
            return
        video_writer = None
        try:
            # Explicit libx264 + yuv420p for a broadly compatible, playable MP4;
            # macro_block_size=1 keeps the exact (already-even) dimensions
            # instead of padding to a multiple of 16.
            video_writer = imageio.get_writer(
                mp4_path,
                fps=self._fps,
                codec="libx264",
                pixelformat="yuv420p",
                macro_block_size=1,
                output_params=["-crf", "20", "-preset", "medium"],
            )
            for img in frames:
                video_writer.append_data(img)
        except Exception as exc:
            warnings.warn(f"Failed to save video {mp4_path}: {exc}")
        finally:
            if video_writer is not None:
                video_writer.close()

    def _prune_futures(self) -> None:
        """Remove finished futures to avoid unbounded growth."""
        self._save_futures = [f for f in self._save_futures if not f.done()]

    def close(self):
        """Wait for pending video writes before closing."""
        self._executor.shutdown(wait=True)
        self._save_futures = []
        return super().close()

    def update_reset_state_ids(self):
        if hasattr(self.env, "update_reset_state_ids"):
            self.env.update_reset_state_ids()
