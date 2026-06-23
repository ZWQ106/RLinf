import json
import os

import gymnasium as gym
import numpy as np

from rlinf.envs.wrappers.collect_episode import CollectEpisode


class _FakeEnv(gym.Env):
    """Minimal gym env exposing the recording hooks."""

    def start_recording(self, svo_dir, tag):
        os.makedirs(svo_dir, exist_ok=True)
        rec = {
            "wrist_1": os.path.join(svo_dir, f"{tag}_wrist_1.svo2"),
            "wrist_2": os.path.join(svo_dir, f"{tag}_wrist_2.svo2"),
        }
        for p in rec.values():
            open(p, "wb").write(b"svo")
        return rec

    def stop_recording(self):
        pass

    def reset(self, *, seed=None, options=None):
        return {"state": np.zeros(8)}, {}


def _ce(tmp_path):
    save_dir = os.path.join(str(tmp_path), "run", "collected_data")
    os.makedirs(save_dir, exist_ok=True)
    ce = CollectEpisode(
        _FakeEnv(), save_dir=save_dir, export_format="pickle",
        only_success=True, record_svo=True,
    )
    return ce, os.path.join(str(tmp_path), "run", "svo")


def test_success_renames_svo_to_episode_index(tmp_path):
    ce, svo_dir = _ce(tmp_path)
    ce.reset()
    ce._finalize_svo(env_idx=0, kept=True, episode_index=0)
    files = set(os.listdir(svo_dir))
    assert {
        "episode_000000_wrist_1.svo2",
        "episode_000000_wrist_2.svo2",
    }.issubset(files)
    idx = json.load(open(os.path.join(svo_dir, "svo_index.json")))
    assert "0" in idx and len(idx["0"]) == 2
    assert set(idx["0"]) == {
        "episode_000000_wrist_1.svo2",
        "episode_000000_wrist_2.svo2",
    }


def test_success_renames_svo_with_nonzero_index(tmp_path):
    ce, svo_dir = _ce(tmp_path)
    ce.reset()
    ce._finalize_svo(env_idx=0, kept=True, episode_index=7)
    files = set(os.listdir(svo_dir))
    assert {
        "episode_000007_wrist_1.svo2",
        "episode_000007_wrist_2.svo2",
    }.issubset(files)
    idx = json.load(open(os.path.join(svo_dir, "svo_index.json")))
    assert "7" in idx and len(idx["7"]) == 2
    assert set(idx["7"]) == {
        "episode_000007_wrist_1.svo2",
        "episode_000007_wrist_2.svo2",
    }


def test_discard_deletes_svo(tmp_path):
    ce, svo_dir = _ce(tmp_path)
    ce.reset()
    ce._finalize_svo(env_idx=0, kept=False)
    leftover = (
        [f for f in os.listdir(svo_dir) if f.endswith(".svo2")]
        if os.path.isdir(svo_dir)
        else []
    )
    assert leftover == []


def test_maybe_flush_failed_terminate_discards_svo(tmp_path):
    # Operator 'a' abort: terminated but NOT a success. _maybe_flush must still
    # finalize (stop + delete) the SVO, else the recording handle leaks.
    ce, svo_dir = _ce(tmp_path)
    ce.reset()  # starts recording (temp .svo2 created)
    ce._episode_success = [False]
    ce._maybe_flush(np.array([True]), np.array([False]))  # done_by_term, not success
    leftover = (
        [f for f in os.listdir(svo_dir) if f.endswith(".svo2")]
        if os.path.isdir(svo_dir)
        else []
    )
    assert leftover == []
    assert ce._svo_active == {}


class _RaisingStartEnv(gym.Env):
    def start_recording(self, svo_dir, tag):
        raise RuntimeError("enable_recording failed")
    def stop_recording(self):
        pass
    def reset(self, *, seed=None, options=None):
        return {"state": np.zeros(8)}, {}


def test_start_svo_failure_does_not_propagate(tmp_path):
    save_dir = os.path.join(str(tmp_path), "run", "collected_data")
    os.makedirs(save_dir, exist_ok=True)
    ce = CollectEpisode(
        _RaisingStartEnv(), save_dir=save_dir, export_format="pickle",
        only_success=True, record_svo=True,
    )
    ce.reset()  # _start_svo's start() raises -> must be caught, not propagate
    assert ce._svo_active == {}
