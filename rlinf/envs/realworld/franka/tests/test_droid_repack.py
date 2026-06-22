import sys
import pathlib

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[5]  # .../vendor/RLinf
sys.path.insert(0, str(_REPO / "examples" / "embodiment"))

from validate_droid_lerobot import rlinf_frame_to_droid  # noqa: E402


def test_repack_splits_state_and_renames_images():
    frame = {
        "state": np.array([0.7, 0.1, -0.6, 0.0, -2.5, 0.0, 1.8, 0.0]),  # [grip, q0..q6]
        "actions": np.arange(8, dtype=np.float32),
        "image": np.zeros((180, 320, 3), dtype=np.uint8),
        "extra_view_image": np.ones((180, 320, 3), dtype=np.uint8),
        "task": "pick up the cube",
    }
    out = rlinf_frame_to_droid(frame)
    np.testing.assert_allclose(out["observation/joint_position"], frame["state"][1:8])
    np.testing.assert_allclose(out["observation/gripper_position"], frame["state"][0:1])
    assert out["observation/exterior_image_1_left"].shape == (180, 320, 3)
    assert out["observation/wrist_image_left"].shape == (180, 320, 3)
    assert out["actions"].shape == (8,)
    assert out["prompt"] == "pick up the cube"
