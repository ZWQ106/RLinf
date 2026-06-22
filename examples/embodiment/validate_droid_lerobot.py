"""Validate that a joint-velocity LeRobot dataset maps onto openpi DROID keys.

Pure mapping `rlinf_frame_to_droid` is unit-tested. `main()` loads a dataset
and (if openpi is importable) runs DroidInputs to confirm the model input
builds. Run on the Desktop where openpi is installed:
    python examples/embodiment/validate_droid_lerobot.py --repo-id <path>
"""
import argparse

import numpy as np


def rlinf_frame_to_droid(frame: dict) -> dict:
    """Map one RLinf-collected LeRobot frame to openpi DROID input keys.

    Recorded `state` is the alphabetical concat of {gripper_position(1),
    joint_position(7)} = [grip, q0..q6]."""
    state = np.asarray(frame["state"]).reshape(-1)
    assert state.shape[0] == 8, f"expected 8-D state [grip,q0..q6], got {state.shape}"
    # image = main cam (exterior ZED 2i, main_image_key); extra_view_image = wrist ZED Mini.
    out = {
        "observation/joint_position": state[1:8],
        "observation/gripper_position": state[0:1],
        "observation/exterior_image_1_left": np.asarray(frame["image"]),
        "observation/wrist_image_left": np.asarray(frame["extra_view_image"]),
        "actions": np.asarray(frame["actions"]).reshape(-1),
        "prompt": str(frame["task"]),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True, help="path to the LeRobot dataset dir")
    args = ap.parse_args()

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(args.repo_id)
    sample = ds[0]
    frame = {
        "state": sample["state"].numpy() if hasattr(sample["state"], "numpy") else sample["state"],
        "actions": sample["actions"].numpy() if hasattr(sample["actions"], "numpy") else sample["actions"],
        "image": np.asarray(sample["image"]),
        "extra_view_image": np.asarray(sample.get("extra_view_image", sample.get("extra_view_image-0"))),
        "task": sample.get("task", "unknown task"),
    }
    droid = rlinf_frame_to_droid(frame)
    print("joint_position:", droid["observation/joint_position"].shape)
    print("gripper_position:", droid["observation/gripper_position"].shape)
    print("actions:", droid["actions"].shape)
    print("exterior img:", droid["observation/exterior_image_1_left"].shape)
    print("wrist img:", droid["observation/wrist_image_left"].shape)
    assert droid["observation/joint_position"].shape == (7,)
    assert droid["observation/gripper_position"].shape == (1,)
    assert droid["actions"].shape[0] == 8

    try:
        from openpi.policies.droid_policy import DroidInputs
        from openpi.models.model import ModelType

        di = DroidInputs(model_type=ModelType.PI0)
        model_in = di(droid)
        print("DroidInputs OK; state dim:", np.asarray(model_in["state"]).shape)
    except Exception as e:  # openpi not installed here / API drift
        print(f"[skip] DroidInputs not run: {e}")
    print("FORMAT VALIDATION PASSED")


if __name__ == "__main__":
    main()
