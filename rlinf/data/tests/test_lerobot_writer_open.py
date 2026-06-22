import numpy as np
from rlinf.data.lerobot_writer import LeRobotDatasetWriter

FEATS_KW = dict(
    robot_type="panda",
    fps=10,
    state_dim=4,
    action_dim=4,
    has_image=False,
    has_intervene_flag=False,
)

TASK = "pick up the cube"


def _make_episode(val, n=3):
    return [
        {
            "state": np.full(4, val, np.float32),
            "actions": np.full(4, val, np.float32),
            "done": np.array([False], dtype=bool),
            "is_success": np.array([False], dtype=bool),
            "task": TASK,
        }
        for _ in range(n)
    ]


def test_open_or_create_accumulates_across_instances(tmp_path):
    root = str(tmp_path / "ds")
    rid = "tasl/ds_test"

    w1 = LeRobotDatasetWriter()
    w1.open_or_create(repo_id=rid, root=root, **FEATS_KW)
    w1.add_episode(_make_episode(1.0))
    w1.add_episode(_make_episode(2.0))
    assert w1.dataset.meta.total_episodes == 2
    del w1

    w2 = LeRobotDatasetWriter()
    w2.open_or_create(repo_id=rid, root=root, **FEATS_KW)
    assert w2.dataset.meta.total_episodes == 2
    w2.add_episode(_make_episode(3.0))
    assert w2.dataset.meta.total_episodes == 3

    import json
    import pathlib

    info = json.loads((pathlib.Path(root) / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 3
