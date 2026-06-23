import json, os
from rlinf.envs.wrappers.collect_episode import CollectEpisode


def test_existing_total_episodes_reads_info(tmp_path):
    root = tmp_path / "ds"; (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": 5}))
    assert CollectEpisode._existing_total_episodes(str(root)) == 5


def test_existing_total_episodes_missing_returns_zero(tmp_path):
    assert CollectEpisode._existing_total_episodes(None) == 0
    assert CollectEpisode._existing_total_episodes(str(tmp_path / "nope")) == 0
