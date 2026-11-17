"""Episode logging round-trip."""

from __future__ import annotations

from r4r.logging_ import EpisodeLog, EpisodeLogger


def test_logger_write_and_read(tmp_path):
    path = tmp_path / "episodes.jsonl"
    logger = EpisodeLogger(str(path))
    logger.write(EpisodeLog(episode_id="a1", instruction="Put the red mug in the bin.",
                            success=True))
    logger.write(EpisodeLog(episode_id="b2", instruction="Throw the mug.",
                            success=False, failure_category="plan_refused"))
    logs = logger.read_all()
    assert [l.episode_id for l in logs] == ["a1", "b2"]
    assert logs[0].success is True
    assert logs[1].failure_category == "plan_refused"
