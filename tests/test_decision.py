import math

from aps.decision.decision import ThreatDecider
from aps.types import TtcResult


def result(ttc, on_course=True, track_id=1):
    return TtcResult(track_id, ttc, on_course, 0.0, 0.0)


def test_requires_low_ttc_and_on_course(cfg):
    decider = ThreatDecider(cfg["decision"])
    assert decider.update([result(0.3, on_course=False)], 0.0, 0) == []
    assert decider.update([result(0.6)], 0.1, 1) == []
    assert decider.update([result(math.inf)], 0.2, 2) == []
    assert len(decider.update([result(0.3)], 0.3, 3)) == 1


def test_threshold_is_exclusive(cfg):
    threshold = cfg["decision"]["ttc_threshold_s"]
    assert ThreatDecider(cfg["decision"]).update([result(threshold)], 0.0, 0) == []


def test_fires_once_on_rising_edge_and_rearms(cfg):
    decider = ThreatDecider(cfg["decision"])
    received = []
    decider.subscribe(received.append)
    for i, ttc in enumerate([0.4, 0.3, 0.2, 2.0, 0.3]):
        decider.update([result(ttc)], i / 30, i)
    assert [e.frame_index for e in received] == [0, 4]


def test_tracks_are_independent(cfg):
    decider = ThreatDecider(cfg["decision"])
    events = decider.update([result(0.3, track_id=1), result(0.2, track_id=2)], 0.0, 0)
    assert {e.track_id for e in events} == {1, 2}
    assert decider.active == {1, 2}
