from aps.decision.decision import ThreatDecider
from aps.fc_link.mock import MockFcLink
from aps.types import TtcResult


def test_mock_receives_threat_through_callback(cfg):
    decider = ThreatDecider(cfg["decision"])
    link = MockFcLink()
    decider.subscribe(link.evade)
    decider.update([TtcResult(7, 0.3, True, 0.0, 0.0)], 1.5, 45)
    assert [(e.track_id, e.frame_index) for e in link.calls] == [(7, 45)]
