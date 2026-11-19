from aps.catch.simulator import CatchSimulator
from aps.net_link.base import NetFireEvent
from aps.net_link.mock import MockNetLink


def event(meet_m=6.0):
    return NetFireEvent(1, 0.0, 0.5, 5.0, meet_m)


def link(cfg, **over):
    return MockNetLink({**cfg["net_link"], **over})


def test_blocked_when_disarmed_by_default(cfg):
    mock = link(cfg)
    assert mock.fire(event()) is False
    assert mock.calls == [] and mock.rejected[0][1] == "disarmed"


def test_fires_when_armed_then_out_of_nets(cfg):
    mock = link(cfg, armed=True)
    assert mock.fire(event()) is True
    assert mock.fire(event()) is False and mock.rejected[-1][1] == "no net left"
    assert len(mock.calls) == 1


def test_too_close_blocked_without_using_the_net(cfg):
    mock = link(cfg, armed=True)
    assert mock.fire(event(meet_m=0.6)) is False and mock.rejected[0][1] == "too close"
    assert mock.fire(event(meet_m=6.0)) is True  # 막힌 요청은 그물을 쓰지 않음


def test_disarm_blocks_again(cfg):
    mock = link(cfg, armed=True)
    mock.disarm()
    assert mock.fire(event()) is False


def test_simulator_delivers_event_to_link(cfg):
    import numpy as np

    from aps.tracker.tracker import Tracker
    from aps.ttc.estimator import TtcEstimator
    from synthetic import FRAME_SHAPE, approach_center, generate

    mock = link(cfg, armed=True, min_meet_m=0.0)  # 합성 궤적은 가까이서 끝나므로 거리 조건은 끈다
    sim = CatchSimulator({**cfg["catch"], "fov_deg": 76.6})
    sim.subscribe(mock.fire)
    tracker, ttc = Tracker(cfg["tracker"]), TtcEstimator(cfg["ttc"])
    for sample in generate(approach_center(), 2.0, noise_px=1.0):
        tracks = tracker.update([sample.detection] if sample.detection else [], sample.t, FRAME_SHAPE)
        sim.update(tracks, ttc.estimate(tracks, FRAME_SHAPE), FRAME_SHAPE, sample.t)
    assert len(mock.calls) == 1
    assert mock.calls[0].ttc <= cfg["catch"]["fire_ttc_s"]
