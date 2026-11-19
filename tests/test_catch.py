import numpy as np
import pytest

from aps.catch.draw import draw_net
from aps.catch.simulator import CatchSimulator
from aps.tracker.tracker import Tracker
from aps.ttc.estimator import TtcEstimator
from synthetic import FOCAL_PX, FRAME_SHAPE, WIDTH, approach_center, approach_constant_bearing, generate, passing, stationary


def run(cfg, position, duration=2.0, seed=0):
    import math

    catch_cfg = {**cfg["catch"], "fov_deg": math.degrees(2 * math.atan(WIDTH / 2 / FOCAL_PX))}
    tracker, ttc, sim = Tracker(cfg["tracker"]), TtcEstimator(cfg["ttc"]), CatchSimulator(catch_cfg)
    states, image = [], np.zeros(FRAME_SHAPE, dtype=np.uint8)
    for sample in generate(position, duration, noise_px=1.0, seed=seed):
        tracks = tracker.update([sample.detection] if sample.detection else [], sample.t, FRAME_SHAPE)
        states.append((sample.t, sim.update(tracks, ttc.estimate(tracks, FRAME_SHAPE), FRAME_SHAPE, sample.t), sample))
        draw_net(image, sim, states[-1][1], sample.t)  # 그리기가 예외 없이 도는지
    return states


@pytest.mark.parametrize("position", [approach_center(), approach_constant_bearing(0.05, 0.02)])
def test_fires_near_threshold_and_catches_head_on(cfg, position):
    states = run(cfg, position)
    fired = next(s for s in states if s[1].phase != "idle")
    assert fired[2].true_ttc <= cfg["catch"]["fire_ttc_s"] + 0.1
    assert any(s.caught is True for _, s, _ in states)


@pytest.mark.parametrize("position", [passing(), stationary()])
def test_no_fire_without_collision_course(cfg, position):
    assert all(s.phase == "idle" for _, s, _ in run(cfg, position))


def test_only_one_shot(cfg):
    states = run(cfg, approach_center(), duration=3.0)
    assert len({s.fire_t for _, s, _ in states if s.phase != "idle"}) == 1


def _sim(cfg):
    return CatchSimulator({**cfg["catch"], "fov_deg": 76.6})


def test_net_size_constant_while_opening_then_shrinks(cfg):
    sim = _sim(cfg)
    assert sim.net_half_px(0.0, 1280) == sim.net_half_px(cfg["catch"]["open_dist_m"], 1280) > 0
    assert sim.net_half_px(9.0, 1280) < sim.net_half_px(4.5, 1280)


def test_judge_miss_and_lost(cfg):
    from aps.catch.simulator import CatchState
    from aps.types import Track

    def judge(track):
        sim = _sim(cfg)
        sim.state = CatchState("flying", 0.0, 1, 5.0, 0.3, 6.0)
        return sim._judge([track] if track else [], 1280, 720, 0.3).caught

    def trk(cx, missed=0):
        return Track(1, cx, 360, 100.0, 0, 0, 0, (0, 0, 1, 1), missed)

    assert judge(trk(640)) is True
    assert judge(trk(1200)) is False  # 중심에서 크게 벗어남
    assert judge(trk(640, missed=2)) is None  # 예측만 남은 트랙
    assert judge(None) is None
