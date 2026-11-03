import math

import numpy as np
import pytest

from aps.tracker.tracker import Tracker
from aps.ttc.estimator import TtcEstimator
from aps.types import Track
from synthetic import (
    FRAME_SHAPE,
    HEIGHT,
    WIDTH,
    accelerating,
    approach_center,
    approach_constant_bearing,
    generate,
    passing,
    receding,
    stationary,
)


def track(cx=WIDTH / 2, cy=HEIGHT / 2, area=100.0, vx=0.0, vy=0.0, va=0.0):
    return Track(1, cx, cy, area, vx, vy, va, (0, 0, 1, 1), 0)


def run(cfg, samples):
    """(실제 TTC, TtcResult) 목록을 반환한다."""
    tracker, estimator = Tracker(cfg["tracker"]), TtcEstimator(cfg["ttc"])
    out = []
    for s in samples:
        tracks = tracker.update([s.detection] if s.detection else [], s.t, FRAME_SHAPE)
        out += [(s.true_ttc, r) for r in estimator.estimate(tracks, FRAME_SHAPE)]
    return out


def test_formula(cfg):
    assert TtcEstimator(cfg["ttc"]).estimate([track(area=100, va=400)], FRAME_SHAPE)[0].ttc == pytest.approx(0.5)


@pytest.mark.parametrize("va", [0.0, -50.0])
def test_not_approaching_is_inf(cfg, va):
    assert math.isinf(TtcEstimator(cfg["ttc"]).estimate([track(va=va)], FRAME_SHAPE)[0].ttc)


@pytest.mark.parametrize("position", [approach_center(), approach_constant_bearing(), passing()])
def test_ttc_matches_ground_truth_without_noise(cfg, position):
    errors = [abs(r.ttc - true) / true for true, r in run(cfg, generate(position, 2.0)) if 0.3 <= true <= 1.0]
    assert errors and max(errors) < 0.05


def test_ttc_within_tolerance_with_noise(cfg):
    ratios = []
    for seed in range(30):
        samples = generate(approach_center(), 2.0, noise_px=1.0, seed=seed)
        ratios += [r.ttc / true for true, r in run(cfg, samples) if 0.4 <= true <= 0.6]
    assert abs(np.median(ratios) - 1) < 0.05
    assert np.percentile(ratios, 5) > 0.75 and np.percentile(ratios, 95) < 1.25


def test_ttc_recovers_after_sudden_speed_change(cfg):
    # 필터의 일정 속도 가정이 깨져도 임계값에 도달하기 전에 따라잡아야 한다
    results = run(cfg, generate(accelerating(), 2.0))
    ratios = [r.ttc / true for true, r in results if 0.4 <= true <= 0.6]
    assert ratios and abs(np.median(ratios) - 1) < 0.25


@pytest.mark.parametrize("position", [stationary(), stationary(z=8.0), receding()])
def test_no_collision_ttc_stays_above_threshold_with_noise(cfg, position):
    threshold = cfg["decision"]["ttc_threshold_s"]
    for seed in range(30):
        assert all(r.ttc >= threshold for _, r in run(cfg, generate(position, 2.0, noise_px=1.0, seed=seed)))


@pytest.mark.parametrize("position", [approach_center(), approach_constant_bearing()])
def test_collision_course_is_on_course_near_impact(cfg, position):
    for seed in range(30):
        near = [r for true, r in run(cfg, generate(position, 2.0, noise_px=1.0, seed=seed)) if 0.2 <= true <= 0.5]
        assert near and all(r.on_course for r in near)


@pytest.mark.parametrize("position", [passing(), passing(2.0, 0.5)])
def test_passing_is_off_course_near_threshold(cfg, position):
    for seed in range(30):
        near = [r for true, r in run(cfg, generate(position, 2.0, noise_px=1.0, seed=seed)) if 0.2 <= true <= 0.5]
        assert near and not any(r.on_course for r in near)


def test_truncated_box_does_not_bias_ttc(cfg):
    # 화면 오른쪽으로 잘려 면적이 실제보다 작게 측정되는 접근 궤적. 보호가 없으면 TTC가 최대 1.4배까지 커진다
    samples = generate(approach_constant_bearing(x_ratio=0.78, y_ratio=0.0), 2.0)
    truncated_from = next(s.true_ttc for s in samples if s.detection.xyxy[2] >= WIDTH - 1)
    ratios = [r.ttc / true for true, r in run(cfg, samples) if 0.1 <= true <= truncated_from]
    assert ratios and all(abs(ratio - 1) < 0.1 for ratio in ratios)
