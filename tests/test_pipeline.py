import csv

import numpy as np
import pytest

from aps.fc_link.mock import MockFcLink
from aps.metrics import COLUMNS, LatencyLogger
from aps.pipeline import Pipeline
from aps.types import DetectionResult, Frame
from synthetic import FRAME_SHAPE, approach_center, approach_constant_bearing, generate, passing, receding, stationary


class ReplayDetector:
    """합성 궤적을 차례로 돌려주는 가짜 탐지기. 모델이 필요 없다."""

    def __init__(self, samples):
        self._samples = iter(samples)

    def detect(self, image):
        detection = next(self._samples).detection
        return DetectionResult([detection] if detection else [], 0.5, 1.0)


def run(cfg, samples, metrics=None):
    pipeline = Pipeline(cfg, ReplayDetector(samples), metrics)
    link = MockFcLink()
    pipeline.decider.subscribe(link.evade)
    image = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    for i, sample in enumerate(samples):
        pipeline.process(Frame(image, sample.t, i))
    return link.calls


@pytest.mark.parametrize("position", [approach_center(), approach_constant_bearing()])
def test_collision_course_fires_once_near_threshold(cfg, position):
    for seed in range(10):
        samples = generate(position, 2.0, noise_px=1.0, seed=seed)
        calls = run(cfg, samples)
        assert len(calls) >= 1
        true_ttc = samples[calls[0].frame_index].true_ttc
        assert 0.35 <= true_ttc <= 0.7


@pytest.mark.parametrize("position", [passing(), passing(2.0, 0.5), stationary(), stationary(z=8.0), receding()])
def test_non_threats_never_fire(cfg, position):
    for seed in range(10):
        assert run(cfg, generate(position, 2.0, noise_px=1.0, seed=seed)) == []


def test_writes_latency_row_per_frame(cfg, tmp_path):
    path = tmp_path / "latency.csv"
    samples = generate(approach_center(), 1.0)
    metrics = LatencyLogger(path, "cpu", 960, cfg["metrics"]["budget_ms"])
    run(cfg, samples, metrics)
    metrics.close()

    rows = list(csv.DictReader(path.open()))
    assert len(rows) == len(samples)
    assert list(rows[0]) == COLUMNS
    assert float(rows[0]["preprocess_ms"]) == 0.5 and float(rows[0]["inference_ms"]) == 1.0
    assert all(float(r["total_ms"]) >= 0 for r in rows)
