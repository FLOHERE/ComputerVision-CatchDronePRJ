from dataclasses import dataclass
from time import perf_counter

from aps.decision.decision import ThreatDecider
from aps.detector.base import Detector
from aps.metrics import LatencyLogger
from aps.tracker.tracker import Tracker
from aps.ttc.estimator import TtcEstimator
from aps.types import Frame, ThreatEvent, Track, TtcResult


@dataclass(frozen=True)
class FrameResult:
    tracks: list[Track]
    ttc_results: list[TtcResult]
    events: list[ThreatEvent]
    threat_ids: set[int]
    total_ms: float


class Pipeline:
    def __init__(self, cfg: dict, detector: Detector, metrics: LatencyLogger | None = None):
        self.detector = detector
        self.tracker = Tracker(cfg["tracker"])
        self.ttc = TtcEstimator(cfg["ttc"])
        self.decider = ThreatDecider(cfg["decision"])
        self.metrics = metrics

    def process(self, frame: Frame) -> FrameResult:
        start = perf_counter()
        detection = self.detector.detect(frame.image)

        before_tracking = perf_counter()
        tracks = self.tracker.update(detection.detections, frame.t, frame.image.shape)

        before_ttc = perf_counter()
        ttc_results = self.ttc.estimate(tracks, frame.image.shape)

        after_ttc = perf_counter()
        events = self.decider.update(ttc_results, frame.t, frame.index)
        end = perf_counter()

        total_ms = (end - start) * 1000
        if self.metrics is not None:
            self.metrics.log(
                frame.index, detection.preprocess_ms, detection.inference_ms,
                (before_ttc - before_tracking) * 1000, (after_ttc - before_ttc) * 1000, total_ms,
                len(detection.detections), len(tracks),
            )  # fmt: skip
        return FrameResult(tracks, ttc_results, events, self.decider.active, total_ms)
