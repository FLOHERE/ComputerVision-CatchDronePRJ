from itertools import count

import numpy as np

from aps.tracker.kalman import AreaKalman
from aps.types import Detection, Track


def _measurement(det: Detection) -> np.ndarray:
    x1, y1, x2, y2 = det.xyxy
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * (y2 - y1)])


class _TrackState:
    def __init__(self, track_id: int, det: Detection, kalman_cfg: dict, truncated: bool):
        self.track_id = track_id
        self.kf = AreaKalman(_measurement(det), kalman_cfg)
        if truncated:
            self.kf.P[2, 2] *= kalman_cfg["truncated_area_scale"] ** 2
        self.bbox = det.xyxy
        self.hits = 1
        self.missed = 0

    def snapshot(self) -> Track:
        cx, cy, area, vx, vy, va = (float(v) for v in self.kf.x)
        return Track(self.track_id, cx, cy, area, vx, vy, va, self.bbox, self.missed)


class Tracker:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._tracks: list[_TrackState] = []
        self._next_id = count(1)
        self._last_t: float | None = None

    def update(self, detections: list[Detection], t: float, frame_shape: tuple[int, ...]) -> list[Track]:
        """확정된 트랙만 반환한다. 미탐지 중인 트랙은 max_age까지 예측값으로 유지한다."""
        if self._last_t is not None:
            dt = t - self._last_t
            if dt <= 0:
                raise ValueError(f"프레임 시간이 증가하지 않습니다: {self._last_t} -> {t}")
            for track in self._tracks:
                track.kf.predict(dt)
        self._last_t = t

        matches, unmatched = self._associate(detections)
        for track, det in matches:
            track.kf.update(_measurement(det), self._is_truncated(det, frame_shape))
            track.bbox = det.xyxy
            track.hits += 1
            track.missed = 0

        matched = {id(track) for track, _ in matches}
        for track in self._tracks:
            if id(track) not in matched:
                track.missed += 1
        self._tracks = [track for track in self._tracks if track.missed <= self.cfg["max_age"]]

        for det in unmatched:
            truncated = self._is_truncated(det, frame_shape)
            self._tracks.append(_TrackState(next(self._next_id), det, self.cfg["kalman"], truncated))

        return [track.snapshot() for track in self._tracks if track.hits >= self.cfg["min_hits"]]

    def _associate(self, detections: list[Detection]) -> tuple[list[tuple[_TrackState, Detection]], list[Detection]]:
        # 작고 빠른 드론은 프레임 사이에 박스가 겹치지 않으므로 IoU 대신 중심 거리를 쓴다
        # shortcut: greedy, 트랙이 많아 오매칭이 생기면 헝가리안으로 교체
        candidates = []
        for ti, track in enumerate(self._tracks):
            gate = max(self.cfg["gate_min_px"], self.cfg["gate_size_ratio"] * np.sqrt(max(track.kf.x[2], 0.0)))
            for di, det in enumerate(detections):
                z = _measurement(det)
                distance = float(np.hypot(*(z[:2] - track.kf.x[:2])))
                if distance <= gate:
                    candidates.append((distance, ti, di))

        used_tracks, used_dets, matches = set(), set(), []
        for _, ti, di in sorted(candidates):
            if ti not in used_tracks and di not in used_dets:
                used_tracks.add(ti)
                used_dets.add(di)
                matches.append((self._tracks[ti], detections[di]))
        return matches, [det for di, det in enumerate(detections) if di not in used_dets]

    def _is_truncated(self, det: Detection, frame_shape: tuple[int, ...]) -> bool:
        height, width = frame_shape[:2]
        margin = self.cfg["edge_margin_px"]
        x1, y1, x2, y2 = det.xyxy
        return x1 <= margin or y1 <= margin or x2 >= width - margin or y2 >= height - margin
