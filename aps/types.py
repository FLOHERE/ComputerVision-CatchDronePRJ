"""모듈 사이에서 주고받는 데이터 형식."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    t: float  # 초
    index: int


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    conf: float
    cls: int


@dataclass(frozen=True)
class DetectionResult:
    detections: list[Detection]
    preprocess_ms: float
    inference_ms: float


@dataclass(frozen=True)
class Track:
    track_id: int
    cx: float
    cy: float
    area: float
    vx: float
    vy: float
    va: float
    bbox: tuple[float, float, float, float]  # 마지막으로 매칭된 탐지 박스
    missed: int  # 연속 미탐지 프레임 수


@dataclass(frozen=True)
class TtcResult:
    track_id: int
    ttc: float  # 접근하지 않으면 inf
    on_course: bool
    r: float  # 화면 중앙까지 거리 (반대각선 대비)
    dr_dt: float  # r의 변화율 (1/s)


@dataclass(frozen=True)
class ThreatEvent:
    track_id: int
    t: float
    frame_index: int
    ttc: float
