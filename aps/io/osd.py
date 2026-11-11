"""
영상에 박힌 화면 고정 오버레이(OSD 조준점, 글자, 로고, 화면에 걸린 자기 기체)에서 나온 탐지를 버린다.

오버레이는 기체가 움직여도 화면 위치와 밝기가 거의 변하지 않는다. 최근 calib_frames 프레임(축소·간격 추출)을
보관해 두고 update_every 프레임마다 '창 평균과 밝기가 거의 같은 프레임 비율'이 높고 주변보다 튀는 픽셀을 찾아,
글자 단위로 묶은 사각형을 오버레이 영역으로 삼는다.
박스의 대부분이 이 영역 안에 있는 탐지만 버리고 영상 자체는 건드리지 않으므로,
조준점 바로 옆의 원거리 표적이 지워지거나 흐려지지 않는다. 영상 중간에 새로 뜨는 자막·로고도 몇 초 뒤부터 걸러진다.
"""

import logging
from collections import deque
from typing import Iterable, Iterator

import cv2
import numpy as np

from aps.detector.base import Detector
from aps.types import DetectionResult, Frame

log = logging.getLogger(__name__)


def overlay_region(background: np.ndarray, stable: np.ndarray, cfg: dict) -> np.ndarray | None:
    """배경(창 평균 밝기)과 정지 비율로 오버레이 영역(uint8, 0/1, 오버레이마다 감싼 사각형)을 만든다. 없으면 None."""
    k = cfg["contrast_kernel"]
    contrast = np.abs(background - cv2.blur(background, (k, k)))
    static = ((stable >= cfg["stable_ratio"]) & (contrast > cfg["min_contrast"])).astype(np.uint8)
    if not static.any():
        return None

    h, w = static.shape
    g = max(int(cfg["group_ratio"] * h), 1) | 1  # 글자·아이콘 조각을 한 덩어리로 묶는 팽창 크기
    n, labels = cv2.connectedComponents(cv2.dilate(static, np.ones((g, g), np.uint8)))
    # 사각형은 팽창 전 픽셀에 작은 여백(pad)만 더해 잡는다. 팽창 여백을 다 넣으면 조준점 옆 표적 박스까지 버려지고,
    # 여백이 없으면 조준점보다 조금 크게 잡힌 박스가 남는다
    pad = int(cfg["region_pad_ratio"] * h)
    ys, xs = np.nonzero(static)
    lab = labels[ys, xs]
    y0, x0 = np.full(n, h), np.full(n, w)
    y1, x1 = np.zeros(n, int), np.zeros(n, int)
    np.minimum.at(y0, lab, ys)
    np.minimum.at(x0, lab, xs)
    np.maximum.at(y1, lab, ys)
    np.maximum.at(x1, lab, xs)
    region = np.zeros((h, w), np.uint8)
    for i in range(1, n):
        region[max(y0[i] - pad, 0) : y1[i] + 1 + pad, max(x0[i] - pad, 0) : x1[i] + 1 + pad] = 1
    return region


class OsdOverlaySource:
    """프레임 소스를 감싸 오버레이 영역(self.region)을 계속 갱신한다. 프레임은 그대로 내보낸다.

    시작할 때 calib_frames 프레임을 먼저 읽어 영역을 만든 뒤 내보내므로, 첫 프레임부터 걸러진다.
    """

    def __init__(self, source: Iterable[Frame], cfg: dict):
        self.source = source
        self.cfg = cfg
        self.region: np.ndarray | None = None
        self._window: deque[np.ndarray] = deque(maxlen=cfg["calib_frames"] // cfg["sample_every"])
        self._count = 0

    def update(self, image: np.ndarray) -> None:
        cfg = self.cfg
        self._count += 1
        if self._count % cfg["sample_every"] == 0:
            h, w = image.shape[:2]
            scale = min(1.0, cfg["work_height"] / h)  # 통계는 축소 영상에서 계산해 비용을 줄임
            small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
            self._window.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        if self._count >= cfg["calib_frames"] and self._count % cfg["update_every"] == 0:
            # 창 전체(약 2초)의 평균과 비교한다. 이전 프레임과만 비교하면 천천히 흐르는 구름·하늘도 정지로 잡힘
            stack = np.stack(self._window)
            background = stack.mean(axis=0, dtype=np.float32)  # 움직이는 배경은 흐려져 대비가 낮아짐
            stable = (np.abs(stack - background) < cfg["stable_tol"]).mean(axis=0)
            region = overlay_region(background, stable, cfg)
            # 영역이 넓은데 변하는 픽셀도 적으면 카메라가 거의 멈춘 구간이라 실제 장면(구름, 표적)까지 정지로 잡힌 것이다.
            # 이 창은 믿지 않고 이전 영역을 유지한다. 처음부터 계속 이러면(웹캠 등 고정 카메라) 영역이 없어 거르지 않는다.
            # 넓이만 보면 제목 띠·레터박스가 화면을 넓게 덮는 영상(005.mp4)에서 필터가 계속 꺼지므로 장면 변화도 함께 본다
            moving = (stable < cfg["stable_ratio"]).mean()
            if region is not None and region.mean() > cfg["max_region_ratio"] and moving < cfg["min_moving_ratio"]:
                return
            h, w = image.shape[:2]
            self.region = None if region is None else cv2.resize(region, (w, h), interpolation=cv2.INTER_NEAREST)

    def __iter__(self) -> Iterator[Frame]:
        it = iter(self.source)
        buffered = []
        for frame in it:
            self.update(frame.image)
            buffered.append(frame)
            if len(buffered) >= self.cfg["calib_frames"]:
                break
        if self.region is None and self._window:
            log.info("시작 구간에서 오버레이를 찾지 못했습니다 (없거나 고정 카메라)")
        yield from buffered
        for frame in it:
            self.update(frame.image)
            yield frame


class OverlayFilterDetector:
    """탐지기를 감싸, 박스 면적의 max_overlap 이상이 오버레이 영역에 들어가는 탐지를 버린다."""

    def __init__(self, detector: Detector, overlay: OsdOverlaySource, max_overlap: float):
        self.detector = detector
        self.overlay = overlay
        self.max_overlap = max_overlap

    def detect(self, image: np.ndarray) -> DetectionResult:
        result = self.detector.detect(image)
        region = self.overlay.region
        if region is None:
            return result
        kept = []
        for det in result.detections:
            x0, y0, x1, y1 = (int(round(v)) for v in det.xyxy)
            x0, y0 = max(x0, 0), max(y0, 0)
            area = max((x1 - x0) * (y1 - y0), 1)
            if region[y0:y1, x0:x1].sum() / area < self.max_overlap:
                kept.append(det)
        return DetectionResult(kept, result.preprocess_ms, result.inference_ms)
