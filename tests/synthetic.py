"""3D 점을 핀홀 카메라로 투영해 실제 TTC를 아는 합성 bbox 궤적을 만든다."""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from aps.types import Detection

WIDTH, HEIGHT = 1280, 720
FRAME_SHAPE = (HEIGHT, WIDTH, 3)
FOCAL_PX = 800.0
DRONE_SIZE_M = 0.3
FPS = 30.0
SPEED = 20.0  # 접근 속도 (m/s)
START_Z = 40.0
MIN_Z = 1.0

Position = Callable[[float], tuple[float, float, float]]


@dataclass(frozen=True)
class Sample:
    t: float
    detection: Detection | None
    true_ttc: float


def generate(position: Position, duration: float, noise_px: float = 0.0, seed: int = 0) -> list[Sample]:
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(int(duration * FPS)):
        t = i / FPS
        x, y, z = position(t)
        if z < MIN_Z:
            break
        half = FOCAL_PX * DRONE_SIZE_M / z / 2
        cx, cy = WIDTH / 2 + FOCAL_PX * x / z, HEIGHT / 2 + FOCAL_PX * y / z
        box = np.array([cx - half, cy - half, cx + half, cy + half]) + rng.normal(0, noise_px, 4)
        x1, y1, x2, y2 = np.clip(box, [0, 0, 0, 0], [WIDTH, HEIGHT, WIDTH, HEIGHT])
        visible = x2 - x1 > 1 and y2 - y1 > 1
        detection = Detection((float(x1), float(y1), float(x2), float(y2)), 0.9, 0) if visible else None
        closing_speed = (z - position(t + 1e-3)[2]) / 1e-3
        samples.append(Sample(t, detection, z / closing_speed if closing_speed > 1e-6 else float("inf")))
    return samples


def approach_center(x0: float = 4.0, y0: float = 2.0) -> Position:
    """화면 중앙으로 수렴: 측면 오프셋이 z보다 빠르게 줄어 r ∝ 남은 시간."""
    def position(t):
        z = START_Z - SPEED * t
        scale = (z / START_Z) ** 2
        return x0 * scale, y0 * scale, z
    return position


def approach_constant_bearing(x_ratio: float = 0.5, y_ratio: float = 0.2) -> Position:
    """직선으로 곧장 날아옴: 같은 픽셀에 머문 채 커진다."""
    def position(t):
        z = START_Z - SPEED * t
        return x_ratio * z, y_ratio * z, z
    return position


def passing(x_miss: float = 4.0, y_miss: float = 1.0) -> Position:
    """측면 거리를 유지한 채 지나감: 면적은 커지지만 바깥으로 벌어진다."""
    def position(t):
        return x_miss, y_miss, START_Z - SPEED * t
    return position


def stationary(z: float = 20.0, x: float = 0.0, y: float = 0.0) -> Position:
    def position(t):
        return x, y, z
    return position


def receding(z0: float = 5.0, x: float = 0.0, y: float = 0.0) -> Position:
    def position(t):
        return x, y, z0 + SPEED * t
    return position


def accelerating(switch_t: float = 1.0, speed_after: float = 40.0) -> Position:
    """비행 중 접근 속도가 갑자기 바뀌는 기동. 필터의 일정 속도 가정을 깨는 경우."""
    def position(t):
        if t < switch_t:
            return 0.0, 0.0, 60.0 - SPEED * t
        return 0.0, 0.0, 60.0 - SPEED * switch_t - speed_after * (t - switch_t)
    return position
