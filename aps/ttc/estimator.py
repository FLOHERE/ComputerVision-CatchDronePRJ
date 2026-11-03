import math

from aps.types import Track, TtcResult


class TtcEstimator:
    def __init__(self, cfg: dict):
        self.center_radius = cfg["center_radius"]
        self.converge_eps = cfg["converge_eps"]

    def estimate(self, tracks: list[Track], frame_shape: tuple[int, ...]) -> list[TtcResult]:
        height, width = frame_shape[:2]
        half_diagonal = math.hypot(width, height) / 2
        return [self._estimate_one(track, width / 2, height / 2, half_diagonal) for track in tracks]

    def _estimate_one(self, track: Track, center_x: float, center_y: float, half_diagonal: float) -> TtcResult:
        # 면적 ∝ 1/거리² 이므로 TTC ≈ 2A / (dA/dt)
        ttc = 2 * track.area / track.va if track.area > 0 and track.va > 0 else math.inf

        dx, dy = track.cx - center_x, track.cy - center_y
        distance = math.hypot(dx, dy)
        dr_dt = (dx * track.vx + dy * track.vy) / distance / half_diagonal if distance > 0 else 0.0
        r = distance / half_diagonal

        # 곧장 날아오는 물체는 같은 픽셀에 머문 채 커지므로, 수렴뿐 아니라 "벌어지지 않음"도 충돌 코스로 본다
        on_course = r <= self.center_radius or dr_dt <= self.converge_eps
        return TtcResult(track.track_id, ttc, on_course, r, dr_dt)
