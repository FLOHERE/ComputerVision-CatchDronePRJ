import math
from dataclasses import dataclass
from typing import Callable

from aps.net_link.base import NetFireEvent
from aps.types import Track, TtcResult


@dataclass(frozen=True)
class CatchState:
    phase: str  # idle | flying | result
    fire_t: float = 0.0
    target_id: int = -1
    distance_m: float = 0.0  # 발사 시 추정 거리 (화면 표시용)
    arrive_s: float = 0.0  # 발사 후 그물이 표적과 만나기까지 시간
    meet_m: float = 0.0  # 만나는 지점의 내 기체로부터 거리
    result_t: float = 0.0
    caught: bool | None = None


class CatchSimulator:
    """TTC 기준으로 가상 발사하고, 도달 시점에 표적이 그물 안인지 판정한다. 거리는 표시용 추정값만 쓴다."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.state = CatchState("idle")
        self._shots = 0
        self._streak: dict[int, int] = {}
        self._callbacks: list[Callable[[NetFireEvent], object]] = []

    def subscribe(self, callback: Callable[[NetFireEvent], object]) -> None:
        self._callbacks.append(callback)

    def focal_px(self, width: int) -> float:
        return width / (2 * math.tan(math.radians(self.cfg["fov_deg"]) / 2))

    def net_half_px(self, dist_m: float, width: int) -> float:
        """그물 반폭의 화면 크기. 펴지는 구간에서는 폭이 거리에 비례해 커지므로 크기가 일정하다."""
        c = self.cfg
        return c["net_width_m"] / 2 * self.focal_px(width) / max(dist_m, c["open_dist_m"])

    def update(self, tracks: list[Track], results: list[TtcResult], shape: tuple[int, ...], t: float) -> CatchState:
        height, width = shape[:2]
        s = self.state
        if s.phase == "idle":
            self._try_fire(tracks, results, width, height, t)
        elif s.phase == "flying" and t - s.fire_t >= s.arrive_s:
            self.state = self._judge(tracks, width, height, t)
        elif s.phase == "result" and t - s.result_t > self.cfg["show_result_s"]:
            self.state = CatchState("idle")
        return self.state

    def _try_fire(self, tracks, results, width, height, t):
        c = self.cfg
        if self._shots >= c["max_shots"]:
            return
        f = self.focal_px(width)
        by_id = {r.track_id: r for r in results}
        ready = {}
        for track in tracks:
            r = by_id[track.track_id]
            off_deg = math.degrees(math.atan(math.hypot(track.cx - width / 2, track.cy - height / 2) / f))
            if track.missed == 0 and r.on_course and r.ttc <= c["fire_ttc_s"] and off_deg <= c["max_offcenter_deg"]:
                ready[track.track_id] = self._streak.get(track.track_id, 0) + 1
        self._streak = ready
        candidates = [tr for tr in tracks if ready.get(tr.track_id, 0) >= c["min_frames"]]
        if candidates:  # 동시에 여럿이면 TTC가 가장 짧은 표적
            target = min(candidates, key=lambda tr: by_id[tr.track_id].ttc)
            self._fire(target, by_id[target.track_id], f, t)

    def _fire(self, track: Track, r: TtcResult, f: float, t: float):
        c = self.cfg
        box_w = max(track.bbox[2] - track.bbox[0], 1.0)
        dist = c["target_size_m"] * f / box_w
        closing = dist / r.ttc if r.ttc > 0 else float("inf")  # 상대 속도 추정
        flight_ttc = max(r.ttc - c["latency_s"], 0.0)
        v_n = c["net_speed_mps"]
        arrive = flight_ttc * closing / (v_n + closing) if math.isfinite(closing) else 0.0
        self._shots += 1
        self._streak = {}
        self.state = CatchState("flying", t, track.track_id, dist, c["latency_s"] + arrive, v_n * arrive)
        event = NetFireEvent(track.track_id, t, r.ttc, dist, v_n * arrive)
        for callback in self._callbacks:
            callback(event)

    def _judge(self, tracks, width, height, t) -> CatchState:
        s = self.state
        # 칼만 예측으로만 남은 트랙(missed > 0)은 실측이 아니므로 판정하지 않는다
        target = next((tr for tr in tracks if tr.track_id == s.target_id and tr.missed == 0), None)
        caught = None  # None = 표적을 놓침
        if target is not None:
            miss_px = math.hypot(target.cx - width / 2, target.cy - height / 2)
            caught = miss_px <= self.net_half_px(s.meet_m, width)
        return CatchState("result", s.fire_t, s.target_id, s.distance_m, s.arrive_s, s.meet_m, t, caught)
