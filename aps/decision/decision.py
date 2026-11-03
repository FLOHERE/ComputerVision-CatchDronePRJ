from typing import Callable

from aps.types import ThreatEvent, TtcResult


class ThreatDecider:
    def __init__(self, cfg: dict):
        self.ttc_threshold = cfg["ttc_threshold_s"]
        self._callbacks: list[Callable[[ThreatEvent], None]] = []
        self._active: set[int] = set()

    def subscribe(self, callback: Callable[[ThreatEvent], None]) -> None:
        self._callbacks.append(callback)

    def update(self, results: list[TtcResult], t: float, frame_index: int) -> list[ThreatEvent]:
        """위협 상태로 새로 진입한 트랙(상승 에지)에 대해서만 이벤트를 낸다."""
        threats = {r.track_id: r for r in results if r.ttc < self.ttc_threshold and r.on_course}
        events = [
            ThreatEvent(track_id, t, frame_index, r.ttc)
            for track_id, r in threats.items()
            if track_id not in self._active
        ]
        self._active = set(threats)
        for event in events:
            for callback in self._callbacks:
                callback(event)
        return events

    @property
    def active(self) -> set[int]:
        return set(self._active)
