import logging

from aps.net_link.base import NetFireEvent

logger = logging.getLogger(__name__)


class MockNetLink:
    """실제 발사 없이 요청을 기록한다. 실기체 구현도 같은 안전 조건을 지켜야 한다."""

    def __init__(self, cfg: dict):
        self.min_meet_m = cfg["min_meet_m"]
        self.max_shots = cfg["max_shots"]
        self.armed = cfg["armed"]
        self.calls: list[NetFireEvent] = []  # 발사한 요청
        self.rejected: list[tuple[NetFireEvent, str]] = []  # 막힌 요청과 이유

    def arm(self) -> None:
        self.armed = True

    def disarm(self) -> None:
        self.armed = False

    def fire(self, event: NetFireEvent) -> bool:
        reason = self._block_reason(event)
        if reason:
            self.rejected.append((event, reason))
            logger.warning("MOCK net BLOCKED (%s): track=%d ttc=%.3fs meet=%.1fm", reason, event.track_id, event.ttc, event.meet_m)
            return False
        self.calls.append(event)
        logger.warning("MOCK net FIRE: track=%d ttc=%.3fs meet=%.1fm", event.track_id, event.ttc, event.meet_m)
        return True

    def _block_reason(self, event: NetFireEvent) -> str | None:
        if not self.armed:
            return "disarmed"
        if len(self.calls) >= self.max_shots:
            return "no net left"
        if event.meet_m < self.min_meet_m:
            return "too close"  # 그물이 펴지기 전에 만남
        return None
