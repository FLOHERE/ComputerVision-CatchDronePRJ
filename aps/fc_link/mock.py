import logging

from aps.types import ThreatEvent

logger = logging.getLogger(__name__)


class MockFcLink:
    """실제 송신 없이 회피 요청을 기록한다."""

    def __init__(self):
        self.calls: list[ThreatEvent] = []

    def evade(self, event: ThreatEvent) -> None:
        self.calls.append(event)
        logger.warning("MOCK evade: track=%d ttc=%.3fs frame=%d", event.track_id, event.ttc, event.frame_index)
