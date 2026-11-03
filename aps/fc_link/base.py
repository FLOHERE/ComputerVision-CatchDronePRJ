from typing import Protocol

from aps.types import ThreatEvent


class FcLink(Protocol):
    def evade(self, event: ThreatEvent) -> None: ...
