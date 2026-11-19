from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class NetFireEvent:
    track_id: int
    t: float
    ttc: float
    distance_m: float  # 박스 크기로 추정한 값. 참고용
    meet_m: float  # 그물과 표적이 만날 것으로 계산된 지점의 내 기체로부터 거리


class NetLink(Protocol):
    def fire(self, event: NetFireEvent) -> bool:
        """발사 요청. 실제로 발사 신호를 냈으면 True, 안전 조건에 막혔으면 False."""
        ...
