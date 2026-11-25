"""ArduPilot과 MAVLink로 통신하여 회피 명령을 보낸다."""

import logging
import time
from pymavlink import mavutil

from aps.types import ThreatEvent

logger = logging.getLogger(__name__)


class MavlinkFcLink:
    """
    THREAT 이벤트 발생 시 GUIDED 모드로 전환하여 상승 회피 명령을 보낸 뒤 원래 모드로 복귀한다.

    회피 전략:
    - GUIDED 모드 전환
    - SET_POSITION_TARGET_LOCAL_NED로 상승 속도 명령 (NED 좌표계에서 vz < 0 = 상승)
    - 지정된 시간 동안 명령 유지
    - 원래 모드로 복귀
    """

    def __init__(self, cfg: dict):
        """
        Args:
            cfg: config.yaml의 fc_link 섹션
                - connection_string: MAVLink 연결 문자열 (예: "udp:127.0.0.1:14550")
                - evade_vz_mps: 상승 회피 속도 (m/s, 양수)
                - evade_duration_s: 회피 명령 지속 시간 (초)
                - timeout_s: 연결/명령 타임아웃 (초)
        """
        self.connection_string = cfg["connection_string"]
        self.evade_vz = -abs(cfg["evade_vz_mps"])  # NED 좌표계에서 음수=상승
        self.duration = cfg["evade_duration_s"]
        self.timeout = cfg["timeout_s"]

        self._master = None
        self._connected = False
        self._original_mode = None

        # 연결 시도
        self._connect()

    def _connect(self) -> bool:
        """MAVLink 연결을 시도한다."""
        try:
            logger.info(f"MAVLink 연결 시도: {self.connection_string}")
            self._master = mavutil.mavlink_connection(
                self.connection_string,
                timeout=self.timeout
            )

            # Heartbeat 대기
            logger.info("Heartbeat 대기 중...")
            self._master.wait_heartbeat(timeout=self.timeout)

            self._connected = True
            logger.info(
                f"MAVLink 연결 성공: system={self._master.target_system}, "
                f"component={self._master.target_component}"
            )
            return True

        except Exception as e:
            logger.error(f"MAVLink 연결 실패: {e}")
            self._connected = False
            return False

    def _get_mode(self) -> int:
        """현재 비행 모드를 가져온다."""
        if not self._connected or not self._master:
            return -1

        try:
            msg = self._master.recv_match(type='HEARTBEAT', blocking=True, timeout=1.0)
            if msg:
                return msg.custom_mode
            return -1
        except Exception as e:
            logger.error(f"모드 조회 실패: {e}")
            return -1

    def _set_mode(self, mode_name: str) -> bool:
        """
        비행 모드를 변경한다.

        Args:
            mode_name: 모드 이름 (예: "GUIDED", "AUTO")

        Returns:
            성공 여부
        """
        if not self._connected or not self._master:
            return False

        try:
            # 모드 ID 가져오기
            mode_id = self._master.mode_mapping().get(mode_name)
            if mode_id is None:
                logger.error(f"알 수 없는 모드: {mode_name}")
                return False

            # 모드 변경 명령
            self._master.mav.set_mode_send(
                self._master.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_id
            )

            # 모드 변경 확인 (최대 2초 대기)
            start_time = time.time()
            while time.time() - start_time < 2.0:
                msg = self._master.recv_match(type='HEARTBEAT', blocking=True, timeout=0.5)
                if msg and msg.custom_mode == mode_id:
                    logger.info(f"모드 변경 성공: {mode_name}")
                    return True

            logger.warning(f"모드 변경 타임아웃: {mode_name}")
            return False

        except Exception as e:
            logger.error(f"모드 변경 실패 ({mode_name}): {e}")
            return False

    def _send_velocity_ned(self, vx: float, vy: float, vz: float) -> bool:
        """
        NED 좌표계 속도 명령을 보낸다.

        Args:
            vx: North 방향 속도 (m/s)
            vy: East 방향 속도 (m/s)
            vz: Down 방향 속도 (m/s, 음수=상승)

        Returns:
            성공 여부
        """
        if not self._connected or not self._master:
            return False

        try:
            self._master.mav.set_position_target_local_ned_send(
                0,  # time_boot_ms (0 = 무시)
                self._master.target_system,
                self._master.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                0b0000111111000111,  # 속도만 사용 (type_mask)
                0, 0, 0,  # x, y, z 위치 (무시)
                vx, vy, vz,  # 속도
                0, 0, 0,  # 가속도 (무시)
                0, 0  # yaw, yaw_rate (무시)
            )
            return True

        except Exception as e:
            logger.error(f"속도 명령 송신 실패: {e}")
            return False

    def _get_mode_name(self, mode_id: int) -> str:
        """모드 ID를 모드 이름으로 변환한다."""
        if not self._master:
            return "UNKNOWN"

        mode_mapping = self._master.mode_mapping()
        for name, id_ in mode_mapping.items():
            if id_ == mode_id:
                return name
        return f"CUSTOM({mode_id})"

    def evade(self, event: ThreatEvent) -> None:
        """
        THREAT 이벤트에 대응하여 상승 회피 명령을 보낸다.

        Args:
            event: THREAT 이벤트 (track_id, t, frame_index, ttc 포함)
        """
        logger.warning(
            f"THREAT 감지: track={event.track_id} ttc={event.ttc:.3f}s frame={event.frame_index}"
        )

        # 연결되지 않았으면 경고만 출력
        if not self._connected or not self._master:
            logger.error("MAVLink 연결 없음. 회피 명령 송신 불가")
            return

        # 1. 현재 모드 저장
        self._original_mode = self._get_mode()
        original_mode_name = self._get_mode_name(self._original_mode)
        logger.info(f"현재 모드 저장: {original_mode_name}")

        # 2. GUIDED 모드로 전환
        if not self._set_mode("GUIDED"):
            logger.error("GUIDED 모드 전환 실패. 회피 중단")
            return

        # 3. 상승 회피 명령 송신
        logger.info(f"상승 회피 명령: vz={self.evade_vz:.2f} m/s, {self.duration:.2f}초")

        start_time = time.time()
        while time.time() - start_time < self.duration:
            if not self._send_velocity_ned(0, 0, self.evade_vz):
                logger.error("속도 명령 송신 실패")
                break
            time.sleep(0.1)  # 10Hz로 명령 반복

        # 4. 속도 명령 정지 (hover)
        self._send_velocity_ned(0, 0, 0)
        logger.info("회피 완료. hover 상태")

        # 5. 원래 모드로 복귀
        if self._original_mode >= 0:
            original_mode_name = self._get_mode_name(self._original_mode)
            if self._set_mode(original_mode_name):
                logger.info(f"원래 모드로 복귀: {original_mode_name}")
            else:
                logger.warning(f"원래 모드 복귀 실패: {original_mode_name}")
        else:
            logger.warning("원래 모드 불명. AUTO 모드로 전환 시도")
            self._set_mode("AUTO")

    def close(self) -> None:
        """MAVLink 연결을 종료한다."""
        if self._master:
            self._master.close()
            self._connected = False
            logger.info("MAVLink 연결 종료")
