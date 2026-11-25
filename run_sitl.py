"""
Edge Vision APS + ArduPilot SITL 통합 테스트

SITL과 연동하여 THREAT 이벤트 발생 시 실제 MAVLink 회피 명령을 보낸다.

사용법:
    # SITL 먼저 실행 (별도 터미널)
    cd ~/ardupilot/ArduCopter
    sim_vehicle.py --console --map

    # 이 스크립트 실행
    python run_sitl.py --source video.mp4 --sitl
    python run_sitl.py --source 0 --sitl --no-display

    # mock 모드 (SITL 없이 테스트)
    python run_sitl.py --source video.mp4
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import cv2
import yaml

from aps.detector.yolo import YoloDetector
from aps.device import select_device
from aps.fc_link.mock import MockFcLink
from aps.fc_link.mavlink import MavlinkFcLink
from aps.io.osd import OsdOverlaySource, OverlayFilterDetector
from aps.io.overlay import draw_overlay
from aps.io.sources import open_source
from aps.metrics import LatencyLogger
from aps.pipeline import Pipeline


def parse_args():
    p = argparse.ArgumentParser(description="APS + SITL 통합 테스트")
    p.add_argument("--source", required=True, help="영상 파일 경로 또는 웹캠 번호")
    p.add_argument("--config", default="config.yaml", help="설정 파일 경로")
    p.add_argument("--sitl", action="store_true", help="SITL MAVLink 연결 활성화 (없으면 mock)")
    p.add_argument("--no-display", action="store_true", help="오버레이 창을 띄우지 않음")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    device = select_device(cfg["device"])
    source = open_source(args.source)
    detector = YoloDetector(cfg["detector"], device)

    if cfg["osd"]["enabled"]:
        source = OsdOverlaySource(source, cfg["osd"])
        detector = OverlayFilterDetector(detector, source, cfg["osd"]["max_overlap"])

    csv_path = Path(cfg["metrics"]["csv_dir"]) / f"latency_sitl_{datetime.now():%Y%m%d_%H%M%S}.csv"
    metrics = LatencyLogger(csv_path, device, cfg["detector"]["imgsz"], cfg["metrics"]["budget_ms"])

    pipeline = Pipeline(cfg, detector, metrics)

    # fc_link 선택: --sitl 플래그에 따라 MavlinkFcLink 또는 MockFcLink
    if args.sitl:
        logging.info("SITL 모드: MavlinkFcLink 사용")
        fc_link = MavlinkFcLink(cfg["fc_link"])
        pipeline.decider.subscribe(fc_link.evade)
    else:
        logging.info("Mock 모드: MockFcLink 사용 (SITL 연결 없음)")
        mock_fc = MockFcLink()
        pipeline.decider.subscribe(mock_fc.evade)

    try:
        for frame in source:
            result = pipeline.process(frame)
            if not args.no_display:
                canvas = draw_overlay(frame.image, result.tracks, result.ttc_results, result.threat_ids)

                # SITL 연결 상태 표시
                status_text = "SITL: CONNECTED" if args.sitl else "SITL: MOCK"
                cv2.putText(
                    canvas, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if args.sitl else (128, 128, 128), 2
                )

                cv2.imshow("APS + SITL", canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        logging.info("사용자 중단")
    finally:
        # MavlinkFcLink 연결 종료
        if args.sitl and hasattr(fc_link, 'close'):
            fc_link.close()

        metrics.close()
        cv2.destroyAllWindows()

    print(f"\nCSV: {csv_path}")
    print(f"요약: {metrics.summary()}")

    # Mock 모드에서 호출 로그 출력
    if not args.sitl:
        print(f"\nMock fc_link 호출 횟수: {len(mock_fc.calls)}")
        for i, event in enumerate(mock_fc.calls, 1):
            print(f"  {i}. track={event.track_id} ttc={event.ttc:.3f}s frame={event.frame_index}")


if __name__ == "__main__":
    main()
