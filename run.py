"""
Edge Vision APS 비전 파이프라인 실행

사용법:
    python run.py --source video.mp4
    python run.py --source 0 --no-display
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
from aps.io.osd import OsdOverlaySource, OverlayFilterDetector
from aps.io.overlay import draw_overlay
from aps.io.sources import open_source
from aps.metrics import LatencyLogger
from aps.pipeline import Pipeline


def parse_args():
    p = argparse.ArgumentParser(description="APS 비전 파이프라인")
    p.add_argument("--source", required=True, help="영상 파일 경로 또는 웹캠 번호")
    p.add_argument("--config", default="config.yaml", help="설정 파일 경로")
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
    csv_path = Path(cfg["metrics"]["csv_dir"]) / f"latency_{datetime.now():%Y%m%d_%H%M%S}.csv"
    metrics = LatencyLogger(csv_path, device, cfg["detector"]["imgsz"], cfg["metrics"]["budget_ms"])

    pipeline = Pipeline(cfg, detector, metrics)
    pipeline.decider.subscribe(MockFcLink().evade)

    try:
        for frame in source:
            result = pipeline.process(frame)
            if not args.no_display:
                canvas = draw_overlay(frame.image, result.tracks, result.ttc_results, result.threat_ids)
                cv2.imshow("APS", canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        metrics.close()
        cv2.destroyAllWindows()

    print(f"CSV: {csv_path}")
    print(f"요약: {metrics.summary()}")


if __name__ == "__main__":
    main()
