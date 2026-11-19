"""
기존 파이프라인(탐지·트래커·TTC) 위에 그물 발사 애니메이션만 얹어 보는 시험

사용법:
    python run_catch.py --source vdieos/002.mp4 --save runs/catch.mp4
"""

import argparse
import logging
from pathlib import Path

import cv2
import yaml

from aps.catch.draw import draw_net
from aps.catch.simulator import CatchSimulator
from aps.detector.yolo import YoloDetector
from aps.device import select_device
from aps.io.osd import OsdOverlaySource, OverlayFilterDetector
from aps.io.overlay import draw_overlay
from aps.io.sources import open_source
from aps.net_link.mock import MockNetLink
from aps.pipeline import Pipeline


def main():
    p = argparse.ArgumentParser(description="그물 발사 애니메이션 시험")
    p.add_argument("--source", required=True)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--save", help="결과 영상을 저장할 경로 (mp4)")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--arm", action="store_true", help="무장 상태로 시작 (mock 발사 요청을 허용)")
    p.add_argument("--start", type=float, default=0.0, help="저장 시작 시각(초). 처리는 처음부터 하되 이 구간만 저장")
    p.add_argument("--end", type=float, default=float("inf"), help="저장 끝 시각(초)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    device = select_device(cfg["device"])
    source = open_source(args.source)
    fps = getattr(source, "fps", 30.0)
    detector = YoloDetector(cfg["detector"], device)
    if cfg["osd"]["enabled"]:
        source = OsdOverlaySource(source, cfg["osd"])
        detector = OverlayFilterDetector(detector, source, cfg["osd"]["max_overlap"])
    pipeline, sim = Pipeline(cfg, detector), CatchSimulator(cfg["catch"])
    link = MockNetLink({**cfg["net_link"], "armed": args.arm or cfg["net_link"]["armed"]})
    sim.subscribe(link.fire)

    writer, last_phase = None, "idle"
    try:
        for frame in source:
            result = pipeline.process(frame)
            state = sim.update(result.tracks, result.ttc_results, frame.image.shape, frame.t)
            if state.phase != last_phase:
                print(f"frame {frame.index} t={frame.t:.2f}s {state.phase} {state}")
                last_phase = state.phase
            canvas = draw_overlay(frame.image, result.tracks, result.ttc_results, result.threat_ids)
            canvas = draw_net(canvas, sim, state, frame.t)
            if args.save and args.start <= frame.t <= args.end:
                if writer is None:
                    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
                    h, w = canvas.shape[:2]
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(canvas)
            if not args.no_display:
                cv2.imshow("APS catch", canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
