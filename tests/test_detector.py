from pathlib import Path

import cv2
from ultralytics.utils import ASSETS

from aps.detector.yolo import YoloDetector
from aps.device import select_device

ROOT = Path(__file__).parent.parent


def test_select_device():
    assert select_device("cpu") == "cpu"
    assert select_device("auto") in {"mps", "cuda", "cpu"}


def test_yolo_detects_bus_with_class_filter(cfg):
    detector_cfg = {**cfg["detector"], "weights": str(ROOT / cfg["detector"]["weights"]), "conf": 0.25, "classes": [5]}
    image = cv2.imread(str(ASSETS / "bus.jpg"))
    result = YoloDetector(detector_cfg, select_device("auto")).detect(image)

    assert result.detections, "bus.jpg에서 버스를 찾아야 한다"
    assert all(d.cls == 5 for d in result.detections)
    x1, y1, x2, y2 = result.detections[0].xyxy
    assert 0 <= x1 < x2 <= image.shape[1] and 0 <= y1 < y2 <= image.shape[0]
    assert result.preprocess_ms > 0 and result.inference_ms > 0
