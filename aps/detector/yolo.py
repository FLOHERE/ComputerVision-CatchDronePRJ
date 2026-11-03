import numpy as np
from ultralytics import YOLO

from aps.types import Detection, DetectionResult


class YoloDetector:
    def __init__(self, cfg: dict, device: str):
        self.model = YOLO(cfg["weights"])
        self.imgsz = cfg["imgsz"]
        self.conf = cfg["conf"]
        self.classes = cfg["classes"]
        self.device = device

    def detect(self, image: np.ndarray) -> DetectionResult:
        result = self.model.predict(
            image,
            imgsz=self.imgsz,
            conf=self.conf,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )[0]
        boxes = result.boxes
        detections = [
            Detection(tuple(xyxy), conf, int(cls))
            for xyxy, conf, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist())
        ]
        speed = result.speed
        return DetectionResult(detections, speed["preprocess"], speed["inference"] + speed["postprocess"])
