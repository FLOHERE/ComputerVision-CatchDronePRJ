from typing import Protocol

import numpy as np

from aps.types import DetectionResult


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> DetectionResult: ...
