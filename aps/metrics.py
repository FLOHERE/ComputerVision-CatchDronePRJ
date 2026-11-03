import csv
import logging
import platform
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

COLUMNS = [
    "frame", "platform", "device", "imgsz",
    "preprocess_ms", "inference_ms", "tracking_ms", "ttc_ms", "total_ms",
    "n_dets", "n_tracks",
]  # fmt: skip


class LatencyLogger:
    """프레임별 지연시간 CSV. 장치 정보를 행마다 넣어 PC와 Pi 결과를 그대로 합칠 수 있게 한다."""

    def __init__(self, path: str | Path, device: str, imgsz: int, budget_ms: float):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.budget_ms = budget_ms
        self._common = {"platform": f"{platform.system()}-{platform.machine()}", "device": device, "imgsz": imgsz}
        self._totals: list[float] = []
        self._file = path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=COLUMNS)
        self._writer.writeheader()

    def log(self, frame: int, preprocess_ms: float, inference_ms: float, tracking_ms: float, ttc_ms: float,
            total_ms: float, n_dets: int, n_tracks: int) -> None:  # fmt: skip
        self._writer.writerow({
            "frame": frame, **self._common,
            "preprocess_ms": f"{preprocess_ms:.3f}", "inference_ms": f"{inference_ms:.3f}",
            "tracking_ms": f"{tracking_ms:.3f}", "ttc_ms": f"{ttc_ms:.3f}", "total_ms": f"{total_ms:.3f}",
            "n_dets": n_dets, "n_tracks": n_tracks,
        })  # fmt: skip
        self._totals.append(total_ms)
        logger.info(
            "frame=%d pre=%.1f infer=%.1f track=%.2f ttc=%.2f total=%.1fms",
            frame, preprocess_ms, inference_ms, tracking_ms, ttc_ms, total_ms,
        )  # fmt: skip

    def summary(self) -> dict[str, float]:
        if not self._totals:
            return {"frames": 0}
        totals = np.array(self._totals)
        return {
            "frames": len(totals),
            "median_ms": float(np.median(totals)),
            "p95_ms": float(np.percentile(totals, 95)),
            "over_budget_ratio": float((totals > self.budget_ms).mean()),
        }

    def close(self) -> None:
        self._file.close()
