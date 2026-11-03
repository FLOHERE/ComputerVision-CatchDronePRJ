import math

import cv2
import numpy as np

from aps.types import Track, TtcResult

_NORMAL = (0, 255, 0)
_THREAT = (0, 0, 255)


def draw_overlay(image: np.ndarray, tracks: list[Track], results: list[TtcResult], threat_ids: set[int]) -> np.ndarray:
    canvas = image.copy()
    by_id = {r.track_id: r for r in results}
    for track in tracks:
        threat = track.track_id in threat_ids
        color = _THREAT if threat else _NORMAL
        x1, y1, x2, y2 = (int(v) for v in track.bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3 if threat else 1)

        ttc = by_id[track.track_id].ttc
        ttc_text = "inf" if math.isinf(ttc) else f"{ttc:.2f}s"
        label = f"#{track.track_id} TTC {ttc_text}" + (" THREAT" if threat else "")
        cv2.putText(canvas, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return canvas
