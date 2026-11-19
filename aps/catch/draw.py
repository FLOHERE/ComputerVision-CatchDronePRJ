import cv2
import numpy as np

from aps.catch.simulator import CatchSimulator, CatchState

_GRID = 8
_NET = (255, 255, 0)
_HIT = (0, 255, 0)
_MISS = (0, 0, 255)


def draw_net(image: np.ndarray, sim: CatchSimulator, state: CatchState, t: float) -> np.ndarray:
    """발사기가 카메라 광축에 고정이라 그물은 화면 중앙에서 날아가 멀어지며 작아진다."""
    if state.phase == "idle":
        return image
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    dist = state.meet_m if state.phase == "result" else sim.cfg["net_speed_mps"] * max(t - state.fire_t - sim.cfg["latency_s"], 0.0)
    half = int(min(sim.net_half_px(max(dist, 1e-3), width), max(width, height)))

    layer = image.copy()
    for k in range(_GRID + 1):
        o = int(-half + 2 * half * k / _GRID)
        cv2.line(layer, (center[0] + o, center[1] - half), (center[0] + o, center[1] + half), _NET, 1, cv2.LINE_AA)
        cv2.line(layer, (center[0] - half, center[1] + o), (center[0] + half, center[1] + o), _NET, 1, cv2.LINE_AA)
    canvas = cv2.addWeighted(layer, 0.6, image, 0.4, 0)
    cv2.rectangle(canvas, (center[0] - half, center[1] - half), (center[0] + half, center[1] + half), _NET, 2)

    if state.phase == "flying":
        text, color = f"NET FIRED  D~{state.distance_m:.1f}m  meet {state.meet_m:.1f}m", _NET
    else:
        text, color = {True: ("CAUGHT", _HIT), False: ("MISS", _MISS), None: ("TARGET LOST", _MISS)}[state.caught]
    cv2.putText(canvas, text, (center[0] - half, max(center[1] - half - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return canvas
