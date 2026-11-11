import cv2
import numpy as np
import pytest

from aps.io.overlay import draw_overlay
from aps.io.sources import VideoFileSource, WebcamSource, open_source
from aps.types import Track, TtcResult


def write_video(path, frames=5, fps=10.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    assert writer.isOpened()
    for _ in range(frames):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()


def test_video_source_uses_video_time(tmp_path):
    path = tmp_path / "clip.mp4"
    write_video(path)
    frames = list(VideoFileSource(str(path)))
    assert [f.index for f in frames] == list(range(5))
    assert [f.t for f in frames] == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])
    assert frames[0].image.shape == (48, 64, 3)


def test_missing_video_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        VideoFileSource(str(tmp_path / "none.mp4"))


def test_open_source_dispatch(tmp_path):
    path = tmp_path / "clip.mp4"
    write_video(path)
    assert isinstance(open_source("0"), WebcamSource)
    assert isinstance(open_source(str(path)), VideoFileSource)


def test_overlay_draws_without_mutating_input():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    tracks = [Track(1, 50, 50, 400, 0, 0, 0, (40, 40, 60, 60), 0), Track(2, 150, 50, 400, 0, 0, 0, (140, 40, 160, 60), 0)]
    results = [TtcResult(1, float("inf"), False, 0.0, 0.0), TtcResult(2, 0.3, True, 0.0, 0.0)]
    canvas = draw_overlay(image, tracks, results, {2})
    assert image.sum() == 0
    assert tuple(canvas[40, 50]) == (0, 255, 0)
    assert tuple(canvas[40, 150]) == (0, 0, 255)


def osd_frames(n=80, size=(120, 240)):
    """흔들리는 배경 위에 고정 조준점(+), 그 오른쪽에서 프레임마다 1px씩 움직이는 표적."""
    from aps.types import Frame

    rng = np.random.default_rng(0)
    h, w = size
    frames = []
    for i in range(n):
        img = rng.integers(60, 100, (h, w, 3), dtype=np.uint8)
        cv2.line(img, (w // 2 - 6, h // 2), (w // 2 + 6, h // 2), (255, 255, 255), 1)
        cv2.line(img, (w // 2, h // 2 - 6), (w // 2, h // 2 + 6), (255, 255, 255), 1)
        tx = w // 2 + 12 + i
        img[h // 2 - 3 : h // 2 + 3, tx : tx + 6] = 0
        frames.append(Frame(img, i * 0.1, i))
    return frames


def test_osd_filter_drops_reticle_keeps_target_and_image(cfg):
    from aps.io.osd import OsdOverlaySource, OverlayFilterDetector
    from aps.types import Detection, DetectionResult

    frames = osd_frames()
    h, w = frames[0].image.shape[:2]
    src = OsdOverlaySource(frames, cfg["osd"])
    out = list(src)
    assert [f.index for f in out] == list(range(len(frames)))
    assert out[-1].image is frames[-1].image  # 영상은 건드리지 않음 (흐림으로 원거리 표적을 잃지 않게)
    assert src.region[h // 2, w // 2] and not src.region[h // 2, w // 2 + 30]

    reticle = Detection((w / 2 - 7, h / 2 - 7, w / 2 + 7, h / 2 + 7), 0.9, 0)
    tx = w // 2 + 12 + len(frames) - 1
    target = Detection((tx, h / 2 - 3, tx + 6, h / 2 + 3), 0.3, 0)

    class Fake:
        def detect(self, image):
            return DetectionResult([reticle, target], 1.0, 2.0)

    result = OverlayFilterDetector(Fake(), src, cfg["osd"]["max_overlap"]).detect(out[-1].image)
    assert result.detections == [target]
    assert (result.preprocess_ms, result.inference_ms) == (1.0, 2.0)


def test_osd_detects_overlay_that_appears_later(cfg):
    """시작 구간에 없던 자막이 중간에 떠도 calib_frames 정도 지나면 영역에 들어간다."""
    from aps.io.osd import OsdOverlaySource
    from aps.types import Frame

    rng = np.random.default_rng(2)
    n0 = cfg["osd"]["calib_frames"]
    src = OsdOverlaySource([], cfg["osd"])
    for i in range(n0 * 4):
        img = rng.integers(0, 255, (120, 240, 3), dtype=np.uint8)
        if i >= n0:
            img[20:30, 20:60] = 255
            img[22:28, 22:58] = 0
        src.update(img)
    assert src.region is not None and src.region[25, 40] and not src.region[90, 200]


def _feed(src, images):
    for im in images:
        src.update(im)
    return src.region


def test_osd_region_disabled_for_static_scene(cfg):
    from aps.io.osd import OsdOverlaySource

    img = np.zeros((120, 160, 3), dtype=np.uint8)
    img[:, ::4] = 255  # 장면 전체가 정지한 고대비 패턴 = 고정 카메라
    assert _feed(OsdOverlaySource([], cfg["osd"]), [img] * (cfg["osd"]["calib_frames"] * 2)) is None


def test_osd_region_none_without_overlay(cfg):
    from aps.io.osd import OsdOverlaySource

    rng = np.random.default_rng(1)
    images = [rng.integers(0, 255, (120, 160, 3), dtype=np.uint8) for _ in range(cfg["osd"]["calib_frames"] * 2)]
    assert _feed(OsdOverlaySource([], cfg["osd"]), images) is None


def test_osd_region_kept_with_wide_banner_over_moving_scene(cfg):
    """005.mp4처럼 고정 제목 띠가 화면의 20%를 넘게 덮어도, 아래 장면이 움직이면 필터를 끄지 않는다."""
    from aps.io.osd import OsdOverlaySource

    rng = np.random.default_rng(3)
    images = []
    for _ in range(cfg["osd"]["calib_frames"] * 2):
        img = rng.integers(0, 255, (160, 120, 3), dtype=np.uint8)
        img[:40] = 40  # 위아래 띠 (화면의 50%)
        img[120:] = 40
        img[10:30:4, 5:115] = 255  # 띠 위의 고정 글자 줄
        img[130:150:4, 5:115] = 255
        images.append(img)
    region = _feed(OsdOverlaySource([], cfg["osd"]), images)
    assert region is not None and region.mean() > cfg["osd"]["max_region_ratio"]
    assert region[20, 60] and region[140, 60] and not region[80, 60]
