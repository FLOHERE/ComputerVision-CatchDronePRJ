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
