import pytest

from aps.tracker.tracker import Tracker
from aps.types import Detection
from synthetic import FPS, FRAME_SHAPE, approach_center, generate


def box(cx, cy, size=20.0):
    return Detection((cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2), 0.9, 0)


def test_unconfirmed_until_min_hits(cfg):
    tracker = Tracker(cfg["tracker"])
    min_hits = cfg["tracker"]["min_hits"]
    for i in range(min_hits - 1):
        assert tracker.update([box(300 + i, 300)], i / FPS, FRAME_SHAPE) == []
    assert len(tracker.update([box(300 + min_hits, 300)], min_hits / FPS, FRAME_SHAPE)) == 1


def test_keeps_id_through_missed_detections(cfg):
    tracker = Tracker(cfg["tracker"])
    samples = generate(approach_center(), 1.0)
    ids = set()
    for i, sample in enumerate(samples):
        missing = 15 <= i < 15 + cfg["tracker"]["max_age"]
        tracks = tracker.update([] if missing else [sample.detection], sample.t, FRAME_SHAPE)
        ids |= {t.track_id for t in tracks}
    assert len(ids) == 1


def test_track_removed_after_max_age(cfg):
    tracker = Tracker(cfg["tracker"])
    t = 0.0
    for i in range(cfg["tracker"]["min_hits"]):
        t = i / FPS
        tracks = tracker.update([box(300, 300)], t, FRAME_SHAPE)
    assert len(tracks) == 1
    for i in range(cfg["tracker"]["max_age"] + 1):
        t += 1 / FPS
        tracks = tracker.update([], t, FRAME_SHAPE)
    assert tracks == []


def test_crossing_tracks_keep_ids(cfg):
    tracker = Tracker(cfg["tracker"])
    final = {}
    first = {}
    for i in range(60):
        t = i / FPS
        # 프레임 수 기준 화면 중앙에서 교차하는 두 궤적
        tracks = tracker.update([box(200 + 10 * i, 300), box(800 - 10 * i, 340)], t, FRAME_SHAPE)
        for track in tracks:
            first.setdefault(track.track_id, track.vx)
            final[track.track_id] = track.vx
    assert len(final) == 2
    for track_id, vx in final.items():
        assert (vx > 0) == (first[track_id] > 0)


def test_fast_small_box_without_overlap_is_matched(cfg):
    # 프레임 사이 이동량이 박스 크기보다 커서 IoU로는 연결할 수 없는 경우
    tracker = Tracker(cfg["tracker"])
    for i in range(cfg["tracker"]["min_hits"]):
        tracks = tracker.update([box(100 + 30 * i, 300, size=8.0)], i / FPS, FRAME_SHAPE)
    assert len(tracks) == 1


def test_rejects_non_increasing_time(cfg):
    tracker = Tracker(cfg["tracker"])
    tracker.update([], 1.0, FRAME_SHAPE)
    with pytest.raises(ValueError):
        tracker.update([], 1.0, FRAME_SHAPE)


def low(cx, cy, size=20.0):
    return Detection((cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2), 0.07, 0)


def test_low_conf_box_never_starts_track(cfg):
    tracker = Tracker(cfg["tracker"])
    for i in range(cfg["tracker"]["min_hits"] * 3):
        assert tracker.update([low(300 + i, 300)], i / FPS, FRAME_SHAPE) == []


def test_low_conf_box_keeps_confirmed_track(cfg):
    tracker = Tracker(cfg["tracker"])
    n = cfg["tracker"]["min_hits"]
    for i in range(n):
        tracks = tracker.update([box(300 + i, 300)], i / FPS, FRAME_SHAPE)
    track_id = tracks[0].track_id
    # max_age보다 길게 저신뢰 박스만 들어와도 같은 트랙이 유지되어야 함
    for i in range(n, n + cfg["tracker"]["max_age"] * 3):
        tracks = tracker.update([low(300 + i, 300)], i / FPS, FRAME_SHAPE)
        assert [t.track_id for t in tracks] == [track_id] and tracks[0].missed == 0
