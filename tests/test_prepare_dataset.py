import zipfile

import pytest

from tools.prepare_dataset import _extract, is_valid_label, split_images, split_sequence, voc_to_yolo

CFG = {"seed": 42, "train_images": 6, "val_images": 2, "negative_ratio": 0.5, "test_negatives": 2}


@pytest.mark.parametrize(
    "text,expected",
    [
        ("0 0.5 0.5 0.1 0.1\n", True),
        ("0 0.5 0.5 0.1 0.1\n0 0.2 0.2 0.1 0.1\n", True),
        ("", False),
        ("1 0.5 0.5 0.1 0.1\n", False),
        ("0 0.5 0.5 0.1\n", False),
        ("0 1.5 0.5 0.1 0.1\n", False),
        ("0 0.5 0.5 0 0.1\n", False),
        ("0 a 0.5 0.1 0.1\n", False),
    ],
)
def test_label_validation(text, expected):
    assert is_valid_label(text) is expected


def test_splits_do_not_overlap_and_are_reproducible():
    positives = [f"p{i}" for i in range(10)]
    negatives = [f"n{i}" for i in range(10)]
    splits = split_images(positives, negatives, CFG)

    assert len(splits["train"]) == 6 + 3
    assert len(splits["val"]) == 2 + 1
    assert len(splits["test_negatives"]) == 2
    all_images = splits["train"] + splits["val"] + splits["test_negatives"]
    assert len(all_images) == len(set(all_images))
    assert split_images(list(reversed(positives)), negatives, CFG) == splits


def test_too_few_images_raises():
    with pytest.raises(ValueError):
        split_images(["p"], [f"n{i}" for i in range(10)], CFG)
    with pytest.raises(ValueError):
        split_images([f"p{i}" for i in range(10)], ["n"], CFG)


def test_extract_creates_target_and_ignores_archive_paths(tmp_path):
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("nested/1.jpg", b"one")
        z.writestr("../../evil.jpg", b"two")
        z.writestr("skip.jpg", b"three")
    target = tmp_path / "new" / "dir"

    _extract(archive, target, {"1.jpg", "evil.jpg"})

    assert sorted(p.name for p in target.iterdir()) == ["1.jpg", "evil.jpg"]
    assert (target / "1.jpg").read_bytes() == b"one"
    assert not (tmp_path / "evil.jpg").exists()


VOC = """<annotation><size><width>200</width><height>100</height></size>
<object><name>x</name><bndbox><xmin>20</xmin><ymin>10</ymin><xmax>60</xmax><ymax>30</ymax></bndbox></object>
</annotation>"""


def test_voc_to_yolo():
    assert voc_to_yolo(VOC) == "0 0.200000 0.200000 0.200000 0.200000\n"


def test_voc_to_yolo_clips_and_rejects_invalid_boxes():
    clipped = VOC.replace("<xmax>60</xmax>", "<xmax>500</xmax>")
    assert voc_to_yolo(clipped).split()[3] == "0.900000"
    assert voc_to_yolo(VOC.replace("<xmax>60</xmax>", "<xmax>10</xmax>")) is None


def test_split_sequence_uses_time_blocks_without_overlap():
    frames = [f"f{i:03d}" for i in range(100)]
    splits = split_sequence(list(reversed(frames)), {"val_ratio": 0.1, "test_ratio": 0.1, "train_stride": 2})
    assert splits["train"] == frames[:80:2]
    assert splits["val"] == frames[80:90]
    assert splits["test"] == frames[90:]
