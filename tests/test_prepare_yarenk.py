from tools.prepare_yarenk import source_name, split_by_source

PATHS = [f"/d/yarenk/dataset/train/images/{i}_jpg.rf.{h}.jpg" for i in range(20) for h in ("a1", "b2", "c3")]


def test_source_name_strips_roboflow_suffix():
    assert source_name("/x/12_jpg.rf.6fe528cf.jpg") == "12_jpg"


def test_split_keeps_augmented_copies_together():
    train, test = split_by_source(PATHS, 0.2, 42)
    assert len(test) == 12 and len(train) == 48
    assert not {source_name(p) for p in train} & {source_name(p) for p in test}
    assert set(split_by_source(list(reversed(PATHS)), 0.2, 42)[1]) == set(test)
