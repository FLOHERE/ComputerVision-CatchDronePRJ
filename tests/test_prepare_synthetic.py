from tools.prepare_synthetic import compose_train

PATHS = [f"/d/seraphim/{i}.jpg" for i in range(5)] + [f"/d/synth/{i}.png" for i in range(5)] + ["/d/negatives/n.jpg"]


def test_compose_limits_sources_and_keeps_negatives():
    train = compose_train(PATHS, {"seraphim": 2, "synth": 3}, 42)

    assert sum("/seraphim/" in p for p in train) == 2
    assert sum("/synth/" in p for p in train) == 3
    assert "/d/negatives/n.jpg" in train
    assert compose_train(list(reversed(PATHS)), {"seraphim": 2, "synth": 3}, 42) == train


def test_compose_limit_larger_than_group_uses_all():
    assert sum("/seraphim/" in p for p in compose_train(PATHS, {"seraphim": 99}, 42)) == 5
