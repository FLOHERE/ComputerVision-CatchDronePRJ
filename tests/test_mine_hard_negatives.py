from tools.mine_hard_negatives import split_pool


def test_split_pool_is_disjoint_and_reproducible():
    paths = [f"n{i}" for i in range(10)]
    mine, dev = split_pool(paths, 0.5, 42)

    assert len(mine) == 5 and len(dev) == 5
    assert set(mine) | set(dev) == set(paths) and not set(mine) & set(dev)
    assert split_pool(list(reversed(paths)), 0.5, 42) == (mine, dev)
