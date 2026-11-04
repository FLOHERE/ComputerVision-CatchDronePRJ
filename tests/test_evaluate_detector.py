import numpy as np

from tools.evaluate_detector import match_boxes, size_bin

EDGES = np.array([0.1, 0.5])


def box(x, size):
    return [x, 0.0, x + size, size]


def test_size_bin_boundaries():
    assert size_bin(box(0, 0.05), EDGES) == 0
    assert size_bin(box(0, 0.2), EDGES) == 1
    assert size_bin(box(0, 0.6), EDGES) == 2


def test_each_gt_matches_only_one_prediction():
    gt = np.array([box(0, 0.2), box(0.5, 0.05)])
    pred = np.array([box(0, 0.2), box(0, 0.2)])

    totals, hits, false_positives = match_boxes(gt, pred, 0.5, EDGES)

    assert totals.tolist() == [1, 1, 0]
    assert hits.tolist() == [0, 1, 0]
    assert false_positives == 1


def test_low_iou_is_false_positive_and_miss():
    totals, hits, false_positives = match_boxes(np.array([box(0, 0.2)]), np.array([box(0.5, 0.2)]), 0.5, EDGES)

    assert hits.sum() == 0 and totals.sum() == 1 and false_positives == 1


def test_empty_inputs():
    empty = np.empty((0, 4))
    assert match_boxes(empty, np.array([box(0, 0.2)]), 0.5, EDGES)[2] == 1
    totals, hits, false_positives = match_boxes(np.array([box(0, 0.2)]), empty, 0.5, EDGES)
    assert totals.sum() == 1 and hits.sum() == 0 and false_positives == 0
