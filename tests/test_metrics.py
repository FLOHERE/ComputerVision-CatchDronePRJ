import csv

import pytest

from aps.metrics import COLUMNS, LatencyLogger


def test_csv_format_and_summary(tmp_path):
    path = tmp_path / "sub" / "latency.csv"
    logger = LatencyLogger(path, "mps", 960, budget_ms=30.0)
    logger.log(0, 1.0, 10.0, 0.1, 0.01, 20.0, 1, 1)
    logger.log(1, 1.0, 30.0, 0.1, 0.01, 40.0, 2, 1)
    summary = logger.summary()
    logger.close()

    rows = list(csv.DictReader(path.open()))
    assert list(rows[0]) == COLUMNS
    assert rows[0]["device"] == "mps" and rows[0]["imgsz"] == "960" and rows[0]["platform"]
    assert float(rows[1]["total_ms"]) == 40.0
    assert summary["frames"] == 2
    assert summary["median_ms"] == pytest.approx(30.0)
    assert summary["over_budget_ratio"] == pytest.approx(0.5)


def test_summary_without_frames(tmp_path):
    logger = LatencyLogger(tmp_path / "x.csv", "cpu", 640, budget_ms=30.0)
    assert logger.summary() == {"frames": 0}
    logger.close()
