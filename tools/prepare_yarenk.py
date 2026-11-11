"""
공대공 고정익 데이터(github.com/yarenk/fixed-wing-uav-dataset-and-model, CC BY 4.0)로 재학습 목록을 만든다.
Roboflow 증강으로 원본 한 장마다 사본이 3장씩 있으므로, 시험셋은 원본 이름 단위로 떼어 학습과 겹치지 않게 한다.

사용법:
    python tools/prepare_yarenk.py   # 먼저 datasets/yarenk/dataset/ 에 release의 dataset.zip을 풀어 둔다
"""

import argparse
import random
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.prepare_dataset import is_valid_label
from tools.prepare_synthetic import compose_train


def source_name(path: str) -> str:
    """'12_jpg.rf.<hash>.jpg' → '12_jpg'. 같은 원본의 증강 사본은 같은 이름이 된다."""
    return Path(path).name.split(".rf.")[0]


def split_by_source(paths: list[str], test_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    sources = sorted({source_name(p) for p in paths})
    random.Random(seed).shuffle(sources)
    test_sources = set(sources[: int(len(sources) * test_ratio)])
    test = [p for p in paths if source_name(p) in test_sources]
    return [p for p in paths if source_name(p) not in test_sources], test


def main():
    p = argparse.ArgumentParser(description="공대공 고정익 데이터 준비")
    p.add_argument("--config", default="config.yaml", help="설정 파일 경로")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    training = cfg["training"]
    yarenk = training["yarenk"]
    root = Path(training["dataset_root"]).resolve()
    data_dir = root / "drone"

    valid = []
    for image in sorted((root / "yarenk" / "dataset").glob("*/images/*.jpg")):
        label = Path(str(image).replace("/images/", "/labels/")).with_suffix(".txt")
        # 라벨이 빈 이미지는 음성인지 라벨 누락인지 알 수 없어 뺀다
        if label.exists() and is_valid_label(label.read_text(encoding="utf-8")):
            valid.append(str(image))

    rest, test = split_by_source(valid, yarenk["test_ratio"], training["seed"])
    base = (data_dir / "train.txt").read_text(encoding="utf-8").split()
    train = compose_train(base + rest, yarenk["train_limits"], training["seed"])
    (data_dir / "train_yarenk.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (data_dir / "test_yarenk.txt").write_text("\n".join(test) + "\n", encoding="utf-8")
    data = yaml.safe_load((data_dir / "data.yaml").read_text(encoding="utf-8"))
    (data_dir / "data_yarenk.yaml").write_text(yaml.safe_dump({**data, "train": "train_yarenk.txt"}), encoding="utf-8")
    print(f"유효 이미지 {len(valid)}장, 시험 {len(test)}장, 학습 목록 {len(train)}장")


if __name__ == "__main__":
    main()
