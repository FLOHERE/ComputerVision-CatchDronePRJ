"""
학습에 쓰지 않은 음성 이미지 중 현재 모델이 오탐하는 이미지를 골라 재학습용 목록을 만든다.
나머지 음성 이미지는 conf 조정용(dev_negatives.txt)으로 남긴다. test_negatives는 건드리지 않는다.

사용법:
    python tools/mine_hard_negatives.py
"""

import argparse
import random
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aps.device import select_device


def split_pool(paths, mine_ratio, seed):
    """입력 순서와 무관하게 같은 결과가 나오도록 정렬 후 섞어 (채굴용, 조정용)으로 나눈다."""
    shuffled = sorted(paths)
    random.Random(seed).shuffle(shuffled)
    n_mine = int(len(shuffled) * mine_ratio)
    return shuffled[:n_mine], shuffled[n_mine:]


def main():
    p = argparse.ArgumentParser(description="오탐 이미지 선별")
    p.add_argument("--config", default="config.yaml", help="설정 파일 경로")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    training = cfg["training"]
    finetune = training["finetune"]
    root = Path(training["dataset_root"]).resolve()
    data_dir = root / "drone"
    read = lambda name: (data_dir / f"{name}.txt").read_text(encoding="utf-8").split()

    train = read("train")
    used = set(train) | set(read("val")) | set(read("test_negatives"))
    pool = [str(p) for d in ("coco", "visdrone") for p in (root / "negatives" / "images" / d).iterdir()]
    mine, dev = split_pool([p for p in pool if p not in used], finetune["mine_ratio"], training["seed"])

    mine_file = data_dir / "mine_negatives.txt"
    mine_file.write_text("\n".join(mine) + "\n", encoding="utf-8")
    model = YOLO(finetune["base_weights"])
    results = model.predict(
        str(mine_file), imgsz=finetune["imgsz"], conf=finetune["hard_neg_conf"], device=select_device(cfg["device"]),
        batch=training["batch"], stream=True, verbose=False,
    )  # fmt: skip
    hard = [r.path for r in results if len(r.boxes)]

    (data_dir / "train_hard.txt").write_text("\n".join(train + hard) + "\n", encoding="utf-8")
    (data_dir / "dev_negatives.txt").write_text("\n".join(dev) + "\n", encoding="utf-8")
    data = yaml.safe_load((data_dir / "data.yaml").read_text(encoding="utf-8"))
    (data_dir / "data_hard.yaml").write_text(yaml.safe_dump({**data, "train": "train_hard.txt"}), encoding="utf-8")
    print(f"풀 {len(mine) + len(dev)}장: 채굴 {len(mine)}, 조정용 {len(dev)}, 추가된 오탐 이미지 {len(hard)}")


if __name__ == "__main__":
    main()
