"""
드론 탐지 모델 학습. 끝나면 best.pt를 training.output_weights로 복사한다.

사용법:
    python tools/train_detector.py [--data 작은.yaml --epochs 1]  # 뒤의 두 옵션은 동작 확인용
    python tools/train_detector.py --finetune finetune_yarenk --name drone_v4
"""

import argparse
import shutil
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aps.device import select_device


def main():
    p = argparse.ArgumentParser(description="드론 탐지 모델 학습")
    p.add_argument("--config", default="config.yaml", help="설정 파일 경로")
    p.add_argument("--epochs", type=int, help="config의 epochs 대신 사용")
    p.add_argument("--data", help="기본 data.yaml 대신 사용")
    p.add_argument("--name", default="drone", help="runs/train 아래 실행 이름")
    p.add_argument("--finetune", nargs="?", const="finetune", help="training.<이름> 설정으로 재학습 (기본 finetune)")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    training = cfg["training"]
    if args.finetune:
        training = {**training, **training[args.finetune]}
    model = YOLO(training["base_weights"])
    results = model.train(
        data=args.data or training["data"],
        epochs=args.epochs or training["epochs"],
        patience=training["patience"],
        imgsz=training["imgsz"],
        batch=training["batch"],
        optimizer=training["optimizer"],
        lr0=training["lr0"],
        workers=training["workers"],
        seed=training["seed"],
        device=select_device(cfg["device"]),
        project=str(Path("runs/train").resolve()),
        name=args.name,
        exist_ok=False,
    )

    output = Path(training["output_weights"])
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(results.save_dir) / "weights" / "best.pt", output)
    print(f"가중치 저장: {output}")


if __name__ == "__main__":
    main()
