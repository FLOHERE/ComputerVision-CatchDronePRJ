"""
합성 드론 데이터(Kaggle banderastepan/drone-detection, CC0)를 풀고 짧은 재학습용 목록을 만든다.
검증과 시험에 쓰는 기존 목록은 바꾸지 않고, 합성 이미지 일부는 test_synth.txt로 남긴다.

사용법:
    python tools/prepare_synthetic.py
"""

import argparse
import random
import sys
import zipfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.prepare_dataset import is_valid_label


def compose_train(paths, limits, seed):
    """음성 이미지는 모두 두고, limits에 있는 출처({폴더 이름: 장수})만 재현 가능하게 줄인다."""
    rng = random.Random(seed)
    selected = []
    for name, limit in limits.items():
        group = sorted(p for p in paths if f"/{name}/" in p)
        selected += rng.sample(group, min(limit, len(group)))
    return selected + [p for p in paths if not any(f"/{name}/" in p for name in limits)]


def main():
    p = argparse.ArgumentParser(description="합성 드론 데이터 준비")
    p.add_argument("--config", default="config.yaml", help="설정 파일 경로")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    training = cfg["training"]
    synth = training["synthetic"]
    root = Path(training["dataset_root"]).resolve()
    data_dir = root / "drone"
    out = root / "synth"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    valid = []
    with zipfile.ZipFile(out / "raw" / "drone-detection.zip") as z:
        names = set(z.namelist())
        for name in sorted(n for n in names if n.startswith("dataset/images/") and n.endswith(".png")):
            stem = Path(name).stem
            label = f"dataset/labels/{stem}.txt"
            text = z.read(label).decode() if label in names else ""
            if not is_valid_label(text):
                continue
            image = out / "images" / f"{stem}.png"
            if not image.exists():
                image.write_bytes(z.read(name))
            (out / "labels" / f"{stem}.txt").write_text(text, encoding="utf-8")
            valid.append(str(image))

    random.Random(training["seed"]).shuffle(valid)
    n_test = int(len(valid) * synth["test_ratio"])
    test, rest = valid[:n_test], valid[n_test:]

    train_hard = (data_dir / "train_hard.txt").read_text(encoding="utf-8").split()
    train = compose_train(train_hard + rest, synth["train_limits"], training["seed"])
    (data_dir / "train_synth.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (data_dir / "test_synth.txt").write_text("\n".join(test) + "\n", encoding="utf-8")
    data = yaml.safe_load((data_dir / "data.yaml").read_text(encoding="utf-8"))
    (data_dir / "data_synth.yaml").write_text(yaml.safe_dump({**data, "train": "train_synth.txt"}), encoding="utf-8")
    print(f"유효 합성 이미지 {len(valid)}장, 시험 {len(test)}장, 학습 목록 {len(train)}장")


if __name__ == "__main__":
    main()
