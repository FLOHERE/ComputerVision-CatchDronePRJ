"""
Seraphim(드론) + MMFW 구조 고정익 데이터 + COCO/VisDrone(음성) 이미지를 압축 해제하고 학습용 목록과 data.yaml을 만든다.

사용법:
    python tools/prepare_dataset.py
"""

import argparse
import json
import random
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import cv2
import numpy as np
import yaml


def is_valid_label(text: str) -> bool:
    """YOLO 라벨 한 파일이 단일 클래스 형식과 0~1 범위를 지키는지 검사한다."""
    rows = [row.split() for row in text.splitlines() if row.strip()]
    if not rows:
        return False
    for row in rows:
        if len(row) != 5 or row[0] != "0":
            return False
        try:
            values = [float(v) for v in row[1:]]
        except ValueError:
            return False
        if not all(0.0 <= v <= 1.0 for v in values) or values[2] <= 0 or values[3] <= 0:
            return False
    return True


def split_images(positives: list[str], negatives: list[str], cfg: dict) -> dict[str, list[str]]:
    """train과 val은 겹치지 않고, 음성 이미지는 test_negatives를 먼저 떼어 학습에 쓰지 않는다."""
    rng = random.Random(cfg["seed"])
    positives, negatives = sorted(positives), sorted(negatives)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    n_train, n_val = cfg["train_images"], cfg["val_images"]
    n_train_neg = int(n_train * cfg["negative_ratio"])
    n_val_neg = int(n_val * cfg["negative_ratio"])
    n_test_neg = cfg["test_negatives"]
    if len(positives) < n_train + n_val:
        raise ValueError(f"드론 이미지가 부족합니다: {len(positives)} < {n_train + n_val}")
    if len(negatives) < n_train_neg + n_val_neg + n_test_neg:
        raise ValueError(f"음성 이미지가 부족합니다: {len(negatives)} < {n_train_neg + n_val_neg + n_test_neg}")

    test_negatives = negatives[:n_test_neg]
    val_negatives = negatives[n_test_neg : n_test_neg + n_val_neg]
    train_negatives = negatives[n_test_neg + n_val_neg : n_test_neg + n_val_neg + n_train_neg]
    return {
        "train": positives[:n_train] + train_negatives,
        "val": positives[n_train : n_train + n_val] + val_negatives,
        "test_negatives": test_negatives,
    }


def voc_to_yolo(xml_text: bytes | str) -> str | None:
    """VOC 주석을 단일 클래스 YOLO 라벨로 바꾼다. 유효한 박스가 없으면 None."""
    root = ET.fromstring(xml_text)
    size = root.find("size")
    width, height = float(size.find("width").text), float(size.find("height").text)
    rows = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        x1, y1, x2, y2 = (float(box.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, x2 = max(x1, 0.0), min(x2, width)
        y1, y2 = max(y1, 0.0), min(y2, height)
        if x2 <= x1 or y2 <= y1:
            continue
        rows.append(f"0 {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
    return "\n".join(rows) + "\n" if rows else None


def split_sequence(frames: list[str], cfg: dict) -> dict[str, list[str]]:
    """연속 프레임을 시간 구간으로 나눠 거의 같은 장면이 분할을 넘나들지 않게 한다."""
    frames = sorted(frames)
    n_test = int(len(frames) * cfg["test_ratio"])
    n_val = int(len(frames) * cfg["val_ratio"])
    n_train = len(frames) - n_val - n_test
    return {
        "train": frames[: n_train][:: cfg["train_stride"]],
        "val": frames[n_train : n_train + n_val],
        "test": frames[n_train + n_val :],
    }


def _prepare_fixedwing(root: Path, cfg: dict) -> dict[str, list[str]]:
    # 같은 프레임의 Zoom, Wide, Infrared는 같은 장면이므로 프레임 단위로 묶어 나눈다
    pattern = re.compile(r"^[^/]+/(?P<view>[^/]+)/(?P<modality>[^/_]+)_Imgs/(?P<frame>[^/.][^/]*)\.jpg$")
    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for archive in sorted((root / "fixedwing" / "raw").glob("*.zip")):
        with zipfile.ZipFile(archive) as z:
            names = set(z.namelist())
            groups: dict[tuple[str, str], list[re.Match]] = {}
            for name in names:
                match = pattern.match(name)
                if match:
                    groups.setdefault((match["view"], match["frame"]), []).append(match)
            views: dict[str, list[str]] = {}
            for view, frame in groups:
                views.setdefault(view, []).append(frame)

            for view, frames in views.items():
                for split, selected in split_sequence(frames, cfg).items():
                    image_dir = root / "fixedwing" / "images" / split
                    label_dir = root / "fixedwing" / "labels" / split
                    image_dir.mkdir(parents=True, exist_ok=True)
                    label_dir.mkdir(parents=True, exist_ok=True)
                    for frame in selected:
                        for match in groups[(view, frame)]:
                            stem = f"{archive.stem}_{view}_{match['modality']}_{frame}"
                            image_out = image_dir / f"{stem}.jpg"
                            label_out = label_dir / f"{stem}.txt"
                            prefix = match.string.split("/")[0]
                            ann = f"{prefix}/{view}/{match['modality']}_Anns/{frame}.xml"
                            if not image_out.exists():
                                label = voc_to_yolo(z.read(ann)) if ann in names else None
                                if label is None:
                                    continue
                                image = cv2.imdecode(np.frombuffer(z.read(match.string), np.uint8), cv2.IMREAD_COLOR)
                                if image is None:
                                    continue
                                scale = cfg["max_side_px"] / max(image.shape[:2])
                                if scale < 1:
                                    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                                label_out.write_text(label, encoding="utf-8")
                                cv2.imwrite(str(image_out), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                            out[split].append(str(image_out.resolve()))
    return out


def _extract(archive: Path, target: Path, members: set[str] | None = None) -> None:
    # 파일명만 사용하므로 압축 파일 안의 경로가 target 밖을 가리켜도 밖으로 나가지 않는다
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            name = Path(info.filename).name
            if info.is_dir() or not name or name.startswith("."):
                continue
            if members is not None and name not in members:
                continue
            out = target / name
            if not out.exists():
                out.write_bytes(z.read(info))


def _prepare_seraphim(root: Path) -> None:
    raw = root / "seraphim" / "raw"
    for split in ("train", "test"):
        for archive in sorted((raw / split / "images").glob("*.zip")):
            _extract(archive, root / "seraphim" / "images" / split)
        for archive in sorted((raw / split / "labels").glob("*.zip")):
            _extract(archive, root / "seraphim" / "labels" / split)


def _valid_positives(root: Path, split: str) -> tuple[list[str], int]:
    images, rejected = [], 0
    label_dir = root / "seraphim" / "labels" / split
    for image in (root / "seraphim" / "images" / split).iterdir():
        label = label_dir / f"{image.stem}.txt"
        if label.exists() and is_valid_label(label.read_text(encoding="utf-8", errors="replace")):
            images.append(str(image.resolve()))
        else:
            rejected += 1
    return images, rejected


def _prepare_negatives(root: Path, excluded_categories: list[str]) -> list[str]:
    raw = root / "negatives" / "raw"
    with zipfile.ZipFile(raw / "annotations_trainval2017.zip") as z:
        coco = json.loads(z.read("annotations/instances_val2017.json"))
    excluded_ids = {c["id"] for c in coco["categories"] if c["name"] in excluded_categories}
    excluded_images = {a["image_id"] for a in coco["annotations"] if a["category_id"] in excluded_ids}
    coco_names = {i["file_name"] for i in coco["images"] if i["id"] not in excluded_images}
    _extract(raw / "val2017.zip", root / "negatives" / "images" / "coco", coco_names)

    # VisDrone은 zip 안의 images/ 폴더만 사용한다. 주석 파일은 필요 없다
    visdrone = root / "negatives" / "images" / "visdrone"
    visdrone.mkdir(parents=True, exist_ok=True)
    for name in ("VisDrone2019-DET-train.zip", "VisDrone2019-DET-val.zip"):
        with zipfile.ZipFile(raw / name) as z:
            for info in z.infolist():
                path = Path(info.filename)
                if path.parent.name == "images" and path.suffix.lower() == ".jpg" and not path.name.startswith("."):
                    out = visdrone / path.name
                    if not out.exists():
                        out.write_bytes(z.read(info))

    return [str(p.resolve()) for d in ("coco", "visdrone") for p in (root / "negatives" / "images" / d).iterdir()]


def main():
    p = argparse.ArgumentParser(description="학습 데이터 준비")
    p.add_argument("--config", default="config.yaml", help="설정 파일 경로")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["training"]

    root = Path(cfg["dataset_root"])
    _prepare_seraphim(root)
    negatives = _prepare_negatives(root, cfg["excluded_coco_categories"])
    positives, rejected_train = _valid_positives(root, "train")
    test, rejected_test = _valid_positives(root, "test")
    splits = split_images(positives, negatives, cfg)
    splits["test"] = sorted(test)
    fixedwing = _prepare_fixedwing(root, cfg["fixedwing"])
    splits["train"] += fixedwing["train"]
    splits["val"] += fixedwing["val"]
    splits["test_fixedwing"] = sorted(fixedwing["test"])

    out = root / "drone"
    out.mkdir(exist_ok=True)
    for name, images in splits.items():
        (out / f"{name}.txt").write_text("\n".join(images) + "\n", encoding="utf-8")
    data = {
        "path": str(out.resolve()),
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "names": {0: "drone"},
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")

    print(f"드론 이미지: train 후보 {len(positives)} (제외 {rejected_train}), test {len(test)} (제외 {rejected_test})")
    print(f"음성 이미지 후보: {len(negatives)}")
    print({name: len(images) for name, images in splits.items()})


if __name__ == "__main__":
    main()
