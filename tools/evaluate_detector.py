"""
학습된 탐지 모델 평가. 드론/고정익 test의 mAP와 크기별 재현율, 음성 이미지 오탐률을 JSON으로 저장한다.

사용법:
    python tools/evaluate_detector.py [--weights weights/drone_yolo11n.pt] [--imgsz 640 960]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.data.utils import img2label_paths
from ultralytics.utils.metrics import box_iou

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aps.device import select_device


def size_bin(box, edges):
    """정규화 xyxy 박스의 √면적이 edges 구간 중 어디에 속하는지 반환한다."""
    return int(np.searchsorted(edges, float(((box[2] - box[0]) * (box[3] - box[1])) ** 0.5), side="right"))


def match_boxes(gt, pred, iou_thres, edges):
    """신뢰도 내림차순 pred를 gt에 1:1로 매칭해 (구간별 gt 수, 구간별 탐지 수, 오탐 수)를 반환한다."""
    totals = np.zeros(len(edges) + 1, dtype=int)
    hits = np.zeros(len(edges) + 1, dtype=int)
    for box in gt:
        totals[size_bin(box, edges)] += 1
    if not len(pred):
        return totals, hits, 0
    if not len(gt):
        return totals, hits, len(pred)

    iou = box_iou(torch.as_tensor(pred, dtype=torch.float32), torch.as_tensor(gt, dtype=torch.float32)).numpy()
    free = np.ones(len(gt), dtype=bool)
    false_positives = 0
    for row in iou:
        candidates = np.where(free & (row >= iou_thres))[0]
        if not len(candidates):
            false_positives += 1
            continue
        best = candidates[row[candidates].argmax()]
        free[best] = False
        hits[size_bin(gt[best], edges)] += 1
    return totals, hits, false_positives


def load_gt(image_path):
    """YOLO 라벨을 정규화 xyxy 배열로 읽는다."""
    rows = np.loadtxt(img2label_paths([image_path])[0], ndmin=2)
    cx, cy, w, h = rows[:, 1], rows[:, 2], rows[:, 3], rows[:, 4]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def predict(model, list_file, imgsz, conf, device, batch):
    # Ultralytics는 .txt 소스의 각 줄을 이미지 경로로 읽는다
    return model.predict(str(list_file), imgsz=imgsz, conf=conf, device=device, batch=batch, stream=True, verbose=False)


def evaluate_positive(model, name, data_dir, out_dir, imgsz, device, cfg):
    data_yaml = out_dir / f"{name}.yaml"
    data_yaml.write_text(
        yaml.safe_dump({"path": str(data_dir), "train": "train.txt", "val": "val.txt", "test": f"{name}.txt", "names": {0: "drone"}}),
        encoding="utf-8",
    )
    metrics = model.val(
        data=str(data_yaml), split="test", imgsz=imgsz, batch=cfg["batch"], device=device,
        project=str(out_dir), name=f"{name}_{imgsz}", exist_ok=True, plots=False, verbose=False,
    )

    edges = np.array(cfg["eval_size_bins_px"], dtype=float) / cfg["imgsz"]
    totals = np.zeros(len(edges) + 1, dtype=int)
    hits = np.zeros(len(edges) + 1, dtype=int)
    false_positives = 0
    for result in predict(model, data_dir / f"{name}.txt", imgsz, cfg["eval_conf"], device, cfg["batch"]):
        t, h, fp = match_boxes(load_gt(result.path), result.boxes.xyxyn.cpu().numpy(), cfg["eval_iou"], edges)
        totals += t
        hits += h
        false_positives += fp

    labels = [f"<{cfg['eval_size_bins_px'][0]}"]
    labels += [f"{a}-{b}" for a, b in zip(cfg["eval_size_bins_px"], cfg["eval_size_bins_px"][1:])]
    labels += [f">={cfg['eval_size_bins_px'][-1]}"]
    return {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "at_eval_conf": {
            "precision": float(hits.sum() / max(hits.sum() + false_positives, 1)),
            "recall": float(hits.sum() / max(totals.sum(), 1)),
        },
        "recall_by_size": {
            label: {"gt": int(t), "recall": float(h / t) if t else None} for label, t, h in zip(labels, totals, hits)
        },
    }


def evaluate_negative(model, data_dir, imgsz, device, cfg, confs):
    counts = {conf: {"images_with_fp": 0, "fp_boxes": 0} for conf in confs}
    n_images = 0
    for result in predict(model, data_dir / "test_negatives.txt", imgsz, min(confs), device, cfg["batch"]):
        n_images += 1
        scores = result.boxes.conf.cpu().numpy()
        for conf in confs:
            n = int((scores >= conf).sum())
            counts[conf]["fp_boxes"] += n
            counts[conf]["images_with_fp"] += n > 0
    return {"images": n_images, **{f"conf_{conf}": c for conf, c in counts.items()}}


def main():
    p = argparse.ArgumentParser(description="드론 탐지 모델 평가")
    p.add_argument("--config", default="config.yaml", help="설정 파일 경로")
    p.add_argument("--weights", help="기본 output_weights 대신 사용")
    p.add_argument("--imgsz", type=int, nargs="+", help="평가 입력 크기. 기본은 학습 크기와 탐지 크기")
    p.add_argument("--output", default="runs/eval/drone_v1", help="결과 폴더")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    training = cfg["training"]
    device = select_device(cfg["device"])
    model = YOLO(args.weights or training["output_weights"])
    data_dir = (Path(training["dataset_root"]) / "drone").resolve()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for imgsz in args.imgsz or [training["imgsz"], cfg["detector"]["imgsz"]]:
        results[str(imgsz)] = {
            "test": evaluate_positive(model, "test", data_dir, out_dir, imgsz, device, training),
            "test_fixedwing": evaluate_positive(model, "test_fixedwing", data_dir, out_dir, imgsz, device, training),
            "test_negatives": evaluate_negative(
                model, data_dir, imgsz, device, training, sorted({cfg["detector"]["conf"], training["eval_conf"]})
            ),
        }

    path = out_dir / "results.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"결과 저장: {path}")


if __name__ == "__main__":
    main()
