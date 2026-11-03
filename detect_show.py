"""
YOLO 감지 결과를 창(window)으로 띄워서 보여주기
- 사람(person)과 버스(bus)를 감지해서 박스를 그림
- 이미지마다 창이 뜨고, 아무 키나 누르면 다음 이미지로, q 누르면 종료

사용법:
    python detect_show.py --source images/
    python detect_show.py --source images/bus.jpg
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

# COCO 데이터셋 클래스 번호
CLASSES = {
    0: "person",
    5: "bus",
}


def parse_args():
    p = argparse.ArgumentParser(description="YOLO 감지 결과 창으로 보기")
    p.add_argument("--source", required=True, help="이미지 파일 또는 폴더 경로")
    p.add_argument("--model", default="yolo11n.pt", help="사전학습 가중치")
    p.add_argument("--conf", type=float, default=0.25, help="신뢰도 임계값 (0~1)")
    return p.parse_args()


def collect_images(source: str):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    path = Path(source)
    if path.is_dir():
        return sorted(f for f in path.iterdir() if f.suffix.lower() in exts)
    if path.is_file():
        return [path]
    raise FileNotFoundError(f"경로를 찾을 수 없습니다: {source}")


def main():
    args = parse_args()
    model = YOLO(args.model)
    images = collect_images(args.source)

    print("창이 뜨면: 방향키 오른쪽 = 다음 사진 / q = 종료\n")

    for img_path in images:
        results = model.predict(
            source=str(img_path),
            conf=args.conf,
            classes=list(CLASSES.keys()),  # 사람 + 버스만
            verbose=False,
        )
        result = results[0]

        # 클래스별 개수 세기
        counts = {name: 0 for name in CLASSES.values()}
        for cls_id in result.boxes.cls.tolist():
            counts[CLASSES[int(cls_id)]] += 1
        summary = ", ".join(f"{name} {n}" for name, n in counts.items())
        print(f"{img_path.name}: {summary}")

        # 박스가 그려진 이미지
        annotated = result.plot()

        # 화면보다 큰 사진은 보기 좋게 축소
        h, w = annotated.shape[:2]
        max_side = 900
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            annotated = cv2.resize(annotated, (int(w * scale), int(h * scale)))

        window_name = f"YOLO - {img_path.name}  ({summary})"
        cv2.imshow(window_name, annotated)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(window_name)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
