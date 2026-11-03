"""
YOLO를 이용한 사진 속 사람 감지
- COCO로 사전학습된 YOLO 모델을 그대로 사용
- 'person'(클래스 0)만 필터링해서 박스를 그리고 저장
- 감지된 사람 수를 출력

사용법:
    python detect_person.py --source images/           # 폴더 안의 모든 사진
    python detect_person.py --source photo.jpg         # 사진 한 장
    python detect_person.py --source photo.jpg --conf 0.5
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

PERSON_CLASS_ID = 0  # COCO 데이터셋에서 'person' 클래스 번호


def parse_args():
    p = argparse.ArgumentParser(description="YOLO 사람 감지")
    p.add_argument("--source", required=True, help="이미지 파일 또는 폴더 경로")
    p.add_argument("--model", default="yolo11n.pt", help="사전학습 가중치 (자동 다운로드)")
    p.add_argument("--conf", type=float, default=0.25, help="신뢰도 임계값 (0~1)")
    p.add_argument("--out", default="runs/person", help="결과 저장 폴더")
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
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)  # 없으면 자동으로 내려받음
    images = collect_images(args.source)
    print(f"총 {len(images)}장 처리 시작\n")

    total_people = 0
    for img_path in images:
        # classes=[0] 으로 사람만 감지하도록 제한
        results = model.predict(
            source=str(img_path),
            conf=args.conf,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )
        result = results[0]
        count = len(result.boxes)
        total_people += count

        # 박스가 그려진 이미지 (BGR numpy 배열) 저장
        annotated = result.plot()
        save_path = out_dir / f"{img_path.stem}_detected.jpg"
        cv2.imwrite(str(save_path), annotated)

        print(f"{img_path.name}: 사람 {count}명 -> {save_path}")

    print(f"\n완료. 전체 사진에서 감지된 사람 수 합계: {total_people}명")
    print(f"결과 폴더: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
