import time
from typing import Iterator, Protocol

import cv2

from aps.types import Frame


class FrameSource(Protocol):
    def __iter__(self) -> Iterator[Frame]: ...


class VideoFileSource:
    def __init__(self, path: str):
        self.path = path
        cap = cv2.VideoCapture(path)
        try:
            if not cap.isOpened():
                raise FileNotFoundError(f"영상을 열 수 없습니다: {path}")
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                raise ValueError(f"영상 fps를 읽을 수 없습니다: {path}")
        finally:
            cap.release()

    def __iter__(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(self.path)
        try:
            index = 0
            while True:
                ok, image = cap.read()
                if not ok:
                    return
                # 처리 속도와 무관하게 TTC가 영상 시간 기준으로 계산되도록 한다
                yield Frame(image, index / self.fps, index)
                index += 1
        finally:
            cap.release()


class WebcamSource:
    def __init__(self, camera_index: int):
        self.camera_index = camera_index

    def __iter__(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"웹캠을 열 수 없습니다: {self.camera_index}")
        try:
            index = 0
            while True:
                ok, image = cap.read()
                if not ok:
                    return
                yield Frame(image, time.monotonic(), index)
                index += 1
        finally:
            cap.release()


def open_source(source: str) -> FrameSource:
    return WebcamSource(int(source)) if source.isdigit() else VideoFileSource(source)
