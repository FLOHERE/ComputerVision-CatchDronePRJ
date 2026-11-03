"""OS 종속 장치 선택은 이 파일에만 둔다."""

import torch


def select_device(preference: str = "auto") -> str:
    if preference != "auto":
        return preference
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
