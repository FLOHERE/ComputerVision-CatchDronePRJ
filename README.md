# ComputerVision-CatchDronePRJ
컴퓨터비전 개인 프로젝트 리포지토리 입니다!

후방/사각지대에서 접근하는 FPV 드론을 탐지하고, TTC(충돌 예상 시간)로 "단순 통과"와 "충돌 코스"를 구분해 위험 신호(THREAT)를 냅니다.

## Quick Start

### 1. 설치
```bash
git clone https://github.com/FLOHERE/ComputerVision-CatchDronePRJ.git
cd ComputerVision-CatchDronePRJ
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
학습된 드론 가중치 `weights/drone_yolo11n.pt`는 저장소에 들어 있습니다.

### 2. 테스트
```bash
python3 -m pytest
```
데이터셋 없이 합성 bbox 궤적(접근/통과/정지)으로 TTC·트래커·판단 로직을 검증합니다.

### 3. 파이프라인 실행
venv를 켠 터미널(`source .venv/bin/activate`)에서 실행합니다. macOS에는 `python` 명령이 없을 수 있으니 `python3`를 씁니다.
```bash
python3 run.py --source vdieos/002.mp4     # 영상 파일 + 오버레이 창
python3 run.py --source 0                  # 웹캠 + 오버레이 창
```
- 오버레이 창(bbox·TTC 표시)이 뜨고, 창을 클릭한 뒤 `q`를 누르면 종료합니다.
- `--no-display`를 붙이면 **창이 뜨지 않고** 로그와 CSV만 남습니다.
- 영상 파일은 저장소에 없습니다(`vdieos/`는 gitignore). 직접 넣어 경로를 지정하세요.
- **시험 영상은 OSD(조준점·고도/배터리 숫자·자막·로고)가 없는 영상을 직접 준비해 테스트하세요.** FPV 조종 화면 녹화 영상은 OSD 글자를 드론으로 잘못 탐지하는 경우가 많고, 실제 APS 후방 카메라 영상에는 OSD가 없습니다. 카메라를 고정하고 드론이 카메라 쪽으로 다가오는 영상이 실제 상황과 가장 비슷합니다.
- 웹캠은 처음 실행할 때 macOS 카메라 권한을 허용해야 합니다(시스템 설정 → 개인정보 보호 및 보안 → 카메라 → 터미널).
- 단계별 지연시간(전처리/추론/트래킹/TTC)은 `runs/aps/latency_*.csv`로 저장되고, 종료 시 요약이 출력됩니다.
- 임계값과 장치(`device: auto | mps | cuda | cpu`)는 `config.yaml`에서 바꿉니다.
- 시험 영상에 박힌 화면 고정 오버레이(조준점·글자·로고)에서 나온 박스는 버립니다(`osd.enabled`). 영상은 건드리지 않고, 최근 2초 동안 화면에서 움직이지 않은 부분을 오버레이로 봅니다. OSD가 없는 영상·실제 카메라에서는 `false`로 두세요. 값이 바뀌는 숫자(고도·배터리)는 이 필터로 걸러지지 않습니다.

### 4. (선택) 탐지 모델 재학습
데이터셋을 `datasets/`에 준비한 뒤:
```bash
python3 tools/prepare_dataset.py
python3 tools/train_detector.py
python3 tools/evaluate_detector.py
```

## 구조
| 모듈 | 역할 |
|------|------|
| `aps/detector/` | 프레임 → bbox (YOLO11n) |
| `aps/tracker/` | track ID 유지 + 칼만 필터 |
| `aps/ttc/` | TTC ≈ 2·A / (dA/dt) + 중앙 수렴 판단 |
| `aps/decision/` | TTC < 0.5s 이고 충돌 코스면 THREAT 이벤트 |
| `aps/io/` | 영상/웹캠 입력, bbox·TTC 오버레이 |
| `aps/fc_link/` | 비행제어 연동 (현재 mock) |
