# ComputerVision-CatchDronePRJ
컴퓨터비전 개인 프로젝트 리포지토리 입니다!

**프로젝트 목표**: 후방/사각지대에서 접근하는 적 FPV 드론을 비전으로 탐지하여 **그물로 포획**합니다.

## 대응 전략
1. **1차 대응 (그물 발사)**: TTC(충돌 예상 시간) ≤ 0.55초 + 조준 조건 만족 시 그물 발사 (`aps/catch`, `net_link`)
2. **2차 대응 (회피 기동)**: 그물이 빗나갔거나 발사 불가 시 비행제어 회피 명령 (`fc_link`)

## 비전 파이프라인
후방 카메라 영상에서 접근하는 드론을 탐지하고, TTC로 "단순 통과"와 "충돌 코스"를 구분해 위험 신호(THREAT)를 냅니다.

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

### 4. (선택) ArduPilot SITL 시뮬레이션 테스트

실제 드론 없이 소프트웨어 시뮬레이터에서 비전 파이프라인 → MAVLink 회피 명령 → 가상 드론 반응을 테스트합니다.

**⚠️ 중요: 테스트 영상 한계**
- 현재 `vdieos/002.mp4`, `003.mp4`는 **앞쪽 드론을 쫓는** 영상입니다.
- 실제 APS는 **후방에서 접근당하는** 시나리오를 가정합니다.
- 따라서 현재 테스트는 **파이프라인 동작 확인용**이며, 실제 후방 접근 영상으로 재검증이 필요합니다.
- 실제 테스트는: 후방 카메라 고정 + 적 드론이 카메라 쪽으로 접근하는 영상이 필요합니다.

#### 4-1. Mock 모드 (SITL 설치 없이)
SITL 연결 없이 THREAT 이벤트 발생 시 로그만 출력합니다.
```bash
python3 run_sitl.py --source vdieos/003.mp4 --no-display
```
종료 후 Mock fc_link 호출 로그가 출력됩니다:
```
Mock fc_link 호출 횟수: 5
  1. track=18 ttc=0.469s frame=422
  2. track=28 ttc=0.485s frame=441
  ...
```

#### 4-2. SITL 모드 (실제 MAVLink 연동)

**사전 준비: ArduPilot SITL 설치**

자세한 설치 가이드는 [`docs/sitl_setup.md`](docs/sitl_setup.md)를 참고하세요. 간단 요약:

**macOS:**
```bash
# ArduPilot 소스 클론
cd ~/
git clone https://github.com/ArduPilot/ardupilot
cd ardupilot
git submodule update --init --recursive

# 환경 설정 (Homebrew 필요)
Tools/environment_install/install-prereqs-mac.sh -y

# PATH 추가 (~/.zshrc 또는 ~/.bash_profile)
echo 'export PATH=$PATH:$HOME/ardupilot/Tools/autotest' >> ~/.zshrc
source ~/.zshrc

# 빌드 테스트
cd ~/ardupilot/ArduCopter
sim_vehicle.py -w
```

**Ubuntu/Debian:**
```bash
cd ~/
git clone https://github.com/ArduPilot/ardupilot
cd ardupilot
git submodule update --init --recursive
Tools/environment_install/install-prereqs-ubuntu.sh -y
echo 'export PATH=$PATH:$HOME/ardupilot/Tools/autotest' >> ~/.bashrc
source ~/.bashrc
cd ~/ardupilot/ArduCopter
sim_vehicle.py -w
```

**통합 테스트 실행:**

```bash
# 터미널 1: SITL 실행 (먼저 실행하고 Heartbeat 확인)
cd ~/ardupilot/ArduCopter
sim_vehicle.py --console --map

# 터미널 2: 비전 파이프라인 + MAVLink 연동
cd /path/to/ComputerVision-CatchDronePRJ
source .venv/bin/activate
python3 run_sitl.py --source vdieos/003.mp4 --sitl --no-display
```

**확인 사항:**
- THREAT 이벤트 발생 시 터미널 2에서 `MAVLink 연결 성공` 로그 확인
- THREAT 발생 시 `GUIDED 모드 전환 → 상승 회피 → 원래 모드 복귀` 로그 확인
- 터미널 1 MAVProxy 콘솔에서 `Mode GUIDED` 메시지 확인
- MAVProxy 맵에서 가상 드론이 상승하는 것을 육안 확인

**📌 현재 구현 상태:**
- ✅ 그물 발사 시뮬레이션 (`run_catch.py` - 영상 위 가상 발사)
- ✅ 회피 기동 MAVLink 연동 (`run_sitl.py --sitl` - SITL 드론 제어)
- ⚠️ 그물 실패 시 자동 회피 연동: 아직 미구현 (향후 작업)
  - 현재: 그물 발사와 회피 기동이 별도 스크립트로 분리
  - 목표: 그물 빗나감 판정 후 자동으로 fc_link 회피 호출

**문제 해결:**
- `ModuleNotFoundError: No module named 'pymavlink'` → `pip install pymavlink`
- `MAVLink 연결 실패` → SITL이 먼저 실행 중인지 확인 (`sim_vehicle.py` 실행 후 Heartbeat 대기)
- `Address already in use (14550 포트)` → `lsof -i :14550`로 포트 사용 프로세스 확인 후 종료

**설정 변경:**
- MAVLink 연결 주소, 회피 속도(기본 3m/s 상승), 지속시간(기본 1초)은 `config.yaml`의 `fc_link:` 섹션에서 변경 가능

### 5. (선택) 탐지 모델 재학습
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
| `aps/fc_link/` | 비행제어 연동 (MAVLink SITL / mock) |
| `aps/net_link/` | 그물 발사 인터페이스 (mock) |
| `aps/catch/` | 그물 발사 시뮬레이션 (영상 위 가상 발사) |
