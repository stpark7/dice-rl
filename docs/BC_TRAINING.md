# DexJoCo image BC training

새 Linux GPU 서버에서 **설치 → Hugging Face 데이터 다운로드 → Flow Matching 기반 image BC 사전학습**을 실행하는 가이드다. 기존 BC 체크포인트 없이 처음부터 학습한다. 아래 명령은 Bash 기준이며, 설치 이후에는 모두 `dice-rl/`에서 실행한다.

## 1. Installation

### 1.1. 코드 준비

필요한 디렉터리 구조는 다음과 같다. `dexjoco/` 안의 두 번째 `dexjoco/`가 설치할 Python 패키지다.

```text
latent_action/
├── dice-rl/
│   ├── cfg/dexjoco/pretrain/
│   ├── script/download_dexjoco.py
│   └── docs/bc_training.md
└── dexjoco/
    └── dexjoco/
        ├── pyproject.toml
        └── dexjoco/
```

**현재 작업본을 옮기는 경우:** 이 문서 작성 시점(2026-09-21)에는 학습 YAML과 action wrapper 수정 등 미커밋 변경이 두 저장소에 있다. Git clone만으로는 현재 작업본을 재현할 수 없으므로, 새 서버에서 아래처럼 기존 서버의 소스를 복사한다. `SOURCE_HOST`를 기존 서버의 SSH 접속 주소로 바꾼다. 새 서버의 경로는 자유롭게 바꿀 수 있다.

```bash
export BC_WORKSPACE="$HOME/code/latent_action"
SOURCE_HOST=user@source-server
mkdir -p "$BC_WORKSPACE/dice-rl" "$BC_WORKSPACE/dexjoco/dexjoco"

rsync -az \
  --exclude='.git' --exclude='__pycache__' --exclude='*.egg-info' \
  --exclude='.venv*' --exclude='data_dir' --exclude='log_dir' \
  --exclude='hf_release' --exclude='wandb' \
  "$SOURCE_HOST:/home/sangtae_park/code/latent_action/dice-rl/" \
  "$BC_WORKSPACE/dice-rl/"

rsync -az \
  --exclude='.git' --exclude='__pycache__' --exclude='*.egg-info' \
  "$SOURCE_HOST:/home/sangtae_park/code/latent_action/dexjoco/dexjoco/" \
  "$BC_WORKSPACE/dexjoco/dexjoco/"

cd "$BC_WORKSPACE/dice-rl"
test -f cfg/dexjoco/pretrain/pick_bucket/pre_flow_matching_unet_img.yaml
test -f ../dexjoco/dexjoco/pyproject.toml
```

수정 사항이 원격 저장소에 모두 올라간 이후에는 위 복사 대신 아래 방법을 사용할 수 있다. 학습 설정과 wrapper가 포함된 branch/commit을 사용해야 한다.

```bash
mkdir -p "$HOME/code/latent_action"
cd "$HOME/code/latent_action"
git clone https://github.com/stpark7/dice-rl.git
git clone https://github.com/brave-eai/dexjoco.git
cd dice-rl
```

### 1.2. Python 및 패키지 설치

Conda/Miniconda와 NVIDIA 드라이버가 설치된 서버를 전제로 한다. 먼저 `nvidia-smi`가 정상 동작하는지 확인한다. **Python 3.11**을 사용한다. 원본 DICE-RL README의 Python 3.8과 달리 DexJoCo는 Python `>=3.10, <3.12`를 요구한다.

Ubuntu/Debian에서 렌더링용 시스템 라이브러리가 없다면 설치한다.

```bash
sudo apt-get update
sudo apt-get install -y build-essential git rsync libegl1 libgl1 libglfw3 libosmesa6 ffmpeg

conda create -n dice-bc python=3.11 -y
conda activate dice-bc
python -m pip install --upgrade pip setuptools wheel

# CUDA 12.1 wheel 예시: 서버 드라이버가 해당 CUDA runtime을 지원해야 한다.
python -m pip install torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu121

# dice-rl/에서 실행. 두 프로젝트의 의존성을 함께 해결한다.
python -m pip install -e . -e ../dexjoco/dexjoco \
  'numpy==1.26.4' 'numcodecs<0.16' 'opencv-python==4.11.0.86'
python -m pip check
```

`torchvision`은 이미지 모델에서 필요하지만 DICE-RL의 기본 의존성에는 빠져 있어 별도로 설치한다. `torch==2.4.0` / `torchvision==0.19.0` 조합과 CUDA wheel 선택지는 [PyTorch 공식 설치 표](https://pytorch.org/get-started/previous-versions/#v240)를 따른다. GPU/드라이버가 이 버전을 지원하는지 아래 확인 단계에서 점검한다.

이 경로는 DexJoCo의 `mujoco==3.4.0`을 사용한다. Robomimic extra, `mujoco-py`, OpenPI, 원본 데이터 변환기는 이 BC 학습에 필요하지 않다. DICE-RL의 `pyproject.toml`에 따라 SciPy는 `1.10.1`, Zarr는 `2.18.3`으로 설치된다. 기존 서버의 전체 Conda 환경을 그대로 복사하는 대신 이 전용 환경을 사용한다.

### 1.3. 경로 및 GPU 설정

아래 파일을 한 번 만든다. 데이터/로그를 별도 디스크에 저장하려면 생성 전에 두 경로를 변경한다.

```bash
export DICE_RL_DATA_DIR="$PWD/data_dir"
export DICE_RL_LOG_DIR="$PWD/log_dir"
mkdir -p "$DICE_RL_DATA_DIR" "$DICE_RL_LOG_DIR"

cat > "$CONDA_PREFIX/etc-dice-bc.sh" <<EOF
export DICE_RL_DATA_DIR="$DICE_RL_DATA_DIR"
export DICE_RL_LOG_DIR="$DICE_RL_LOG_DIR"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1
EOF
source "$CONDA_PREFIX/etc-dice-bc.sh"
export CUDA_VISIBLE_DEVICES=0
```

**새 터미널마다** `conda activate dice-bc`, `cd ~/code/latent_action/dice-rl`, `source "$CONDA_PREFIX/etc-dice-bc.sh"`를 실행하고 `CUDA_VISIBLE_DEVICES`를 지정한다. 프로세스당 물리 GPU 하나를 노출하고 학습 설정의 `device=cuda:0`은 그대로 둔다. 예를 들어 물리 GPU 2를 사용하려면 `CUDA_VISIBLE_DEVICES=2`로 실행한다.

GPU 및 headless 렌더링을 확인한다.

```bash
python - <<'PY'
import torch
import torchvision
import mujoco
from dexjoco.tasks.mappings import CONFIG_MAPPING

print("torch:", torch.__version__, "torchvision:", torchvision.__version__)
assert torch.cuda.is_available(), "CUDA를 사용할 수 없습니다. 드라이버와 torch 설치를 확인하세요."
print("GPU:", torch.cuda.get_device_name(0))
x = torch.ones((32, 32), device="cuda:0")
print("CUDA matmul:", (x @ x).mean().item())
model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size="0.1"/></worldbody></mujoco>')
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
with mujoco.Renderer(model, height=96, width=96) as renderer:
    renderer.update_scene(data)
    print("MuJoCo RGB:", renderer.render().shape)
assert "pick_bucket" in CONFIG_MAPPING
PY
```

예상 RGB shape은 `(96, 96, 3)`이다. EGL 초기화에 실패하면 서버의 NVIDIA EGL 라이브러리와 컨테이너 GPU/graphics 접근 설정을 확인한다. 임시로 CPU 렌더링을 확인하려면 `export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`로 바꾼 뒤 재실행한다. OSMesa에서는 평가 렌더링이 느려질 수 있다.

## 2. Download datasets from Hugging Face

데이터 저장소는 [robopark/dexjoco-image-data](https://huggingface.co/datasets/robopark/dexjoco-image-data)이며, 접근 권한이 있는 Hugging Face 계정이 필요하다. 아래 로그인 프롬프트에 read 권한 토큰을 입력한다.

```bash
python -c 'from huggingface_hub import login; login()'

# 로컬 설정에서 참조하는 데이터 릴리스
export BC_DATA_REVISION=68a239b1ddabe7228026f94ce57845d10f7d98f2

# 첫 학습에 사용할 데이터만 다운로드
bash script/download_hf.sh dexjoco \
  --revision "$BC_DATA_REVISION" --task pick_bucket
```

전체 6개 작업이 필요하면 `--task`를 생략한다. 여러 작업만 선택하려면 `--task`를 반복한다.

```bash
# 전체 다운로드
bash script/download_hf.sh dexjoco --revision "$BC_DATA_REVISION"

# 선택 다운로드 예시
bash script/download_hf.sh dexjoco --revision "$BC_DATA_REVISION" \
  --task hammer_nail --task bimanual_hanoi
```

**반드시 `dexjoco` 인자를 붙인다.** 인자 없는 `bash script/download_hf.sh`는 원본 Robomimic 데이터와 체크포인트를 다운로드한다. DexJoCo BC 체크포인트는 현재 배포하지 않으며, 위 명령은 데이터만 받는다.

다운로더는 고정된 HF commit의 archive SHA-256을 확인한 뒤 압축을 해제한다. 캐시 archive와 압축 해제된 데이터가 함께 저장되므로 두 용량을 모두 확보한다. 캐시 위치는 `--cache-dir /path/to/hf-cache`, 데이터 위치는 `--data-dir /path/to/data`로 지정할 수 있다. 후자를 사용하면 학습 시 `DICE_RL_DATA_DIR`도 같은 경로로 설정한다.

```text
$DICE_RL_DATA_DIR/dexjoco/pick_bucket-img/
├── episodes.json
├── images.zarr/
├── ph_pretrain/
│   ├── train.npz
│   └── normalization.npz
└── ph_finetune/
    ├── train.npz
    └── normalization.npz
```

BC 학습은 `ph_pretrain`을 사용한다. NPZ가 `../images.zarr`를 참조하므로 작업 폴더 전체를 함께 보관한다. 데이터는 이미 변환·정규화되어 있어 추가 전처리가 필요 없다. 다운로드를 반복하면 동일한 설치 영수증이 있는 데이터를 재사용하며, 다른 기존 데이터는 덮어쓰지 않는다.

## 3. Pretraining

### 3.1. 작업 선택과 설정 확인

학습 설정은 `cfg/dexjoco/pretrain/<task>/pre_flow_matching_unet_img.yaml`에 있다.

| Task | State / action 차원 | 카메라 순서 | 기본 학습 episode 수 | Epoch | 평가 환경 수 |
| --- | --- | --- | ---: | ---: | ---: |
| `hammer_nail` | 23 / 22 | front, wrist | 50 | 8000 | 10 |
| `pick_bucket` | 23 / 22 | front, wrist | 50 | 8000 | 10 |
| `bimanual_assembly` | 46 / 44 | ego, wrist_left, wrist_right | 50 | 8000 | 10 |
| `bimanual_hanoi` | 46 / 44 | ego, wrist_left, wrist_right | 50 | 8000 | 10 |
| `bimanual_microwave_cook` | 46 / 44 | ego, wrist_left, wrist_right | 100 | 5000 | 20 |

`fold_glasses`는 데이터 다운로드만 지원하며 아직 해당 학습 YAML은 없다. 위 5개 작업 중 하나를 선택한다. 각 데이터에는 100개 episode가 있지만 기본 설정은 표의 개수만 앞에서부터 사용한다.

```bash
export BC_TASK=pick_bucket
python script/run.py \
  --config-dir="cfg/dexjoco/pretrain/$BC_TASK" \
  --config-name=pre_flow_matching_unet_img \
  wandb=null --cfg job --resolve
```

이 명령은 학습을 시작하지 않고 최종 설정을 출력한다. `train_dataset_path`, `normalization_path`, `logdir`가 새 서버의 경로인지 확인한다. 모든 실행 예시는 `wandb=null`로 WandB 없이 동작한다.

공통 기본값은 seed 42, batch size 512, learning rate `1e-4`, action horizon/execution 8, observation history 1, flow steps 10이다. RGB 96×96에 84×84 crop을 적용하며, 데이터셋은 CPU에 두고 batch를 GPU로 옮긴다. 카메라 순서와 state/action 차원은 작업별 YAML에서 이미 맞춰져 있다.

### 3.2. 짧은 학습 확인

처음에는 2개 episode, batch 8, 평가 환경 1개로 1 epoch만 학습한다.

```bash
CUDA_VISIBLE_DEVICES=0 python script/run.py \
  --config-dir="cfg/dexjoco/pretrain/$BC_TASK" \
  --config-name=pre_flow_matching_unet_img \
  wandb=null device=cuda:0 \
  train.n_epochs=1 train.batch_size=8 train.save_model_freq=1 \
  train_dataset.max_n_episodes=2 env.n_envs=1 \
  "logdir=$DICE_RL_LOG_DIR/bc-smoke/$BC_TASK/$(date +%Y-%m-%d_%H-%M-%S)"
```

`1: train loss ...`와 `Saved model to .../checkpoint/state_1.pt`가 출력되면 데이터 로딩, GPU 학습, 저장이 완료된 것이다. 이 과정에서도 시뮬레이터 환경을 생성하지만, **1 epoch 실행에는 정책 rollout 평가가 포함되지 않는다.** 평가 연결까지 확인하려면 같은 작은 설정에서 `train.n_epochs=100`으로 실행한다. 양팔 작업을 학습할 서버에서는 `BC_TASK=bimanual_hanoi` 등으로 바꾸고 해당 데이터를 받은 뒤 같은 확인을 수행한다.

### 3.3. 본 학습

```bash
export BC_TASK=pick_bucket
CUDA_VISIBLE_DEVICES=0 python script/run.py \
  --config-dir="cfg/dexjoco/pretrain/$BC_TASK" \
  --config-name=pre_flow_matching_unet_img \
  wandb=null device=cuda:0 seed=42
```

다른 작업은 `BC_TASK`만 바꿔 실행한다. 여러 GPU에서는 터미널별로 다른 `CUDA_VISIBLE_DEVICES`와 작업을 지정한다. 현재 명령은 GPU 하나당 독립 학습 프로세스 하나이며 DDP 학습은 아니다.

GPU 메모리가 부족하면 명령 끝에 `train.batch_size=128 env.n_envs=2`를 추가한다. 학습 episode 100개를 모두 사용하려면 `train_dataset.max_n_episodes=100`을 추가한다. 이는 기본 실험 설정을 변경하므로 결과 비교 시 함께 기록한다. 기본 평가 환경 수는 프로세스별로 생성되며, BC 학습에도 시뮬레이터가 필요하다.

SSH 접속이 끊겨도 계속 학습하려면 `tmux new -s bc-pick-bucket`으로 세션을 연 뒤 환경 활성화·경로 설정·학습 명령을 실행한다. `Ctrl-b`, `d`로 빠져나오고 `tmux attach -t bc-pick-bucket`으로 다시 접속한다. `tmux`가 없다면 먼저 설치한다.

WandB 기록이 필요하면 `wandb login`을 실행하고 학습 명령의 `wandb=null`을 제거한다. 기본 project는 `dexjoco-<task>-pretrain`이다. 팀을 지정하려면 `export WANDB_ENTITY=<team-or-user>`를 사용한다.

### 3.4. 로그와 체크포인트

기본 출력 구조는 다음과 같다.

```text
$DICE_RL_LOG_DIR/dexjoco-pretrain/
└── <task>_pre_flow_matching_unet_img_ta8_td10/
    └── <YYYY-MM-DD>_<HH-MM-SS>_42/
        ├── .hydra/
        │   ├── config.yaml
        │   └── overrides.yaml
        ├── run.log
        ├── checkpoint/
        │   ├── state_200.pt
        │   ├── state_400.pt
        │   └── ...
        └── render/
```

체크포인트는 200 epoch마다, 그리고 마지막 epoch에 저장한다. 파일에는 `epoch`, `model`, `ema`가 들어 있다. optimizer/scheduler 상태는 없고 실행 스크립트에 자동 resume 기능도 연결되어 있지 않으므로, 같은 명령을 다시 실행하면 처음부터 새 학습을 시작한다.

현재 이미지 trainer는 **100 epoch마다 model 가중치로 평가**한다. `train.val_freq`를 override해도 이미지 trainer 내부의 100 설정이 우선하며, `train.render.freq`도 평가 주기를 바꾸지 않는다. 환경 자동 reset을 사용하는 현재 집계 방식의 로그 수치는 고정된 episode 수로 별도 측정한 성공률과 구분해서 해석한다. 기본 설정에는 validation split과 평가 영상 저장이 없다.

재현을 위해 실행에 사용한 코드 버전/작업본, HF revision, GPU 모델, `.hydra/` 설정과 아래 패키지 목록을 함께 보관한다.

```bash
python -m pip freeze > "$DICE_RL_LOG_DIR/bc-packages-$(date +%Y-%m-%d_%H-%M-%S).txt"
```

### 3.5. 자주 발생하는 문제

| 증상 | 확인할 내용 |
| --- | --- |
| HF 401/403 또는 repository not found | `robopark/dexjoco-image-data` 접근 권한과 HF 로그인 확인 |
| `MissingConfigException` | `dice-rl/`에서 실행했는지, 현재 작업본의 `cfg/dexjoco/`까지 복사했는지 확인 |
| `ModuleNotFoundError: dexjoco` / `torchvision` | `which python`으로 `dice-bc` 환경 확인 후 1.2의 설치 명령 실행 |
| `images.zarr` / `normalization.npz`를 찾지 못함 | `DICE_RL_DATA_DIR`와 작업 폴더 구조 확인; NPZ만 따로 옮기지 않기 |
| EGL / OpenGL 초기화 실패 | 1.3의 단독 렌더링 확인 및 서버 EGL 설정 점검 |
| CUDA out of memory | `train.batch_size`와 `env.n_envs`를 함께 줄이고 GPU 중복 사용 확인 |
| 다운로드 시 기존 폴더 덮어쓰기 거부 | 새 `--data-dir`에 받고 `DICE_RL_DATA_DIR`도 변경 |

이 문서는 현재 코드와 설정을 기준으로 작성했다. 새 서버에서의 의존성 설치, 데이터 접근, GPU/EGL 동작은 위 확인 명령과 짧은 학습으로 검증한 뒤 본 학습을 실행한다.
