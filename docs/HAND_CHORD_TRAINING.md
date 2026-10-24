# Hand Chord 학습

기존 사전학습과 동일하게 Hydra가 설정을 읽고
[`TrainChunkHandChordAgent`](../agent/hand_chord/train_chunk_hand_chord_agent.py)의
`run()`을 실행한다. Agent가 데이터 분할, 통계 추정, 최적화, 검증과 저장을
관리하며, 모델 구조와 손 동작 복원은
[`ChunkHandActionChord`](../model/hand_chord/chunk_hand_chord.py)이 담당한다.
시뮬레이터나 이미지 인코더를 실행하지 않는 오프라인 학습이다.

## 모델 구조

[`ChunkHandActionChord`](../model/hand_chord/chunk_hand_chord.py)는
`torch.nn.Module`을 직접 상속하는 조건부 VAE다. 하나의 클래스에서
좌표 변환, 정규화·통계 관리, Transformer 인코더와 디코더, VAE 손실과
재구성, action packing, 모델 설정과 체크포인트 저장·로드 검증을 담당한다.
학습 Agent는 이 모델을 생성해 사용하며 데이터 분할, optimizer와
학습·검증 주기를 관리한다.

모델은 청크 시작의 실제 손 관절 상태를 조건으로 손 동작 전체를 하나의
`[B, K]` latent에 압축한다. 팔 action은 frame 순서대로 보존한다.
`encode()`는 팔 action과 표준화된 손 latent를 합친
`[B, H * arm_dim + K]`를 반환하고, `decode()`는 이를
`[B, H, raw_action_dim]`으로 복원한다. `H`는 청크 길이,
`K`는 `latent_dim`이다.

학습 Agent와 Hydra 설정은 이 클래스를 사용한다. 기존
`ChunkHandActionChord` import 경로와 체크포인트 형식은 유지한다.

## 실행

저장소 루트에서 PyTorch, NumPy, Hydra 등 프로젝트 의존성이 설치된 Python
환경을 사용한다. 경로 환경변수를 생략하면 데이터는 `data_dir`, 결과는
`log_dir` 아래에서 찾는다.

```bash
export DICE_RL_DATA_DIR="$PWD/data_dir"
export DICE_RL_LOG_DIR="$PWD/log_dir"

python script/run.py \
  --config-name=pre_chunk_hand_chord \
  --config-dir=cfg/dexjoco/hand_chord/pretrain/pick_bucket/
```

Hammer Nail은 `--config-dir=cfg/dexjoco/hand_chord/pretrain/hammer_nail/`로 실행한다.
같은 데이터 형식을 사용하는 다른 단일 팔 과제는 `env_name`을 바꾸면
데이터 경로와 실험 이름에 반영된다.

```bash
python script/run.py \
  --config-name=pre_chunk_hand_chord \
  --config-dir=cfg/dexjoco/hand_chord/pretrain/pick_bucket/ \
  env_name=hammer_nail latent_dim=12 horizon_steps=16 \
  train.learning_rate=1e-4 train.n_steps=10000 seed=7
```

설정만 확인하려면 명령 끝에 `--cfg job --resolve`를 붙인다.
PickBucket 설정은 WandB를 사용하며 `wandb=null`로 끌 수 있다.
Hammer Nail 설정은 기본적으로 꺼져 있으며 다음 override로 사용할 수 있다.

```bash
python script/run.py \
  --config-name=pre_chunk_hand_chord \
  --config-dir=cfg/dexjoco/hand_chord/pretrain/pick_bucket/ \
  'wandb={project:dexjoco-hand-chord,run:pick_bucket_k8_seed42}'
```

## 데이터와 학습 설정

입력 차원과 손 관절 위치는 각 YAML의 `model`에서 지정한다.

```yaml
model:
  raw_action_dim: 22
  hand_dim: 16
  state_dim: 23
  hand_state_start: 7
```

손 action은 마지막 `hand_dim`개 열이며, 앞의
`raw_action_dim - hand_dim`개 팔 action은 그대로 보존한다.
측정 손 관절은 state의 `[hand_state_start:hand_state_start + hand_dim]`을
사용한다. 데이터와 normalizer의 차원도 이 설정과 일치해야 한다.
차원 설정은 체크포인트에 함께 저장되어 로딩할 때 복원된다.

기본 입력 경로는
`$DICE_RL_DATA_DIR/dexjoco/<env_name>-img/ph_pretrain/` 아래의
`train.npz`와 `normalization.npz`다. 직접 지정할 수도 있다.

```bash
python script/run.py \
  --config-name=pre_chunk_hand_chord \
  --config-dir=cfg/dexjoco/hand_chord/pretrain/pick_bucket/ \
  train_dataset_path=/path/to/train.npz \
  normalization_path=/path/to/normalization.npz
```

`train.npz`에는 이어 붙인 `actions: [N, 22]`, `states: [N, 23]`,
에피소드별 길이인 `traj_lengths`가 필요하다. 이미지는 읽지 않는다.
이 chord는 팔 6차원과 손 16차원인 단일 팔 DexJoCo 형식을 사용하므로
action이 44차원인 bimanual 데이터에는 이 설정을 사용할 수 없다.
`normalization.npz`에는 action 정규화용 `action_min`, `action_max` (각 22차원)와
state 정규화용 `obs_min`, `obs_max` (각 23차원)가 필요하다.

처음 50개 에피소드를 사용하며 10%를 검증용으로 나눈다.
`train_dataset.max_n_episodes`, `train_dataset.val_fraction`,
`train_dataset.split_seed`로 변경할 수 있다. 에피소드 경계를 넘지 않는
완전한 chunk만 구성하고, chord의 상대 손 동작 통계와 latent 통계는
학습 분할에서 추정한다. 데이터의 기존 min/max 정규화 정보는 입력 파일을
그대로 사용하고, frame 순서와 간격을 바꾸는 재샘플링은 수행하지 않는다.
검증에서는 모델 선택에 사용하는 손 관절 평균 복원 오차(MAE, degree)만 계산한다.

공통 기본값은 horizon 16, batch 128, AdamW 학습률 `3e-4`다.
PickBucket은 chunk 전체의 latent 32차원으로 20,000 optimizer step을,
Hammer Nail은 latent 8차원으로 30,000 step을 학습한다. 검증과 주기 저장은
1,000 step마다 수행한다. KL 가중치는 처음 4,000 step 동안 0이고,
다음 4,000 step 동안 `3e-4`까지 증가한다. 손 위치 복원 손실은 Huber를
사용하며 `model.reconstruction_weight`, `model.huber_delta`로 조절한다.
CPU 연산 스레드는 코드에서 가용 CPU 수 내 최대 4개로 제한한다.

학습 길이 `train.n_steps`는 optimizer update 횟수다. DataLoader를 모두
소진하면 다시 shuffle한 iterator를 만들며, 마지막 epoch 도중에도 지정된
step 수에 도달하면 종료한다.

학습 로그에는 해당 step의 batch loss, reconstruction loss, KL과 `beta`를
기록한다. 별도의 구간 평균이나 learning rate, gradient norm 기록은 만들지
않는다. Gradient clipping은 그대로 적용하며, loss tensor는 로그 시점에만
Python scalar로 변환한다. 검증 이력은 `step`, `mae_deg` 두 값만 저장한다.

검증 DataLoader는 학습과 독립적인 난수 generator를 사용한다. 검증에서는
`reconstruct()`로 posterior mean 기반 복원값을 구한다.
이 경로는 `encode()`·`decode()`와 같은 posterior mean을
복원하며, packed latent 표준화의 왕복 연산을 생략한다.

GPU 없이 작은 실행을 확인하려면 준비된 데이터 경로를 지정하고 다음처럼
설정을 줄인다. 최소 두 에피소드와 horizon 길이 이상의 에피소드가 필요하다.

```bash
python script/run.py \
  --config-name=pre_chunk_hand_chord \
  --config-dir=cfg/dexjoco/hand_chord/pretrain/pick_bucket/ \
  device=cpu wandb=null train.n_steps=2 train.batch_size=2 \
  train.val_freq=1 train.save_model_freq=1 train.log_freq=1 \
  train_dataset.max_n_episodes=2 model.width=16 model.layers=1 model.heads=2
```

## 결과 사용

결과는 기본적으로
`$DICE_RL_LOG_DIR/dexjoco-handchord/pretrain/<name>/<날짜>_<시간>_<seed>/`에 저장된다.

학습 Agent는 `agent/hand_chord/`, 사전학습 설정은
`cfg/dexjoco/hand_chord/pretrain/<task>/`에서 관리한다.

| 파일 | 내용 |
|---|---|
| `config.yaml` | 경로와 변수를 해석한 Hydra 설정 |
| `history.jsonl` | 검증 시점별 `step`, `mae_deg` |
| `summary.json` | RL 스크립트에 필요한 설정, 최적 step·MAE, 체크포인트 경로, normalizer 경로·해시 |
| `latest.pt` | 주기마다 덮어쓰는 최근 학습 step의 chord 가중치와 통계 |
| `best.pt`, `chord.pt` | 검증으로 선택한 최적 모델과 학습 분할에서 추정한 latent 통계 |

번호별 체크포인트와 검증 예측 배열은 저장하지 않는다. 통계 배열과 모델
메타데이터는 `.pt`에 포함하며 summary에 중복 저장하지 않는다. 종료 시에는
선택한 모델의 latent 통계만 학습 분할에서 추정하고, 검증을 다시 돌리지 않는다.

모든 `.pt`는 추론용 chord export이며 optimizer 상태나 학습 재개 기능은
포함하지 않는다. 학습 설정이나 원본 normalizer를 다시 전달하지 않고 읽는다.

```python
from model.hand_chord.chunk_hand_chord import ChunkHandActionChord

chord = ChunkHandActionChord.load("/path/to/run/chord.pt", device="cpu")
# actions: [B, H, 22], state: [B, 23]; 데이터와 동일한 정규화 좌표
packed = chord.encode(actions, state)  # [B, H * 6 + K]
reconstructed = chord.decode(packed, state)  # [B, H, 22]
```

팔 action은 frame 순서대로 보존하고, 손 action 전체는 chunk 시작의 실제
손 관절 상태를 조건으로 하나의 K차원 latent에 담는다.

## 검증

```bash
WANDB_MODE=disabled OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-dexjoco-bc/bin/python -m unittest discover -s tests -v
```

학습 테스트는 합성 데이터를 사용해 CPU에서 seed 재현성, 검증 주기에
독립적인 학습, episode 경계를 넘지 않는 chunk, 학습 분할의 통계,
입력 검증과 체크포인트 왕복을 확인한다. CLI smoke는 `PYTHONPATH` 없이
PickBucket 설정을 실행하며 seed 7, horizon 4, latent 3, width 16,
layer 1, head 2, batch 4, 2 optimizer step으로 export를 확인한다.

검증 환경은 `.venv-dexjoco-bc/bin/python`의 Python 3.11.15,
PyTorch 2.7.0+cu128, NumPy 1.26.4, Hydra 1.3.4, OmegaConf 2.3.1이다.
GPU 장기 학습의 수렴과 정책 성공률은 이 CPU smoke의 검증 범위에 포함하지 않는다.

2026-10-24 VAE 커밋 대상만 추출한 디렉터리에서 전체 unittest 114개가
통과했다. RL 작업을 포함한 원래 작업 디렉터리의 전체 unittest 156개도
통과했다. 데이터 길이·좌표 차원·유한값, 비어 있는 chunk 분할,
모델 관절 배치와 학습 주기 검증을 보완했다.

실제 PickBucket 데이터의 처음 두 에피소드(1,113 frame, 학습 window 710개,
검증 window 373개)로 seed 42, CPU, horizon 16, latent 32, width 16,
layer 1, head 2, batch 2, 2 optimizer step을 실행했다.
`chord.pt`를 다시 읽어 encode/decode의 유한값과 팔 action 보존을 확인했다.
PickBucket과 Hammer Nail 설정의 `--cfg job --resolve` 및 로그 경로 override도
확인했다. 장기 학습은 새로 실행하지 않았다.
