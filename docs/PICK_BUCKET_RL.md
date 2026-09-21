# Pick Bucket residual RL

`cfg/dexjoco/finetune/pick_bucket/ft_distill_residual_flow_unet_img.yaml`은
`cfg/robomimic/finetune/can/ft_distill_residual_flow_unet_img.yaml`을 복사해
동일한 섹션 순서와 학습 구조를 유지한 설정이다.

기존 BC와 동일한 이미지 기반 정책이다. 로봇 자신의 자세 정보 23차원은 이미지와
함께 쓰는 입력이며, 물체의 정답 위치는 정책에 제공하지 않는다.
`state_1000.pt`의 `state`는 체크포인트 파일 이름이지 상태 전용 정책이라는 뜻이 아니다.

## 시작점과 기본 예산

- Base: `log_dir/dexjoco-launch-20260920-env10/pick_bucket/checkpoint/state_1000.pt`.
  사용자가 지정한 체크포인트의 `model` 가중치를 로드한다(EMA 아님).
  같은 run의 `.hydra/config.yaml`도 필요하다.
- Base policy와 visual encoder는 고정하고 residual actor와 critic만 업데이트한다.
- 관측 23, 행동 22, horizon/act steps 8, flow steps 10, noise 176.
  카메라는 `[front, wrist]`, 입력은 96×96×6, 내부 feature 포함 상태는 151차원이다.
- BC와 같은 앞 50개 데모와 `ph_pretrain/normalization.npz`를 사용한다.
- Actor/critic: `[1024, 1024, 1024]`, critic ensemble 10,
  learning rate 각각 `1e-4`, batch 256, BC/residual 규제 계수 50.
- Expert 비율은 rollout 호출 0–10,000 사이 0.7에서 0.1로 감소한다.
  adaptive 설정을 끄면 `expert_ratio: 0.3`이 적용된다.
- Rollout마다 critic 업데이트 10회, 매 두 번째 rollout에서 actor 업데이트 10회.
  replay에 batch 크기 이상의 online chunk가 쌓인 뒤 시작한다.
  온라인과 expert 모두 3-step return, gamma 0.99를 사용한다.
- 예산 `train.num_train_steps: 150000`은 **벡터 rollout 호출 수**다.
  환경 16개 × act steps 8 기준 최대 19,200,000 simulator steps이며,
  조기 종료 시 실제 상호작용 수는 작아진다. 평가 상호작용은 이 예산에 포함되지 않는다.
  최대 critic 150만 회 / actor 75만 회 업데이트다.
- `train.save_freq: 1000`이 실제 residual trainer 저장 주기다.
  원본의 `save_model_freq`는 유지했지만 이 루프의 저장 주기를 제어하지 않는다.

## 데이터와 보상 확인

2026-09-21 로컬 데이터 확인 결과:

- 전체 100개 중 앞 50개: 20,912 steps, 길이 276–725.
- BC/RL의 state·action 및 정규화 값은 동일하다.
- 선택한 50개 각각 마지막 step에 reward=1, terminal=1이 한 번씩 있고
  나머지는 모두 0이다. 변환기의 `_save_datasets`가 넣은 합성 라벨이다.
- 온라인 성공은 음식 위치가 버킷 내부이고 모든 버킷 바닥 기준점이
  reset 시점보다 15cm 이상 상승한 경우다. reward=1이며 즉시 종료한다.
  Wrapper의 성공 종료 횟수도 1이다. 시간 제한은 BC와 같은 1100 steps다.
- **실제 성공 시점과 데모 마지막 step의 일치는 확인되지 않았다.**
  이 archive에는 시점별 물체 상태나 실측 success 라벨이 없다.
  로컬 raw `replay.zarr` 50개도 확인했다. success/reward 라벨이 없고,
  학습에 남긴 구간의 추가 15차원 물체 필드는 모두 일정한 초기 pose/높이여서
  실제 버킷 상승 시점을 계산할 수 없다.
  따라서 보상 형식은 맞지만, 온라인과 성공 시점까지 일치하는 데이터라고
  해석하면 안 된다. 이를 확정하려면 원본 성공 라벨 또는 초기 물체 상태를
  복원한 재생 결과로 최초 성공 시점을 찾아 데모를 잘라야 한다.

이번 연결에서 함께 수정한 실행/학습 오류:

1. DexJoCo async dummy observation space에 `multi_step_full` history 축 반영.
2. CPU expert state와 GPU visual feature를 같은 장치에서 결합.
3. Expert 3-step return의 next state/image를 3개 chunk 뒤로 이동하고,
   마지막 partial chunk의 성공 보상을 보존하며 에피소드 경계를 넘지 않도록 수정.
4. `run_eval=false`의 마지막 평가 호출 제거 및 종료 시 최종 checkpoint 저장.
5. Base 로딩 시 현재 RL device를 사용하도록 수정.
6. Base 평가에서 자동 reset 후 다른 에피소드의 성공을 포함하던 집계 수정.
7. CPU 실행 시 불필요한 CUDA 가용성 조회를 피하도록 수정.
8. DexJoCo가 `TimeLimit.truncated=false`를 반환하더라도 RL wrapper의
   `max_episode_steps`에서 종료하도록 수정. 이전에는 이 플래그 때문에 제한이
   무시되어 짧은 검증에서 episode 완료와 업데이트가 발생하지 않았다.

## 실행

저장소 루트에서 `dice-rl` conda 환경을 사용한다. GPU 번호는 실행 시 여유에 맞춘다.
체크포인트에 저장된 BC 설정이 환경변수를 참조하므로 데이터 경로도 설정한다.

```bash
conda activate dice-rl
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export DICE_RL_DATA_DIR="$PWD/data_dir"
export DICE_RL_LOG_DIR="$PWD/log_dir"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

# 짧은 동작 검증: GPU 번호는 예시
CUDA_VISIBLE_DEVICES=5 MUJOCO_EGL_DEVICE_ID=3 \
python script/smoke_dexjoco_rl.py

# GPU 드라이버/렌더러 없이 CPU에서 같은 동작 경로 검증
CUDA_VISIBLE_DEVICES=-1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
python script/smoke_dexjoco_rl.py device=cpu

# 본 학습
CUDA_VISIBLE_DEVICES=5 MUJOCO_EGL_DEVICE_ID=3 \
python script/run.py \
  --config-dir=cfg/dexjoco/finetune/pick_bucket \
  --config-name=ft_distill_residual_flow_unet_img
```

Smoke는 실제 50개 데모를 읽고 그중 고르게 고른 64개 transition만 feature로
변환한다. 실제 환경 1개, episode cap 16, rollout 4회, batch 4,
rollout당 gradient step 1, replay capacity 128로 검증하며 WandB·평가는 끈다.
Base의 파라미터 및 buffer가 그대로인지, actor/critic이 변했는지,
loss가 유한한지, online episode가 replay에 들어왔는지 확인하고
`smoke_result.json`과 실제 override를 반영한 `smoke_config.yaml`을 저장한다.
성공률 검증이나 전체 데이터 전처리 성능 측정은 아니다.

## 비교 평가

평가 카메라·정규화·환경 제한·flow steps는 BC와 동일하다.
`num_eval_episodes: 3`, `eval_n_envs: 10`이므로 총 30회,
환경 seed 10000–10029로 평가한다. `evaluate_strategy: standard`로
BC와 같은 단일 noise sampling을 사용한다. 온라인 탐색은 원본처럼 `max_q_min`이다.
더 많은 평가가 필요하면 `num_eval_episodes=10`으로 총 100회 평가할 수 있다.

본 학습 시작 시 frozen base를 먼저 평가하고, 이후 같은 환경 seed와 설정으로
RL 평가를 `evaluation_results.csv`에 기록한다. 기존 53%는 비교 참고값이며
그 평가의 episode 수·policy RNG 상태가 모두 기록되어 있지 않으면 정확한 재현값으로
간주하지 않는다. 특히 위 자동 reset 집계 수정 후의 base 재평가를 기준으로 비교한다.
초기/후기 BC 체크포인트 비교는 `base_policy_path`만 변경하고
동일한 seed, 데모, rollout 예산, 업데이트 수 및 평가 설정을 사용한다.

```bash
python -m unittest discover -s tests -v
```

## 검증 결과 (2026-09-21)

- 위 unittest 명령: 56 tests 통과.
- Hydra YAML 전체 interpolation 해석 통과. 모델 설정은 robomimic 원본과 동일하다.
- 실제 데이터의 앞 50개 모두에서 마지막 20-step 구간의 3-step 반환값이
  `0.99²`이고 terminal인 것을 확인했다. 선택한 front/wrist 입력도 `(1, 6, 96, 96)`이다.
- 실제 `state_1000.pt`, seed 42, CPU/OSMesa로 smoke 완료:
  rollout 4회, online episode 2개, update 3회.
  Base state_dict 해시 불변, actor/critic 해시 변경, 유한한 loss 확인.
  마지막 actor loss 5.70590, critic loss 17.14479.
- 결과: `log_dir/pick-bucket-rl-smoke-state1000-sandbox-20260921/`의
  `smoke_result.json`, `smoke_config.yaml`, `dataset_audit.json`, `tests.log`.
- GPU 실행은 호스트 NVIDIA 드라이버의 `os_acquire_rwlock_read` 대기 때문에
  완료하지 못했다. 본 학습과 53% 기준선 재평가는 아직 실행하지 않았다.

## 병렬 환경 RAM 측정 (2026-09-21)

사용자 요청으로 본 학습 기본값을 `env.n_envs: 16`으로 변경했다. GPU는 2번을
사용하도록 `log_dir/pick-bucket-env-memory-20260921/launch-gpu2.sh`에 준비했다.
기존 Pick Bucket BC 프로세스와 관련 worker는 종료된 것을 확인했다.
첫 시도에서는 종료 후에도 새 GPU 2 CUDA 초기화가 NVIDIA 드라이버의
`os_acquire_rwlock_read/write`에서 대기하여 새 RL 본 학습을 시작하지 못했다.

동일한 이미지/행동/환경 wrapper와 seed 42로 실제 병렬 환경을 구성하고,
평가 환경 10개를 유지한 상태에서 학습 환경별 4 rollout 호출(각 8 steps)을
CPU/OSMesa에서 실행했다. 측정은 부모와 모든 worker의 PSS 합계이며,
공유 메모리 중복 합산을 피한다. **모델·optimizer·replay와 GPU/EGL 메모리는
포함하지 않으므로 GPU 학습 전체 RAM의 실측값은 아니다.**

| 학습 환경 | 평가 환경 | 최대 RAM (PSS) | 최소 서버 가용 RAM |
|---:|---:|---:|---:|
| 8 | 10 | 18.29 GiB | 332.97 GiB |
| 12 | 10 | 22.34 GiB | 339.03 GiB |
| 16 | 10 | 26.37 GiB | 335.63 GiB |

서버 총 RAM은 566.52 GiB다. 환경 RAM 기준으로는 16개도 여유가 크며,
측정 결과를 확인한 사용자의 요청에 따라 기본값을 16개로 올렸다.
GPU 2에서 실제 업데이트 및 VRAM 검증은 별도로 필요하다.
150,000 rollout 호출 예산은 8/12/16개에서 각각 최대
9.6M/14.4M/19.2M simulator steps가 된다. `gradient_steps: 10`을 유지했으므로
환경 수를 늘리면 수집 transition당 업데이트 비율이 낮아진다.

원시 결과와 재실행 스크립트는 `log_dir/pick-bucket-env-memory-20260921/`의
`results.json`, `probe.py`, `probe.log`, `runtime.json`에 있다.
Hydra 설정 전체 해석과 실행 shell 구문 검사도 통과했다.

### 16개 환경 실행

GPU 2 CUDA 초기화는 재시도에서 통과했으나, EGL 환경 worker 16개가
드라이버 잠금에서 계속 대기하여 해당 실행을 종료했다. OSMesa CPU 렌더링과
GPU 2 정책/학습 조합으로 재실행했고 학습 환경 16개 및 평가 환경 10개
생성을 완료했다. CPU renderer는 `LP_NUM_THREADS=1`을 사용한다.
이 CPU 렌더링 실행은 속도 문제로 사용자가 중지를 요청하여 종료했다.
`launch-gpu2.sh`는 다시 EGL GPU 렌더링으로 고정했으며 CPU로 자동 전환하지 않는다.
재시도는 GPU 상태를 상속하지 않도록 별도 실행 진입점 `run_spawn.py`에서
worker 시작 방식을 `spawn`으로 설정한다.

`spawn`에서 공유 observation buffer의 semaphore가 worker 초기화 전에 해제되는
오류도 재현했다. `AsyncVectorEnv`가 buffer를 인스턴스 속성으로 유지하도록
수정했으며, 실제 spawn worker 2개의 reset/step 회귀 테스트가 통과했다.
GPU 2 단일 EGL 컨텍스트에서 96×96 이미지를 약 1.24초에 생성했고,
EGL 장치 2가 `CUDA_VISIBLE_DEVICES=2`의 논리 CUDA 장치 0에 매핑됨을 확인했다.

동시 EGL 초기화도 드라이버에서 지연되어, `run_spawn.py`에서 공유 lock으로
환경 생성만 순차 수행한다. 생성 후 rollout은 16개 worker가 병렬 실행한다.
렌더링 해상도는 기존 640×640 그대로다 (`GymRenderingSpec`이 XML의
2048×2048을 덮어씀). 최종 정책 입력은 기존 96×96이다.

현재 GPU 렌더링 재시도:
`log_dir/pick-bucket-rl-env16-gpu2-egl-serial-20260921-144049/`.
이 디렉터리의 `console.log`, `launch.json`, `memory-latest.json`을 확인한다.
14:44 확인 시 환경 순차 초기화 중이며 rollout/optimizer 업데이트는 아직
시작 전이다. CPU 렌더링으로 자동 전환하지 않는다.

종료된 CPU 렌더링 실행: `log_dir/pick-bucket-rl-env16-gpu2-osmesa-20260921-142924/`.
실제 명령/시작 시각/PID는 `launch.json`, 진행 상황은 `console.log`,
15초 간격의 전체 process tree RAM(PSS/RSS)은 `memory.jsonl`과
`memory-latest.json`에 기록한다. 환경 생성 직후 PSS는 약 26.6 GiB였으며,
이는 데이터 준비와 업데이트까지 완료한 정상 학습 구간의 최대값은 아니다.
