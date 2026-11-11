# 다음 에이전트에게 전달할 실험 실행 프롬프트

아래 내용을 서버에서 작업할 에이전트에게 그대로 전달하세요.

---

`stpark7/dice-rl` 저장소의 `temporal-hand-chord-pick-bucket-fold-glasses` 브랜치를
기준으로, 아래 비교 실험을 서버에서 실제로 학습하고 평가해줘. 계획이나 실행
명령만 제안하고 끝내지 말고, 설정 검증, 실행 큐 구성, 실제 실행, 체크포인트와
평가 결과 확인까지 진행해줘. 현재 로컬에서는 RL 본 학습을 시작하지 않았다.

먼저 `AGENTS.md`, `docs/research/temporal-hand-chord-server-handoff.md`,
`docs/research/temporal-hand-chord-training.md`와 관련 코드·테스트를 읽어줘.
기존 사용자 변경은 보존하고, 필요하면 이 브랜치에서 실험용 하위 브랜치를 만들어줘.

## 1. 사용자가 확정한 비교군과 실행 범위

- 작업: `fold_glasses`, `pick_bucket`.
- 기존 압축 방식: `global_chunk`, K=8, `ChunkHandActionChord`.
- 새 압축 방식: `temporal_chunk`, K=9, `TemporalHandActionChord`.
- paired seed: **42, 142, 242, 342**. 두 방식에 동일한 네 seed를 적용해줘.
- 주 실험은 2개 작업 × 2개 방식 × 4개 seed = **16개 RL 실행**이야.
- 각 작업·방식·seed의 codec도 아래 공통 조건으로 준비해줘. codec seed와 RL
  seed는 대응시키고, seed가 달라도 데이터 episode split은 항상 0으로 고정해줘.
- 사용자 승인 없이 K8↔K8, K128 control, 추가 seed 등 다른 비교군으로 바꾸거나
  실험 수를 늘리지 말아줘.

비교군 간 허용되는 차이는 codec 종류, 선택한 K(8 대 9), 해당 codec checkpoint,
그리고 이로부터 자동 결정되는 actor 입출력 차원이야. 나머지 파라미터는 같은
작업·seed 쌍에서 동일해야 해. Global K8은 손 latent 8개/actor 출력104,
Temporal K9는 손 latent144개/actor 출력240이므로, 결과는 이 두 후보 전체의
성능 비교로 해석해줘. 시간축 보존만의 효과나 동일 용량 비교로 단정하지 말아줘.

## 2. RL 하이퍼파라미터의 기준

사용자가 지정한 기준은 다음 프로젝트의 **seed 42 실행 `zbsqw25b`**야:

https://wandb.ai/gist_ailab/dexjoco-pick_bucket_latent_residual_batch256/runs/zbsqw25b

원래 제공된 프로젝트 주소:
https://forge.coreweave.com/wandb/gist_ailab/dexjoco-pick_bucket_latent_residual_batch256?nw=nwuserstpark5013

API로 확인한 네 기존 실행은 seed와 run/output 이름 외 학습 설정이 같았어.
이미 추가된 각 작업의
`cfg/dexjoco/finetune/<task>/ft_temporal_hand_chord_residual_flow_unet_img.yaml`과
`tests/test_temporal_hand_chord_rl_config.py`의 reference snapshot을 기준으로
비교용 설정을 만들어줘. 기존 YAML 전체를 복제하기보다는 같은 부모 설정을
공유하고 codec/checkpoint/seed/실험명만 전환할 수 있도록 구성해줘.

다음 값은 두 방식, 두 작업 모두 그대로 유지해줘:

| 항목 | 기준값 |
| --- | --- |
| Training envs | 10 |
| RL 학습 예산 | 10,000 rollout calls |
| Batch size / gradient steps | 256 / 10 |
| Replay capacity | 1,000,000 chunks, CPU storage, pinned batch staging |
| Actor/critic hidden dims | `[1024,1024,1024]` |
| Critic ensemble size | 10 |
| Actor 후보 / next-noise 후보 / exploration 후보 | 16 / 4 / 16 |
| Actor/critic learning rate / weight decay | 1e-4 / 1e-5 |
| Actor/target update frequency | 2 / 2 |
| Gamma / tau / n-step | 0.99 / 0.01 / 3 |
| BC weight / critic weight | 50 / 1 |
| Q-filter warmup / replay-flow warmup | 1,000 / 3,000 rollout calls |
| Adaptive expert ratio | 0.7 → 0.1, 10,000 rollout calls |
| Evaluation / save frequency | 매 1,000 rollout calls |
| Evaluation episodes | 50 rounds × 4 envs = 200 episodes |
| Flow / horizon / act steps | 10 / 16 / 16 |
| Raw action / noise dimension | 22 / 352 |

LR scheduler, Q filtering/normalization, expert sampling, 탐색 전략,
`success_steps_before_termination=1`, image keys `[front,wrist]` 등 표에 없는
공통 설정도 reference snapshot과 동일하게 유지해줘. 이름과 저장 경로, 로깅
프로젝트는 실행별로 구분해도 돼. 로깅 방식은 모든 비교군에서 동일하게 해줘.

메모리 때문에 batch, gradient steps, 후보 수, hidden dims, replay 크기,
학습/평가 episode 수 등을 줄이지 말아줘. 메모리가 부족하면 다른 작업이
완료되기를 기다리거나 여유 있는 서버/GPU를 사용해줘. 기존 다른 학습을
중지하거나 자원을 임의로 빼앗지 말아줘. 메모리 여유가 확인되지 않은 GPU에는
한 번에 RL 작업 하나만 배치해줘.

## 3. 같은 작업 안에서 반드시 고정할 입력

- PickBucket: base BC `pretrained_bc_policy_pick_bucket_img/checkpoint/state_2000.pt`,
  expert demos98, max episode steps700.
- FoldGlasses: base BC `pretrained_bc_policy_fold_glasses_img/checkpoint/state_1100.pt`,
  expert demos100, max episode steps900.
- 각 BC run의 `.hydra/config.yaml`도 필요해. 동일 작업의 모든 방식·seed에
  같은 BC checkpoint와 model/EMA 선택을 사용하고 fingerprint를 기록해줘.
- dataset과 normalizer도 같은 파일·버전을 사용하고, 두 방식 사이에서 다시
  생성하거나 다르게 정규화하지 말아줘.
- PickBucket normalization SHA256:
  `8496cd72d9e585042fe039c2fb71ca38660f24820f1ffc70250125d5a923e856`.
- FoldGlasses normalization SHA256:
  `b901be560ce2b8f1c24dd8ce050468026d742b781d86666a169010fa129d7b92`.
- `obs_dim=23`은 image trainer가 feature128을 추가하여 runtime151이 돼.
  처음부터151로 지정해서 이중으로 더하지 말아줘.

서버의 `DICE_RL_DATA_DIR`와 `DICE_RL_LOG_DIR`만 실제 위치에 맞춰줘.

## 4. Codec 사전학습도 공통 조건으로 맞춰줘

각 task/seed에 기존 pretrain config의 selector를 사용해줘:

```bash
# Global arm
codec=global_chunk latent_dim=8 seed=<해당 seed>
# Temporal arm
codec=temporal_chunk latent_dim=9 seed=<동일 seed>
```

공통 조건:

- `train.n_steps=20000`, `train.batch_size=128`.
- horizon16, width128, transformer layers3, heads4, dropout0.
- AdamW lr3e-4, weight_decay1e-4, max_grad_norm1.
- `train_dataset.max_n_episodes=100`, `val_fraction=0.1`, **split_seed=0**.
  실제 PickBucket은98개, FoldGlasses는100개 episode를 사용해.
- KL 최대3e-4, zero4,000 steps + ramp4,000 steps.
- KL은 Global과 Temporal 모두 K를 sum하고 나머지 batch/time 축을 mean하는
  현재 정규화를 유지해줘. Temporal이라고 H를 추가로 곱하거나 나누지 말아줘.
- Reconstruction weight1, Huber delta1, 동일 joint weights를 유지하고,
  velocity/acceleration/time-difference loss를 추가하지 말아줘.
- Validation 매500 step, save 매1,000 step. 같은 held-out MAE 기준으로 best를
  선택하고, training episode만으로 latent statistics를 맞춘 `chord.pt`를 사용해줘.

현재 Git에 포함된 Temporal K9 seed42 codec 두 개는 위 조건으로 이미20,000
step 학습됐어. `checkpoints/temporal-hand-chord/k9_seed42/`의 checksum·metadata를
확인하고 일치하면 seed42 Temporal arm에서 재사용해도 돼. 두 export의 MAE는
PickBucket1.052837°, FoldGlasses0.919131°야. 다른 seed의 codec을 이 두 파일로
대체하거나 seed42라고 이름만 바꾸지 말아줘. 기존 Global checkpoint의 학습
조건이 불명확하면 같은 조건으로 Global K8을 다시 학습해서 공정하게 비교해줘.
과거 W&B 성능은 참고치이며, 이번 paired 비교의 누락된 실행을 대신하지 않아.

## 5. 실행 전 검증과 실제 실행

1. 서버 환경·데이터·base BC configs/checkpoints·normalizer 해시를 확인해줘.
2. `python -m unittest discover -s tests -v`를 실행하고 새 설정/큐에 필요한
   집중 검증도 해줘. 두 codec은 같은 raw action/critic/replay 인터페이스를
   사용해야 하고, 팔 값 보존·frozen decoder 경유 actor gradient를 유지해야 해.
3. 16개 실행의 resolved config를 비교하여 허용된 차이(codec/K/checkpoint,
   그에 따른 actor 차원, seed, 식별/출력 경로) 외의 설정 차이를 검사해줘.
   같은 seed 쌍에서는 seed까지 동일해야 해. 작업별 차이는 위에서 명시한 것만 허용해.
4. task/codec/K/seed/stage를 경로와 실행명에 포함하고, config·commit·코드 diff·
   데이터/모델 fingerprint·환경 버전·GPU·시작/종료 시각·종료 코드를 기록해줘.
   이전 run을 덮어쓰지 말고, 실패한 run의 오래된 summary를 새 성공으로 취급하지 말아줘.
5. `script/train_temporal_hand_chord_rl.sh`는 Temporal K9 seed42의 두 작업을
   순차 실행하는 출발점이야. 전체 matrix를 실행하려면 이를 적절히 확장하거나
   별도 launcher를 만들어줘. 현재 스크립트만 실행하고16개를 완료했다고 말하지 말아줘.
6. tmux 또는 작업 스케줄러로 세션 종료 후에도 유지되는 실행 큐를 실제로 시작해줘.
   실패는 원인을 조사하고, 파라미터를 몰래 줄여 통과시키지 말아줘.
   코드 오류를 고쳤으면 같은 수정본으로 모든 비교군을 검증해줘.
7. 실제 optimizer update와 주기적 checkpoint가 생기는지 확인하고, 진행표를
   파일에 유지해서 다른 에이전트가 중복 실행 없이 이어받을 수 있게 해줘.

## 6. 평가와 최종 산출물

- Raw BC, codec projection `D(E(base))`(residual=0), 학습된 RL을 따로 측정해줘.
  랜덤 초기화 actor를 zero-residual baseline으로 취급하지 말아줘.
- 현재 trainer는 `evaluation_results.csv`에 `pretrained`, `codec_projection`,
  `finetuned`를 구분하고 initial raw/projection의 환경 seed·flow noise를 맞춰.
  현재 noise pairing은 한 run 안의 initial raw/projection에 적용돼. 두 codec
  방식 사이의 평가 noise도 명시적인 공통 seed/generator로 맞추고, 평가 RNG를
  fork/restore하여 추가 평가가 training RNG를 바꾸지 않도록 검증해줘.
- 평가 환경 seed10000–10199를 사용하고, 주 평가 metric은 최종10,000 rollout
  checkpoint의200 episode success rate로 고정해줘. 중간 best를 선택한 결과는
  별도 보조 지표로만 표시해줘. 판정 기준과 checkpoint 선택 규칙을 결과 보고
  나서 변경하지 말아줘.
- 각 작업·방식·seed의 success rate, return, codec MAE, 학습량, runtime,
  checkpoint 경로를 남겨줘. 실제 episode 수와 성공 횟수도 같이 저장해줘.
- 작업별로 두 방식의 네 seed 결과 원본, 평균±표본 표준편차(ddof=1), 같은 seed
  쌍의 성능 차이(Temporal−Global)를 표와 그래프로 정리해줘. 환경 episode를
  독립 training seed처럼 합쳐서 seed 수를 부풀리지 말아줘.
- 결과 CSV/JSON, 재현 명령, 설정 차이 보고서, 학습곡선, 실패/재실행 이력과
  간결한 Markdown 보고서를 저장해줘. 미완료 실행이 있으면 상태를 명확히 구분해줘.
- 성공 기준은 16개 RL 실행과 각 최종 평가가 완료되고, 압축 후보 외 조건이
  일치한다는 근거와 paired four-seed 비교 결과를 제공하는 것이야.

---
