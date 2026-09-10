"""
Standalone evaluation of a pretrained flow-matching BC checkpoint with
state or image observations, independent of the finetune agent.

Mirrors `TrainFlowMatchingAgent.evaluate_policy` (action = policy(cond).trajectories[:, :act_steps])
and the fixed-seed episode bookkeeping of `script/eval_rl_checkpoint.py`:
  * evaluates `num_eval_episodes` episodes per environment, for a total of
    `num_eval_episodes * eval_n_envs` episodes,
  * uses deterministic env seeds 10000, 10001, ... (same convention as eval_rl_checkpoint),
  * loads the `model` weights (what DistillResidualRLModel uses as the frozen base).

Usage (from repo root, MUJOCO_GL=egl):
  python script/eval_pretrain_checkpoint.py \
      --ckpt_path log_dir/dexjoco-pretrain/<run>/checkpoint/state_2000.pt \
      --num_eval_episodes 10 --eval_n_envs 10

Prints evaluation results to the console.
"""

import argparse
import math
import os
import time

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from env.gym_utils import make_async

# 학습 당시 설정의 ${eval:...}, ${round_up:...}, ${round_down:...} 표현식을
# 평가에서도 해석할 수 있도록 script/run.py와 같은 resolver를 등록한다.
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil, replace=True)
OmegaConf.register_new_resolver("round_down", math.floor, replace=True)


def find_run_dir(ckpt_path):
    # 체크포인트가 들어 있는 폴더부터 상위로 올라가며 학습 설정을 찾는다.
    # 예: <run>/checkpoint/state_2000.pt -> <run>/.hydra/config.yaml
    d = os.path.dirname(os.path.abspath(ckpt_path))
    while d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, ".hydra", "config.yaml")):
            return d
        d = os.path.dirname(d)
    raise FileNotFoundError(f"No .hydra/config.yaml above {ckpt_path}")


@torch.no_grad()
def _get_flow_action_for_eval(model, obs, act_steps, device, use_image_obs):
    """Prepare observations and sample actions directly from the BC policy."""
    # N: 병렬 환경 수, T: 관측 이력 길이. 이력이 없는 (N, D) 상태에는
    # 시간 축을 추가해 모델 입력을 (N, T, D)로 맞춘다.
    state = torch.as_tensor(obs['state'], dtype=torch.float32, device=device)
    if state.ndim == 2:
        state = state.unsqueeze(1)
    cond = {'state': state}

    if use_image_obs:
        # 이미지도 이력이 없는 (N, H, W, C) 입력에는 시간 축을 추가한다.
        rgb = torch.as_tensor(obs['rgb'], dtype=torch.float32, device=device)
        if rgb.ndim == 4:
            rgb = rgb.unsqueeze(1)
        # 이미지 사전학습과 동일하게 채널 축을 앞으로 옮긴다.
        # (N, T, H, W, C) -> (N, T, C, H, W)
        cond['rgb'] = rgb.permute(0, 1, 4, 2, 3)

    # BC 정책이 생성한 전체 액션 궤적 중 앞쪽 act_steps개만 실행한다.
    # 환경에 전달할 수 있도록 GPU 텐서를 CPU의 NumPy 배열로 변환한다.
    sample = model(cond=cond)
    return sample.trajectories[:, :act_steps].cpu().numpy()


def evaluate(model, eval_venv, num_eval_episodes, eval_n_envs, max_episode_steps,
             act_steps, device, best_reward_threshold_for_success,
             use_image_obs=False):
    """Evaluate fixed-seed batches; BC success uses each episode's peak reward."""
    # eval()은 모델을 평가 모드로 전환하며, 추론 시 gradient 계산은
    # _get_flow_action_for_eval의 @torch.no_grad()에서 비활성화한다.
    model.eval()
    eval_episode_rewards = []
    eval_episode_lengths = []
    eval_episode_peak_rewards = []
    # 환경 step 한 번에 액션 청크를 실행하므로 필요한 정책 호출 수를 올림한다.
    max_eval_steps = math.ceil(max_episode_steps / act_steps)
    start_time = time.time()

    for episode in range(num_eval_episodes):
        # 한 라운드에 환경마다 한 에피소드를 평가한다. 전체 평가 수는
        # num_eval_episodes * eval_n_envs이며 시드는 10000부터 중복 없이 배정한다.
        options_venv = [
            {'seed': 10000 + episode * eval_n_envs + env_idx}
            for env_idx in range(eval_n_envs)
        ]
        obs = eval_venv.reset_arg(options_list=options_venv)
        if isinstance(obs, list):
            # 환경별 관측 딕셔너리 목록을 키별 배치 배열로 합친다.
            obs = {key: np.stack([item[key] for item in obs]) for key in obs[0]}

        # 각 환경의 누적 보상, 실행 액션 수, 최대 반환 보상, 종료 여부를 추적한다.
        # 최대 보상은 0에서 시작하므로 음수 보상만 반환되면 0으로 남는다.
        episode_rewards = np.zeros(eval_n_envs)
        episode_lengths = np.zeros(eval_n_envs)
        episode_peak_rewards = np.zeros(eval_n_envs)
        episode_done = np.zeros(eval_n_envs, dtype=bool)

        for step in range(max_eval_steps):
            # 마지막 청크는 남은 스텝 수만큼 잘라 설정된 길이 상한에 맞춘다.
            remaining_steps = max_episode_steps - step * act_steps
            action_chunk = _get_flow_action_for_eval(
                model, obs, min(act_steps, remaining_steps), device, use_image_obs,
            )
            obs, reward, terminated, truncated, _ = eval_venv.step(action_chunk)
            # 환경 자체의 종료와 시간 제한 등의 중단을 모두 종료로 취급한다.
            done = np.asarray(terminated) | np.asarray(truncated)

            # 먼저 끝난 환경은 자동 리셋되어 다음 에피소드를 시작할 수 있다.
            # 이번 라운드에서 아직 끝나지 않은 환경의 결과만 집계한다.
            active = ~episode_done
            episode_rewards[active] += np.asarray(reward)[active]
            # 성공 판정에는 누적 보상이 아니라 step()이 반환한 보상의 최대값을 쓴다.
            # 청크 내부 보상이 합산되는지는 환경 wrapper의 설정에 따른다.
            episode_peak_rewards[active] = np.maximum(
                episode_peak_rewards[active], np.asarray(reward)[active],
            )
            # 길이는 전달한 액션 수로 집계하므로 청크 도중 종료되었을 때의
            # 실제 실행 스텝 수와는 다를 수 있다.
            episode_lengths[active] += action_chunk.shape[1]
            episode_done |= done | (episode_lengths >= max_episode_steps)
            if episode_done.all():
                break

        # 라운드 결과를 전체 통계에 추가하고 성공 수와 누적 소요 시간을 출력한다.
        eval_episode_rewards.extend(episode_rewards.tolist())
        eval_episode_lengths.extend(episode_lengths.tolist())
        eval_episode_peak_rewards.extend(episode_peak_rewards.tolist())
        successes = episode_peak_rewards >= best_reward_threshold_for_success
        print(
            f"round {episode + 1}/{num_eval_episodes}: "
            f"success {int(successes.sum())}/{eval_n_envs} "
            f"({time.time() - start_time:.0f}s)"
        )

    # 모든 환경과 라운드를 합쳐 통계를 계산한다. 성공한 에피소드가 없으면
    # 성공 에피소드의 평균 길이는 계산할 수 없으므로 NaN으로 반환한다.
    lengths = np.asarray(eval_episode_lengths)
    successes = np.asarray(eval_episode_peak_rewards) >= best_reward_threshold_for_success
    return {
        'return_mean': float(np.mean(eval_episode_rewards)),
        'return_std': float(np.std(eval_episode_rewards)),
        'length_mean': float(lengths.mean()),
        'success_rate': float(successes.mean()),
        'success_length_mean': float(lengths[successes].mean()) if successes.any() else float('nan'),
        'num_episodes': len(eval_episode_rewards),
    }


def main():
    # 체크포인트 위치와 평가 횟수, 병렬 환경 수, 추론 장치를 CLI로 받는다.
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained flow-matching BC checkpoint"
    )
    parser.add_argument(
        "--ckpt_path",
        required=True,
        help="Path to pretrained BC checkpoint file (.pth or .pt)",
    )
    parser.add_argument(
        "--num_eval_episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate per environment",
    )
    parser.add_argument(
        "--eval_n_envs",
        type=int,
        default=10,
        help="Number of parallel evaluation environments",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device to run evaluation on",
    )
    args = parser.parse_args()

    # 그래픽 백엔드가 지정되지 않았다면 화면 없는 서버용 EGL을 사용한다.
    os.environ.setdefault("MUJOCO_GL", "egl")

    # 체크포인트의 학습 설정으로 모델과 환경을 복원
    run_dir = find_run_dir(args.ckpt_path)
    config_path = os.path.join(run_dir, '.hydra', 'config.yaml')
    print(f"Loading config from {config_path}")
    cfg = OmegaConf.load(config_path)
    cfg.device = args.device
    cfg.model.device = args.device
    env_type = cfg.env.get('env_type', None)
    use_image_obs = cfg.env.get('use_image_obs', False)

    print(f"Model type: {'Image-based' if use_image_obs else 'State-based'}")
    print(f"Environment: {cfg.env.name}")
    print(
        f"Evaluating {args.num_eval_episodes} episodes per environment "
        f"with {args.eval_n_envs} parallel envs"
    )

    # eval_rl_checkpoint.py와 동일하게 모델보다 병렬 환경을 먼저 생성한다.
    # 학습 당시 wrapper, 정규화 경로, 관측/액션 차원을 그대로 전달한다.
    print('Creating evaluation environments...')
    eval_venv = make_async(
        cfg.env.name,
        env_type=env_type,
        num_envs=args.eval_n_envs,
        asynchronous=True,
        max_episode_steps=int(cfg.env.max_episode_steps),
        wrappers=cfg.env.get("wrappers", None),
        robomimic_env_cfg_path=cfg.get('robomimic_env_cfg_path', None),
        shape_meta=cfg.get('shape_meta', None),
        use_image_obs=use_image_obs,
        # 화면에 렌더링하지 않아도 이미지 관측을 만들 때는 offscreen 렌더링이 필요하다.
        render=False,
        render_offscreen=use_image_obs,
        normalization_path=cfg.get('normalization_path', None),
        obs_dim=cfg.obs_dim,
        action_dim=cfg.action_dim,
        abs_action=cfg.get('abs_action', False),
        # 환경별 추가 옵션이 있으면 공통 옵션과 함께 전달한다.
        **(cfg.env.specific if "specific" in cfg.env else {}),
    )
    try:
        # 초기 환경 시드를 설정
        eval_venv.seed([10000 + i for i in range(args.eval_n_envs)])

        # Hydra의 _target_ 설정으로 모델을 생성한 뒤 체크포인트를 CPU에서 읽는다.
        # EMA 가중치 대신 'model' 가중치를 복원하고 요청한 추론 장치로 옮긴다.
        print('Initializing model...')
        model = hydra.utils.instantiate(cfg.model)
        print(f"Loading checkpoint from {args.ckpt_path}")
        checkpoint = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        model.to(args.device)

        # 에피소드 길이, 실행 청크 길이, 성공 임계값은 저장된 학습 설정을 따른다.
        metrics = evaluate(
            model=model,
            eval_venv=eval_venv,
            num_eval_episodes=args.num_eval_episodes,
            eval_n_envs=args.eval_n_envs,
            max_episode_steps=int(cfg.env.max_episode_steps),
            act_steps=int(cfg.act_steps),
            device=args.device,
            best_reward_threshold_for_success=float(
                cfg.env.get('best_reward_threshold_for_success', 1),
            ),
            use_image_obs=use_image_obs,
        )
        # 보상 요약과 함께 환경명, 학습 epoch, 성공률, 길이를 한 줄로 출력한다.
        print('Pretrained Policy Results:')
        print(f"  Return: {metrics['return_mean']:.2f} ± {metrics['return_std']:.2f}")
        print(
            f"RESULT {cfg.env.name} epoch={checkpoint.get('epoch', None)} weights=model "
            f"n={metrics['num_episodes']} success_rate={metrics['success_rate']:.3f} "
            f"mean_len={metrics['length_mean']:.1f} "
            f"success_len={metrics['success_length_mean']:.1f}"
        )
    finally:
        # 모델 로딩이나 평가 중 예외가 나더라도 환경과 worker 자원을 정리한다.
        eval_venv.close()


if __name__ == "__main__":
    # 직접 실행할 때만 CLI 평가를 시작하며 import 시에는 함수를 정의한다.
    main()
