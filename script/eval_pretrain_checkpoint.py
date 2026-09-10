"""
Standalone success-rate evaluation of a *pretrained* flow-matching BC checkpoint
(state-based), independent of the finetune agent.

Mirrors `TrainFlowMatchingAgent.evaluate_policy` (action = policy(cond).trajectories[:, :act_steps])
and the fixed-seed episode bookkeeping of `script/eval_rl_checkpoint.py`, but:
  * evaluates `num_episodes` episodes in rounds of `eval_n_envs` parallel envs,
  * uses deterministic env seeds 10000, 10001, ... (same convention as eval_rl_checkpoint),
  * loads the `model` weights by default (what DistillResidualRLModel uses as the
    frozen base); pass --weights ema to evaluate the EMA copy instead.

Usage (from repo root, MUJOCO_GL=egl):
  python script/eval_pretrain_checkpoint.py \
      --ckpt_path log_dir/dexjoco-pretrain/<run>/checkpoint/state_2000.pt \
      --num_episodes 50 --eval_n_envs 10

Appends one row per call to `<run_dir>/eval_pretrain.csv`.
"""

import argparse
import csv
import math
import os
import time

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from env.gym_utils import make_async

# Same custom resolvers as script/run.py (configs use ${eval:...}).
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil, replace=True)
OmegaConf.register_new_resolver("round_down", math.floor, replace=True)


def find_run_dir(ckpt_path):
    d = os.path.dirname(os.path.abspath(ckpt_path))
    while d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, ".hydra", "config.yaml")):
            return d
        d = os.path.dirname(d)
    raise FileNotFoundError(f"No .hydra/config.yaml above {ckpt_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--eval_n_envs", type=int, default=10)
    parser.add_argument("--weights", choices=["model", "ema"], default="model")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed_base", type=int, default=10000)
    parser.add_argument("--torch_seed", type=int, default=0)
    parser.add_argument("--csv", default=None, help="Output csv (default <run_dir>/eval_pretrain.csv)")
    parser.add_argument(
        "--max_episode_steps", type=int, default=None,
        help="Override the config's episode cap (use the env's own truncation limit, "
             "e.g. 1200 for pick_bucket/click_mouse, 1000 for hammer_nail).",
    )
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    torch.manual_seed(args.torch_seed)
    np.random.seed(args.torch_seed)

    run_dir = find_run_dir(args.ckpt_path)
    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    act_steps = int(cfg.act_steps)
    max_episode_steps = int(args.max_episode_steps or cfg.env.max_episode_steps)
    threshold = float(cfg.env.get("best_reward_threshold_for_success", 1))

    print(f"run_dir: {run_dir}\nenv: {cfg.env.name} obs_dim={cfg.obs_dim} action_dim={cfg.action_dim} act_steps={act_steps}")
    venv = make_async(
        cfg.env.name,
        env_type=cfg.env.get("env_type", None),
        num_envs=args.eval_n_envs,
        asynchronous=True,
        max_episode_steps=max_episode_steps,
        wrappers=cfg.env.get("wrappers", None),
        obs_dim=cfg.obs_dim,
        action_dim=cfg.action_dim,
        **(cfg.env.specific if "specific" in cfg.env else {}),
    )
    venv.seed([args.seed_base + i for i in range(args.eval_n_envs)])

    model = hydra.utils.instantiate(cfg.model)
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt[args.weights])
    model.to(args.device)
    model.eval()
    epoch = ckpt.get("epoch", None)

    n_rounds = math.ceil(args.num_episodes / args.eval_n_envs)
    seeds = list(range(args.seed_base, args.seed_base + n_rounds * args.eval_n_envs))
    max_chunks = math.ceil(max_episode_steps / act_steps)

    successes, lengths = [], []
    t0 = time.time()
    for r in range(n_rounds):
        options = [{"seed": seeds[r * args.eval_n_envs + i]} for i in range(args.eval_n_envs)]
        obs = venv.reset_arg(options_list=options)
        if isinstance(obs, list):
            obs = {k: np.stack([o[k] for o in obs]) for k in obs[0].keys()}
        ep_reward_max = np.zeros(args.eval_n_envs)
        ep_len = np.zeros(args.eval_n_envs)
        ep_done = np.zeros(args.eval_n_envs, dtype=bool)
        for _ in range(max_chunks):
            with torch.no_grad():
                cond = {"state": torch.from_numpy(obs["state"]).float().to(args.device)}
                traj = model(cond=cond).trajectories.cpu().numpy()
            obs, reward, terminated, truncated, info = venv.step(traj[:, :act_steps])
            done = np.asarray(terminated) | np.asarray(truncated)
            for i in range(args.eval_n_envs):
                if not ep_done[i]:
                    ep_reward_max[i] = max(ep_reward_max[i], float(reward[i]))
                    ep_len[i] += act_steps
                    if done[i] or ep_len[i] >= max_episode_steps:
                        ep_done[i] = True
            if ep_done.all():
                break
        successes.extend((ep_reward_max >= threshold).tolist())
        lengths.extend(ep_len.tolist())
        print(f"round {r+1}/{n_rounds}: success {int(np.sum(ep_reward_max >= threshold))}/{args.eval_n_envs}  ({time.time()-t0:.0f}s)")

    successes = np.array(successes[: args.num_episodes])
    lengths = np.array(lengths[: args.num_episodes])
    sr = float(successes.mean())
    print(
        f"RESULT {cfg.env.name} epoch={epoch} weights={args.weights} "
        f"n={len(successes)} success_rate={sr:.3f} mean_len={lengths.mean():.1f} "
        f"success_len={lengths[successes].mean() if successes.any() else float('nan'):.1f}"
    )

    out = args.csv or os.path.join(run_dir, "eval_pretrain.csv")
    new = not os.path.exists(out)
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ckpt", "epoch", "weights", "n_episodes", "success_rate", "mean_len", "seed_base", "max_episode_steps"])
        w.writerow([os.path.basename(args.ckpt_path), epoch, args.weights, len(successes), f"{sr:.4f}", f"{lengths.mean():.1f}", args.seed_base, max_episode_steps])
    print(f"appended to {out}")
    venv.close()


if __name__ == "__main__":
    main()
