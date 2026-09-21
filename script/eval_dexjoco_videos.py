"""Record exactly N independent DexJoCo episodes, sorted by actual success.

Uses a saved training config and strict model/EMA loading. Each action in a
chunk is stepped separately so recording, termination, and limits are exact.
"""

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DICE_RL_DATA_DIR", str(ROOT / "data_dir"))
os.environ.setdefault("DICE_RL_LOG_DIR", str(ROOT / "log_dir"))

import hydra
import imageio.v2 as imageio
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
import torch

from env.gym_utils.wrapper.dexjoco_image import DexjocoImageWrapper


def save_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def video_frame(env, task, episode, seed, step, status, epoch):
    frames = env._raw_frames
    width = min(640, frames[0].shape[1])
    height = round(frames[0].shape[0] * width / frames[0].shape[1] / 2) * 2
    canvas = Image.new("RGB", (width * len(frames), height + 64), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), f"{task} | epoch {epoch} | episode {episode:02d} | seed {seed} | step {step} | {status}", fill="white")
    for i, (key, frame) in enumerate(zip(env.image_keys, frames)):
        draw.text((i * width + 12, 38), key, fill="white")
        canvas.paste(Image.fromarray(frame).resize((width, height), Image.Resampling.BILINEAR), (i * width, 64))
    return np.asarray(canvas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode-offset", type=int, default=0,
                        help="Start after this many episodes when splitting an evaluation")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--weights", choices=["model", "ema"], default="model")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.episode_offset < 0:
        parser.error("--episode-offset must be nonnegative")
    torch.set_num_threads(1)
    checkpoint_path = args.checkpoint.resolve()
    cfg = OmegaConf.load(checkpoint_path.parent.parent / ".hydra/config.yaml")
    cfg.device = args.device
    if cfg.env.env_type != "dexjoco":
        raise ValueError("Expected a DexJoCo checkpoint")
    if int(cfg.cond_steps) != int(cfg.img_cond_steps):
        raise ValueError("This recorder requires matching state/image history lengths")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("success", "failure"):
        (args.output / name).mkdir()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    epoch = int(checkpoint["epoch"])
    policy = hydra.utils.instantiate(cfg.model)
    policy.load_state_dict(checkpoint[args.weights], strict=True)
    policy.eval()
    del checkpoint
    metadata = {
        "task": cfg.env_name, "checkpoint": str(checkpoint_path), "epoch": epoch,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "weights": args.weights, "episodes_requested": args.episodes,
        "seeds": list(range(args.seed + args.episode_offset,
                            args.seed + args.episode_offset + args.episodes)),
        "episode_offset": args.episode_offset,
        "max_episode_steps": int(cfg.env.max_episode_steps),
        "act_steps": int(cfg.act_steps), "fps": 30,
        "camera_order": list(cfg.env.wrappers.dexjoco_image.image_keys),
        "randomize": bool(cfg.env.wrappers.dexjoco_image.randomize),
        "success_criterion": "any simulator info.succeed (fallback: reward >= 1)",
        "terminal_hold_frames": 30,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    save_json(args.output / "metadata.json", metadata)
    OmegaConf.save(cfg, args.output / "config.yaml", resolve=True)
    env = DexjocoImageWrapper(**OmegaConf.to_container(cfg.env.wrappers.dexjoco_image, resolve=True))
    results = []
    try:
        for episode in range(args.episode_offset + 1, args.episode_offset + args.episodes + 1):
            started = time.monotonic()
            seed = args.seed + episode - 1
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            obs = env.reset(options={"seed": seed})
            history = deque([obs] * int(cfg.cond_steps), maxlen=int(cfg.cond_steps))
            pending = args.output / f"episode_{episode:02d}_seed_{seed}.partial.mp4"
            success = False
            step = 0
            reward_sum = 0.0
            terminated = truncated = False
            info = {}
            trace = []
            with imageio.get_writer(str(pending), fps=30, codec="libx264", quality=7,
                                    macro_block_size=1, ffmpeg_params=["-threads", "1", "-pix_fmt", "yuv420p"]) as writer:
                writer.append_data(video_frame(env, cfg.env_name, episode, seed, step, "RUNNING", epoch))
                while step < int(cfg.env.max_episode_steps):
                    cond = {key: torch.from_numpy(np.stack([o[key] for o in history])[None]).float().to(args.device)
                            for key in ("state", "rgb")}
                    cond["rgb"] = cond["rgb"].permute(0, 1, 4, 2, 3)
                    with torch.inference_mode():
                        actions = policy(cond=cond).trajectories[0, :int(cfg.act_steps)].cpu().numpy()
                    if not np.isfinite(actions).all():
                        raise ValueError("Policy emitted non-finite actions")
                    for action in actions:
                        obs, reward, terminated, info = env.step(action)
                        step += 1
                        history.append(obs)
                        reward_sum += reward
                        success = success or bool(info.get("succeed", reward >= 1.0))
                        truncated = bool(info.get("TimeLimit.truncated", False))
                        limited = step >= int(cfg.env.max_episode_steps)
                        done = success or terminated or truncated or limited
                        status = ("SUCCESS" if success else "FAILURE") if done else "RUNNING"
                        frame = video_frame(env, cfg.env_name, episode, seed, step, status, epoch)
                        writer.append_data(frame)
                        trace.append({"step": step, "reward": reward, "info": json_value(info)})
                        if done:
                            break
                    if done:
                        break
                for _ in range(30):
                    writer.append_data(frame)
            outcome = "success" if success else "failure"
            filename = f"episode_{episode:02d}_seed_{seed}_steps_{step:04d}.mp4"
            destination = args.output / outcome / filename
            pending.replace(destination)
            reason = "success" if success else ("environment_terminated" if terminated else
                      "environment_truncated" if truncated else "step_limit")
            result = {
                "episode": episode, "seed": seed, "success": success, "steps": step,
                "return": reward_sum, "end_reason": reason,
                "final_info": json_value(info), "video": str(destination.relative_to(args.output)),
                "frames": step + 31, "wall_seconds": round(time.monotonic() - started, 2),
            }
            save_json(destination.with_suffix(".json"), {**result, "trace": trace})
            results.append(result)
            summary = {**metadata, "episodes_completed": len(results),
                       "successes": sum(r["success"] for r in results),
                       "failures": sum(not r["success"] for r in results),
                       "success_rate": sum(r["success"] for r in results) / len(results),
                       "episodes": results}
            save_json(args.output / "summary.json", summary)
            print(json.dumps(result), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
