"""Keep Temporal K9 RL settings aligned with the recorded seed-42 reference."""

from pathlib import Path
import hashlib
import unittest

import hydra
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "cfg/dexjoco/finetune"

# Sanitized snapshot of WandB run zbsqw25b, created 2026-11-05T03:59:09Z:
# gist_ailab/dexjoco-pick_bucket_latent_residual_batch256. No API access needed.
REFERENCE_TRAIN = {
    "tau": 0.01, "gamma": 0.99, "render": {"num": 3, "freq": 0},
    "n_steps": 100, "actor_lr": 1e-4, "log_freq": 1, "val_freq": 10,
    "critic_lr": 1e-4, "eval_freq": 1000, "save_freq": 1000,
    "batch_size": 256, "n_train_itr": 1000, "update_freq": 1,
    "lr_scheduler": {"min_lr": 1e-6, "warmup_steps": 10, "first_cycle_steps": 1000},
    "max_grad_norm": 1, "gradient_steps": 10, "num_train_steps": 10000,
    "save_model_freq": 100, "use_lr_scheduler": True, "actor_update_freq": 2,
    "actor_weight_decay": 1e-5, "critic_weight_decay": 1e-5,
    "critic_target_update_freq": 2,
}
REFERENCE_MODEL = {
    "device": "cuda:0", "n_step": 3,
    "_target_": "model.rl.hand_chord_residual_rl_img.HandChordResidualRLImgModel",
    "noise_dim": 352, "action_dim": 22, "cond_steps": 1, "use_n_step": True,
    "critic_weight": 1, "horizon_steps": 16, "use_layernorm": True,
    "bc_loss_weight": 50, "activation_type": "GELU",
    "actor_hidden_dims": [1024, 1024, 1024],
    "critic_hidden_dims": [1024, 1024, 1024],
    "use_q_normalization": True, "critic_ensemble_size": 10,
    "use_soft_q_filtering": True, "num_next_noise_samples": 4,
    "multi_sample_next_noise": True, "q_filtering_warmup_steps": 1000,
    "replay_flow_warmup_steps": 3000, "num_multi_z_for_actor_loss": 16,
    "q_underestimation_threshold": -0.5, "disable_q_loss_for_expert_data": True,
    "condition_residual_on_base_action": False,
    "always_retain_bc_loss_for_expert_data": True,
}
REFERENCE_REPLAY = {
    "n_step": 3, "max_size": 1000000, "use_n_step": True,
    "expert_n_step": 3, "expert_use_n_step": True,
}
REFERENCE_SHARED = {
    "seed": 42, "device": "cuda:0", "run_eval": True, "use_rlpd": True,
    "act_steps": 16, "noise_dim": 352, "abs_action": False, "action_dim": 22,
    "cond_steps": 1, "flow_steps": 10, "use_6d_rot": False,
    "eval_n_envs": 4, "expert_ratio": 0.3, "horizon_steps": 16,
    "img_cond_steps": 1, "evaluate_strategy": "max_q_min",
    "num_eval_episodes": 50, "visualization_freq": 200000000,
    "log_q_overestimation": True, "visualize_trajectories": False,
    "num_exploration_samples": 16, "online_explore_strategy": "max_q_min",
    "adaptive_expert_ratio_end": 0.1, "use_adaptive_expert_ratio": True,
    "adaptive_expert_ratio_start": 0.7, "adaptive_expert_ratio_steps": 10000,
}


def compose_task(task, overrides=None):
    with hydra.initialize_config_dir(config_dir=str(CONFIG_ROOT / task), version_base=None):
        cfg = hydra.compose(config_name="ft_temporal_hand_chord_residual_flow_unet_img",
                            overrides=overrides or [])
        return OmegaConf.to_container(cfg, resolve=True)


class TemporalHandChordRLConfigTests(unittest.TestCase):
    def test_both_tasks_match_reference_training_model_and_replay_settings(self):
        for task in ("pick_bucket", "fold_glasses"):
            with self.subTest(task=task):
                cfg = compose_task(task)
                for group, expected in (("train", REFERENCE_TRAIN),
                                        ("model", REFERENCE_MODEL),
                                        ("replay_buffer", REFERENCE_REPLAY)):
                    self.assertEqual({key: cfg[group][key] for key in expected}, expected)
                self.assertEqual({key: cfg[key] for key in REFERENCE_SHARED}, REFERENCE_SHARED)
                self.assertEqual(cfg["env"]["n_envs"], 10)
                self.assertEqual(cfg["obs_dim"], 23)
                self.assertEqual(cfg["model"]["obs_dim"], 23)
                self.assertEqual(cfg["replay_buffer"]["storage_device"], "cpu")
                self.assertTrue(cfg["replay_buffer"]["pin_memory"])
                self.assertIsNone(cfg["wandb"])

    def test_task_specific_checkpoints_demonstrations_and_episode_limits_remain(self):
        for task, epoch, episodes, limit in (("pick_bucket", 2000, 98, 700),
                                             ("fold_glasses", 1100, 100, 900)):
            with self.subTest(task=task):
                cfg = compose_task(task)
                self.assertEqual(cfg["env_name"], task)
                self.assertTrue(cfg["base_policy_path"].endswith(
                    f"pretrained_bc_policy_{task}_img/checkpoint/state_{epoch}.pt"))
                self.assertEqual(cfg["model"]["pretrained_flow_policy_path"], cfg["base_policy_path"])
                self.assertEqual(cfg["expert_dataset"]["max_n_episodes"], episodes)
                self.assertTrue(cfg["expert_dataset"]["dataset_path"].endswith(
                    f"dexjoco/{task}-img/ph_finetune/train.npz"))
                self.assertTrue(cfg["normalization_path"].endswith(
                    f"dexjoco/{task}-img/ph_pretrain/normalization.npz"))
                self.assertEqual(cfg["env"]["max_episode_steps"], limit)
                wrappers = cfg["env"]["wrappers"]
                self.assertEqual(wrappers["dexjoco_image"]["max_episode_steps"], limit)
                self.assertEqual(wrappers["multi_step"]["max_episode_steps"], limit)
                self.assertEqual(wrappers["dexjoco_image"]["image_keys"], ["front", "wrist"])
                self.assertTrue(cfg["env"]["save_video"])

    def test_temporal_names_and_codec_root_can_be_overridden_without_changing_base(self):
        for task in ("pick_bucket", "fold_glasses"):
            with self.subTest(task=task):
                cfg = compose_task(task)
                self.assertIn("temporal_k9", cfg["name"])
                self.assertIn(cfg["name"], cfg["logdir"])
                self.assertTrue(cfg["model"]["chord_path"].endswith(
                    f"checkpoints/temporal-hand-chord/k9_seed42/{task}/chord.pt"))
                overridden = compose_task(task, ["chord_run_dir=/tmp/another_codec_run"])
                self.assertEqual(overridden["model"]["chord_path"],
                                 f"/tmp/another_codec_run/{task}/chord.pt")
                self.assertEqual(overridden["base_policy_path"], cfg["base_policy_path"])

    def test_bundled_codecs_match_default_paths_and_integrity_manifest(self):
        from model.hand_chord.codec import load_hand_chord

        root = ROOT / "checkpoints/temporal-hand-chord/k9_seed42"
        for line in (root / "checksums.sha256").read_text().splitlines():
            expected, name = line.split("  ", 1)
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), expected)
        for task in ("pick_bucket", "fold_glasses"):
            with self.subTest(task=task):
                cfg = compose_task(task)
                chord = load_hand_chord(ROOT / cfg["model"]["chord_path"])
                self.assertEqual(chord.metadata()["latent_layout"], "B,H,K")
                self.assertEqual((chord.horizon_steps, chord.latent_dim), (16, 9))
                self.assertEqual((chord.raw_action_dim, chord.policy_dim), (22, 240))


if __name__ == "__main__":
    unittest.main()
