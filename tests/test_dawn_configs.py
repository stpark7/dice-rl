"""Hydra configuration coverage for the DexJoCo DAWN baseline."""

from pathlib import Path
import unittest

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("click_mouse", "fold_glasses", "hammer_nail", "pick_bucket",
         "pinch_tongs", "water_plant")


class DAWNConfigTests(unittest.TestCase):
    def compose_task(self, task, overrides=None, name="ft_dawn_unet_img"):
        directory = ROOT / "cfg" / "dexjoco" / "finetune" / task
        with initialize_config_dir(config_dir=str(directory), version_base=None):
            return compose(config_name=name, overrides=overrides or [])

    def test_all_tasks_preserve_resfit_environment_and_base_policy_settings(self):
        for task in TASKS:
            with self.subTest(task=task):
                cfg = self.compose_task(task)
                reference = OmegaConf.load(
                    ROOT / "cfg" / "dexjoco" / "finetune" / task /
                    "ft_resfit_unet_img.yaml"
                )
                for key in ("env", "shape_meta", "obs_dim", "action_dim",
                            "horizon_steps", "cond_steps", "bc_action_steps",
                            "base_policy_path", "normalization_path",
                            "num_eval_episodes", "eval_n_envs", "eval_seed"):
                    actual = OmegaConf.to_container(cfg[key], resolve=True) if (
                        OmegaConf.is_config(cfg[key])) else cfg[key]
                    expected = OmegaConf.to_container(reference[key], resolve=True) if (
                        OmegaConf.is_config(reference[key])) else reference[key]
                    self.assertEqual(actual, expected, key)
                self.assertEqual(cfg._target_,
                                 "agent.finetune.train_dawn_agent.TrainDAWNAgent")
                self.assertEqual(cfg.model._target_, "model.rl.dawn.DAWNModel")
                self.assertNotIn("expert_dataset", cfg)
                self.assertEqual(cfg.train.n_step, 1)
                self.assertEqual(cfg.train.critic_warmup_steps, 0)
                self.assertEqual(cfg.train.reward_offset, -1.0)
                self.assertEqual(cfg.train.utd_ratio, .25)

    def test_overrides_resolve_without_tracking_or_simulator(self):
        cfg = self.compose_task("pick_bucket", [
            "device=cpu", "wandb=null", "horizon_steps=8", "bc_action_steps=8",
            "base_policy_path=/tmp/native_bc.pt",
            "base_policy_config_path=/tmp/native_bc.yaml",
            "normalization_path=/tmp/normalization.npz",
            "mode=eval", "checkpoint_path=/tmp/dawn.pt",
        ])
        resolved = OmegaConf.to_container(cfg, resolve=True)
        self.assertEqual(resolved["model"]["horizon_steps"], 8)
        self.assertEqual(resolved["model"]["pretrained_policy_path"], "/tmp/native_bc.pt")
        self.assertEqual(resolved["model"]["pretrained_config_path"], "/tmp/native_bc.yaml")
        self.assertEqual(resolved["env"]["wrappers"]["dexjoco_image"]["normalization_path"],
                         "/tmp/normalization.npz")
        self.assertEqual(resolved["env"]["wrappers"]["multi_step"]["n_action_steps"], 1)
        self.assertIsNone(resolved["wandb"])

    def test_run_directories_name_the_algorithm_without_tracking(self):
        # The trainer names the WandB run after this directory.
        for name, label in (("ft_dawn_unet_img", "dawn"), ("ft_resfit_unet_img", "resfit")):
            for task in TASKS:
                with self.subTest(name=name, task=task):
                    cfg = self.compose_task(task, ["wandb=null", "seed=7"], name)
                    self.assertRegex(cfg.logdir, rf"/dexjoco-{label}/{task}_{label}_unet_img/{label}-s7_\d")


if __name__ == "__main__":
    unittest.main()
