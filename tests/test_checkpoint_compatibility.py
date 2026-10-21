"""Load full RL checkpoints with the supported PyTorch versions."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch

from agent.eval.eval_pretrained_flow import load_flow_matching_policy
from agent.finetune.train_distill_residual_flow_agent import TrainDistillResidualFlowAgent
from script import eval_rl_checkpoint


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_flow_policy_loader_reads_embedded_omegaconf_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base.pt"
            expected = torch.nn.Linear(1, 1)
            torch.save({
                "cfg": OmegaConf.create({"model": {
                    "_target_": "torch.nn.Linear", "in_features": 1, "out_features": 1,
                }}),
                "model": expected.state_dict(),
            }, path)

            model = load_flow_matching_policy(str(path), device="cpu")

            self.assertFalse(model.training)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, expected.state_dict()[key])

    def test_residual_agent_loads_checkpoint_with_omegaconf_config(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = object.__new__(TrainDistillResidualFlowAgent)
            agent.logdir = directory
            agent.device = "cpu"
            agent.model = torch.nn.Linear(1, 1)
            agent.actor_optimizer = torch.optim.Adam(agent.model.parameters())
            agent.critic_optimizer = torch.optim.Adam(agent.model.parameters())
            agent.actor_scheduler = agent.critic_scheduler = None
            expected = {key: value.clone() for key, value in agent.model.state_dict().items()}
            torch.save({
                "model_state_dict": expected,
                "actor_optimizer_state_dict": agent.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": agent.critic_optimizer.state_dict(),
                "config": OmegaConf.create({"env_name": "pick_bucket"}),
                "replay_buffer": {"restored": True},
                "step": 7,
            }, Path(directory) / "model_step_7.pth")
            with torch.no_grad():
                for parameter in agent.model.parameters():
                    parameter.zero_()

            agent.load(7)

            for key, value in agent.model.state_dict().items():
                torch.testing.assert_close(value, expected[key])
            self.assertEqual(agent.replay_buffer, {"restored": True})

    def check_evaluation_checkpoint(self, directory, use_image_obs):
        cfg = OmegaConf.create({
            "env_name": "pick_bucket", "device": "cpu",
            "obs_dim": 23, "action_dim": 22, "horizon_steps": 8,
            "env": {"name": "pick_bucket", "env_type": "dexjoco",
                    "max_episode_steps": 16, "use_image_obs": use_image_obs},
        })
        if use_image_obs:
            base_path = Path(directory) / "base.pt"
            cfg.base_policy_path = str(base_path)
            torch.save({"cfg": OmegaConf.create({"model": {"network": {
                "_target_": "model.flow_matching.vision_resnet_mlp_flow_matching.VisionResNetFlowMatchingMLP",
                "visual_feature_dim": 128,
            }}})}, base_path)
        path = Path(directory) / "rl.pth"
        torch.save({"config": cfg}, path)

        # Use the real checkpoint reader; stop before simulator construction.
        with patch("sys.argv", ["eval_rl_checkpoint.py", "--ckpt_path", str(path),
                                "--device", "cpu"]), \
                patch.object(eval_rl_checkpoint, "make_async",
                             side_effect=RuntimeError("evaluation setup reached")) as make_env:
            with self.assertRaisesRegex(RuntimeError, "evaluation setup reached"):
                eval_rl_checkpoint.main()
        self.assertEqual(make_env.call_args.kwargs["obs_dim"],
                         151 if use_image_obs else 23)

    def test_evaluator_loads_rl_checkpoint_with_omegaconf_config(self):
        with tempfile.TemporaryDirectory() as directory:
            self.check_evaluation_checkpoint(directory, use_image_obs=False)

    def test_image_evaluator_loads_embedded_pretrained_config(self):
        with tempfile.TemporaryDirectory() as directory:
            self.check_evaluation_checkpoint(directory, use_image_obs=True)


if __name__ == "__main__":
    unittest.main()
