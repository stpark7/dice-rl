"""Exercise the real image residual agent with a small, explicitly bounded run.

Run from the repository root with PYTHONPATH=. and the BC data environment set.
The 50-episode dataset is loaded; only 64 evenly spaced expert transitions are
encoded for this smoke test. This is not a training or success-rate experiment.
"""

import hashlib
import json
import os
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Subset

from agent.finetune.train_distill_residual_flow_img_agent import TrainDistillResidualFlowImgAgent


def digest(module):
    result = hashlib.sha256()
    for tensor in module.state_dict().values():
        result.update(tensor.detach().cpu().numpy().tobytes())
    return result.hexdigest()


class SmokeAgent(TrainDistillResidualFlowImgAgent):
    def _preprocess_expert_dataset(self, dataset):
        assert len(dataset.traj_lengths) == 50
        self.loaded_expert_transitions = len(dataset)
        indices = np.linspace(0, len(dataset) - 1, 64, dtype=int).tolist()
        return super()._preprocess_expert_dataset(Subset(dataset, indices))

    def update_networks(self, training_step=0):
        result = super().update_networks(training_step)
        if result is not None:
            losses, _ = result
            for key in ('actor_total', 'critic_loss'):
                assert torch.isfinite(losses[key]).all(), key
            self.smoke_updates += 1
            self.last_losses = {key: float(losses[key].detach())
                                for key in ('actor_total', 'critic_loss')}
        return result


@hydra.main(version_base=None, config_path='../cfg/dexjoco/finetune/pick_bucket',
            config_name='ft_distill_residual_flow_unet_img')
def main(cfg):
    if str(cfg.device) == 'cpu':
        # Adam also queries CUDA availability in a CUDA-enabled torch build.
        # With no visible devices, the NVML check avoids initializing a driver.
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'
    cfg.wandb = None
    cfg.run_eval = False
    cfg.env.n_envs = 1
    cfg.env.max_episode_steps = 16
    cfg.train.num_train_steps = 4
    cfg.train.batch_size = 4
    cfg.train.gradient_steps = 1
    cfg.train.use_lr_scheduler = False
    cfg.replay_buffer.max_size = 128
    cfg.log_q_overestimation = False
    # MC-return diagnostics are unused in this short check.
    with open_dict(cfg.expert_dataset):
        cfg.expert_dataset.get_mc_return = False
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    OmegaConf.save(cfg, output_dir / 'smoke_config.yaml', resolve=True)
    agent = SmokeAgent(cfg)
    try:
        modules = dict(base=agent.model.pretrained_flow_policy,
                       actor=agent.model.actor, critic=agent.model.critic)
        before = {key: digest(module) for key, module in modules.items()}
        assert not any(p.requires_grad for p in modules['base'].parameters())
        agent.smoke_updates = 0
        agent.run()
        after = {key: digest(module) for key, module in modules.items()}
        assert before['base'] == after['base'], 'Base policy changed'
        assert before['actor'] != after['actor'], 'Actor did not update'
        assert before['critic'] != after['critic'], 'Critic did not update'
        assert agent.smoke_updates > 0
        assert agent.replay_buffer.num_episodes > 0, 'No online episode collected'
        report = dict(seed=cfg.seed, device=cfg.device,
                      renderer=os.environ.get('MUJOCO_GL'),
                      base_policy_path=cfg.base_policy_path,
                      loaded_expert_transitions=agent.loaded_expert_transitions,
                      encoded_expert_transitions=64,
                      rollout_calls=cfg.train.num_train_steps,
                      online_episodes=agent.replay_buffer.num_episodes,
                      updates=agent.smoke_updates, losses=agent.last_losses,
                      base_unchanged=True, actor_changed=True, critic_changed=True,
                      before=before, after=after)
        output = output_dir / 'smoke_result.json'
        output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    finally:
        agent.venv.close()


if __name__ == '__main__':
    main()
