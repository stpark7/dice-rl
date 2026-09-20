# Repository Guidelines

## Project Structure & Module Organization

DICE-RL is a Python/PyTorch research codebase for finetuning diffusion and flow-based behavior-cloning policies.

- `agent/`: pretraining, finetuning, evaluation, and dataset orchestration.
- `model/`: diffusion, flow-matching, RL, and shared neural-network components.
- `env/` and `util/`: environment adapters, wrappers, replay buffers, and action utilities.
- `cfg/`: Hydra YAML experiment configurations grouped by environment, task, and training stage.
- `script/`: experiment launchers, checkpoint evaluation, downloads, and dataset conversion.
- `tests/`: unit and dataset-conversion tests; `installation/` and `docs/`: setup and walkthroughs; `media/`: README assets.

## Build, Test, and Development Commands

Run commands from the repository root in an activated environment. The README uses Python 3.8.

- `pip install -e .`: install the package and core dependencies for development.
- `pip install -e '.[robomimic]'`: install Robomimic-related dependencies; follow `installation/install_mujoco.md` for simulator setup.
- `source script/set_path.sh`: configure data/log directories and WandB entity; this also appends exports to `~/.bashrc`.
- `python script/run.py --config-name=pre_flow_matching_mlp --config-dir=cfg/robomimic/pretrain/lift/`: pretrain a state-based Lift policy after configuring dataset and normalizer paths.
- `python script/eval_rl_checkpoint.py --ckpt_path <checkpoint> --num_eval_episodes 10 --eval_n_envs 10`: evaluate a finetuned checkpoint and its associated pretrained policy.
- `python -m unittest discover -s tests -v`: run the test suite.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions/modules, and `PascalCase` classes. Match surrounding code and keep changes focused. No formatter or linter is configured in `pyproject.toml`. Follow existing config names such as `pre_flow_matching_mlp.yaml` and `ft_distill_residual_flow_mlp.yaml`; keep Hydra targets aligned with Python classes.

## Testing Guidelines

Tests use standard-library `unittest` and NumPy assertions. Name files `test_*.py` and methods `test_*`. Add focused regression tests for action transformations, dataset conversion, and wrapper behavior. Use temporary directories for generated fixtures. No coverage threshold is configured. For training changes, report the config, seed, runtime dependencies, and smoke-run results.

## Commit & Pull Request Guidelines

Recent commits use short, imperative subjects, such as “Honor configured BC evaluation frequency.” Follow that style. PRs should describe the problem, changed behavior, relevant configs, and validation commands/results. Link related issues and include evaluation metrics when policy behavior changes. Keep datasets, checkpoints, logs, and credentials out of commits.
