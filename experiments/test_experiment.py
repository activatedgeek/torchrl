import pytest
import importlib
import torch
import argparse


@pytest.mark.parametrize('package, class_name, hparam_func', [
    ('experiments.a2c', 'A2CCartpole', 'hparams_a2c_cartpole'),
    ('experiments.ppo', 'PPOPendulum', 'hparams_ppo_pendulum'),
    # ('experiments.prioritized_dqn', 'PERCartpole', 'hparams_per_cartpole'),
])
def test_experiment(package: str, class_name: str, hparam_func: str):
  cls = getattr(importlib.import_module(package), class_name)

  device = torch.device('cpu')

  # NOTE(sanyam): only for reference, all optional
  args = dict(
      seed=1,
      log_interval=1000,
      eval_interval=1000,
      num_eval=1,
  )

  hparams = getattr(cls, hparam_func)()
  hparams.num_total_steps = int(1e3)

  exp = cls(
      hparams,
      argparse.Namespace(**args),
      None,  # disable logging
      device=device,
      show_progress=False,
  )
  exp.run()
