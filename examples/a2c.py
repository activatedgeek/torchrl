from kondo import Spec
import torch
from torchrl.experiments import BaseExperiment
from torchrl.utils.storage import TransitionTupleDataset
from torchrl.contrib.controllers import A2CController


class A2CExperiment(BaseExperiment):
  def __init__(self, gamma=0.99, rollout_steps=5, alpha=0.5,
               lr=3e-4, beta=1e-3, lmbda=1.0, **kwargs):
    self._controller_args = dict(
        lr=lr,
        gamma=gamma,
        lmbda=lmbda,
        alpha=alpha,
        beta=beta
    )

    kwargs['n_train_interval'] = kwargs.get('n_envs', 1) * rollout_steps

    super().__init__(**kwargs)

    self.buffers = [TransitionTupleDataset()
                    for _ in range(self.envs.n_procs)]

  def store(self, transition_list):
    for buffer, transition in zip(self.buffers, transition_list):
      buffer.extend([transition])

  def build_controller(self):
    return A2CController(self.envs.observation_space.shape[0],
                         self.envs.action_space.n,
                         **self._controller_args,
                         device=self.device)

  def train(self):
    all_transitions = [[], [], [], [], []]
    all_returns = []

    for buffer in self.buffers:
      batch = [b.to(self.device) for b in buffer[:]]
      r = self.controller.compute_return(*batch)
      all_returns.append(r)

      for i, b in enumerate(batch):
        all_transitions[i].append(b)

      buffer.truncate()

    all_transitions = [torch.cat(t, dim=0) for t in all_transitions]
    all_returns = torch.cat(all_returns, dim=0)

    return self.controller.learn(*all_transitions, all_returns)

  @staticmethod
  def spec_list():
    return [
        Spec(
            group='a2c',
            exhaustive=True,
            params=dict(
                env_id=['CartPole-v0'],
                n_envs=16,
                n_frames=int(1e6),
                rollout_steps=5,
                gamma=0.99,
                lmbda=1.0,
                alpha=0.5,
                beta=1e-3,
                lr=3e-4
            )
        )
    ]
