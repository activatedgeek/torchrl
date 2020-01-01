import torch
from kondo import Spec
from torchrl.experiments import BaseExperiment
from torchrl.contrib.controllers import DQNController
from torchrl.utils.storage import TransitionTupleDataset


class DQNExperiment(BaseExperiment):
  def __init__(self, double_dqn=False, gamma=.99,
               batch_size=32, lr=1e-3, buffer_size=1000, eps_max=1.0,
               eps_min=1e-2, n_eps_anneal=100, n_update_interval=10, **kwargs):
    self._controller_args = dict(
        double_dqn=double_dqn,
        gamma=gamma,
        lr=lr,
        eps_max=eps_max,
        eps_min=eps_min,
        n_eps_anneal=n_eps_anneal,
        n_update_interval=n_update_interval,
    )

    self.buffer = TransitionTupleDataset(size=buffer_size)
    self.batch_size = batch_size

    super().__init__(**kwargs)

  def store(self, transition_list):
    self.buffer.extend(transition_list)

  def build_controller(self):
    return DQNController(self.rollout_env.observation_space.shape[0],
                         self.rollout_env.action_space.n,
                         **self._controller_args,
                         device=self.device)

  def train(self):
    if len(self.buffer) < self.batch_size:
      return {}

    b_idx = torch.randperm(len(self.buffer))[:self.batch_size]
    b_transition = [b.to(self.device) for b in self.buffer[b_idx]]
    return self.controller.learn(*b_transition)

  @staticmethod
  def spec_list():
    return [
        Spec(
            group='dqn',
            params=dict(
                env_id=['CartPole-v0'],
                gamma=.99,
                n_train_interval=1,
                n_frames=20000,
                batch_size=32,
                buffer_size=1000,
                double_dqn=False,
                eps_max=1.0,
                eps_min=1e-2,
                n_update_interval=10,
                lr=1e-3,
                n_eps_anneal=500,
            ),
            exhaustive=True
        ),
        Spec(
            group='ddqn',
            params=dict(
                env_id=['CartPole-v0'],
                gamma=.99,
                n_train_interval=1,
                n_frames=20000,
                batch_size=32,
                buffer_size=1000,
                double_dqn=True,
                eps_max=1.0,
                eps_min=1e-2,
                n_update_interval=10,
                lr=1e-3,
                n_eps_anneal=500,
            ),
            exhaustive=True
        ),
        Spec(
            group='per',
            params=dict(
                env_id=['CartPole-v0'],
                gamma=.99,
                n_train_interval=1,
                n_frames=20000,
                batch_size=32,
                buffer_size=1000,
                double_dqn=False,
                prioritized=True,
                eps_max=1.0,
                eps_min=1e-2,
                n_update_interval=10,
                lr=1e-3,
                n_eps_anneal=500,
            ),
            exhaustive=True
        ),
    ]

