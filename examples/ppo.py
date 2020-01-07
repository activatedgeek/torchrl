from kondo import Spec
import torch
from torch.utils.data import DataLoader
from torchrl.experiments import BaseExperiment
from torchrl.utils.storage import TensorTupleDataset, TransitionTupleDataset
from torchrl.contrib.controllers import PPOController


class PPOExperiment(BaseExperiment):
  def __init__(self, gamma=0.99, rollout_steps=10, alpha=0.5,
               lr=3e-4, beta=1e-3, lmbda=1.0, clip_ratio=0.2,
               max_grad_norm=1.0, batch_size=32, n_epochs=4,
               **kwargs):
    self._controller_args = dict(
        lr=lr,
        gamma=gamma,
        lmbda=lmbda,
        alpha=alpha,
        beta=beta,
        clip_ratio=clip_ratio,
        max_grad_norm=max_grad_norm
    )

    kwargs['n_train_interval'] = kwargs.get('n_envs', 1) * rollout_steps

    super().__init__(**kwargs)

    self.buffers = [TransitionTupleDataset()
                    for _ in range(self.envs.n_procs)]
    self.batch_size = batch_size
    self.n_epochs = n_epochs

  def store(self, transition_list):
    for buffer, transition in zip(self.buffers, transition_list):
      buffer.extend([transition])

  def build_controller(self):
    return PPOController(self.envs.observation_space.shape[0],
                         self.envs.action_space.shape[0],
                         **self._controller_args,
                         device=self.device)

  def train(self):
    all_transitions = [[], [], [], [], []]
    all_returns = []
    all_log_probs = []
    all_advs = []

    for buffer in self.buffers:
      batch = [b.to(self.device) for b in buffer[:]]
      r, log_probs, values = self.controller.compute_return(*batch)
      all_returns.append(r)
      all_log_probs.append(log_probs)
      all_advs.append(r - values)

      for i, b in enumerate(batch):
        all_transitions[i].append(b)

      buffer.truncate()

    all_transitions = [torch.cat(t, dim=0) for t in all_transitions]
    all_returns = torch.cat(all_returns, dim=0)
    all_log_probs = torch.cat(all_log_probs, dim=0)
    all_advs = torch.cat(all_advs, dim=0)

    ds = TensorTupleDataset(x=[*all_transitions,
                               all_returns, all_log_probs, all_advs])
    loader = DataLoader(ds, batch_size=self.batch_size)

    train_info = dict()
    for _ in range(self.n_epochs):
      for minibatch in loader:
        train_info = self.controller.learn(*minibatch)
    return train_info

  @staticmethod
  def spec_list():
    return [
        Spec(
            group='ppo',
            exhaustive=True,
            params=dict(
                env_id=['Pendulum-v0'],
                n_envs=16,
                n_frames=int(5e6),
                rollout_steps=20,
                gamma=0.99,
                lmbda=0.95,
                alpha=0.5,
                beta=1e-3,
                lr=3e-4,
                batch_size=64,
                clip_ratio=0.2,
                max_grad_norm=1.0,
                n_epochs=4
            )
        )
    ]
