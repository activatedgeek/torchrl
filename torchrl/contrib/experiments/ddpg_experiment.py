import torch
from kondo import Spec
from torchrl.experiments import BaseExperiment
from torchrl.utils.storage import TransitionTupleDataset
from torchrl.contrib.controllers import DDPGController


class DDPGExperiment(BaseExperiment):
  def __init__(self, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99,
               tau=1e-2, batch_size=32, buffer_size=1000,
               n_ou_reset_interval=100000, **kwargs):
    self._controller_args = dict(
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        gamma=gamma,
        tau=tau,
        n_reset_interval=n_ou_reset_interval,
    )

    self.buffer = TransitionTupleDataset(size=buffer_size)
    self.batch_size = batch_size

    super().__init__(**kwargs)

  def store(self, transition_list):
    self.buffer.extend(transition_list)

  def build_controller(self):
    return DDPGController(self.rollout_env.observation_space.shape[0],
                          self.rollout_env.action_space.shape[0],
                          self.rollout_env.action_space.low,
                          self.rollout_env.action_space.high,
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
            group='ddpg',
            params=dict(
                env_id=['Pendulum-v0'],
                gamma=.99,
                n_train_interval=1,
                n_frames=30000,
                batch_size=128,
                buffer_size=int(1e6),
                actor_lr=1e-4,
                critic_lr=1e-3,
                tau=1e-2,
                # n_ou_reset_interval=10000,
                # ou_mu = 0.0
                # ou_theta = 0.15
                # ou_sigma = 0.2
            ),
            exhaustive=True
        )
    ]
