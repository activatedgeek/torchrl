import torch
from kondo import Spec
from torchrl.experiments import BaseExperiment
from torchrl.utils.storage import TransitionTupleDataset
from torchrl.contrib.controllers import SACController


class SACExperiment(BaseExperiment):
  def __init__(self, gamma=.99, tau=0.1,
               buffer_size=-1, batch_size=64, **kwargs):
    self._controller_args = dict(
        gamma=gamma,
        tau=tau
    )

    self.buffer = TransitionTupleDataset(size=buffer_size)
    self.batch_size = batch_size

    super().__init__(**kwargs)

  def store(self, transition_list):
    self.buffer.extend(transition_list)

  def build_controller(self):
    return SACController(self.envs.observation_space.shape[0],
                         self.envs.action_space.shape[0],
                         **self._controller_args,
                         device=self.device)

  def train(self):
    raise NotImplementedError

  @staticmethod
  def spec_list():
    return [
        Spec(
            group='sac',
            exhaustive=True,
            params=dict(
                env_id=['Pendulum-v0'],
                gamma=0.99,
                buffer_size=int(1e6),
            )
        )
    ]
