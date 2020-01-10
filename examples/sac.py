from torch.utils.data import DataLoader 
from kondo import Spec
from torchrl.experiments import BaseExperiment
from torchrl.utils.storage import TransitionTupleDataset
from torchrl.contrib.controllers import SACController


class SACExperiment(BaseExperiment):
  def __init__(self, gamma=.99, tau=0.1, alpha=1e-2, lr=3e-4,
               buffer_size=-1, batch_size=256, n_updates=1,
               **kwargs):
    self._controller_args = dict(
        gamma=gamma,
        tau=tau,
        alpha=alpha,
        lr=lr
    )

    self.buffer = TransitionTupleDataset(size=buffer_size)
    self.batch_size = batch_size
    self.n_updates = n_updates

    super().__init__(**kwargs)

  def store(self, transition_list):
    self.buffer.extend(transition_list)

  def build_controller(self):
    return SACController(self.envs.observation_space.shape[0],
                         self.envs.action_space.shape[0],
                         **self._controller_args,
                         device=self.device)

  def train(self):
    if len(self.buffer) < self.batch_size:
      return {}

    train_info = dict()
    for _ in range(self.n_updates):
      loader = DataLoader(self.buffer, batch_size=self.batch_size)
      for minibatch in loader:
        minibatch = [b.to(self.device) for b in minibatch]
        train_info = self.controller.learn(*minibatch)
        break

    if self.
    return train_info

  @staticmethod
  def spec_list():
    return [
        Spec(
            group='sac',
            exhaustive=True,
            params=dict(
                env_id=['Pendulum-v0'],
                n_frames=int(1e5),
                gamma=0.99,
                buffer_size=int(1e6),
                n_train_interval=1,
                lr=3e-4,
                tau=5e-3,
                batch_size=256,
                n_updates=1,
            )
        )
    ]


if __name__ == "__main__":
  from kondo import HParams

  hp = HParams(SACExperiment)

  _, trial = next(hp.trials(groups=['sac']))

  trial['log_dir'] = 'log/sac'

  SACExperiment(**trial).run()
