from kondo import Experiment
from tqdm.auto import tqdm
import functools
import torch


from torchrl.envs import make_gym_env, ParallelEnvs
from torchrl.utils.storage import Transition
from torchrl.controllers import Controller, RandomController


def log_dict(logger, info: dict, tag: str, step=None):
  for k, v in info.items():
    if v is not None:
      try:
        logger.add_scalar(f'{tag}/{k}', v,
                          global_step=step)
      except AssertionError:
        # NOTE(sanyam): some info may not be scalar and is ignored.
        pass


class BaseExperiment(Experiment):
  def __init__(self, env_id: str = None, n_envs: int = 1,
               n_frames: int = int(1e3), n_rand_frames: int = 0,
               n_train_interval: int = 100, **kwargs):
    assert env_id is not None, '"env_id" cannot be None'

    super().__init__(**kwargs)

    self.device = torch.device('cuda' if self.cuda else 'cpu')

    self.n_frames = n_frames
    self.n_rand_frames = n_rand_frames
    self.n_train_interval = n_train_interval

    self.envs = ParallelEnvs(functools.partial(make_gym_env, env_id),
                             n_envs=n_envs, base_seed=self.seed)

    self.controller = self.build_controller()

    self._cur_frames = 0

  def build_controller(self) -> Controller:
    return RandomController(self.envs.action_space)

  def act(self, obs_list: list) -> list:
    if self._cur_frames < self.n_rand_frames:
      return RandomController(self.envs.action_space).act(obs_list)
    return self.controller.act(obs_list)

  def store(self, transition_list):
    '''Placeholder method for storage related usage.
    '''

  def train(self) -> dict:
    '''Placeholder method for training related usage.
    '''
    return {}

  def run(self):
    with tqdm(initial=self._cur_frames,
              total=self.n_frames, unit='steps') as steps_bar:
      obs_list = self.envs.reset(range(self.envs.n_procs))

      while self._cur_frames < self.n_frames:
        action_list = self.act(obs_list)
        step_list = self.envs.step(list(range(self.envs.n_procs)), action_list)

        steps_bar.update(self.envs.n_procs)

        transition_list = []
        for i, (obs, action, (next_obs, rew, done, _)) in \
          enumerate(zip(obs_list, action_list, step_list)):

          obs_list[i] = next_obs
          self._cur_frames += 1

          transition_list.append(
              Transition(obs=obs, action=action,
                         reward=rew, next_obs=next_obs,
                         done=done))

          if done:
            log_dict(self.logger,
                     self.envs.exec_remote('info', proc_list=[i])[0],
                     tag='episode', step=self._cur_frames)

            obs_list[i] = self.envs.reset([i])[0]

        self.store(transition_list)

        if self._cur_frames >= self.n_rand_frames \
          and self._cur_frames % self.n_train_interval == 0:

          train_info = self.train()
          log_dict(self.logger, train_info, tag='train', step=self._cur_frames)

    self.logger.close()
    self.envs.close()
