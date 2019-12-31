from kondo import Experiment
from tqdm.auto import tqdm
import torch


from torchrl.envs import make_gym_env, TransitionMonitor
from torchrl.controllers import Controller, RandomController


class BaseExperiment(Experiment):
  def __init__(self, env_id: str = None, n_frames: int = int(1e3),
               n_rand_frames: int = 0, **kwargs):
    assert env_id is not None, '"env_id" cannot be None'

    super().__init__(**kwargs)

    self.device = torch.device('cuda' if self.cuda else 'cpu')

    self.n_frames = n_frames
    self.n_rand_frames = n_rand_frames

    self.rollout_env = TransitionMonitor(make_gym_env(env_id, seed=self.seed))
    self.controller = self.build_controller()

    self._cur_frames = 0

  def build_controller(self) -> Controller:
    return RandomController(self.rollout_env.action_space)

  def act(self, obs):  # pylint: disable=unused-argument
    return self.controller.act()

  def _run(self):
    with tqdm(initial=self._cur_frames,
              total=self.n_frames, unit='steps') as steps_bar:
      self.rollout_env.reset()

      while self._cur_frames < self.n_frames:
        action = self.act(self.rollout_env.obs)
        self.rollout_env.step(action)

        self._cur_frames += 1
        steps_bar.update(1)

        if self.rollout_env.is_done:
          for k, v in self.rollout_env.info.items():
            if v is not None:
              try:
                self.logger.add_scalar(f'episode/{k}', v,
                                       global_step=self._cur_frames)
              except AssertionError:
                # NOTE(sanyam): some info may not be scalar and is ignored.
                pass

          self.rollout_env.reset()

    self.logger.close()
    self.rollout_env.close()

  def run(self):
    self._run()
