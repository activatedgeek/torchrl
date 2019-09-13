import time
import gym
import numpy as np

from .base_runner import BaseRunner
from ..agents import BaseAgent
from ..utils import MultiGymEnvs
from ..storage import Transition


class GymRunner(BaseRunner):
  """
  This is a runner for OpenAI Gym Environments.
  It follows the :code:`reset()`, :code:`step()` and :code:`close()`
  API to generate trajectories and renders each parallel trajectory
  in its own thread.

  Args:
      env_id (str): Environment ID registered with Gym.
      seed (int): Optional integer to seed stochastic environments.
      n_envs (int): Number of parallel environments (= trajectories).
      log_level (int): Log levels from :class:`gym.logger`. \
          (DEBUG = 10, INFO = 20, WARN = 30, ERROR = 40, DISABLED = 50)
  """
  def __init__(self, env_id: str, seed: int = None,
               n_envs: int = 1, log_level=gym.logger.ERROR):
    super(GymRunner, self).__init__()

    gym.logger.set_level(log_level)

    self.n_envs = n_envs
    self.env_id = env_id
    self.envs = MultiGymEnvs(self.make_env, n_envs=n_envs, base_seed=seed)

    self.obs = [None] * n_envs

  def make_env(self, seed: int = None) -> gym.Env:
    """
    Create an return the environment.
    See :meth:`~torchrl.runners.base_runner.BaseRunner.make_env` for
    general description.
    """
    env = gym.make(self.env_id)
    env.seed(seed)
    return env

  def maybe_reset(self):
    """
    This helper routine checks any inactive environments
    and resets them to allow future rollouts.
    """
    batch_reset_ids = self._get_active_envs(invert=True)
    if batch_reset_ids:
      new_obs = self.envs.reset(batch_reset_ids)
      for env_id, obs in zip(batch_reset_ids, new_obs):
        self.obs[env_id] = obs

  def compute_action(self, agent: BaseAgent, obs_list: list):
    """
    See :meth:`~torchrl.runners.base_runner.BaseRunner.compute_action`
    for general description.
    """
    return agent.act(obs_list)

  def rollout(self, agent, steps: int = None,
              render: bool = False, fps: int = 30) -> list:
    """
    Rollout trajectories from the Gym environment.
    See :meth:`~torchrl.runners.base_runner.BaseRunner.rollout` for
    general description.
    """
    assert self.obs is not None, """state is not defined,
    please `reset()`
    """

    steps = steps or self.MAX_STEPS

    self.maybe_reset()

    if render:
      env_id_list = self._get_active_envs()
      self.envs.render(env_id_list)
      time.sleep(1. / fps)

    trajectory_list = [[] for _ in range(self.n_envs)]

    while steps:
      env_id_list = self._get_active_envs()
      if not env_id_list:
        break

      obs_list = self._get_obs_list()
      action_list = self.compute_action(agent, obs_list)
      step_list = self.envs.step(env_id_list, action_list)

      for env_id, obs, action, (next_obs, reward, done, _) in zip(
          env_id_list, obs_list, action_list, step_list):

        trajectory_list[env_id].append(
            Transition(obs=obs, action=action, reward=[reward],
                       next_obs=next_obs, done=[done])
        )

        self.obs[env_id] = None if done else next_obs

      if render:
        self.envs.render(env_id_list)
        time.sleep(1. / fps)

      steps -= 1

    # TODO(sanyam): To do this here or leave upto the consumer?
    trajectory_list = [
        Transition(*[
            np.array(item, dtype=np.float)
            for item in zip(*trajectory)
        ])
        for trajectory in trajectory_list
    ]

    # TODO(sanyam): needs to be done for continuous action environments
    for i, traj in enumerate(trajectory_list):
      if traj.action.ndim != 2:
        trajectory_list[i] = Transition(
            obs=traj.obs,
            action=np.squeeze(traj.action, axis=-1),
            reward=traj.reward,
            next_obs=traj.next_obs,
            done=traj.done,
        )

    return trajectory_list

  def close(self):
    """
    Shutdown gym environments.

    See :meth:`~torchrl.runners.base_runner.BaseRunner.close`
    for a general description.
    """
    self.envs.close()

  def _get_active_envs(self, invert=False) -> list:
    """
    Gets a list of active environments.

    Environments are active when their observations
    are not None. The result is complement if invert
    is True. Observations being :code:`None` is
    equivalent to termination by design.

    Args:
        invert (bool): If :code:`True`, get list of all terminated \
          environments.
    """

    target_ids = []

    for env_id, obs in enumerate(self.obs):
      if invert:
        if obs is None:
          target_ids.append(env_id)
        continue

      if obs is not None:
        target_ids.append(env_id)

    return target_ids

  def _get_obs_list(self):
    return list(filter(lambda obs: obs is not None, self.obs))
