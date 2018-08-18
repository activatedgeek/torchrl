import time
import gym

from .base_runner import BaseRunner
from ..agents import BaseAgent
from ..utils import MultiGymEnvs
from ..utils.gym_utils import init_run_history
from ..utils.gym_utils import append_run_history


class GymRunner(BaseRunner):
  """Runner for OpenAI Gym Environments

  This class is a simple wrapper around
  OpenAI Gym environments with essential
  plug points into various steps of the
  rollout.
  """
  def __init__(self, env_id: str, seed: int = None,
               n_envs: int = 1):
    super(GymRunner, self).__init__()

    # Gym throws plenty of warnings with each make.
    gym.logger.set_level(gym.logger.ERROR)

    self.n_envs = n_envs
    self.env_id = env_id
    self.envs = MultiGymEnvs(self.make_env, n_envs=n_envs, base_seed=seed)

    self.obs = [None] * n_envs

  def make_env(self, seed: int = None) -> gym.Env:
    env = gym.make(self.env_id)
    env.seed(seed)
    return env

  def maybe_reset(self):
    batch_reset_ids = self._get_active_envs(invert=True)
    if batch_reset_ids:
      new_obs = self.envs.reset(batch_reset_ids)
      for env_id, obs in zip(batch_reset_ids, new_obs):
        self.obs[env_id] = obs

  def compute_action(self, agent: BaseAgent, obs_list: list):
    """Compute Actions from the agent."""
    return agent.act(obs_list)

  def process_transition(self, history,
                         transition: tuple) -> list:
    if history is None:
      history = init_run_history(self.envs.observation_space,
                                 self.envs.action_space)

    # Rearrange according to convention
    obs, action, next_obs, reward, done, _ = transition
    transition = (obs, action, reward, next_obs, done)

    append_run_history(history, *transition)

    return history

  def rollout(self, agent, steps: int = None,
              render: bool = False, fps: int = 30) -> list:
    assert self.obs is not None, """state is not defined,
    please `reset()`
    """

    steps = steps or self.MAX_STEPS

    self.maybe_reset()

    if render:
      env_id_list = self._get_active_envs()
      self.envs.render(env_id_list)
      time.sleep(1. / fps)

    history_list = [None] * self.n_envs

    while steps:
      env_id_list = self._get_active_envs()
      if not env_id_list:
        break

      obs_list = self._get_obs_list()
      action_list = self.compute_action(agent, obs_list)
      step_list = self.envs.step(env_id_list, action_list)

      transition_list = [
          (obs, action, *step)
          for obs, action, step in zip(obs_list, action_list, step_list)
      ]
      for env_id, transition in zip(env_id_list, transition_list):
        history_list[env_id] = self.process_transition(history_list[env_id],
                                                       transition)

      for env_id, (next_obs, _, done, _) in zip(env_id_list, step_list):
        self.obs[env_id] = None if done else next_obs

      if render:
        self.envs.render(env_id_list)
        time.sleep(1. / fps)

      steps -= 1

    return history_list

  def close(self):
    """Close the environment."""
    self.envs.close()

  def _get_active_envs(self, invert=False) -> list:
    """Gets a list of active environments.

    Environments are active when their observations
    are not None. The result is complement if invert
    is True.
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
