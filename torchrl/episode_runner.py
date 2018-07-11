import time
import gym
import numpy as np
import torch
import functools

from .agents import BaseAgent
from .utils import get_gym_spaces
from .multi_proc_wrapper import MultiProcWrapper


DEFAULT_MAX_STEPS = int(1e6)


class MultiEnvs(MultiProcWrapper):
  """
  A utility class which wraps around multiple environments
  and runs them in subprocesses
  """
  def __init__(self, make_env_fn, n_envs: int = 1, base_seed: int = 0,
               daemon: bool = True, autostart: bool = True):
    obj_fns = [
        functools.partial(self.make_env, make_env_fn,
                          None if base_seed is None else base_seed + rank)
        for rank in range(1, n_envs + 1)
    ]
    super(MultiEnvs, self).__init__(obj_fns, daemon=daemon, autostart=autostart)

  def reset(self, env_ids: list):
    return self.exec_remote('reset', proc_list=env_ids)

  def step(self, env_ids: list, actions: list):
    return self.exec_remote('step', args_list=actions, proc_list=env_ids)

  def close(self):
    self.exec_remote('close')

  @staticmethod
  def make_env(make_env_fn, seed: int = None):
    env = make_env_fn()
    if seed is not None:
      env.seed(seed)
    return env


class MultiEpisodeRunner:
  """
  This class runs environments in separate threads and
   rolls out the trajectories for consumption
  """
  def __init__(self, make_env_fn, n_runners: int = 1,
               max_steps: int = DEFAULT_MAX_STEPS,
               base_seed: int = None, daemon=True, autostart=True):

    self.n_envs = n_runners
    self.max_steps = max_steps
    self.make_env_fn = make_env_fn

    # TODO(sanyam): Assumption of Gym environments
    self.observation_space, self.action_space = get_gym_spaces(make_env_fn)

    self.multi_envs = MultiEnvs(make_env_fn, n_envs=n_runners,
                                base_seed=base_seed,
                                daemon=daemon, autostart=autostart)

    # Internal Vars
    self.is_discrete = self.action_space.__class__.__name__ == 'Discrete'
    self._obs = [None] * n_runners
    self._rollout_duration = 0.0

  # TODO(sanyam): device argument should not be here!
  def get_action_list(self, learner, obs_list, device):
    with torch.no_grad():
      batch_obs_tensor = torch.from_numpy(
          np.array(obs_list)
      ).float().to(device)
      action_list = learner.act(batch_obs_tensor)

    return action_list

  def collect(self, learner: BaseAgent, device: torch.device,
              steps: int = None):
    """This routine collects trajectories from each environment
    until a maximum rollout length of `steps`. Not all trajectories
    might be of the same length if one of the environment reaches a
    terminal state. It will be flagged for reset during the next batch
    of rollouts (call to `.collect()`)"""

    steps = steps or self.max_steps

    history_list = [
        self.init_run_history(self.observation_space, self.action_space)
        for _ in range(self.n_envs)
    ]

    rollout_start = time.time()

    # Reset environments with no observation
    batch_reset_ids = self.get_active_envs(invert=True)
    if batch_reset_ids:
      new_obs = self.multi_envs.reset(batch_reset_ids)
      for env_id, obs in zip(batch_reset_ids, new_obs):
        self._obs[env_id] = obs

    while steps:
      batch_act_ids = self.get_active_envs()
      if not batch_act_ids:
        break

      obs_list = self.get_obs_list()
      action_list = self.get_action_list(learner, obs_list, device)

      step_list = self.multi_envs.step(batch_act_ids, action_list)

      for env_id, obs, action, (next_obs, reward, done, _) in zip(
          batch_act_ids, obs_list, action_list, step_list):
        history_list[env_id] = self.append_history(
            obs, action, next_obs, reward, done, history_list[env_id])

        self._obs[env_id] = None if done else next_obs

      steps -= 1

    self._rollout_duration = time.time() - rollout_start

    return history_list

  def close(self):
    self.multi_envs.close()

  def append_history(self, obs, action, next_obs, reward, done, target_history):
    target_history[0] = np.append(target_history[0],
                                  np.expand_dims(obs, axis=0), axis=0)
    if self.is_discrete:
      action = np.expand_dims(action, axis=0)
    target_history[1] = np.append(target_history[1], action, axis=0)
    target_history[2] = np.append(target_history[2],
                                  np.array([[reward]]), axis=0)
    target_history[3] = np.append(target_history[3],
                                  np.expand_dims(next_obs, axis=0), axis=0)
    target_history[4] = np.append(target_history[4],
                                  np.array([[int(done)]]), axis=0)
    return target_history

  def get_active_envs(self, invert=False) -> list:
    """Gets a list of environment IDs whose observations are
    not None. When invert is True, returns all whose observations
    are None"""

    target_ids = []

    for env_id, obs in enumerate(self._obs):
      if invert:
        if obs is None:
          target_ids.append(env_id)
        continue

      if obs is not None:
        target_ids.append(env_id)

    return target_ids

  def get_obs_list(self) -> list:
    """Get a numpy array of all active observations"""
    return list(filter(lambda obs: obs is not None, self._obs))

  # TODO(sanyam): numpy assumption is wrong. abstract out to
  # utils
  @staticmethod
  def init_run_history(observation_space: gym.Space,
                       action_space: gym.Space) -> list:
    is_discrete = action_space.__class__.__name__ == 'Discrete'

    obs_history = np.empty((0, *observation_space.shape), dtype=np.float)
    action_history = np.empty((0, *((1,) if is_discrete else
                                    action_space.shape)),
                              dtype=np.int if is_discrete else np.float)
    reward_history = np.empty((0, 1), dtype=np.float)
    next_obs_history = np.empty_like(obs_history)
    done_history = np.empty((0, 1), dtype=np.int)

    return [
        obs_history, action_history, reward_history,
        next_obs_history, done_history
    ]

  @property
  def last_rollout_duration(self) -> float:
    return self._rollout_duration
