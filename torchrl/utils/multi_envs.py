import functools
from .multi_proc_wrapper import MultiProcWrapper
from .gym_utils import get_gym_spaces


class MultiGymEnvs(MultiProcWrapper):
  """
  A utility class which wraps around multiple environments
  and runs them in subprocesses
  """
  def __init__(self, make_env_fn, n_envs: int = 1, base_seed: int = 0,
               daemon: bool = True, autostart: bool = True):
    self.observation_space, self.action_space = get_gym_spaces(make_env_fn)
    obj_fns = [
        functools.partial(make_env_fn,
                          None if base_seed is None else base_seed + rank)
        for rank in range(1, n_envs + 1)
    ]
    super(MultiGymEnvs, self).__init__(obj_fns, daemon=daemon,
                                       autostart=autostart)

  def reset(self, env_ids: list):
    return self.exec_remote('reset', proc_list=env_ids)

  def step(self, env_ids: list, actions: list):
    return self.exec_remote('step', args_list=actions, proc_list=env_ids)

  def close(self):
    self.exec_remote('close')

  def render(self, env_ids: list):
    self.exec_remote('render', proc_list=env_ids)
