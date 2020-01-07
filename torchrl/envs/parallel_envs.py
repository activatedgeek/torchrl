import functools
from multiprocessing import Pipe, Process

from .env_utils import get_gym_spaces


def target_fn(conn, obj_fn):
  obj = obj_fn()

  while True:
    fn_string, args, kwargs = conn.recv()

    func = getattr(obj, fn_string)
    if callable(func):
      if args is not None:
        func = functools.partial(func, *args)
      if kwargs is not None:
        func = functools.partial(func, **kwargs)
      result = func()
    else:
      result = func

    conn.send(result)

    if fn_string == 'close':
      break


class MultiProcWrapper:
  """
  A generic wrapper which takes a list of functions to be run inside a process.
  Each function must return an object, see `target_fn` for how it is used.
  Communication between each new process and the parent process happens via
  Pipes.
  """
  def __init__(self, obj_fns, daemon=True, autostart=True):
    self.n_procs = len(obj_fns)
    self.daemon = daemon

    self.p_conn, self.child_conn = zip(*[Pipe() for _ in range(self.n_procs)])

    self.proc_list = [
        Process(target=target_fn, args=(conn, obj_fn))
        for conn, obj_fn in zip(self.child_conn, obj_fns)
    ]

    if autostart:
      self.start()

  def start(self):
    for proc in self.proc_list:
      proc.daemon = self.daemon
      proc.start()

  def stop(self):
    self.exec_remote('stop')

    for proc in self.proc_list:
      if proc.is_alive():
        proc.join()

  def exec_remote(self, fn_string, proc_list=None,
                  args_list=None, kwargs_list=None):
    if proc_list is None:
      proc_list = list(range(self.n_procs))
    if args_list is None:
      args_list = [None] * len(proc_list)
    if kwargs_list is None:
      kwargs_list = [None] * len(proc_list)

    assert len(args_list) == len(proc_list) and \
           len(kwargs_list) == len(proc_list), \
        'Argument list mismatch!'

    target_p_conn = []
    for i, p_conn in enumerate(self.p_conn):
      if i in proc_list:
        target_p_conn.append(p_conn)

    for conn, args, kwargs in zip(target_p_conn, args_list, kwargs_list):
      conn.send((fn_string, args, kwargs))

    return [conn.recv() for conn in target_p_conn]


class ParallelEnvs(MultiProcWrapper):
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
    super().__init__(obj_fns, daemon=daemon, autostart=autostart)

  def reset(self, env_ids: list):
    return self.exec_remote('reset', proc_list=env_ids)

  def step(self, env_ids: list, actions: list):
    action_args = [[a] for a in actions]
    return self.exec_remote('step', args_list=action_args, proc_list=env_ids)

  def close(self):
    self.exec_remote('close')

    for proc in self.proc_list:
      if proc.is_alive():
        proc.terminate()

  def render(self, env_ids: list):
    self.exec_remote('render', proc_list=env_ids)
