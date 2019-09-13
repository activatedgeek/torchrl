import abc
import argparse
import os
import glob
import cloudpickle
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

from ..agents import BaseAgent
from ..utils import set_seeds, Nop
from ..runners import BaseRunner


class HParams:
  """
  This class is friendly wrapper over Python Dictionary
  to represent the named hyperparameters.

  Example:

      One can manually set arbitrary strings as hyperparameters as

      .. code-block:: python

            import torchrl.registry as registry
            hparams = registry.HParams()
            hparams.paramA = 'myparam'
            hparams.paramB = 10

      or just send in a dictionary object containing all the relevant key/value
      pairs.

      .. code-block:: python

            import torchrl.registry as registry
            hparams = registry.HParams({'paramA': 'myparam', 'paramB': 10})
            assert hparams.paramA == 'myparam'
            assert hparams.paramB == 10

      Both form equivalent hyperparameter objects.

      To update/override the hyperparamers, use the `update()` method.

      .. code-block:: python

          hparams.update({'paramA': 20, 'paramB': 'otherparam', 'paramC': 5.0})
          assert hparams.paramA == 20
          assert hparams.paramB == 'otherparam'

  Args:
    kwargs (dict): Python dictionary representing named hyperparameters and
    values.

  """
  def __init__(self, kwargs=None):
    self.update(kwargs or {})

  def __getattr__(self, item):
    return self.__dict__[item]

  def __setattr__(self, key, value):
    self.__dict__[key] = value

  def __iter__(self):
    for key, value in self.__dict__.items():
      yield key, value

  def __repr__(self):
    print_str = ''
    for key, value in self:
      print_str += '{}: {}\n'.format(key, value)
    return print_str

  def update(self, items: dict):
    """
    Merge two Hyperparameter objects, overriding any repeated keys from
    the `items` parameter.

    Args:
        items (dict): Python dictionary containing updated values.
    """
    self.__dict__.update(items)


class Problem(metaclass=abc.ABCMeta):
  """
  This abstract class defines a Reinforcement Learning
  problem.

  Args:
      hparams (:class:`~torchrl.registry.problems.HParams`): Object containing
        all named-hyperparameters.
      problem_args (:class:`argparse.Namespace`): Argparse namespace object
        containing Problem arguments like `seed`, `log_interval`,
        `eval_interval`.
      log_dir (str): Path to log directory.
      device (str): String passed to `torch.device()`.
      show_progress (bool): If true, an animated progress is shown based on
        `tqdm`.
      checkpoint_prefix (str): Prefix for the saved checkpoint files.

  Todo:
      * Remove usage of `argparse.Namespace` for `problem_args` and
        use :class:`~torchrl.registry.problems.HParams` instead. As a temporary
        usage fix, convert any dictionary into `argparse.Namespace` using
        `argparse.Namespace(**mydict)`. Tracked by
        `#61 <https://github.com/activatedgeek/torchrl/issues/61>`_.
  """

  def __init__(self, hparams: HParams,
               problem_args: argparse.Namespace,
               log_dir: str,
               device: str = 'cuda',
               show_progress: bool = True,
               checkpoint_prefix='checkpoint'):
    self.hparams = hparams
    self.args = problem_args
    self.log_dir = log_dir
    self.show_progress = show_progress
    self.device = torch.device(device)
    self.checkpoint_prefix = checkpoint_prefix

    self.start_epoch = 0

    self.logger = SummaryWriter(log_dir=self.log_dir) \
      if self.log_dir else Nop()

    self.runner = self.make_runner(n_envs=self.hparams.num_processes,
                                   seed=self.args.seed)

    self.agent = self.init_agent().to(self.device)

  def load_checkpoint(self, load_dir, epoch=None):
    """
    This method loads the latest checkpoint from a directory.
    It also updates the `self.start_epoch` attribute so that any
    further calls to save_checkpoint don't overwrite the previously
    saved checkpoints. The file name format is
    :code:`<CHECKPOINT_PREFIX>-<EPOCH>.ckpt`.

    Args:
        load_dir (str): Path to directory containing checkpoint files.
        epoch (int): Epoch number to load. If :code:`None`, then the file with
          the latest timestamp is loaded from the given directory.
    """
    if epoch:
      checkpoint_file_path = os.path.join(
          self.log_dir, '{}-{}.ckpt'.format(self.checkpoint_prefix, epoch))
    else:
      checkpoint_files = glob.glob(os.path.join(load_dir,
                                                self.checkpoint_prefix + '*'))
      checkpoint_file_path = max(checkpoint_files, key=os.path.getctime)

    # Parse epoch from the checkpoint path
    self.start_epoch = int(os.path.splitext(
        os.path.basename(checkpoint_file_path))[0].split('-')[1])

    with open(checkpoint_file_path, 'rb') as checkpoint_file:
      self.agent.checkpoint = cloudpickle.load(checkpoint_file)

  def save_checkpoint(self, epoch):
    """
    Save checkpoint at a given epoch. The format is
    :code:`<CHECKPOINT_PREFIX>-<EPOCH>.ckpt`

    Args:
        epoch (int): Value of the epoch number.
    """
    agent_state = self.agent.checkpoint
    if self.log_dir and agent_state:
      checkpoint_file_path = os.path.join(
          self.log_dir, '{}-{}.ckpt'.format(self.checkpoint_prefix, epoch))
      with open(checkpoint_file_path, 'wb') as checkpoint_file:
        cloudpickle.dump(agent_state, checkpoint_file)

  @abc.abstractmethod
  def init_agent(self) -> BaseAgent:
    """
    This method is called by the constructor and **must** be overriden
    by any derived class. Using the hyperparameters and problem arguments,
    one should construct an agent here.

    Returns:
        :class:`~torchrl.agents.base_agent.BaseAgent`: Any derived agent class.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def make_runner(self, n_envs=1, seed=None) -> BaseRunner:
    """
    This method is called by the constructor and **must** be overriden
    by any derived class. Using the hyperparameters and problem arguments,
    one should construct an environment runner here.

    Returns:
        :class:`~torchrl.runners.base_runner.BaseRunner`: Any derived runner
          class.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def train(self, trajectory_list: list) -> dict:
    """
    This method **must** be overridden by the derived
    Problem class and should contain the core idea behind the
    training step.

    There are no restrictions to what comes into this argument as long
    as the derived class takes care of following. Typically this should
    involve a list of rollouts (possibly for each parallel trajectory)
    and all relevant values for each rollout - observation, action, reward,
    next observation, termination flag and potentially other information.
    This raw data must be processed as desired. See
    :meth:`~torchrl.problems.gym_problem.GymProblem.hist_to_tensor` for a
    sample routine.

    .. note::

        It is a good idea to always use
        :meth:`~torchrl.agents.base_agent.BaseAgent.train`
        appropriately here.

    Args:
        trajectory_list (list): A list of histories. This will typically be
          returned by the
          :meth:`~torchrl.runners.base_runner.BaseRunner.rollout` method of the
          runner.

    Returns:
        dict: A Python dictionary containing labeled losses.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def eval(self, epoch):
    """
    This method **must** be overridden by the derived
    Problem class and should contain the core idea behind the
    evaluation of the trained model. This is also responsible
    for any metric logging using the `self.logger` object.

    :code:`self.args.num_eval` should be a helpful variable.

    .. note::

        It is a good idea to always use
        :meth:`~torchrl.agents.base_agent.BaseAgent.train` to
        set training :code:`False` here.

    Args:
        epoch (int): Epoch number in question.
    """
    raise NotImplementedError

  def run(self):
    """
    This is the entrypoint to a problem class and can be overridden
    if desired. However, a common rollout, train and eval loop has
    already been provided here. All variables for logging are prefixed
    with "log\\_".

    :code:`self.args.log_interval` and :code:`self.args.eval_interval`
    should be helpful variables.

    .. note::

        This precoded routine implements the following general steps

          * Set agent to train mode using
            :meth:`~torchrl.agents.base_agent.BaseAgent.train`.

          * Rollout trajectories using runner's
            :meth:`~torchrl.runners.base_runner.BaseRunner.rollout`.

          * Unset agent's train mode.

          * Run the training routine using
            :meth:`~torchrl.registry.problems.Problem.train` which could
            potentially be using agent's
            :meth:`~torchrl.agents.base_agent.BaseAgent.learn`.

          * Evaluate the learned agent using
            :meth:`~torchrl.registry.problems.Problem.eval`.

          * Periodically log and save checkpoints using
            :meth:`~torchrl.registry.problems.Problem.save_checkpoint`.

        Since, this routine handles multiple parallel trajectories, care must be
        taken to reset the environment instances (this should be handled by the
        appropriate runner or as desired).
    """
    params = self.hparams
    set_seeds(self.args.seed)

    n_epochs = params.num_total_steps // params.rollout_steps // params.num_processes  # pylint: disable=line-too-long

    log_n_episodes = 0
    log_n_timesteps = 0
    log_episode_len = [0] * params.num_processes
    log_episode_reward = [0] * params.num_processes

    epoch_iterator = range(self.start_epoch + 1,
                           self.start_epoch + n_epochs + 1)
    if self.show_progress:
      epoch_iterator = tqdm(epoch_iterator, unit='epochs')

    for epoch in epoch_iterator:
      self.agent.train(False)
      traj_list = self.runner.rollout(self.agent, steps=params.rollout_steps)

      self.agent.train(True)
      loss_dict = self.train(traj_list)

      if epoch % self.args.log_interval == 0:
        for loss_label, loss_value in loss_dict.items():
          self.logger.add_scalar('loss/{}'.format(loss_label),
                                 loss_value, global_step=epoch)

      log_rollout_steps = 0

      for i, history in enumerate(traj_list):
        log_rollout_steps += len(history.reward)
        log_episode_len[i] += len(history.reward)
        log_episode_reward[i] += history.reward.sum()

        if history.done[-1] == 1:
          self.agent.reset()

          log_n_episodes += 1
          self.logger.add_scalar('episode/length', log_episode_len[i],
                                 global_step=log_n_episodes)
          self.logger.add_scalar('episode/reward', log_episode_reward[i],
                                 global_step=log_n_episodes)
          log_episode_len[i] = 0
          log_episode_reward[i] = 0

      log_n_timesteps += log_rollout_steps

      if epoch % self.args.log_interval == 0:
        self.logger.add_scalar('episode/timesteps', log_n_timesteps,
                               global_step=epoch)

      if epoch % self.args.eval_interval == 0:
        self.eval(epoch)
        self.save_checkpoint(epoch)

    if self.start_epoch + n_epochs % self.args.eval_interval != 0:
      self.eval(self.start_epoch + n_epochs)
      self.save_checkpoint(self.start_epoch + n_epochs)

    self.runner.close()
    self.logger.close()
