import abc
import warnings
import argparse
import numpy as np
import gym
import os
import torch
import glob
import ruamel.yaml as yaml
import cloudpickle
from tqdm import tqdm
from tensorboardX import SummaryWriter

from .. import MultiEpisodeRunner
from ..agents import BaseAgent
from ..utils import set_seeds, Nop


class HParams:
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
    self.__dict__.update(items)


class Problem(metaclass=abc.ABCMeta):
  """
  An abstract class which defines functions to define
  any RL problem
  """
  hparams_file = 'hparams.yaml'
  args_file = 'args.yaml'
  checkpoint_prefix = 'checkpoint'

  def __init__(self, hparams: HParams,
               problem_args: argparse.Namespace,
               log_dir: str,
               device: str = 'cuda',
               show_progress: bool = True):
    self.hparams = hparams
    self.args = problem_args
    self.log_dir = log_dir
    self.show_progress = show_progress
    self.device = torch.device(device)

    self.logger = None
    self.agent = None
    self.runner = None
    self.start_epoch = 0

    self.init()

  def init(self):
    # Initialize logging directory if possible, else a no-op
    if self.log_dir:
      if os.path.isdir(self.log_dir) and os.listdir(self.log_dir):
        warnings.warn('Directory "{}" not empty!'.format(self.log_dir))
      os.makedirs(self.log_dir, exist_ok=True)

      hparams_file_path = os.path.join(self.log_dir,
                                       self.hparams_file)
      args_file_path = os.path.join(self.log_dir,
                                    self.args_file)

      with open(hparams_file_path, 'w') as hparams_file, \
           open(args_file_path, 'w') as args_file:
        yaml.dump(self.hparams.__dict__, stream=hparams_file,
                  default_flow_style=False)
        yaml.dump(self.args.__dict__, stream=args_file,
                  default_flow_style=False)

      self.logger = SummaryWriter(log_dir=self.log_dir)
    else:
      self.logger = Nop()

    self.agent = self.init_agent()
    self.set_agent_to_device(self.device)

  @staticmethod
  def load_from_dir(load_dir) -> tuple:
    hparams_file_path = os.path.join(load_dir,
                                     Problem.hparams_file)
    args_file_path = os.path.join(load_dir,
                                  Problem.args_file)

    with open(hparams_file_path, 'r') as hparams_file, \
      open(args_file_path, 'r') as args_file:
      params = HParams(yaml.load(hparams_file))
      args = argparse.Namespace(**yaml.load(args_file))

    return params, args

  def load_checkpoint(self, load_dir, epoch=None):
    """
    This method loads the latest checkpoint from the
    load directory
    """
    if epoch:
      checkpoint_file_path = os.path.join(
          self.log_dir, '{}-{}.cpkl'.format(self.checkpoint_prefix, epoch))
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
    agent_state = self.agent.checkpoint
    if agent_state:
      checkpoint_file_path = os.path.join(
          self.log_dir, '{}-{}.cpkl'.format(self.checkpoint_prefix, epoch))
      with open(checkpoint_file_path, 'wb') as checkpoint_file:
        cloudpickle.dump(agent_state, checkpoint_file)

  @abc.abstractmethod
  def init_agent(self) -> BaseAgent:
    """
    Use this method to initialize the learner using `self.args`
    object
    :return: BaseLearner
    """
    raise NotImplementedError

  @abc.abstractmethod
  def make_env(self) -> gym.Env:
    """
    This method should return a Gym-like environment
    :return: gym.Env
    """
    raise NotImplementedError

  def make_runner(self, n_runners=1, base_seed=None) -> MultiEpisodeRunner:
    return MultiEpisodeRunner(self.make_env,
                              max_steps=self.hparams.max_episode_steps,
                              n_runners=n_runners,
                              base_seed=base_seed)

  def set_agent_train_mode(self, flag: bool = True):
    """
    This routine sets the training flag for the models
    returned by the agent
    """
    for model in self.agent.models:
      model.train(flag)

  def set_agent_to_device(self, device: torch.device):
    """
    This routine sends the agent models to desired device
    """
    for model in self.agent.models:
      model.to(device)

  @abc.abstractmethod
  def train(self, history_list: list) -> dict:
    """
    This method is called after a rollout and must
    contain the logic for updating the agent's weights
    :return: dict of key value pairs of losses
    """
    raise NotImplementedError

  def eval(self, epoch):
    """
    This method is called after a rollout and must
    contain the logic for updating the agent's weights
    :return:
    """
    self.set_agent_train_mode(False)

    eval_runner = self.make_runner(n_runners=1)
    eval_rewards = []
    for _ in range(self.args.num_eval):
      eval_history = eval_runner.collect(self.agent, self.device)
      _, _, reward_history, _, _ = eval_history[0]
      eval_rewards.append(np.sum(reward_history, axis=0))
    eval_runner.close()

    log_avg_reward, log_std_reward = np.average(eval_rewards), \
                                     np.std(eval_rewards)
    self.logger.add_scalar('eval_episode/avg_reward', log_avg_reward,
                           global_step=epoch)
    self.logger.add_scalar('eval_episode/std_reward', log_std_reward,
                           global_step=epoch)

    return log_avg_reward, log_std_reward

  def run(self):
    """
    This is the entrypoint to a problem class and can be overridden
    if the train and eval need to be done at a different point in the
    epoch. All variables for statistics are logging with "log_"
    :return:
    """
    self.runner = self.make_runner(n_runners=self.hparams.num_processes,
                                   base_seed=self.args.seed)

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
      self.set_agent_train_mode(False)
      history_list = self.runner.collect(self.agent,
                                         self.device,
                                         steps=params.rollout_steps)

      self.set_agent_train_mode(True)
      loss_dict = self.train(self.hist_to_tensor(history_list))

      if epoch % self.args.log_interval == 0:
        for loss_label, loss_value in loss_dict.items():
          self.logger.add_scalar('loss/{}'.format(loss_label),
                                 loss_value, global_step=epoch)

      log_rollout_steps = 0

      for i, history in enumerate(history_list):

        log_rollout_steps += len(history[2])
        log_episode_len[i] += len(history[2])
        log_episode_reward[i] += history[2].sum()

        if history[-1][-1] == 1:
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
        log_rollout_duration = self.runner.last_rollout_duration
        self.logger.add_scalar('episode/steps per sec',
                               log_rollout_steps / (log_rollout_duration+1e-6),
                               global_step=epoch)
        self.logger.add_scalar('episode/timesteps', log_n_timesteps,
                               global_step=epoch)

      if epoch % self.args.eval_interval == 0:
        self.eval(epoch)
        self.save_checkpoint(epoch)

    self.eval(self.start_epoch + n_epochs)
    self.save_checkpoint(self.start_epoch + n_epochs)

    self.runner.close()
    self.logger.close()

  @staticmethod
  def hist_to_tensor(history_list):

    def from_numpy(item):
      tensor = torch.from_numpy(item)
      if isinstance(tensor, torch.DoubleTensor):
        tensor = tensor.float()
      return tensor

    return [
        tuple([from_numpy(item) for item in history])
        for history in history_list
    ]

  @staticmethod
  def merge_histories(*history_list):
    return tuple([
        torch.cat(hist, dim=0)
        for hist in zip(*history_list)
    ])
