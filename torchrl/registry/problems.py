import abc
import argparse
import numpy as np
import gym
import os
import torch
import glob
import yaml
import cloudpickle
from copy import deepcopy
from tqdm import tqdm
from tensorboardX import SummaryWriter

from .hparams import HParams
from .. import MultiEpisodeRunner, CPUReplayBuffer
from ..learners import BaseLearner
from ..utils import set_seeds, minibatch_generator


class Problem(metaclass=abc.ABCMeta):
  """
  An abstract class which defines functions to define
  any RL problem
  """
  hparams_file = 'hparams.yaml'
  args_file = 'args.yaml'
  checkpoint_prefix = 'checkpoint'

  def __init__(self, params: HParams, args: argparse.Namespace):
    """
    The constructor takes in all the parameters needed by
    the problem.
    :param params:
    """
    self.params = params
    self.args = args

    self.logger = None
    self.agent = None
    self.runner = None

    self.init()

  def init(self):
    # Initialize logging directory
    if os.path.isdir(self.args.log_dir) and os.listdir(self.args.log_dir):
      raise ValueError('Directory "{}" not empty!'.format(self.args.log_dir))
    os.makedirs(self.args.log_dir, exist_ok=True)

    hparams_file_path = os.path.join(self.args.log_dir,
                                     self.hparams_file)
    args_file_path = os.path.join(self.args.log_dir,
                                  self.args_file)

    with open(hparams_file_path, 'w') as hparams_file, \
         open(args_file_path, 'w') as args_file:

      # Remove all directory references before dumping
      args_dict = deepcopy(self.args.__dict__)
      dir_keys = list(filter(lambda key: 'dir' in key, args_dict.keys()))
      for key in dir_keys:
        args_dict.pop(key)

      yaml.dump(self.params.__dict__, stream=hparams_file,
                default_flow_style=False)
      yaml.dump(args_dict, stream=args_file,
                default_flow_style=False)

    self.logger = SummaryWriter(log_dir=self.args.log_dir)

    self.agent = self.init_agent()
    self.set_agent_to_device(torch.device(self.args.device))

    self.runner = self.make_runner(n_runners=self.params.num_processes,
                                   base_seed=self.args.seed)

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

  def load_latest_checkpoint(self, load_dir):
    """
    This method loads the latest checkpoint from the
    load directory
    """
    checkpoint_files = glob.glob(os.path.join(load_dir,
                                              self.checkpoint_prefix + '*'))
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    with open(latest_checkpoint, 'rb') as checkpoint_file:
      self.agent.state = cloudpickle.load(checkpoint_file)

  @abc.abstractmethod
  def init_agent(self) -> BaseLearner:
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
                              max_steps=self.params.max_episode_steps,
                              n_runners=n_runners,
                              base_seed=base_seed)

  def get_gym_spaces(self):
    """
    A utility function to get observation and actions spaces of a
    Gym environment
    """
    env = self.make_env()
    observation_space = env.observation_space
    action_space = env.action_space
    env.close()
    return observation_space, action_space

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
      eval_history = eval_runner.collect(self.agent, self.args.device)
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

  def save(self, epoch):
    checkpoint_file_path = os.path.join(
        self.args.log_dir, '{}-{}.cpkl'.format(self.checkpoint_prefix, epoch))
    with open(checkpoint_file_path, 'wb') as checkpoint_file:
      cloudpickle.dump(self.agent.state, checkpoint_file)

  def run(self):
    """
    This is the entrypoint to a problem class and can be overridden
    if the train and eval need to be done at a different point in the
    epoch. All variables for statistics are logging with "log_"
    :return:
    """
    params = self.params
    set_seeds(self.args.seed)

    n_epochs = params.num_total_steps // params.rollout_steps // params.num_processes  # pylint: disable=line-too-long

    log_n_episodes = 0
    log_n_timesteps = 0
    log_episode_len = [0] * params.num_processes
    log_episode_reward = [0] * params.num_processes

    epoch_iterator = range(1, n_epochs + 1)
    if self.args.progress:
      epoch_iterator = tqdm(epoch_iterator, unit='epochs')

    for epoch in epoch_iterator:
      self.set_agent_train_mode(False)
      history_list = self.runner.collect(self.agent,
                                         self.args.device,
                                         steps=params.rollout_steps)

      self.set_agent_train_mode(True)
      loss_dict = self.train(Problem.hist_to_tensor(history_list))

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
        self.save(epoch)

    self.eval(n_epochs)
    self.save(n_epochs)

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


class DQNProblem(Problem):
  def __init__(self, params, args):
    super(DQNProblem, self).__init__(params, args)

    self.buffer = CPUReplayBuffer(params.buffer_size)

  def train(self, history_list: list):
    # Populate the buffer
    batch_history = Problem.merge_histories(*history_list)
    transitions = list(zip(*batch_history))
    self.buffer.extend(transitions)

    if len(self.buffer) >= self.params.batch_size:
      transition_batch = self.buffer.sample(self.params.batch_size)
      transition_batch = list(zip(*transition_batch))
      transition_batch = [torch.stack(item).to(self.args.device)
                          for item in transition_batch]
      value_loss = self.agent.learn(*transition_batch)
      return {'value_loss': value_loss}
    return {}


class DDPGProblem(DQNProblem):
  def train(self, history_list: list):
    # only overriding to make the return decomposition clear
    loss_dict = super(DDPGProblem, self).train(history_list)
    if 'value_loss' in loss_dict:
      actor_loss, critic_loss = loss_dict['value_loss']
      return {'actor_loss': actor_loss, 'critic_loss': critic_loss}
    return {}


class A2CProblem(Problem):
  def train(self, history_list: list):
    # Merge histories across multiple trajectories
    batch_history = Problem.merge_histories(*history_list)
    batch_history = [item.to(self.args.device) for item in batch_history]
    returns = torch.cat([
        self.agent.compute_returns(*history)
        for history in history_list
    ], dim=0).to(self.args.device)

    actor_loss, critic_loss, entropy_loss = self.agent.learn(*batch_history,
                                                             returns)
    return {'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'entropy_loss': entropy_loss}


class PPOProblem(Problem):
  def train(self, history_list: list):
    # Merge histories across multiple trajectories
    batch_history = Problem.merge_histories(*history_list)
    data = [self.agent.compute_returns(*history) for history in history_list]
    returns, log_probs, values = Problem.merge_histories(*data)
    advantages = returns - values

    # Train the agent
    actor_loss, critic_loss, entropy_loss = None, None, None
    for _ in range(self.params.ppo_epochs):
      for data in minibatch_generator(*batch_history,
                                      returns, log_probs, advantages,
                                      minibatch_size=self.params.batch_size):
        data = [item.to(self.args.device) for item in data]
        actor_loss, critic_loss, entropy_loss = self.agent.learn(*data)

    return {'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'entropy_loss': entropy_loss}

