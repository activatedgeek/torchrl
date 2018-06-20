import abc
import argparse
import numpy as np
import gym
import os
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter

from .hparams import HParams
from .. import MultiEpisodeRunner, EpisodeRunner, CPUReplayBuffer
from ..learners import BaseLearner
from ..utils import set_seeds, minibatch_generator


class Problem(metaclass=abc.ABCMeta):
  """
  An abstract class which defines functions to define
  any RL problem
  """

  def __init__(self, params: HParams, args: argparse.Namespace):
    """
    The constructor takes in all the parameters needed by
    the problem.
    :param params:
    """
    self.params = params
    self.args = args
    self.agent = self.init_agent()
    self.runner = self.make_runner(n_runners=params.num_processes,
                                   base_seed=args.seed)
    self.logger = SummaryWriter(log_dir=args.log_dir)
    if args.load_dir:
      self.agent.load(args.load_dir)

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
    self.agent.eval()

    runner = EpisodeRunner(self.make_env(),
                           max_steps=self.params.max_episode_steps)
    rewards = []
    for _ in range(self.args.num_eval):
      runner.reset()
      _, _, reward_history, _, _ = runner.collect(self.agent, store=True)
      rewards.append(np.sum(reward_history, axis=0))
    runner.stop()

    log_avg_reward, log_std_reward = np.average(rewards), np.std(rewards)
    self.logger.add_scalar('avg eval reward', log_avg_reward,
                           global_step=epoch)
    self.logger.add_scalar('std eval reward', log_std_reward,
                           global_step=epoch)

    return log_avg_reward, log_std_reward

  def save(self, epoch):
    if not self.args.save_dir:
      return

    save_dir = os.path.join(self.args.save_dir, 'epoch-{}'.format(epoch))
    os.makedirs(save_dir, exist_ok=True)
    self.agent.save(save_dir)

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
      epoch_iterator = tqdm(epoch_iterator)

    for epoch in epoch_iterator:
      self.agent.eval()
      history_list = self.runner.collect(self.agent,
                                         steps=params.rollout_steps,
                                         store=True)

      self.agent.train()
      loss_dict = self.train(Problem.hist_to_tensor(history_list))

      if epoch % self.args.log_interval == 0:
        for loss_label, loss_value in loss_dict.items():
          self.logger.add_scalar(loss_label, loss_value, global_step=epoch)

      log_rollout_steps = 0

      for i, history in enumerate(history_list):

        log_rollout_steps += len(history[2])
        log_episode_len[i] += len(history[2])
        log_episode_reward[i] += history[2].sum()

        if history[-1][-1] == 1:
          self.runner.reset(i)
          self.agent.reset()

          log_n_episodes += 1
          self.logger.add_scalar('episode length', log_episode_len[i],
                                 global_step=log_n_episodes)
          self.logger.add_scalar('episode reward', log_episode_reward[i],
                                 global_step=log_n_episodes)
          log_episode_len[i] = 0
          log_episode_reward[i] = 0

      log_n_timesteps += log_rollout_steps

      if epoch % self.args.log_interval == 0:
        log_rollout_duration = np.average(list(map(lambda x: x['duration'],
                                                   self.runner.get_stats())))
        self.logger.add_scalar('steps per sec',
                               log_rollout_steps / (log_rollout_duration+1e-6),
                               global_step=epoch)
        self.logger.add_scalar('total timesteps', log_n_timesteps,
                               global_step=epoch)

      if epoch % self.args.eval_interval == 0:
        self.eval(epoch)
        self.save(epoch)

    self.eval(n_epochs)
    self.save(n_epochs)

    self.runner.stop()
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

  def make_env(self):
    return gym.make(self.params.env)

  def train(self, history_list: list):
    # Populate the buffer
    batch_history = Problem.merge_histories(*history_list)
    transitions = list(zip(*batch_history))
    self.buffer.extend(transitions)

    if len(self.buffer) >= self.params.batch_size:
      transition_batch = self.buffer.sample(self.params.batch_size)
      transition_batch = list(zip(*transition_batch))
      transition_batch = [torch.stack(item) for item in transition_batch]
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
  def make_env(self):
    return gym.make(self.params.env)

  def train(self, history_list: list):
    # Merge histories across multiple trajectories
    batch_history = Problem.merge_histories(*history_list)
    returns = torch.cat([self.agent.compute_returns(*history)
                         for history in history_list], dim=0)

    actor_loss, critic_loss, entropy_loss = self.agent.learn(*batch_history,
                                                             returns)
    return {'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'entropy_loss': entropy_loss}


class PPOProblem(Problem):
  def make_env(self):
    return gym.make(self.params.env)

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
        actor_loss, critic_loss, entropy_loss = self.agent.learn(*data)

    return {'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'entropy_loss': entropy_loss}

