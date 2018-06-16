import abc
import numpy as np
from tensorboardX import SummaryWriter

from .hparams import HParams
from .. import MultiEpisodeRunner, EpisodeRunner
from ..learners import BaseLearner
from ..utils import set_seeds


class Problem(metaclass=abc.ABCMeta):
  """
  An abstract class which defines functions to define
  any RL problem
  """

  def __init__(self, args: HParams):
    """
    The constructor takes in all the parameters needed by
    the problem.
    :param args:
    """
    self.args = args
    self.agent = self.init_agent()
    self.runner = self.make_runner(n_runners=args.num_processes, base_seed=args.seed)
    self.logger = SummaryWriter(log_dir=args.log_dir)

  @abc.abstractmethod
  def init_agent(self) -> BaseLearner:
    """
    Use this method to initialize the learner using `self.args`
    object
    :return: BaseLearner
    """
    raise NotImplementedError

  @abc.abstractmethod
  def make_env(self):
    """
    This method should return a Gym-like environment
    :return: gym.Env
    """
    raise NotImplementedError

  # TODO: this should use `self.make_env`
  def make_runner(self, n_runners=1, base_seed=None) -> MultiEpisodeRunner:
    return MultiEpisodeRunner(self.args.env,
                              max_steps=self.args.max_episode_steps,
                              n_runners=n_runners,
                              base_seed=base_seed)

  @abc.abstractmethod
  def train(self, history_list: list):
    """
    This method is called after a rollout and must
    contain the logic for updating the agent's weights
    :return:
    """
    raise NotImplementedError

  def eval(self):
    """
    This method is called after a rollout and must
    contain the logic for updating the agent's weights
    :return:
    """
    runner = EpisodeRunner(self.make_env(), max_steps=self.args.max_episode_steps)
    rewards = []
    for _ in range(self.args.num_eval):
      runner.reset()
      _, _, reward_history, _, _ = runner.collect(self.agent, store=True)
      rewards.append(np.sum(reward_history, axis=0))
    runner.stop()

    return np.average(rewards)

  def run(self):
    """
    This is the entrypoint to a problem class and can be overridden
    if the train and eval need to be done at a different point in the
    epoch
    :return:
    """
    args = self.args
    set_seeds(args.seed)

    n_epochs = args.num_total_steps // args.rollout_steps // args.num_processes

    for epoch in range(1, n_epochs + 1):
      self.agent.train()
      history_list = self.runner.collect(self.agent,
                                         steps=args.rollout_steps,
                                         store=True)

      self.train(history_list)

      for i, history in enumerate(history_list):
        if history[-1] == 1:
          self.runner.reset(i)
          self.agent.reset()

      if epoch % args.eval_interval == 0:
        # TODO: Move this to logger
        self.agent.eval()
        print('Avg. Reward at Epoch {}: {}'.format(epoch, self.eval()))

      if args.save_dir and epoch % args.save_interval == 0:
        self.agent.save(args.save_dir)

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.runner.stop()
    self.logger.close()
