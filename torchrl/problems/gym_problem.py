import torch
import numpy as np

from ..registry import Problem
from ..runners import GymRunner
from ..runners import BaseRunner


class GymProblem(Problem):
  """
  This problem implements a Problem class to handle
  Gym related environments.

  .. note::

      Any :class:`~torchrl.registry.problems.HParams` supplied
      to this problem must have the ``env_id`` property assigned
      to a valid Gym Environment identifier.
  """

  def make_runner(self, n_envs=1, seed=None) -> BaseRunner:
    """
    Create a set of parallel environments.

    Args:
        n_envs (int): Number of parallel environments.
        seed (int): Optional base integer to seed environments with.

    Returns:
        :class:`~torchrl.runners.gym_runner.GymRunner`: An instantiated \
          runner object.
    """
    return GymRunner(self.hparams.env_id, n_envs=n_envs, seed=seed)

  def eval(self, epoch):
    """
    This preset routine simply takes runs the agent and
    runs some evaluations.

    Args:
        epoch (int): Epoch number.

    Returns:
        tuple: Average reward and standard deviation.
    """
    self.agent.train(False)

    eval_runner = self.make_runner(n_envs=1)
    eval_rewards = []
    for _ in range(self.args.num_eval):
      eval_history = eval_runner.rollout(self.agent)[0]
      eval_rewards.append(np.sum(eval_history.reward, axis=0))
    eval_runner.close()

    log_avg_reward, log_std_reward = np.average(eval_rewards), \
                                     np.std(eval_rewards)
    self.logger.add_scalar('eval_episode/avg_reward', log_avg_reward,
                           global_step=epoch)
    self.logger.add_scalar('eval_episode/std_reward', log_std_reward,
                           global_step=epoch)

    return log_avg_reward, log_std_reward

  @staticmethod
  def hist_to_tensor(history_list, device: torch.device = 'cuda'):
    """
    A utility method to convert list of histories to
    PyTorch Tensors. Additionally, also sends the tensors
    to target device.

    Args:
        history_list (list): List of histories for each parallel trajectory.
        device (:class:`torch.device`): PyTorch device object.

    Returns:
        list: A list of tuples where each tuple represents the history item.
    """
    def from_numpy(item):
      tensor = torch.from_numpy(item)
      if isinstance(tensor, torch.DoubleTensor):
        tensor = tensor.float()
      return tensor.to(device)

    return [
        tuple([from_numpy(item) for item in history])
        for history in history_list
    ]

  @staticmethod
  def merge_histories(*history_list):
    """
    A utility function which merges histories from
    all the parallel environments.

    Args:
        *history_list (list):

    Returns:
        tuple: A single tuple which effectively transposes the history of \
          transition tuples.
    """
    return tuple([
        torch.cat(hist, dim=0)
        for hist in zip(*history_list)
    ])
