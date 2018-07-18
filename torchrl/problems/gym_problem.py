import torch
import numpy as np

from ..registry import Problem
from ..runners import GymRunner
from ..runners import BaseRunner


class GymProblem(Problem):
  """Problems related to Gym Environments.

  This is a base class for problems related to
  Gym environments.

  TODO(sanyamkapoor): `self.env_id` is implicit
  and not really an ideal spec.
  """
  def make_runner(self, n_envs=1, seed=None) -> BaseRunner:
    return GymRunner(self.env_id, n_envs=n_envs, seed=seed)

  def eval(self, epoch):
    """
    This method is called after a rollout and must
    contain the logic for updating the agent's weights
    :return:
    """
    self.set_agent_train_mode(False)

    eval_runner = self.make_runner(n_envs=1)
    eval_rewards = []
    for _ in range(self.args.num_eval):
      eval_history = eval_runner.rollout(self.agent)
      _, _, reward_history, _, _ = eval_history[0]  # pylint: disable=unpacking-non-sequence
      eval_rewards.append(np.sum(reward_history, axis=0))
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
    return tuple([
        torch.cat(hist, dim=0)
        for hist in zip(*history_list)
    ])
