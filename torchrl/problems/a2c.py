import torch

from ..registry import Problem


class A2CProblem(Problem):
  def train(self, history_list: list):
    # Merge histories across multiple trajectories
    batch_history = self.merge_histories(*history_list)
    batch_history = [item.to(self.device) for item in batch_history]
    returns = torch.cat([
        self.agent.compute_returns(*history)
        for history in history_list
    ], dim=0).to(self.device)

    actor_loss, critic_loss, entropy_loss = self.agent.learn(*batch_history,
                                                             returns)
    return {'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'entropy_loss': entropy_loss}
