import torch

from .gym_problem import GymProblem


class A2CProblem(GymProblem):
  def train(self, history_list: list):
    history_list = self.hist_to_tensor(history_list, device=self.device)

    batch_history = self.merge_histories(*history_list)
    returns = torch.cat([
        self.agent.compute_returns(*history)
        for history in history_list
    ], dim=0)

    actor_loss, critic_loss, entropy_loss = self.agent.learn(*batch_history,
                                                             returns)
    return {'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'entropy_loss': entropy_loss}
