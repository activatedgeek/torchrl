from .gym_problem import GymProblem
from ..utils import minibatch_generator


class PPOProblem(GymProblem):
  def train(self, history_list: list):
    history_list = self.hist_to_tensor(history_list, device=self.device)

    batch_history = self.merge_histories(*history_list)
    data = [self.agent.compute_returns(*history) for history in history_list]
    returns, log_probs, values = self.merge_histories(*data)
    advantages = returns - values

    # Train the agent
    actor_loss, critic_loss, entropy_loss = None, None, None
    for _ in range(self.hparams.ppo_epochs):
      for data in minibatch_generator(*batch_history,
                                      returns, log_probs, advantages,
                                      minibatch_size=self.hparams.batch_size):
        actor_loss, critic_loss, entropy_loss = self.agent.learn(*data)

    return {'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'entropy_loss': entropy_loss}
