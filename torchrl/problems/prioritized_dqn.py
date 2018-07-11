import torch
from ..registry import Problem
from ..storage import PrioritizedReplayBuffer


class PrioritizedDQNProblem(Problem):
  def __init__(self, hparams, problem_args, *args, **kwargs):
    super(PrioritizedDQNProblem, self).__init__(
        hparams, problem_args, *args, **kwargs)

    self.buffer = PrioritizedReplayBuffer(self.hparams.buffer_size,
                                          alpha=hparams.alpha,
                                          beta=hparams.beta,
                                          num_steps=hparams.beta_anneal_steps)

  def train(self, history_list: list):
    # Populate the buffer
    batch_history = self.merge_histories(*history_list)
    transitions = list(zip(*batch_history))
    self.buffer.extend(transitions)

    if len(self.buffer) >= self.hparams.batch_size:
      indices, sample_weights, transition_batch = \
        self.buffer.sample(self.hparams.batch_size)
      transition_batch = list(zip(*transition_batch))
      transition_batch = [torch.stack(item).to(self.device)
                          for item in transition_batch]

      current_q_values, expected_q_values = \
        self.agent.compute_q_values(*transition_batch)

      sample_weights = torch.from_numpy(sample_weights).\
        unsqueeze(dim=-1).to(self.device)
      td_error = expected_q_values - current_q_values
      td_error = td_error * sample_weights

      value_loss = self.agent.learn(*transition_batch, td_error)

      self.buffer.update_probs(indices, td_error.abs().detach().cpu().numpy())

      return {'value_loss': value_loss}
    return {}
