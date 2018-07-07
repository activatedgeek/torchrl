import torch
from ..registry import Problem
from ..storage import PrioritizedReplayBuffer


class PrioritizedDQNProblem(Problem):
  def __init__(self, hparams, problem_args, *args, **kwargs):
    super(PrioritizedDQNProblem, self).__init__(
        hparams, problem_args, *args, **kwargs)

    self.buffer = PrioritizedReplayBuffer(self.hparams.buffer_size)

  def train(self, history_list: list):
    # Populate the buffer
    batch_history = self.merge_histories(*history_list)
    transitions = list(zip(*batch_history))
    self.buffer.extend(transitions)

    if len(self.buffer) >= self.hparams.batch_size:
      indices, transition_batch = self.buffer.sample(self.hparams.batch_size)
      transition_batch = list(zip(*transition_batch))
      transition_batch = [torch.stack(item).to(self.device)
                          for item in transition_batch]
      current_q_values, expected_q_values = \
        self.agent.compute_q_values(*transition_batch)
      td_error = (current_q_values - expected_q_values).abs().detach().cpu()
      value_loss = self.agent.learn(*transition_batch,
                                    current_q_values, expected_q_values)
      self.buffer.update_probs(indices, td_error.numpy())
      return {'value_loss': value_loss}
    return {}
