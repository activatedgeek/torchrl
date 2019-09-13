import torch
from torchrl.problems import GymProblem
from torchrl.storage import ReplayBuffer


class DDPGProblem(GymProblem):
  def __init__(self, hparams, problem_args, *args, **kwargs):
    super(DDPGProblem, self).__init__(hparams, problem_args, *args, **kwargs)

    self.buffer = ReplayBuffer(self.hparams.buffer_size)

  def train(self, history_list: list):
    history_list = self.hist_to_tensor(history_list, device=torch.device('cpu'))

    batch_history = self.merge_histories(*history_list)
    transitions = list(zip(*batch_history))
    self.buffer.extend(transitions)

    if len(self.buffer) >= self.hparams.batch_size:
      transition_batch = self.buffer.sample(self.hparams.batch_size)
      transition_batch = list(zip(*transition_batch))
      transition_batch = [torch.stack(item).to(self.device)
                          for item in transition_batch]
      actor_loss, critic_loss = self.agent.learn(*transition_batch)
      return {'actor_loss': actor_loss, 'critic_loss': critic_loss}
    return {}
