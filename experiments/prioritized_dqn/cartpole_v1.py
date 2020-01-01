from collections import deque
import numpy as np
import torch
import random
from torchrl.utils.multi_envs import get_gym_spaces
from torchrl.contrib.agents import BaseDQNAgent
from torchrl.problems import GymProblem
from torchrl.utils import LinearSchedule

from ..dqn.cartpole_v1 import DQNCartpole


DEFAULT_BUFFER_SIZE = int(1e6)

class ReplayBuffer:
  def __init__(self, size=DEFAULT_BUFFER_SIZE):
    self.buffer = deque(maxlen=size)

  def push(self, item):
    self.buffer.append(item)

  def extend(self, *items):
    self.buffer.extend(*items)

  def clear(self):
    self.buffer.clear()

  def sample(self, batch_size):
    self._assert_batch_size(batch_size)
    return random.sample(self.buffer, batch_size)

  def _assert_batch_size(self, batch_size):
    assert batch_size <= self.__len__(), \
      'Unable to sample {} items, current buffer size {}'.format(
          batch_size, self.__len__())

  def __len__(self):
    return len(self.buffer)


class PrioritizedReplayBuffer(ReplayBuffer):
  def __init__(self, size=DEFAULT_BUFFER_SIZE,
               epsilon=0.01, alpha=0.6, beta=0.4,
               num_steps=int(1e6)):
    super(PrioritizedReplayBuffer, self).__init__(size=size)

    self.size = size
    self.epsilon = epsilon
    self.alpha = alpha
    self.beta = LinearSchedule(min_val=beta, max_val=1.0, num_steps=num_steps)
    self.probs = deque(maxlen=size)

  def push(self, item):
    super(PrioritizedReplayBuffer, self).push(item)
    self.probs.append(self.compute_max_prob())

  def extend(self, *items):
    super(PrioritizedReplayBuffer, self).extend(*items)
    max_prob = self.compute_max_prob()
    self.probs.extend([max_prob] * len(items))

  def clear(self):
    super(PrioritizedReplayBuffer, self).clear()
    self.probs.clear()

  def sample(self, batch_size):
    self._assert_batch_size(batch_size)

    probs = np.array(self.probs, dtype=np.float32)
    probs /= np.sum(probs)

    indices = np.random.choice(range(self.__len__()), size=batch_size,
                               replace=False, p=probs)

    sample_weights = np.power(probs[indices], - self.beta.value)  # pylint: disable=assignment-from-no-return
    sample_weights /= np.max(sample_weights)

    batch = [self.buffer[i] for i in indices]
    return indices, sample_weights, batch

  def update_probs(self, indices: np.array, td_error: np.array):
    new_prob = np.power(td_error + self.epsilon, self.alpha)  # pylint: disable=assignment-from-no-return
    updated_probs = np.array(self.probs)
    updated_probs[indices] = np.squeeze(new_prob, axis=-1)

    self.probs = deque(updated_probs, maxlen=self.size)

  def compute_max_prob(self):
    max_prob = max(self.probs) if self.probs else 1.0
    return max_prob


class PrioritizedDQNProblem(GymProblem):
  def __init__(self, hparams, problem_args, *args, **kwargs):
    super(PrioritizedDQNProblem, self).__init__(
        hparams, problem_args, *args, **kwargs)

    self.buffer = PrioritizedReplayBuffer(self.hparams.buffer_size,
                                          alpha=hparams.alpha,
                                          beta=hparams.beta,
                                          num_steps=hparams.beta_anneal_steps)

  def train(self, history_list: list):
    history_list = self.hist_to_tensor(history_list, device=torch.device('cpu'))

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


class PERCartpole(PrioritizedDQNProblem):
  def init_agent(self):
    observation_space, action_space = get_gym_spaces(self.runner.make_env)

    agent = BaseDQNAgent(
        observation_space,
        action_space,
        double_dqn=self.hparams.double_dqn,
        lr=self.hparams.actor_lr,
        gamma=self.hparams.gamma,
        num_eps_steps=self.hparams.num_eps_steps,
        target_update_interval=self.hparams.target_update_interval)

    return agent

  @staticmethod
  def hparams_per_cartpole():
    params = DQNCartpole.hparams_dqn_cartpole()

    params.alpha = 0.6
    params.beta = 0.4
    params.beta_anneal_steps = 1000

    return params
