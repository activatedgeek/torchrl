from collections import deque
import numpy as np
from .replay_buffer import ReplayBuffer
from ..utils import LinearSchedule

DEFAULT_BUFFER_SIZE = int(1e6)

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

    sample_weights = np.power(probs[indices], - self.beta.value)
    sample_weights /= np.max(sample_weights)

    batch = [self.buffer[i] for i in indices]
    return indices, sample_weights, batch

  def update_probs(self, indices: np.array, td_error: np.array):
    new_prob = np.power(td_error + self.epsilon, self.alpha)
    updated_probs = np.array(self.probs)
    updated_probs[indices] = np.squeeze(new_prob, axis=-1)

    self.probs = deque(updated_probs, maxlen=self.size)

  def compute_max_prob(self):
    max_prob = max(self.probs) if self.probs else 1.0
    return max_prob
