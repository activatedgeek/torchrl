import random
from collections import deque
import numpy as np


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
               epsilon=0.01, alpha=1.0):
    super(PrioritizedReplayBuffer, self).__init__(size=size)

    self.size = size
    self.epsilon = epsilon
    self.alpha = alpha
    self.probs = deque(maxlen=size)

  def push(self, item):
    super(PrioritizedReplayBuffer, self).push(item)
    self.probs.append(1.0)

  def extend(self, *items):
    super(PrioritizedReplayBuffer, self).extend(*items)
    self.probs.extend([1.0] * len(items))

  def clear(self):
    super(PrioritizedReplayBuffer, self).clear()
    self.probs.clear()

  def sample(self, batch_size):
    self._assert_batch_size(batch_size)

    probs = np.array(self.probs)
    probs /= np.sum(probs)

    indices = np.random.choice(range(self.__len__()), size=batch_size,
                               replace=False, p=probs)
    batch = list(map(lambda i: self.buffer[i], indices))
    return indices, batch

  def update_probs(self, indices: np.array, td_error: np.array):
    new_prob = np.power(td_error + self.epsilon, self.alpha)
    updated_probs = np.array(self.probs)
    updated_probs[indices] = np.squeeze(new_prob, axis=-1)

    self.probs = deque(updated_probs, maxlen=self.size)
