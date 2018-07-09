import random
from collections import deque

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
