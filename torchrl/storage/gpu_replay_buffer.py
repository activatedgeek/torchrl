import torch


class GPUReplayBuffer:
  """
  This class implements a GPU-ready replay buffer
  """
  def __init__(self, state_shape, action_shape, size=int(1e6)):
    self.state_shape = state_shape
    self.action_shape = action_shape
    self.buffer_size = size

    self.state_buffer = torch.zeros(0, *state_shape)
    self.action_buffer = torch.zeros(0, *action_shape).long()
    self.reward_buffer = torch.zeros(0, 1)
    self.next_state_buffer = torch.zeros(0, *state_shape)
    self.done_buffer = torch.zeros(0, 1).long()

    self._size = 0

  def cuda(self):
    self.state_buffer.cuda()
    self.action_buffer.cuda()
    self.reward_buffer.cuda()
    self.next_state_buffer.cuda()
    self.done_buffer.cuda()

  def push(self, state, action, reward, next_state, done):
    incoming_size = state.shape[0]

    assert incoming_size == action.shape[0] and \
           incoming_size == reward.shape[0] and \
           incoming_size == next_state.shape[0] and \
           incoming_size == done.shape[0], \
      'Input tensors shape mismatch, all arguments must have same dimension 0'

    self._size += incoming_size

    if self._size > self.buffer_size:
      overflow = self._size - self.buffer_size

      self.state_buffer = self.state_buffer[overflow:]
      self.action_buffer = self.action_buffer[overflow:]
      self.reward_buffer = self.reward_buffer[overflow:]
      self.next_state_buffer = self.next_state_buffer[overflow:]
      self.done_buffer = self.done_buffer[overflow:]

      self._size = self.buffer_size

    self.state_buffer = torch.cat([self.state_buffer, state], dim=0)
    self.action_buffer = torch.cat([self.action_buffer, action], dim=0)
    self.reward_buffer = torch.cat([self.reward_buffer, reward], dim=0)
    self.next_state_buffer = torch.cat([self.next_state_buffer, next_state],
                                       dim=0)
    self.done_buffer = torch.cat([self.done_buffer, done], dim=0)

  def sample(self, batch_size):
    total_size = self.__len__()
    assert batch_size <= total_size, \
      'Unable to sample {} items, current buffer size {}'.format(
          batch_size, total_size)

    batch_index = (torch.rand(batch_size) * total_size).long()

    state_batch = self.state_buffer.index_select(0, batch_index)
    action_batch = self.action_buffer.index_select(0, batch_index)
    reward_batch = self.reward_buffer.index_select(0, batch_index)
    next_state_batch = self.next_state_buffer.index_select(0, batch_index)
    done_batch = self.done_buffer.index_select(0, batch_index)

    return state_batch, action_batch, reward_batch, next_state_batch, done_batch

  def __len__(self):
    return self._size
