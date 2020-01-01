import torch
from torch.utils.data import Dataset
from typing import List
from collections import namedtuple


def truncated_cat(a, b, maxsize=-1):
  # NOTE(sanyam): this may overflow a little over size
  # but keeps logic simple.
  res = torch.cat([a, b], dim=0)

  diff = res.size(0) - maxsize
  if maxsize > 0 and diff > 0:
    res = res[diff:]

  return res


class TensorTupleDataset(Dataset):
  '''Store vectorized tuples of tensors
  '''
  def __init__(self, size: int = -1):
    self.size = size

    self._raw_x = None

  def extend(self, *tensor_list: List[torch.Tensor]):
    if self._raw_x is None:
      self._raw_x = [None] * len(tensor_list)

    assert len(self._raw_x) == len(tensor_list)

    for i, new_x in enumerate(tensor_list):
      if self._raw_x[i] is None:
        self._raw_x[i] = torch.zeros(0, *new_x.shape[1:])

      self._raw_x[i] = truncated_cat(self._raw_x[i], new_x, maxsize=self.size)

  def __len__(self) -> int:
    if self._raw_x is None:
      return 0
    return self._raw_x[0].size(0)

  def __getitem__(self, index) -> List[torch.Tensor]:
    return [x[index] for x in self._raw_x]


Transition = namedtuple('Transition', [
    'obs',
    'action',
    'reward',
    'next_obs',
    'done',
])


class TransitionTupleDataset(TensorTupleDataset):
  def extend(self, transition_list: List[Transition]):
    if len(transition_list) == 0:
      return

    transition_batch = Transition(*[
        torch.Tensor(b) for b in zip(*transition_list)
    ])
    transition_batch = [
        b.unsqueeze(-1) if b.dim() == 1 else b
        for b in transition_batch
    ]
    super().extend(*transition_batch)
