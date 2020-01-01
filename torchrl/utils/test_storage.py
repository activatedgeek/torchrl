import pytest
import random
import torch
from torchrl.utils.storage import TensorTupleDataset


@pytest.mark.parametrize('n', [10, 20, 30, 40, 50])
def test_unbounded(n):
  ds = TensorTupleDataset(size=-1)

  assert len(ds) == 0

  x = [
      torch.randn(n, 2, 19),
      torch.randn(n, 4),
      torch.randn(n, 8),
      torch.randn(n, 16),
      torch.randn(n, 32)
  ]

  ds.extend(*x)

  assert len(ds) == n

  for v, ref in zip(ds[random.randint(0, n - 1)], x):
    assert v.dim() == len(ref.shape[1:])
    assert v.size(-1) == ref.size(-1)


@pytest.mark.parametrize('n,size', [
    (10, 9),
    (20, 100),
    (30, 10),
    (40, 40),
    (50, 12)
])
def test_bounded(n, size):
  ds = TensorTupleDataset(size=size)

  assert ds.size == size
  assert len(ds) == 0

  x = [
      torch.randn(n, 2),
      torch.randn(n, 4),
      torch.randn(n, 8, 12),
      torch.randn(n, 16),
      torch.randn(n, 32)
  ]

  ds.extend(*x)

  assert len(ds) == min(n, size)

  for v, ref in zip(ds[random.randint(0, len(ds) - 1)], x):
    assert v.dim() == len(ref.shape[1:])
    assert v.size(-1) == ref.size(-1)
