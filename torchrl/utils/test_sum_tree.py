# pylint: disable=redefined-outer-name

import pytest
import random
from torchrl.utils.sum_tree import SumTree


@pytest.fixture(scope='function')
def tree():
  yield SumTree(capacity=16)

def test_clear(tree: SumTree):
  for _ in range(tree.capacity + 1):
    value = random.random()
    tree.add(value)

  assert tree.max_value > 0
  assert tree.sum_value > 0.0
  assert tree.next_free_index == 1

  tree.clear()

  assert tree.max_value == 0.0
  assert tree.sum_value == 0.0
  assert tree.next_free_index == 0

def test_sum(tree: SumTree):
  sum_value = 0.0
  for _ in range(tree.capacity):
    value = random.random()
    tree.add(value)

    sum_value += value
    assert tree.sum_value == sum_value

def test_max(tree: SumTree):
  max_value = 0.0
  for _ in range(tree.capacity):
    value = random.random()
    tree.add(value)

    max_value = max(value, max_value)
    assert tree.max_value == max_value

def test_overflow(tree: SumTree):
  for _ in range(tree.capacity):
    tree.add(random.random())

  max_value = 0.0
  for _ in range(tree.capacity):
    value = random.random()
    tree.add(value)

    max_value = max(value, max_value)

  assert tree.max_value == max_value

def test_get(tree: SumTree):
  values = list(range(1, tree.capacity + 1))

  for value in values:
    tree.add(value)

  assert tree.get(10.0) == 3
  assert tree.get(11.0) == 4
  assert tree.get(150) == 15

def test_power_of_two():
  with pytest.raises(AssertionError):
    SumTree(42)
