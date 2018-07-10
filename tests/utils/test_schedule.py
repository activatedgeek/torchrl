# pylint: disable=redefined-outer-name

import pytest
from torchrl.utils.schedule import LinearSchedule


@pytest.fixture(scope='function')
def schedule():
  yield LinearSchedule(min_val=0.25, max_val=0.85,
                       num_steps=10)


@pytest.fixture(scope='function')
def inverted_schedule():
  yield LinearSchedule(min_val=0.25, max_val=0.85,
                       num_steps=10, invert=True)


def test_five_steps(schedule: LinearSchedule):
  val = None
  for _ in range(6):
    val = schedule.value

  assert val == 0.55


def test_overflow_steps(schedule: LinearSchedule):
  val = 0.0
  for _ in range(100):
    val = schedule.value

  assert val == schedule.max_val

def test_underflow_steps(inverted_schedule: LinearSchedule):
  val = 0.0
  for _ in range(100):
    val = inverted_schedule.value

  assert val == inverted_schedule.min_val
