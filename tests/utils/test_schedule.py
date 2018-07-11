# pylint: disable=redefined-outer-name

import pytest
from torchrl.utils.schedule import LinearSchedule, ExpDecaySchedule


@pytest.fixture(scope='function')
def schedule():
  yield LinearSchedule(min_val=0.25, max_val=0.85,
                       num_steps=10)


@pytest.fixture(scope='function')
def inverted_schedule():
  yield LinearSchedule(min_val=0.25, max_val=0.85,
                       num_steps=10, invert=True)

@pytest.fixture(scope='function')
def exp_decay_schedule():
  yield ExpDecaySchedule(start=1.0, end=0.1, num_steps=1000)


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

def test_exp_decay(exp_decay_schedule: ExpDecaySchedule):
  assert exp_decay_schedule.value == exp_decay_schedule.start

def test_exp_decay_asymptote(exp_decay_schedule: ExpDecaySchedule):
  val = None
  for _ in range(exp_decay_schedule.num_steps * 15):
    val = exp_decay_schedule.value

  assert (val - exp_decay_schedule.end) < 1e-6
