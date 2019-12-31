import pytest
import random

from torchrl.envs import make_gym_env, TransitionMonitor


@pytest.mark.parametrize('spec_id', [
    'Acrobot-v1',
    'CartPole-v1',
    'MountainCar-v0',
    'MountainCarContinuous-v0',
    'Pendulum-v0',
])
def test_transition_monitor(spec_id: str):
  env = TransitionMonitor(make_gym_env(spec_id))

  for _ in range(3):
    env.reset()

    info = env.info
    assert not env.is_done
    assert len(env.transitions) == 0
    assert info.get('len') == 0
    assert info.get('return') == 0.0

    flushed_transitions = []
    while not env.is_done:
      env.step(env.action_space.sample())
      if random.random() < 0.2:  # Flush with probability 0.2
        flushed_transitions += env.flush()

    flushed_transitions += env.flush()

    info = env.info
    assert info.get('return') is not None
    assert info.get('len') > 0
    assert info.get('len') == len(flushed_transitions)
    assert len(env.transitions) == 0

  env.close()
