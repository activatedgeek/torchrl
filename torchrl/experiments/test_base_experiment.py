import pytest

from torchrl.experiments import BaseExperiment


@pytest.mark.parametrize('spec_id, n_frames', [
    ('Acrobot-v1', 200),
    ('CartPole-v1', 600),
    ('MountainCar-v0', 800),
    ('MountainCarContinuous-v0', 400),
    ('Pendulum-v0', 1000),
])
def test_base_exp(spec_id: str, n_frames: str):
  exp = BaseExperiment(env_id=spec_id, n_frames=n_frames)
  exp.run()

  assert exp._cur_frames == n_frames
  