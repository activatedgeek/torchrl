import pytest
from torchrl.contrib.experiments import DQNExperiment
from torchrl.contrib.experiments import DDPGExperiment


@pytest.mark.parametrize('spec_id, n_frames, buffer_size', [
    ('Acrobot-v1', 200, 500),
    ('CartPole-v1', 600, 500),
    ('MountainCar-v0', 800, 500),
])
def test_dqn_exp(spec_id: str, n_frames: str, buffer_size: int):
  exp = DQNExperiment(env_id=spec_id, n_frames=n_frames,
                      buffer_size=buffer_size)
  exp.run()

  assert len(exp.buffer) == min(buffer_size, n_frames)
  assert exp._cur_frames == n_frames


@pytest.mark.parametrize('spec_id, n_frames, buffer_size', [
    ('Pendulum-v0', 800, 500),
])
def test_ddpg_exp(spec_id: str, n_frames: str, buffer_size: int):
  exp = DDPGExperiment(env_id=spec_id, n_frames=n_frames,
                       buffer_size=buffer_size)
  exp.run()

  assert len(exp.buffer) == min(buffer_size, n_frames)
  assert exp._cur_frames == n_frames
