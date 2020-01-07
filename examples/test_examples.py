import pytest
from dqn import DQNExperiment
from ddpg import DDPGExperiment
from a2c import A2CExperiment
from ppo import PPOExperiment


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


@pytest.mark.parametrize('spec_id, n_frames', [
    ('Acrobot-v1', 200),
    ('CartPole-v1', 600),
    ('MountainCar-v0', 800),
])
def test_a2c_exp(spec_id: str, n_frames: str):
  exp = A2CExperiment(env_id=spec_id, n_frames=n_frames)
  exp.run()

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


@pytest.mark.parametrize('spec_id, n_frames', [
    ('Pendulum-v0', 800),
])
def test_ppo_exp(spec_id: str, n_frames: str):
  exp = PPOExperiment(env_id=spec_id, n_frames=n_frames)
  exp.run()

  assert exp._cur_frames == n_frames
