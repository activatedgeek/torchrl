from .. import registry


def base():
  params = registry.HParams()

  params.num_processes = 1

  params.gamma = 0.99
  params.rollout_steps = 100
  params.max_episode_steps = 2500
  params.num_total_steps = int(1e3)
  params.batch_size = 128
  params.buffer_size = int(1e6)

  return params


def base_pg():
  params = base()

  params.alpha = 0.5
  params.beta = 1e-3
  params.actor_lr = 1e-4
  params.critic_lr = 1e-3
  params.clip_grad_norm = 10.0

  return params


def base_ddpg():
  params = base_pg()

  params.tau = 1e-2
  params.ou_mu = 0.0
  params.ou_theta = 0.15
  params.ou_sigma = 0.2

  return params


def base_ppo():
  params = base_pg()

  params.lmbda = 1.0
  params.clip_ratio = 0.2
  params.ppo_epochs = 5
  params.max_grad_norm = 1.0

  return params


def base_dqn():
  params = base()

  params.double_dqn = False
  params.eps_max = 1.0
  params.eps_min = 0.1
  params.target_update_interval = 2

  return params
