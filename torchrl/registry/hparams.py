import torch
from . import registry


class HParams:
  def __getattr__(self, item):
    return self.__dict__[item]

  def __setattr__(self, key, value):
    self.__dict__[key] = value

  def __repr__(self):
    print_str = ''
    for key, value in self.__dict__.items():
      print_str += '{}: {}\n'.format(key, value)
    return print_str


@registry.register_hparam
def base():
  params = HParams()

  params.num_processes = 1
  params.seed = None
  params.cuda = torch.cuda.is_available()
  params.save_interval = 100
  params.eval_interval = 100
  params.num_eval = 10

  params.log_dir = None
  params.save_dir = None
  params.load_dir = None

  params.gamma = 0.99
  params.rollout_steps = 100
  params.max_episode_steps = 2500
  params.num_total_steps = int(1e6)
  params.batch_size = 128
  params.buffer_size = int(1e6)

  return params


@registry.register_hparam
def base_pg():
  params = base()

  params.alpha = 0.5
  params.beta = 1e-3
  params.actor_lr = 1e-4
  params.critic_lr = 1e-3
  params.clip_grad_norm = 10.0

  return params


@registry.register_hparam
def base_ddpg():
  params = base_pg()

  params.tau = 1e-2
  params.ou_mu = 0.0
  params.ou_theta = 0.15
  params.ou_sigma = 0.2

  return params


@registry.register_hparam
def base_ppo():
  params = base_pg()

  params.lmbda = 1.0
  params.clip_ratio = 0.2
  params.ppo_epochs = 5

  return params


@registry.register_hparam
def base_dqn():
  params = base()

  params.eps_max = 1.0
  params.eps_min = 0.1
  params.target_update_interval = 2

  return params
