import torchrl.registry as registry
import torchrl.registry.hparams as hparams
from torchrl.registry.problems import DDPGProblem
from torchrl.learners import BaseDDPGLearner


@registry.register_problem('ddpg-pendulum-v0')
class PendulumDDPGProblem(DDPGProblem):
  def __init__(self, params, args):
    params.env = 'Pendulum-v0'
    super(PendulumDDPGProblem, self).__init__(params, args)

  def init_agent(self):
    params = self.params

    observation_space, action_space = self.get_gym_spaces()

    agent = BaseDDPGLearner(
        observation_space,
        action_space,
        actor_lr=params.actor_lr,
        critic_lr=params.critic_lr,
        gamma=params.gamma,
        tau=params.tau)

    return agent


@registry.register_hparam('ddpg-pendulum')
def hparam_ddpg_pendulum():
  params = hparams.base_ddpg()

  params.num_processes = 1

  params.rollout_steps = 1
  params.max_episode_steps = 500
  params.num_total_steps = 20000

  params.gamma = 0.99
  params.buffer_size = int(1e6)

  params.batch_size = 128
  params.tau = 1e-2
  params.actor_lr = 1e-4
  params.critic_lr = 1e-3

  return params
