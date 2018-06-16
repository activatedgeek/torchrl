import torchrl.registry as registry
import torchrl.registry.hparams as hparams
from torchrl.registry.problems import DQNProblem
from torchrl.learners import BaseDQNLearner


class CartPoleDQNLearner(BaseDQNLearner):
  def learn(self, obs, action, reward, next_obs, done, **kwargs):
    for i in range(len(reward)):
      if done[i] == 1:
        reward[i] = -1.0

    return super(CartPoleDQNLearner, self).learn(obs, action, reward, next_obs, done)


@registry.register_problem('dqn-cartpole-v1')
class CartPoleDQNProblem(DQNProblem):
  def __init__(self, args):
    args.env = 'CartPole-v1'
    super(CartPoleDQNProblem, self).__init__(args)

  def init_agent(self):
    args = self.args

    observation_space, action_space = self.get_gym_spaces()

    agent = CartPoleDQNLearner(
      observation_space,
      action_space,
      lr=args.actor_lr,
      gamma=args.gamma,
      target_update_interval=args.target_update_interval)

    if args.cuda:
        agent.cuda()

    return agent


@registry.register_hparam('dqn-cartpole')
def hparam_dqn_cartpole():
  params = hparams.base_dqn()

  params.seed = 1
  params.rollout_steps = 1
  params.num_processes = 1
  params.actor_lr = 1e-3
  params.gamma = 0.8
  params.target_update_interval = 5
  params.eps_min = 0.1
  params.buffer_size = 5000
  params.batch_size = 64
  params.num_total_steps = 12000

  params.eval_interval = 500

  return params
