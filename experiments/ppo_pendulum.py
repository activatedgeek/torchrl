import torchrl.registry as registry
import torchrl.registry.hparams as hparams
from torchrl.registry.problems import PPOProblem
from torchrl.learners import BasePPOLearner


@registry.register_problem('ppo-pendulum-v0')
class PendulumPPOProblem(PPOProblem):
  def __init__(self, args):
    args.env = 'Pendulum-v0'
    super(PendulumPPOProblem, self).__init__(args)

  def init_agent(self):
    args = self.args

    observation_space, action_space = self.get_gym_spaces()

    agent = BasePPOLearner(
        observation_space,
        action_space,
        lr=args.actor_lr,
        gamma=args.gamma,
        lmbda=args.lmbda,
        alpha=args.alpha,
        beta=args.beta)

    if args.cuda:
      agent.cuda()

    return agent


@registry.register_hparam('ppo-pendulum')
def hparam_ppo_pendulum():
  params = hparams.base_ppo()

  params.seed = 1

  params.rollout_steps = 20
  params.num_processes = 16
  params.num_total_steps = int(7e6)
  params.eval_interval = 500

  params.batch_size = 5

  params.actor_lr = 3e-4

  params.alpha = 0.5
  params.gamma = 0.99
  params.beta = 1e-3
  params.lmbda = 0.95

  params.clip_ratio = 0.2
  params.max_grad_norm = 1.0
  params.ppo_epochs = 4

  return params
