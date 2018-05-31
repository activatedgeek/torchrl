import argparse
import torch
import importlib


def parse_common(parser):
    # Generic arguments
    parser.add_argument('--env', type=str, required=True, help='Gym environment id')
    parser.add_argument('--algo', type=str, required=True, help='Algorithm to run')
    parser.add_argument('--num-processes', type=int, metavar='', default=1,
                        help='Number of parallel trajectories to run')
    parser.add_argument('--seed', type=int, metavar='', default=None, help='Random Seed')
    parser.add_argument('--cuda', dest='cuda', action='store_true', help='Enable CUDA')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='Disable CUDA')
    parser.set_defaults(cuda=True)
    parser.add_argument('--log-dir', type=str, metavar='', default='log', help='Directory to store logs')
    parser.add_argument('--save-dir', type=str, metavar='', help='Directory to store models')
    parser.add_argument('--save-interval', type=int, metavar='', default=100, help='Save interval for the model')
    parser.add_argument('--eval-interval', type=int, metavar='', default=100, help='Eval interval for the model')
    parser.add_argument('--num-eval', type=int, metavar='', default=10, help='Number of model evaluations')

    # General training arguments
    parser.add_argument('--gamma', type=float, metavar='', default=0.99, help='Discount factor')
    parser.add_argument('--rollout-steps', type=int, metavar='', default=100, help='Number of rollout steps')
    parser.add_argument('--max-episode-steps', type=int, metavar='', default=2500,
                        help='Maximum number of episode steps')
    parser.add_argument('--num-total-steps', type=int, metavar='', default=int(1e6),
                        help='Total number of trajectory steps to see before complete training')
    parser.add_argument('--batch-size', type=int, metavar='', default=128,
                        help='Batch size for training')
    parser.add_argument('--buffer-size', type=int, metavar='', default=int(1e6), help='Size of the replay buffer')


def parse_ddpg(parser):
    parser.add_argument('--tau', type=float, metavar='', default=1e-2, help='Soft Parameter update coefficient')
    parser.add_argument('--ou-mu', type=float, metavar='', default=0.0, help='Mean for OU Noise')
    parser.add_argument('--ou-theta', type=float, metavar='', default=0.15, help='Theta for OU Noise')
    parser.add_argument('--ou-sigma', type=float, metavar='', default=0.2, help='Standard Deviation for OU Noise')


def parse_ppo(parser):
    parser.add_argument('--lambda', type=float, metavar='', dest='lmbda', default=1e-2,
                        help='Parameter for generalized advantage estimate')


def parse_pg(parser):
    parser.add_argument('--alpha', type=float, metavar='', default=0.5, help='Critic Loss coefficient')
    parser.add_argument('--beta', type=float, metavar='', default=1e-3, help='Entropy Loss coefficient')
    parser.add_argument('--actor-lr', type=float, metavar='', default=1e-4, help='Learning rate for actor')
    parser.add_argument('--critic-lr', type=float, metavar='', default=1e-3, help='Learning rate for critic')
    parser.add_argument('--clip-grad-norm', type=float, metavar='', default=10.0,
                        help='Gradient norm clipping parameter')


def parse_dqn(parser):
    parser.add_argument('--eps-max', type=float, metavar='', default=1.0,
                        help='Maximum value for epsilon-greedy')
    parser.add_argument('--eps-min', type=float, metavar='', default=0.01,
                        help='Minimum value for epsilon-greedy')
    parser.add_argument('--target-update-interval', type=int, metavar='', default=2,
                        help='Number of training steps before updating target Q-network')


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser._optionals.title = 'Common Arguments'

    parse_common(parser)

    pg_parser = parser.add_argument_group('Policy Gradient Arguments')
    parse_pg(pg_parser)

    ddpg_parser = parser.add_argument_group('DDPG Arguments')
    parse_ddpg(ddpg_parser)

    ppo_parser = parser.add_argument_group('PPO Arguments')
    parse_ppo(ppo_parser)

    dqn_parser = parser.add_argument_group('DQN Arguments')
    parse_dqn(dqn_parser)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        args.cuda = False

    return args


if __name__ == '__main__':
    args = parse_args()
    algo = importlib.import_module(args.algo)
    algo.main(args)
