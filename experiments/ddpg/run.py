import gym
import argparse
import torch
import time
import numpy as np
from copy import deepcopy
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from torchrl import EpisodeRunner, MultiEpisodeRunner, CPUReplayBuffer
from torchrl.utils import set_seeds, OUNoise

from learner import BaseDDPGLearner


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Generic arguments
    parser.add_argument('--env', type=str, metavar='', default='MountainCarContinuous-v0',
                        help='Gym environment id')
    parser.add_argument('--num-processes', type=int, metavar='', default=1,
                        help='Number of parallel trajectories to run')
    parser.add_argument('--seed', type=int, metavar='', default=0, help='Random Seed')
    parser.add_argument('--cuda', dest='cuda', action='store_true', help='Enable CUDA')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='Disable CUDA')
    parser.set_defaults(cuda=True)
    parser.add_argument('--log-dir', type=str, metavar='', default='log', help='Directory to store logs')
    parser.add_argument('--save-dir', type=str, metavar='', help='Directory to store models')
    parser.add_argument('--save-interval', type=int, metavar='', default=100, help='Save interval for the model')

    # General training arguments
    parser.add_argument('--gamma', type=float, metavar='', default=0.995, help='Discount factor')
    parser.add_argument('--rollout-steps', type=int, metavar='', default=100, help='Number of rollout steps')
    parser.add_argument('--max-episode-steps', type=int, metavar='', default=2500,
                        help='Maximum number of episode steps')
    parser.add_argument('--batch-size', type=int, metavar='', default=128, help='Size of batch size of transitions')
    parser.add_argument('--num-total-steps', type=int, metavar='', default=int(1e6),
                        help='Total number of trajectory steps to see before complete training')

    # DDPG specific arguments
    parser.add_argument('--tau', type=float, metavar='', default=1e-2, help='Soft Parameter update coefficient')
    parser.add_argument('--actor-lr', type=float, metavar='', default=1e-4, help='Learning rate for actor')
    parser.add_argument('--critic-lr', type=float, metavar='', default=1e-3, help='Learning rate for critic')
    parser.add_argument('--buffer-size', type=int, metavar='', default=int(1e6), help='Size of the replay buffer')
    parser.add_argument('--ou-mu', type=float, metavar='', default=0.0, help='Mu parameter for OU Noise')
    parser.add_argument('--ou-theta', type=float, metavar='', default=0.15, help='Theta parameter for OU Noise')
    parser.add_argument('--ou-sigma', type=float, metavar='', default=0.2, help='Std Deviation for OU Noise')

    args = parser.parse_args()

    if not torch.cuda.is_available():
        args.cuda = False

    return args


class DDPGRunner(EpisodeRunner):
    def act(self, learner):
        obs_tensor = Variable(torch.from_numpy(self._obs).float(), volatile=True).unsqueeze(0)
        if learner.is_cuda:
            obs_tensor = obs_tensor.cuda()
        action = learner.act(obs_tensor)
        action = action.squeeze(0).cpu().data.numpy()
        action += learner.noise()
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        return action


def train(env, agent, runner, logger, buffer):
    num_epochs = args.num_total_steps // args.rollout_steps // args.num_processes
    num_episodes = 0
    num_timesteps = 0

    agent.train()

    for epoch in range(1, num_epochs + 1):
        rollout_steps = 0
        episode_len = [0] * args.num_processes
        episode_reward = [0] * args.num_processes
        episode_actions = [np.zeros((0, env.action_space.shape[0])) for _ in range(args.num_processes)]

        rollout_start = time.time()
        history_list = runner.run(agent, steps=rollout_steps, store=True)
        rollout_duration = time.time() - rollout_start

        done = runner.is_done()
        for i, history in enumerate(history_list):
            transitions = list(zip(*history))
            buffer.extend(transitions)

            rollout_steps += len(history[2])
            episode_len[i] += len(history[2])
            episode_reward[i] += np.array(history[2]).sum()
            episode_actions[i] = np.concatenate([episode_actions[i], np.array(history[1])])

            if done[i]:
                num_episodes += 1

                logger.add_scalar('episode length', episode_len[i], global_step=num_episodes)
                logger.add_scalar('episode reward', episode_reward[i], global_step=num_episodes)
                logger.add_histogram('agent actions', episode_actions[i], global_step=num_episodes)

                episode_len[i] = 0
                episode_reward[i] = 0
                episode_actions[i] = np.zeros((0, env.action_space.shape[0]))

                runner.reset(i)
                agent.reset()

        num_timesteps += rollout_steps

        if len(buffer) >= args.batch_size:
            transition_batch = buffer.sample(args.batch_size)
            actor_loss, critic_loss = agent.learn(transition_batch)

            logger.add_scalar('actor loss', actor_loss, global_step=epoch)
            logger.add_scalar('critic loss', critic_loss, global_step=epoch)

        logger.add_scalar('steps per sec', rollout_steps / rollout_duration, global_step=epoch)

        if args.save_dir and epoch % args.save_interval == 0:
            agent.save(args.save_dir)


def main(args):
    set_seeds(args.seed)

    env = gym.make(args.env)
    env.seed(args.seed)

    agent = BaseDDPGLearner(
        env.observation_space,
        env.action_space,
        OUNoise(
            mu=args.ou_mu * np.ones(env.action_space.shape[0]),
            sigma=args.ou_sigma * np.ones(env.action_space.shape[0]),
            theta=args.ou_theta
        ),
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        tau=args.tau)
    if args.cuda:
        agent.cuda()

    runner = MultiEpisodeRunner([
        lambda: DDPGRunner(deepcopy(env), max_steps=args.max_episode_steps)
        for _ in range(args.num_processes)
    ])

    buffer = CPUReplayBuffer(args.buffer_size)

    logger = SummaryWriter(args.log_dir)

    train(env, agent, runner, logger, buffer)

    runner.stop()
    env.close()
    logger.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
