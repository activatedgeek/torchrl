import gym
import argparse
import torch
import time
import numpy as np
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from torchrl import EpisodeRunner, CPUReplayBuffer
from torchrl.utils import set_seeds, OUNoise

from learner import BaseDDPGLearner


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env', type=str, metavar='', default='MountainCarContinuous-v0',
                        help='Gym environment id')
    parser.add_argument('--seed', type=int, metavar='', default=1, help='Random Seed')
    parser.add_argument('--cuda', dest='cuda', action='store_true', help='Enable CUDA')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='Disable CUDA')
    parser.set_defaults(cuda=True)
    parser.add_argument('--log-dir', type=str, metavar='', default='log', help='Directory to store logs')
    parser.add_argument('--save-dir', type=str, metavar='', help='Directory to store models')
    parser.add_argument('--save-interval', type=int, metavar='', default=100, help='Save interval for the model')

    parser.add_argument('--gamma', type=float, metavar='', default=0.99, help='Discount factor')
    parser.add_argument('--tau', type=float, metavar='', default=1e-3, help='Soft Parameter update coefficient')
    parser.add_argument('--actor-lr', type=float, metavar='', default=1e-4, help='Learning rate for actor')
    parser.add_argument('--critic-lr', type=float, metavar='', default=1e-3, help='Learning rate for critic')
    parser.add_argument('--buffer-size', type=int, metavar='', default=int(1e6), help='Size of the replay buffer')
    parser.add_argument('--batch-size', type=int, metavar='', default=64, help='Size of batch size of transitions')
    parser.add_argument('--rollout-steps', type=int, metavar='', default=100, help='Number of rollout steps')
    parser.add_argument('--max-episode-steps', type=int, metavar='', default=2500,
                        help='Maximum number of episode steps')
    parser.add_argument('--train-steps', type=int, metavar='', default=50,
                        help='Number of training steps after rollout')
    parser.add_argument('--epochs', type=int, metavar='', default=1000, help='Number of training epochs')

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


def main(args):
    set_seeds(args.seed)

    env = gym.make(args.env)

    noise = OUNoise(
        mu=args.ou_mu * np.ones(env.action_space.shape[0]),
        sigma=args.ou_sigma * np.ones(env.action_space.shape[0]),
        theta=args.ou_theta)
    agent = BaseDDPGLearner(
        env.observation_space,
        env.action_space,
        noise,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        tau=args.tau)
    if args.cuda:
        agent.cuda()

    buffer = CPUReplayBuffer(args.buffer_size)
    runner = DDPGRunner(env, max_steps=args.max_episode_steps)

    logger = SummaryWriter(args.log_dir)

    num_episodes = 0
    num_train_steps = 0

    for e in range(args.epochs):
        rollout_start = time.time()
        rollout_steps = args.rollout_steps

        episode_reward = 0
        episode_actions = np.zeros((0, env.action_space.shape[0]))

        while rollout_steps:
            history = runner.run(agent, steps=rollout_steps, store=True)
            transitions = list(zip(*history))
            buffer.extend(transitions)

            rollout_steps -= len(history)
            episode_reward += np.array(history[2]).sum()
            episode_actions = np.concatenate([episode_actions, np.array(history[1])])

            if runner.is_done():
                num_episodes += 1

                logger.add_scalar('total reward', episode_reward, global_step=num_episodes)
                logger.add_histogram('agent action', episode_actions, global_step=num_episodes)

                episode_reward = 0
                episode_actions = np.zeros((0, env.action_space.shape[0]))

                runner.reset()
                noise.reset()

        rollout_duration = time.time() - rollout_start
        logger.add_scalar('steps per sec', args.rollout_steps / rollout_duration, global_step=e)

        for _ in range(args.train_steps):
            transition_batch = buffer.sample(args.batch_size)
            actor_loss, critic_loss = agent.learn(transition_batch)

            num_train_steps += 1
            logger.add_scalar('actor loss per train step', actor_loss, global_step=num_train_steps)
            logger.add_scalar('critic loss per train step', critic_loss, global_step=num_train_steps)

        if args.save_dir and e % args.save_interval == 0:
            agent.save(args.save_dir)

    env.close()
    logger.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
