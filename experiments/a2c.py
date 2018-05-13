import gym
import time
import numpy as np
import torch
from copy import deepcopy
from tensorboardX import SummaryWriter
from torch.autograd import Variable

from torchrl import EpisodeRunner, MultiEpisodeRunner
from torchrl.utils import set_seeds

from a2c_learner import BaseA2CLearner


class A2CRunner(EpisodeRunner):
    def act(self, learner: BaseA2CLearner):
        obs_tensor = torch.from_numpy(self._obs).unsqueeze(0).float()
        obs_tensor = Variable(obs_tensor, volatile=True)
        if learner.is_cuda:
            obs_tensor = obs_tensor.cuda()

        action = learner.act(obs_tensor)
        return action[0]


def train(args, env, agent, runner, logger):
    n_epochs = args.num_total_steps // args.rollout_steps // args.num_processes
    n_episodes = 0
    n_timesteps = 0

    episode_len = [0] * args.num_processes
    episode_reward = [0] * args.num_processes
    episode_actions = [np.zeros((0, 1)) for _ in range(args.num_processes)]

    agent.train()

    for epoch in range(1, n_epochs + 1):
        # Generate rollouts
        rollout_start = time.time()

        history_list = runner.run(agent, steps=args.rollout_steps, store=True)
        done_list = runner.is_done()

        rollout_duration = time.time() - rollout_start

        # Merge histories across multiple trajectories
        batch_history = EpisodeRunner.merge_histories(env.observation_space, env.action_space, *history_list)

        # Train the agent
        agent.learn(*batch_history)

        # Stats Collection for this epoch
        epoch_rollout_steps = 0

        for i, (history, done) in enumerate(zip(history_list, done_list)):
            epoch_rollout_steps += len(history[2])
            episode_len[i] += len(history[2])
            episode_reward[i] += history[2].sum()
            episode_actions[i] = np.append(episode_actions[i], history[1], axis=0)

            if done:
                n_episodes += 1

                logger.add_scalar('episode length', episode_len[i], global_step=n_episodes)
                logger.add_scalar('episode reward', episode_reward[i], global_step=n_episodes)
                logger.add_histogram('agent actions', episode_actions[i], global_step=n_episodes)

                episode_len[i] = 0
                episode_reward[i] = 0
                episode_actions[i] = np.zeros((0, 1))

                runner.reset(i)
                agent.reset()

        n_timesteps += epoch_rollout_steps

        logger.add_scalar('total timesteps', n_timesteps, global_step=epoch)
        logger.add_scalar('steps per sec', epoch_rollout_steps / rollout_duration, global_step=epoch)

        # Save Agent
        if args.save_dir and epoch % args.save_interval == 0:
            agent.save(args.save_dir)


def main(args):
    set_seeds(args.seed)

    env = gym.make(args.env)
    env.seed(args.seed)

    agent = BaseA2CLearner(
        env.observation_space,
        env.action_space,
        lr=args.actor_lr,
        gamma=args.gamma)
    if args.cuda:
        agent.cuda()

    runner = MultiEpisodeRunner([
        lambda: A2CRunner(deepcopy(env), max_steps=args.max_episode_steps)
        for _ in range(args.num_processes)
    ])

    logger = SummaryWriter(args.log_dir)

    train(args, env, agent, runner, logger)

    runner.stop()
    env.close()
    logger.close()
