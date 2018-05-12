import gym
import numpy as np
import time
from copy import deepcopy
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from torchrl import EpisodeRunner, MultiEpisodeRunner, CPUReplayBuffer
from torchrl.utils import set_seeds
from torchrl.policies import epsilon_greedy

from dqn_learner import BaseDQNLearner


class DQNRunner(EpisodeRunner):
    def act(self, learner):
        obs_tensor = Variable(torch.from_numpy(self._obs).float(), volatile=True).unsqueeze(0)
        if learner.is_cuda:
            obs_tensor = obs_tensor.cuda()
        action = learner.act(obs_tensor)
        action = action.max(dim=1)[1].cpu().data.numpy()
        action, _ = epsilon_greedy(self.env.action_space.n, action, learner.eps)
        return action


class CartPoleDQNLearner(BaseDQNLearner):
    def learn(self, batch, **kwargs):

        for i in range(len(batch)):
            if batch[i][-1] == 1:
                batch[i] = (*batch[i][:2], -1.0, *batch[i][3:])

        return super(CartPoleDQNLearner, self).learn(batch)


def train(args, env, agent, runner, logger, buffer):
    num_epochs = args.num_total_steps // args.rollout_steps // args.num_processes
    num_episodes = 0
    num_timesteps = 0

    episode_len = [0] * args.num_processes
    episode_reward = [0] * args.num_processes
    episode_actions = [np.zeros((0, 1)) for _ in range(args.num_processes)]

    agent.train()

    for epoch in range(1, num_epochs + 1):
        epoch_rollout_steps = 0

        rollout_start = time.time()
        history_list = runner.run(agent, steps=args.rollout_steps, store=True)
        rollout_duration = time.time() - rollout_start

        done = runner.is_done()
        for i, history in enumerate(history_list):
            transitions = list(zip(*history))
            buffer.extend(transitions)

            epoch_rollout_steps += len(history[2])
            episode_len[i] += len(history[2])
            episode_reward[i] += np.array(history[2]).sum()
            episode_actions[i] = np.concatenate([episode_actions[i], np.expand_dims(np.array(history[1]), axis=1)])

            if done[i]:
                num_episodes += 1

                logger.add_scalar('episode length', episode_len[i], global_step=num_episodes)
                logger.add_scalar('episode reward', episode_reward[i], global_step=num_episodes)
                logger.add_histogram('agent actions', episode_actions[i], global_step=num_episodes)

                episode_len[i] = 0
                episode_reward[i] = 0
                episode_actions[i] = np.zeros((0, 1))

                runner.reset(i)
                agent.reset()

        if len(buffer) >= args.batch_size:
            transition_batch = buffer.sample(args.batch_size)
            value_loss = agent.learn(transition_batch)

            logger.add_scalar('value loss', value_loss, global_step=epoch)

        num_timesteps += epoch_rollout_steps
        logger.add_scalar('steps per sec', epoch_rollout_steps / rollout_duration, global_step=epoch)

        if args.save_dir and epoch % args.save_interval == 0:
            agent.save(args.save_dir)


def main(args):
    set_seeds(args.seed)

    env = gym.make(args.env)
    env.seed(args.seed)

    agent = CartPoleDQNLearner(
        env.observation_space,
        env.action_space,
        lr=args.actor_lr,
        gamma=args.gamma,
        target_update_interval=args.target_update_interval)
    if args.cuda:
        agent.cuda()

    runner = MultiEpisodeRunner([
        lambda: DQNRunner(deepcopy(env), max_steps=args.max_episode_steps)
        for _ in range(args.num_processes)
    ])

    buffer = CPUReplayBuffer(args.buffer_size)

    logger = SummaryWriter(args.log_dir)

    train(args, env, agent, runner, logger, buffer)

    runner.stop()
    env.close()
    logger.close()
