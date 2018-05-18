import numpy as np
import time
from tensorboardX import SummaryWriter

from torchrl import EpisodeRunner, MultiEpisodeRunner, CPUReplayBuffer
from torchrl.utils import set_seeds, get_gym_spaces

from dqn_learner import BaseDQNLearner


class CartPoleDQNLearner(BaseDQNLearner):
    def learn(self, obs, action, reward, next_obs, done, **kwargs):
        for i in range(len(reward)):
            if done[i] == 1:
                reward[i] = -1.0

        return super(CartPoleDQNLearner, self).learn(obs, action, reward, next_obs, done)


def train(args, agent: CartPoleDQNLearner, runner: MultiEpisodeRunner, logger: SummaryWriter, buffer: CPUReplayBuffer):
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

        history_list = runner.collect(agent, steps=args.rollout_steps, store=True)
        done_list = runner.is_done()

        rollout_duration = time.time() - rollout_start

        # Populate the buffer
        batch_history = EpisodeRunner.merge_histories(agent.observation_space, agent.action_space, *history_list)
        transitions = list(zip(*batch_history))
        buffer.extend(transitions)

        # Train the agent
        if len(buffer) >= args.batch_size:
            transition_batch = buffer.sample(args.batch_size)
            transition_batch = list(zip(*transition_batch))
            transition_batch = [np.array(item) for item in transition_batch]
            value_loss = agent.learn(*transition_batch)

            logger.add_scalar('value loss', value_loss, global_step=epoch)

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

    observation_space, action_space = get_gym_spaces(args.env)

    agent = CartPoleDQNLearner(
        observation_space,
        action_space,
        lr=args.actor_lr,
        gamma=args.gamma,
        target_update_interval=args.target_update_interval)
    if args.cuda:
        agent.cuda()

    runner = MultiEpisodeRunner(args.env, max_steps=args.max_episode_steps,
                                n_runners=args.num_processes, base_seed=args.seed)

    buffer = CPUReplayBuffer(args.buffer_size)

    logger = SummaryWriter(args.log_dir)

    train(args, agent, runner, logger, buffer)

    runner.stop()
    logger.close()
