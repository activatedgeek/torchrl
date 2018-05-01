import os
import gym
from torch.optim import Adam
from tensorboardX import SummaryWriter

from torchrl.models import SimpleQNet
from torchrl.learners import DeepQLearner
from torchrl import EpisodeRunner

NUM_EPISODES = 300
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'log', 'dqn_cartpole')


class CartPoleLearner(DeepQLearner):
    """
    A DeepQLearner with some reward shaping - penalize when the
    cart pole falls (i.e. episode ends)
    """
    def transition(self, state, action, reward, next_state, done, episode_id=None):
        if done[-1] == 1:
            reward[-1] = -1
        super(CartPoleLearner, self).transition(state, action, reward, next_state, done, episode_id)


def create_learner(env):
    q_net = SimpleQNet(env.observation_space.shape[0], env.action_space.n)

    adam = Adam(q_net.parameters(), lr=1e-4)

    learner = CartPoleLearner(q_net, adam,
                              env.observation_space.shape, (env.action_space.n,),
                              gamma=0.8, eps_max=1.0, eps_min=0.1, temperature=2000.0,
                              memory_size=5000, batch_size=64, target_update_freq=5)
    return learner


def main():
    env = gym.make('CartPole-v1')
    runner = EpisodeRunner(env, max_steps=1000)
    learner = create_learner(env)
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    logger = SummaryWriter(LOG_DIR)

    for i in range(1, NUM_EPISODES + 1):
        runner.reset()

        steps = 0
        reward = 0

        while not runner.is_done():
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = runner.run(learner, steps=1)
            learner.transition(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
            learner.learn()
            steps += len(reward_batch)
            reward += sum(reward_batch)

        logger.add_scalar('episode length', steps, i + 1)
        logger.add_scalar('reward', steps, i + 1)

    logger.close()
    env.close()


if __name__ == '__main__':
    main()
