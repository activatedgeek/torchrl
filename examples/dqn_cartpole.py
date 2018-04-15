import gym
from torch.nn import SmoothL1Loss
from torch.optim import Adam

from torchrl.agents import DQN
from torchrl.archs import SimpleQNet
from torchrl.learners import DeepQLearner
from torchrl import EpisodeRunner

NUM_EPISODES = 400


class CartPoleLearner(DeepQLearner):
    """
    A DeepQLearner with some reward shaping - penalize when the
    cart pole falls (i.e. episode ends)
    """
    def transition(self, episode_id, state, action, reward, next_state, done):
        if done:
            reward = -1
        super(CartPoleLearner, self).transition(episode_id, state, action, reward, next_state, done)


def main():
    env = gym.make('CartPole-v1')
    runner = EpisodeRunner(env, max_steps=1000)

    qnet = SimpleQNet(env.observation_space.shape[0], env.action_space.n)
    agent = DQN(qnet)

    smooth_loss = SmoothL1Loss()
    adam = Adam(qnet.parameters(), lr=1e-4)

    learner = CartPoleLearner(agent, smooth_loss, adam, env.action_space.n,
                              gamma=0.8, eps_max=1.0, eps_min=0.1, temperature=2000.0,
                              memory_size=5000, batch_size=64)

    for i in range(1, NUM_EPISODES + 1):
        runner.reset()
        reward = 0
        while not runner.run(learner, steps=1):
            learner.learn()
            reward += 1

        if i % 10 == 0:
            print('Episode {}: {} steps'.format(i, reward))

    env.close()


if __name__ == '__main__':
    main()
