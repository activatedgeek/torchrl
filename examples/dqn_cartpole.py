import gym
from torch.nn import SmoothL1Loss
from torch.optim import Adam

from torchrl.models import SimpleQNet
from torchrl.learners import DeepQLearner
from torchrl import EpisodeRunner

NUM_EPISODES = 350


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

    smooth_loss = SmoothL1Loss()
    adam = Adam(q_net.parameters(), lr=1e-4)

    learner = CartPoleLearner(q_net, smooth_loss, adam,
                              env.observation_space.shape, (env.action_space.n,),
                              gamma=0.8, eps_max=1.0, eps_min=0.1, temperature=2000.0,
                              memory_size=5000, batch_size=64)
    return learner


def main():
    env = gym.make('CartPole-v1')
    runner = EpisodeRunner(env, max_steps=1000)
    learner = create_learner(env)

    for i in range(1, NUM_EPISODES + 1):
        runner.reset()
        reward = 0

        while not runner.is_done():
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = runner.run(learner, steps=1)
            learner.transition(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
            learner.learn()
            reward += sum(reward_batch)

        if i % 10 == 0:
            print('Episode {} reward: {}'.format(i, reward))

    env.close()


if __name__ == '__main__':
    main()
