"""
Work-in-progress (Runs but does not work)
"""
import gym
from torch.nn import MSELoss
from torch.optim import RMSprop

from torchrl.archs import SimplePolicyNet
from torchrl.learners import A2CLearner
from torchrl import EpisodeRunner

NUM_EPISODES = 350


class MountainCarLearner(A2CLearner):
    pass


def main():
    env = gym.make('MountainCar-v0')
    runner = EpisodeRunner(env, max_steps=1000)

    policy_net = SimplePolicyNet(env.observation_space.shape[0], env.action_space.n)

    mse_loss = MSELoss()
    rms_prop = RMSprop(policy_net.parameters(), lr=1e-3, weight_decay=0.99)

    learner = MountainCarLearner(policy_net, mse_loss, rms_prop, env.action_space.n,
                                 gamma=0.99, eps_max=1.0, eps_min=0.1, temperature=2000.0)

    for i in range(1, NUM_EPISODES + 1):
        runner.reset()

        # @TODO
        reward = 0
        while not runner.run(learner, steps=1):
            reward -= 1
        learner.learn()

        if i % 10 == 0:
            print('Episode {}: {} steps'.format(i, reward))

    env.close()


if __name__ == '__main__':
    main()
