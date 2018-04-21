"""
Work-in-progress (Runs but does not work)
"""
import gym
from torch.nn import MSELoss
from torch.optim import RMSprop

from torchrl.models import SimplePolicyNet
from torchrl.learners import A2CLearner
from torchrl import EpisodeRunner

NUM_EPISODES = 1000


class MountainCarLearner(A2CLearner):
    pass


def create_learner(env):
    policy_net = SimplePolicyNet(env.observation_space.shape[0], env.action_space.n)

    mse_loss = MSELoss()
    rms_prop = RMSprop(policy_net.parameters(), lr=1e-3, weight_decay=0.99)

    learner = MountainCarLearner(policy_net, mse_loss, rms_prop, (env.action_space.n,),
                                 gamma=0.99, eps_max=1.0, eps_min=0.1, temperature=2000.0)

    return learner


def main():
    env = gym.make('CartPole-v1')
    runner = EpisodeRunner(env, max_steps=1000)
    learner = create_learner(env)

    for i in range(1, NUM_EPISODES + 1):
        runner.reset()
        reward = 0

        while not runner.is_done():
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
                runner.run(learner, steps=5, render=False)
            learner.transition(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
            learner.learn()
            reward += sum(reward_batch)

        if i % 10 == 0:
            print('Episode {} reward: {}'.format(i, reward))

    env.close()


if __name__ == '__main__':
    main()
