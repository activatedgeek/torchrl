import gym
import torch
import visdom
from torch.nn import SmoothL1Loss
from torch.optim import Adam

from rl_baselines.agents import DQN
from rl_baselines.archs import SimpleQNet
from rl_baselines.algorithms import QLearning

viz = visdom.Visdom()
window = None


def main():
    env = gym.make('CartPole-v1')
    qnet = SimpleQNet(env.observation_space.shape[0], env.action_space.n)
    agent = DQN(qnet)
    criterion = SmoothL1Loss()
    optimizer = Adam(qnet.parameters(), lr=1e-4)

    num_episodes = 1000
    runner = QLearning(env,
                       agent,
                       criterion,
                       optimizer,
                       gamma=0.8,
                       eps_max=1.0,
                       eps_min=0.01,
                       temperature=1000.0,
                       memory_size=1000,
                       batch_size=32)
    history = runner.run(num_episodes, store_history=True, render=False)

    reward_list = list(map(lambda h: len(h), history))

    # @TODO A general method to collect training stats?
    if viz.check_connection():
        global window
        window = viz.line(torch.FloatTensor(reward_list), torch.FloatTensor(list(range(num_episodes))),
                          win=window, name='training_episode_rewards', update='replace' if window else None,
                          opts={'title': 'Training Rewards', 'xlabel': 'Episode',
                                'ylabel': 'Reward', 'width': 800, 'height': 400})

    env.close()


if __name__ == '__main__':
    main()
