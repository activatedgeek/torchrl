import gym
import torch
import visdom
from torch.nn import SmoothL1Loss
from torch.optim import Adam

from torchrl.agents import DQN
from torchrl.archs import SimpleQNet
from torchrl.algorithms import QLearning

viz = visdom.Visdom()
window = None


def main():
    env = gym.make('CartPole-v1')
    qnet = SimpleQNet(env.observation_space.shape[0], env.action_space.n)
    agent = DQN(qnet)
    criterion = SmoothL1Loss()
    optimizer = Adam(qnet.parameters(), lr=1e-4)

    num_episodes = 1000
    runner = QLearning(env, agent, criterion, optimizer,
                       gamma=0.8, eps_max=1.0, eps_min=0.1, temperature=2000.0,
                       memory_size=100000, batch_size=64)
    history = runner.run(num_episodes, store_history=True, render=False)

    reward_list = list(map(lambda h: len(h), history))

    # @TODO A general method to collect training stats?
    if viz.check_connection():
        global window
        window = viz.line(torch.FloatTensor(reward_list), torch.FloatTensor(list(range(1, num_episodes + 1))),
                          win=window, name='training_episode_rewards', update='replace' if window else None,
                          opts={'title': 'Training Rewards', 'xlabel': 'Episode',
                                'ylabel': 'Reward', 'width': 800, 'height': 400})

    env.close()


if __name__ == '__main__':
    main()
