import gym
import torch
import visdom
from torch.nn import MSELoss
from torch.optim import RMSprop

from rl_baselines.agents import DQN
from rl_baselines.archs import SimpleQNet
from rl_baselines.algorithms import A2C

viz = visdom.Visdom()
window = None


def main():
    env = gym.make('CartPole-v1')
    qnet = SimpleQNet(env.observation_space.shape[0], env.action_space.n)
    agent = DQN(qnet)
    criterion = MSELoss()
    optimizer = RMSprop(qnet.parameters(), lr=1e-3, weight_decay=0.99)

    num_episodes = 15000
    runner = A2C(env, agent, criterion, optimizer,
                 gamma=0.99, eps_max=1.0, eps_min=0.05, temperature=15000.0)
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
