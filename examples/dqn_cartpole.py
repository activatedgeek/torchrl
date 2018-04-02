import gym
import time
import torch
import visdom
from torch.nn import SmoothL1Loss
from torch.optim import Adam

from rl_baselines import Transition
from rl_baselines.agents import DQN
from rl_baselines.archs import SimpleQNet

viz = visdom.Visdom()
window = None


def run_episode(env, agent, criterion, optimizer, render=False, fps=24):
    t = 0

    state = env.reset()
    done = False
    while not done:
        if render:
            env.render()
            time.sleep(1. / fps)

        action = agent.act(torch.FloatTensor(state))
        next_state, reward, done, info = env.step(action)
        reward = -1 if done else reward

        agent.remember(Transition(state, action, reward, next_state, done))

        optimizer.zero_grad()
        agent.grad(criterion)
        optimizer.step()

        state = next_state
        t += 1

    return t


def main():
    env = gym.make('CartPole-v1')
    qnet = SimpleQNet(env.observation_space.shape[0], env.action_space.n)
    agent = DQN(qnet, gamma=0.8, eps_max=1.0, eps_min=0.05, batch_size=32, temperature=3500.0)
    criterion = SmoothL1Loss()
    optimizer = Adam(qnet.parameters(), lr=1e-4)

    episodes = 350
    score_list = []
    eps_list = []
    for idx in range(1, episodes + 1):
        score = run_episode(env, agent, criterion, optimizer, render=False)
        score_list.append(score)
        eps_list.append(agent._eps)

        # @TODO Fix this mess
        if viz.check_connection():
            global window
            window = viz.line(torch.FloatTensor(score_list), torch.FloatTensor(list(range(idx))),
                              win=window, name='training_episode_rewards', update='replace' if window else None,
                              opts={'title': 'Training Rewards', 'xlabel': 'Episode',
                                    'ylabel': 'Reward', 'width': 800, 'height': 400})
            window = viz.line(torch.FloatTensor(eps_list), torch.FloatTensor(list(range(idx))),
                              win=window, name='training_episode_eps', update='replace' if window else None,
                              opts={'title': 'Training Epsilon Decay', 'xlabel': 'Episode',
                                    'ylabel': '$\epsilon$', 'width': 800, 'height': 400})
    env.close()


if __name__ == '__main__':
    main()
