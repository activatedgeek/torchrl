import gym
import time
import torch
import visdom
from torch.nn import MSELoss
from torch.optim import RMSprop

from torchrl.agents import DQN
from torchrl.archs import SimpleQNet
from torchrl.learners import A2CLearner
from torchrl import Episode


def run_episode(env, learner, **kwargs):
    max_steps = kwargs.get('max_steps', 1000000)
    render = kwargs.get('render', False)
    fps = kwargs.get('fps', 30)

    episode = Episode()

    state, done = env.reset(), False
    if render:
        env.render()
        time.sleep(1. / fps)

    for step in range(1, max_steps + 1):
        action = learner.step(state, env.action_space.n)
        next_state, reward, done, info = env.step(action)
        reward = -1 if done else reward  # Penalize for termination

        episode.append(state, action, reward, next_state, done)

        if render:
            env.render()
            time.sleep(1. / fps)

        if done:
            break

        state = next_state

    return episode


def main():
    viz = visdom.Visdom()
    window = None

    env = gym.make('CartPole-v1')

    qnet = SimpleQNet(env.observation_space.shape[0], env.action_space.n)
    agent = DQN(qnet)
    criterion = MSELoss()
    optimizer = RMSprop(qnet.parameters(), lr=1e-3, weight_decay=0.99)

    learner = A2CLearner(agent, criterion, optimizer,
                         gamma=0.9, eps_max=1.0, eps_min=0.1, temperature=2000.0)

    num_episodes = 2
    reward_list = []
    for i in range(num_episodes):
        episode = run_episode(env, learner, render=False)
        learner.learn(episode)
        reward_list.append(len(episode))

    # @TODO A general method to collect training stats?
    if viz.check_connection():
        window = viz.line(torch.FloatTensor(reward_list), torch.FloatTensor(list(range(1, num_episodes + 1))),
                          win=window, name='training_episode_rewards', update='replace' if window else None,
                          opts={'title': 'Training Rewards', 'xlabel': 'Episode',
                                'ylabel': 'Reward', 'width': 800, 'height': 400})

    env.close()


if __name__ == '__main__':
    main()
