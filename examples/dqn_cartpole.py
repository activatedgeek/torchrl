import gym
import time
import torch
import visdom
from rl_baselines import Transition
from rl_baselines.agents import DQN

viz = visdom.Visdom()
window = None


def run_episode(env, agent, render=False):
    t = 0

    state = env.reset()
    done = False
    while not done:
        if render:
            env.render()
            time.sleep(0.1)

        action = agent.act(torch.FloatTensor(state))
        next_state, reward, done, info = env.step(action)
        reward = -1 if done else reward

        agent.remember(Transition(state, action, reward, next_state, done))

        agent.learn()

        state = next_state
        t += 1

    return t


def main():
    env = gym.make('CartPole-v1')
    agent = DQN(env.observation_space.shape[0], env.action_space.n)

    episodes = 350
    score_list = []
    eps_list = []
    for idx in range(1, episodes + 1):
        score = run_episode(env, agent, render=False)
        score_list.append(score)
        eps_list.append(agent.eps)

        # @TODO This is a mess
        if idx % 25 == 0 and viz.check_connection():
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
