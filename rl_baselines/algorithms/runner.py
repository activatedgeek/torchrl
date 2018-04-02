import abc
import time


class Runner(metaclass=abc.ABCMeta):
    def __init__(self, env, agent, criterion, optimizer):
        self.env = env
        self.agent = agent
        self.criterion = criterion
        self.optimizer = optimizer

    @abc.abstractmethod
    def step(self, state):
        raise NotImplementedError

    def run_episode(self, render=False, fps=30):
        total_reward = 0

        state = self.env.reset()
        done = False
        while not done:
            if render:
                self.env.render()
                time.sleep(1. / fps)

            next_state, reward, done, info = self.step(state)

            state = next_state
            total_reward += reward

        return total_reward

    def run(self, num_episodes, render=False, fps=30):
        reward_list = []
        for _ in range(num_episodes):
            reward = self.run_episode(render, fps)
            reward_list.append(reward)

        return reward_list
