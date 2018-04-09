import numpy as np
import torch
from torch.autograd import Variable
from . import EpisodeRunner
from ..memory import Transition


class A2C(EpisodeRunner):
    def __init__(self, env, agent, criterion, optimizer, **kwargs):
        super(A2C, self).__init__(env, agent, criterion, optimizer)

        # Hyper-Parameters
        self.gamma = kwargs.get('gamma', 0.99)
        self.eps_max = kwargs.get('eps_max', 1.0)
        self.eps_min = kwargs.get('eps_min', 0.01)
        self.temperature = kwargs.get('temperature', 3500.0)
        self.tmax = kwargs.get('tmax', 5)

        # Internal State
        self._eps = self.eps_max
        self._steps = 0
        self._episode = None

    def epsilon_greedy(self, state):
        best_action = self.agent.act(state)
        probs = np.ones(self.env.action_space.n) * self._eps / self.env.action_space.n
        probs[best_action] += 1 - self._eps
        action = np.random.choice(np.arange(self.env.action_space.n), p=probs)
        return action

    def step(self, state):
        action = self.epsilon_greedy(state)
        next_state, reward, done, info = self.env.step(action)
        reward = -10 if done else reward

        transition = Transition(state, action, reward, next_state, done)

        self._steps += 1
        return transition

    def process_episode(self, episode):
        self._episode = episode
        self.learn()
        self._episode = None

    def learn(self):
        expected_reward = Variable(torch.FloatTensor([0.0]), requires_grad=True)
        if not self._episode[-1].done:
            state_tensor = torch.FloatTensor(self._episode[-1].state)
            q_values = self.agent.forward(Variable(state_tensor))
            expected_reward, _ = q_values.max()
            expected_reward = Variable(expected_reward, requires_grad=True)

        for transition in self._episode[::-1][1:]:
            expected_reward = transition.reward + self.gamma * expected_reward
            state_tensor = torch.FloatTensor(self._episode[-1].state)
            q_values = self.agent.forward(Variable(state_tensor, volatile=True))

            loss = self.criterion(expected_reward, q_values[transition.action])
            loss.backward()

        if self._steps % self.tmax == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
