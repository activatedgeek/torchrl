import torch
from torch.autograd import Variable
from . import BaseLearner
from ..policies import epsilon_greedy


class A2C(BaseLearner):
    def __init__(self, agent, criterion, optimizer, **kwargs):
        super(A2C, self).__init__(agent, criterion, optimizer)

        # Hyper-Parameters
        self.gamma = kwargs.get('gamma', 0.99)
        self.eps_max = kwargs.get('eps_max', 1.0)
        self.eps_min = kwargs.get('eps_min', 0.01)
        self.temperature = kwargs.get('temperature', 3500.0)
        self.tmax = kwargs.get('tmax', 5)

        # Internal State
        self._eps = self.eps_max

    def step(self, state, n, *args, **kwargs):
        best_action = self.agent.act(state)
        action, _ = epsilon_greedy(n, best_action, eps=self._eps)
        return action

    def learn(self, episode, **kwargs):
        expected_reward = Variable(torch.FloatTensor([0.0]), requires_grad=True)
        if not episode[-1].done:
            state_tensor = torch.FloatTensor(episode[-1].state)
            q_values = self.agent.forward(Variable(state_tensor))
            expected_reward, _ = q_values.max()
            expected_reward = Variable(expected_reward, requires_grad=True)

        for transition in episode[::-1][1:]:
            expected_reward = transition.reward + self.gamma * expected_reward
            state_tensor = torch.FloatTensor(episode[-1].state)
            q_values = self.agent.forward(Variable(state_tensor, volatile=True))

            loss = self.criterion(expected_reward, q_values[transition.action])
            loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()
