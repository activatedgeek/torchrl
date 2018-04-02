import torch
from . import Runner
from ..memory import Transition


class QLearning(Runner):
    def __init__(self, env, agent, criterion, optimizer):
        super(QLearning, self).__init__(env, agent, criterion, optimizer)

    def step(self, state):
        action = self.agent.act(torch.FloatTensor(state))
        next_state, reward, done, info = self.env.step(action)
        reward = -1 if done else reward

        transition = Transition(state, action, reward, next_state, done)
        self.agent.remember(transition)

        self.optimizer.zero_grad()
        self.agent.grad(self.criterion)
        self.optimizer.step()

        return transition
