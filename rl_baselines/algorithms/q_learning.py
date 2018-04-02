import math
import torch
import random
from torch.autograd import Variable
from . import Runner
from ..memory import Transition, ReplayMemory


class QLearning(Runner):
    def __init__(self, env, agent, criterion, optimizer, **kwargs):
        super(QLearning, self).__init__(env, agent, criterion, optimizer)

        # Hyper-Parameters
        self.gamma = kwargs.get('gamma', 0.99)
        self.eps_max = kwargs.get('eps_max', 1.0)
        self.eps_min = kwargs.get('eps_min', 0.01)
        self.temperature = kwargs.get('temperature', 3500.0)
        self.memory_size = kwargs.get('memory_size', 1000)
        self.batch_size = kwargs.get('batch_size', 32)

        # Internal State
        self._memory = ReplayMemory(size=self.memory_size)
        self._steps = 0
        self._eps = self.eps_max

    def step(self, state):
        action = random.randrange(self.env.action_space.n) if random.random() < self._eps \
            else self.agent.act(state)

        next_state, reward, done, info = self.env.step(action)
        reward = -1 if done else reward

        transition = Transition(state, action, reward, next_state, done)
        self._memory.add(transition)

        self.optimizer.zero_grad()
        self.learn()
        self.optimizer.step()

        return transition

    def learn(self):
        if len(self._memory) < self.batch_size:
            return

        batch = self._memory.sample(self.batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = list(zip(*batch))

        batch_state = Variable(torch.cat(map(lambda s: torch.FloatTensor([s]), batch_state)))
        batch_action = Variable(torch.cat(list(map(lambda a: torch.LongTensor([[a]]), batch_action))))
        batch_reward = Variable(torch.cat(map(lambda r: torch.FloatTensor([[r]]), batch_reward)))
        batch_next_state = Variable(
            torch.cat(map(lambda s: torch.FloatTensor([s]), batch_next_state)), volatile=True)

        current_q_values = self.agent.forward(batch_state).gather(1, batch_action)
        max_next_q_values = self.agent.forward(batch_next_state).max(1)[0].unsqueeze(1)
        expected_q_values = batch_reward + self.gamma * max_next_q_values

        loss = self.criterion(current_q_values, expected_q_values)
        loss.backward()

        self._steps += 1
        self._eps = self.eps_min + \
                    (self.eps_max - self.eps_min) * math.exp(-float(self._steps) * 1. / self.temperature)
