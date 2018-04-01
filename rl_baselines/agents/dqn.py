import torch
import random
import math
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import SmoothL1Loss
from torch.optim import Adam
from ..memory import ReplayMemory, Transition


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()

        self._input_size = input_size
        self.action_size = output_size
        self._steps = 0

        # Hyper-Parameters
        self.gamma = 0.8
        self.eps = 0.9
        self.eps_max = 0.9
        self.eps_min = 0.05
        self.eps_decay = 1. / 3500
        self.memory = ReplayMemory(size=1000)
        self.memory_batch_size = 32

        # A simple one hidden layer network IN4-FC512-OUT2
        self.fc = nn.Sequential(OrderedDict([
            ('f1', nn.Linear(self._input_size, 512)),
            ('relu1', nn.ReLU()),
            ('f2', nn.Linear(512, self.action_size)),
        ]))

        # Huber Loss
        self.criterion = SmoothL1Loss()

        # Adam Optimizer for SGD
        self.optimizer = Adam(self.parameters(), lr=1e-4)

    def forward(self, obs):
        values = self.fc(obs)
        return values

    def _optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._steps += 1

    def act(self, state):
        if random.random() < self.eps:
            return random.randrange(self.action_size)

        action_values = self.forward(Variable(state.unsqueeze(0), volatile=True))
        value, action = action_values.max(1)
        return action.data[0]

    def remember(self, transition):
        assert isinstance(transition, Transition)
        self.memory.add(transition)

    def learn(self):
        if len(self.memory) < self.memory_batch_size:
            return

        batch = self.memory.sample(self.memory_batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = list(zip(*batch))

        batch_state = Variable(torch.cat(map(lambda s: torch.FloatTensor([s]), batch_state)))
        batch_action = Variable(torch.cat(list(map(lambda a: torch.LongTensor([[a]]), batch_action))))
        batch_reward = Variable(torch.cat(map(lambda r: torch.FloatTensor([[r]]), batch_reward)))
        batch_next_state = Variable(
            torch.cat(map(lambda s: torch.FloatTensor([s]), batch_next_state)), volatile=True)

        current_q_values = self.forward(batch_state).gather(1, batch_action)
        max_next_q_values = self.forward(batch_next_state).max(1)[0].unsqueeze(1)
        expected_q_values = batch_reward + self.gamma * max_next_q_values

        loss = self.criterion(current_q_values, expected_q_values)

        self._optimize(loss)

        self.eps = self.eps_min + (self.eps_max - self.eps_min) * math.exp(-float(self._steps) * self.eps_decay)
