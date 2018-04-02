import math
import torch
import random
from torch.autograd import Variable
from ..memory import ReplayMemory, Transition


class DQN:
    def __init__(self,
                 network,
                 gamma=0.99,
                 eps_max=1.0,
                 eps_min=0.01,
                 temperature=3500.0,
                 batch_size=32):
        super(DQN, self).__init__()

        # Hyper-Parameters
        self._gamma = gamma
        self._eps_max = eps_max
        self._eps_min = eps_min
        self._temperature = temperature
        self._batch_size = batch_size

        # Network Architecture
        self._network = network

        # Internal state
        self._memory = ReplayMemory(size=1000)
        self._steps = 0
        self._eps = self._eps_max

    def act(self, state):
        if random.random() < self._eps:
            return random.randrange(self._network._output_size)

        action_values = self._network(Variable(state.unsqueeze(0), volatile=True))
        value, action = action_values.max(1)
        return action.data[0]

    def remember(self, transition):
        assert isinstance(transition, Transition)
        self._memory.add(transition)

    def grad(self, criterion):
        if len(self._memory) < self._batch_size:
            return

        batch = self._memory.sample(self._batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = list(zip(*batch))

        batch_state = Variable(torch.cat(map(lambda s: torch.FloatTensor([s]), batch_state)))
        batch_action = Variable(torch.cat(list(map(lambda a: torch.LongTensor([[a]]), batch_action))))
        batch_reward = Variable(torch.cat(map(lambda r: torch.FloatTensor([[r]]), batch_reward)))
        batch_next_state = Variable(
            torch.cat(map(lambda s: torch.FloatTensor([s]), batch_next_state)), volatile=True)

        current_q_values = self._network(batch_state).gather(1, batch_action)
        max_next_q_values = self._network(batch_next_state).max(1)[0].unsqueeze(1)
        expected_q_values = batch_reward + self._gamma * max_next_q_values

        loss = criterion(current_q_values, expected_q_values)
        loss.backward()

        self._steps += 1
        self._eps = self._eps_min + \
                    (self._eps_max - self._eps_min) * math.exp(-float(self._steps) * 1. / self._temperature)
