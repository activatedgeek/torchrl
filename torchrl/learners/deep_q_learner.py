import torch
import random
import numpy as np
from torch.autograd import Variable
from . import BaseLearner
from .. import ReplayBuffer


class DeepQLearner(BaseLearner):
    def __init__(self, q_net, criterion, optimizer, observation_space, action_shape,
                 gamma=0.99,
                 eps_max=1.0,
                 eps_min=0.01,
                 temperature=3500.0,
                 memory_size=5000,
                 batch_size=32):
        super(DeepQLearner, self).__init__(criterion, optimizer)

        self.q_net = q_net
        self.action_shape = action_shape
        self.observation_shape = observation_space

        # Hyper-Parameters
        self.gamma = gamma
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.temperature = temperature
        self.memory_size = memory_size
        self.batch_size = batch_size

        # Internal State
        self._memory = ReplayBuffer(observation_space, action_shape, size=self.memory_size)
        self._steps = 0
        self._eps = self.eps_max

    def act(self, state):
        if random.random() < self._eps:
            return random.randrange(self.action_shape[0])

        action_values = self.q_net(Variable(torch.FloatTensor([state]), volatile=True))
        value, action = action_values.max(1)
        return action.data[0]

    def transition(self, state, action, reward, next_state, done, episode_id=None):
        self._memory.push(
            torch.FloatTensor(state),
            torch.LongTensor([action]),
            torch.FloatTensor([reward]),
            torch.FloatTensor(next_state),
            torch.LongTensor([done])
        )

    def learn(self, *args, **kwargs):
        if len(self._memory) <= self.batch_size:
            return

        batch_state, batch_action, batch_reward, batch_next_state, batch_done = \
            self._memory.sample(self.batch_size)

        batch_state = Variable(batch_state)
        batch_action = Variable(batch_action)
        batch_reward = Variable(batch_reward)
        batch_next_state = Variable(batch_next_state, volatile=True)

        current_q_values = self.q_net(batch_state).gather(1, batch_action)
        max_next_q_values = self.q_net(batch_next_state).max(1)[0].unsqueeze(1)
        expected_q_values = batch_reward + self.gamma * max_next_q_values

        loss = self.criterion(current_q_values, expected_q_values)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self._steps += 1
        self._eps = self.eps_min + \
                    (self.eps_max - self.eps_min) * np.exp(-float(self._steps) * 1. / self.temperature)
