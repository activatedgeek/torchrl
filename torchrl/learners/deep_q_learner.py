import torch
import numpy as np
from copy import deepcopy
from torch.autograd import Variable
from torch.nn import MSELoss
from . import BaseLearner
from .. import ReplayBuffer
from ..policies import epsilon_greedy


class DeepQLearner(BaseLearner):
    def __init__(self, q_net, optimizer, observation_space, action_shape,
                 gamma=0.99,
                 eps_max=1.0,
                 eps_min=0.01,
                 temperature=3500.0,
                 memory_size=5000,
                 batch_size=32,
                 target_update_freq=5):
        """
        :param q_net: Q-value network
        :param criterion: Loss function criterion
        :param optimizer: Optimizer for stepping
        :param observation_space: shape of the observation space
        :param action_shape: shape of the action space
        :param gamma: discount factor
        :param eps_max: starting value of epsilon
        :param eps_min: terminal value of epsilon during annealing
        :param temperature: temperature for annealing
        :param memory_size: size of replay buffer
        :param batch_size: batch size to sample from replay buffer
        :param target_update_freq: number of steps to update target network parameters after
        """
        super(DeepQLearner, self).__init__(optimizer)

        self.q_net = q_net
        self.target_q_net = deepcopy(q_net)

        self.action_shape = action_shape
        self.observation_shape = observation_space

        # Hyper-Parameters
        self.gamma = gamma
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.temperature = temperature
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Internal State
        self._memory = ReplayBuffer(observation_space, action_shape, size=self.memory_size)
        self._steps = 0
        self._eps = self.eps_max

    def act(self, state):
        action_values = self.q_net(Variable(torch.FloatTensor([state]), volatile=True))
        _, action = action_values.max(1)
        action, _ = epsilon_greedy(self.action_shape[0], action.data[0], eps=self._eps)
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
        max_next_q_values = self.target_q_net(batch_next_state).max(1)[0].unsqueeze(1)
        expected_q_values = batch_reward + self.gamma * max_next_q_values

        loss = MSELoss()(current_q_values, expected_q_values)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self._steps += 1
        self._eps = self.eps_min + \
                    (self.eps_max - self.eps_min) * np.exp(-float(self._steps) * 1. / self.temperature)

        if self._steps % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        return loss
