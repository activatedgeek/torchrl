import numpy as np
import torch
import random
from torch.autograd import Variable
from torch.distributions import Categorical
from . import BaseLearner
from .. import Episode


class A2CLearner(BaseLearner):
    def __init__(self, policy_net, criterion, optimizer, action_space,
                 gamma=0.99,
                 eps_max=1.0,
                 eps_min=0.01,
                 temperature=3500.0,
                 tmax=5):
        super(A2CLearner, self).__init__(criterion, optimizer)

        self.policy_net = policy_net
        self.action_space = action_space

        # Hyper-Parameters
        self.gamma = gamma
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.temperature = temperature
        self.tmax = tmax

        # Internal State
        self._eps = self.eps_max
        self._cur_episode = Episode()
        self._t = 0

    def act(self, state, *args, **kwargs):
        action_values, policy = self.policy_net(Variable(torch.FloatTensor([state]), volatile=True))
        dist = Categorical(policy)
        action = torch.LongTensor([random.randrange(self.action_space)]) if random.random() < self._eps else dist.sample()
        return action[0], dist.log_prob(action)

    def transition(self, episode_id, state, action, reward, next_state, done):
        self._cur_episode.append(state, action, reward, next_state, done)
        self._t += 1

    def learn(self, **kwargs):
        episode = self._cur_episode

        expected_return = Variable(torch.FloatTensor([0.0]), requires_grad=True)
        if not episode[-1].done:
            state_tensor = torch.FloatTensor(episode[-1].state)
            value, policy = self.policy_net(Variable(state_tensor))
            expected_return = Variable(value, requires_grad=True)

        for transition in episode[::-1][1:]:
            expected_return = transition.reward + self.gamma * expected_return
            state_tensor = torch.FloatTensor(episode[-1].state)
            value, policy = self.policy_net(Variable(state_tensor))

            value_loss = self.criterion(expected_return, value.detach())
            policy_loss = Variable(transition.action_log_prob) * (expected_return - value)

            (value_loss + policy_loss).backward()

        self.optimizer.step()
        self.optimizer.zero_grad()
        self._cur_episode.clear()

        self._eps = self.eps_min + \
                    (self.eps_max - self.eps_min) * np.exp(-float(self._t) * 1. / self.temperature)

