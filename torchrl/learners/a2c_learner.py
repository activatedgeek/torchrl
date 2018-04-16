import torch
from torch.autograd import Variable
from . import BaseLearner
from ..policies import epsilon_greedy
from ..utils import Episode


class A2CLearner(BaseLearner):
    def __init__(self, q_net, criterion, optimizer, action_space,
                 gamma=0.99,
                 eps_max=1.0,
                 eps_min=0.01,
                 temperature=3500.0,
                 tmax=5):
        super(A2CLearner, self).__init__(criterion, optimizer)

        self.q_net = q_net
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

    def act(self, state, *args, **kwargs):
        action_values = self.q_net(Variable(torch.FloatTensor([state]), volatile=True))
        value, action = action_values.max(1)
        best_action = action.data[0]
        action, _ = epsilon_greedy(self.action_space, best_action, eps=self._eps)
        return action

    def transition(self, episode_id, state, action, reward, next_state, done):
        self._cur_episode.append(state, action, reward, next_state, done)

    def learn(self, **kwargs):
        episode = self._cur_episode

        expected_return = Variable(torch.FloatTensor([0.0]), requires_grad=True)
        if not episode[-1].done:
            state_tensor = torch.FloatTensor(episode[-1].state)
            q_values = self.q_net.forward(Variable(state_tensor))
            expected_return, _ = q_values.max()
            expected_return = Variable(expected_return, requires_grad=True)

        for transition in episode[::-1][1:]:
            expected_return = transition.reward + self.gamma * expected_return
            state_tensor = torch.FloatTensor(episode[-1].state)
            q_values = self.q_net.forward(Variable(state_tensor, volatile=True))

            loss = self.criterion(expected_return, q_values[transition.action])
            loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()
        self._cur_episode.clear()
