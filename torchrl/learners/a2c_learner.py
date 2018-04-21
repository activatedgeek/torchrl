import torch
from copy import deepcopy
from torch.autograd import Variable
from torch.distributions import Categorical
from . import BaseLearner


class A2CLearner(BaseLearner):
    def __init__(self, policy_net, criterion, optimizer, action_space,
                 gamma=0.99,
                 eps_max=1.0,
                 eps_min=0.01,
                 temperature=3500.0,
                 target_update_freq=5):
        super(A2CLearner, self).__init__(criterion, optimizer)

        self.policy_net = policy_net
        self.target_policy_net = deepcopy(policy_net)
        self.action_space = action_space

        # Hyper-Parameters
        self.gamma = gamma
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.temperature = temperature
        self.target_update_freq = target_update_freq

        # Internal State
        self._eps = self.eps_max
        self._steps = 0

        self._state_batch = []
        self._action_batch = []
        self._reward_batch = []
        self._next_state_batch = []
        self._done_batch = []
        self._log_probs_batch = []

    def _clear(self):
        self._state_batch.clear()
        self._action_batch.clear()
        self._reward_batch.clear()
        self._next_state_batch.clear()
        self._done_batch.clear()
        self._log_probs_batch.clear()

    def act(self, state, *args, **kwargs):
        action_values, policy = self.policy_net(Variable(torch.FloatTensor([state]), volatile=True))
        dist = Categorical(policy)
        action = dist.sample()
        self._log_probs_batch.append(dist.log_prob(action)[0])
        return action[0]

    def transition(self, state, action, reward, next_state, done, episode_id=None):
        self._state_batch.extend(state)
        self._action_batch.extend(action)
        self._reward_batch.extend(reward)
        self._next_state_batch.extend(next_state)
        self._done_batch.extend(done)

    def learn(self, **kwargs):
        state_batch = self._state_batch
        action_batch = self._action_batch
        reward_batch = self._reward_batch
        next_state_batch = self._next_state_batch
        done_batch = self._done_batch
        log_probs_batch = self._log_probs_batch

        expected_return = 0.0
        if not done_batch[-1]:
            state_tensor = torch.FloatTensor([state_batch[-1]])
            expected_return, _ = self.target_policy_net(Variable(state_tensor, volatile=True))
            expected_return = expected_return.data[0][0]

        for state, action, reward, next_state, log_prob in \
                zip(state_batch[::-1][1:], action_batch[::-1][1:], reward_batch[::-1][1:],
                    next_state_batch[::-1][1:], log_probs_batch[::-1][1:]):
            expected_return = reward + self.gamma * expected_return
            state_tensor = torch.FloatTensor([state])
            value, policy = self.policy_net(Variable(state_tensor, requires_grad=True))

            value_loss = self.criterion(value, Variable(torch.FloatTensor([[expected_return]])))
            policy_loss = log_prob * (expected_return - value)

            (value_loss + policy_loss).backward()

        self.optimizer.step()
        self.optimizer.zero_grad()
        self._clear()

        self._steps += 1
        # self._eps = self.eps_min + \
        #             (self.eps_max - self.eps_min) * np.exp(-float(self._steps) * 1. / self.temperature)

        if self._steps % self.target_update_freq == 0:
            self.target_policy_net.load_state_dict(self.policy_net.state_dict())
