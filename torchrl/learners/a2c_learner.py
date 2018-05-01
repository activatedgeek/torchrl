import torch
from torch.autograd import Variable
from torch.distributions import Categorical
from . import BaseLearner


class A2CLearner(BaseLearner):
    def __init__(self, actor_critic_net, optimizer, action_space,
                 gamma=0.99,
                 tau=1.0,
                 beta=0.01,
                 clip_grad_norm=10):
        """
        :param actor_critic_net: The Actor-Critic Network
        :param optimizer: Optimizer for stepping
        :param action_space: shape of the action space
        :param gamma: Reward discount factor
        :param tau: Discount factor for generalized advantage estimator
        :param beta: Parameter for the entropy loss term
        :param clip_grad_norm: Max value of the gradient norms
        """
        super(A2CLearner, self).__init__(optimizer)

        self.ac_net = actor_critic_net

        self.action_space = action_space

        # Hyper-Parameters
        self.gamma = gamma
        self.tau = tau
        self.beta = beta
        self.clip_grad_norm = clip_grad_norm

        # Internal State
        self._steps = 0

        self._next_state_batch = []
        self._value_batch = []
        self._log_prob_batch = []
        self._entropy_batch = []
        self._reward_batch = []
        self._done_batch = []

    def _clear(self):
        self._next_state_batch.clear()
        self._value_batch.clear()
        self._log_prob_batch.clear()
        self._entropy_batch.clear()
        self._reward_batch.clear()
        self._done_batch.clear()

    def act(self, state, *args, **kwargs):
        value, prob = self.ac_net(Variable(torch.FloatTensor([state])))
        log_prob = torch.log(prob)
        entropy = -(log_prob * prob).sum(1)

        dist = Categorical(prob)
        action = dist.sample()

        self._entropy_batch.append(entropy)
        self._value_batch.append(value)
        self._log_prob_batch.append(dist.log_prob(action))

        return action.data[0]

    def transition(self, state, action, reward, next_state, done, episode_id=None):
        self._next_state_batch.extend(next_state)
        self._reward_batch.extend(reward)
        self._done_batch.extend(done)

    def learn(self, **kwargs):
        value_batch = self._value_batch
        next_state_batch = self._next_state_batch
        reward_batch = self._reward_batch
        done_batch = self._done_batch
        log_prob_batch = self._log_prob_batch
        entropy_batch = self._entropy_batch

        expected_return = torch.zeros(1, 1)
        if not done_batch[-1]:
            state_tensor = torch.FloatTensor([next_state_batch[-1]])
            expected_return, _ = self.ac_net(Variable(state_tensor))
            expected_return = expected_return.data

        policy_loss = 0
        value_loss = 0

        expected_return = Variable(expected_return)
        value_batch.append(expected_return)

        gae = torch.zeros(1, 1)
        for i in reversed(range(len(reward_batch))):
            expected_return = reward_batch[i] + self.gamma * expected_return
            advantage = expected_return - value_batch[i]

            value_loss += advantage.pow(2)

            td_error = reward_batch[i] + self.gamma * value_batch[i + 1].data - value_batch[i].data
            gae = td_error + self.gamma * self.tau * gae

            policy_loss -= log_prob_batch[i] * Variable(gae) + self.beta * entropy_batch[i]

        (policy_loss + value_loss).backward()
        torch.nn.utils.clip_grad_norm(self.ac_net.parameters(), self.clip_grad_norm)

        self.optimizer.step()
        self.optimizer.zero_grad()
        self._clear()

        self._steps += 1
