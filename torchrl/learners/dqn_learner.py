from copy import deepcopy
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam

from torchrl.learners import BaseLearner
from torchrl.policies import epsilon_greedy
from torchrl.models import QNet


class BaseDQNLearner(BaseLearner):
  def __init__(self, observation_space, action_space,
               gamma=0.8,
               lr=1e-4,
               eps_max=1.0,
               eps_min=0.1,
               temperature=2000.0,
               target_update_interval=5):
    super(BaseDQNLearner, self).__init__(observation_space, action_space)

    self.q_net = QNet(observation_space.shape[0], action_space.n)
    self.target_q_net = deepcopy(self.q_net)
    self.q_net_optim = Adam(self.q_net.parameters(), lr=lr)

    self.gamma = gamma
    self.eps_max = eps_max
    self.eps_min = eps_min
    self.temperature = temperature
    self.target_update_interval = target_update_interval

    self._steps = 0
    self.eps = eps_max

    self.train()

  def act(self, obs):
    actions = self.q_net(obs)
    actions = actions.max(dim=1)[1].cpu().numpy()
    actions = epsilon_greedy(self.action_space.n, actions, self.eps)
    return actions

  def learn(self, obs, action, reward, next_obs, done):  # pylint: disable=unused-argument
    current_q_values = self.q_net(obs).gather(1, action)
    max_next_q_values = self.target_q_net(next_obs)
    max_next_q_values = max_next_q_values.max(1)[0].unsqueeze(1).detach()
    expected_q_values = reward + self.gamma * max_next_q_values

    loss = F.mse_loss(current_q_values, expected_q_values)

    loss.backward()
    self.q_net_optim.step()
    self.q_net_optim.zero_grad()

    self._steps += 1
    self.eps = self.eps_min + \
           (self.eps_max - self.eps_min) * \
               np.exp(-float(self._steps) * 1. / self.temperature)

    if self._steps % self.target_update_interval == 0:
      self.target_q_net.load_state_dict(self.q_net.state_dict())

    return loss.detach().cpu().item()

  def cuda(self):
    self.q_net.cuda()
    self.target_q_net.cuda()
    self.is_cuda = True

  def train(self):
    self.q_net.train()
    self.target_q_net.train()
    self.training = True

  def eval(self):
    self.q_net.eval()
    self.target_q_net.eval()
    self.training = False
