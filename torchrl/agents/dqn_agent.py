from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam

from .base_agent import BaseAgent
from ..policies import epsilon_greedy
from ..models import QNet


class BaseDQNAgent(BaseAgent):
  def __init__(self, observation_space, action_space,
               double_dqn=False,
               gamma=0.8,
               lr=1e-4,
               eps_max=1.0,
               eps_min=0.1,
               temperature=2000.0,
               target_update_interval=5):
    super(BaseDQNAgent, self).__init__(observation_space, action_space)

    self.q_net = QNet(observation_space.shape[0], action_space.n)
    self.target_q_net = deepcopy(self.q_net)
    self.q_net_optim = Adam(self.q_net.parameters(), lr=lr)

    self.double_dqn = double_dqn
    self.gamma = gamma
    self.eps_max = eps_max
    self.eps_min = eps_min
    self.temperature = temperature
    self.target_update_interval = target_update_interval

    self._steps = 0
    self.eps = eps_max

  @property
  def models(self):
    return [self.q_net, self.target_q_net]

  @property
  def checkpoint(self):
    return {
        'q_net': self.q_net.state_dict(),
        'steps': self._steps,
        'eps': self.eps,
    }

  @checkpoint.setter
  def checkpoint(self, cp):
    self.q_net.load_state_dict(cp['q_net'])
    self.target_q_net = deepcopy(self.q_net)
    self._steps = cp['steps']
    self.eps = cp['eps']

  def act(self, obs):
    actions = self.q_net(obs)
    actions = actions.max(dim=1)[1].cpu().numpy()
    actions = epsilon_greedy(self.action_space.n, actions, self.eps)
    return actions

  def compute_q_values(self, obs, action, reward, next_obs, done):  # pylint: disable=unused-argument
    current_q_values = self.q_net(obs).gather(1, action)

    with torch.no_grad():
      if self.double_dqn:
        _, next_actions = self.q_net(next_obs).max(1, keepdim=True)
        max_next_q_values = self.target_q_net(next_obs).gather(1, next_actions)
      else:
        max_next_q_values = self.target_q_net(next_obs)
        max_next_q_values = max_next_q_values.max(1)[0].unsqueeze(1)

      expected_q_values = reward + self.gamma * max_next_q_values

    return current_q_values, expected_q_values

  def learn(self, obs, action, reward, next_obs, done,  # pylint: disable=unused-argument
            td_error):
    loss = td_error.pow(2).mean()

    self.q_net_optim.zero_grad()
    loss.backward()
    self.q_net_optim.step()

    self._steps += 1
    self.eps = self.eps_min + \
           (self.eps_max - self.eps_min) * \
               np.exp(-float(self._steps) * 1. / self.temperature)

    if self._steps % self.target_update_interval == 0:
      self.target_q_net.load_state_dict(self.q_net.state_dict())

    return loss.detach().cpu().item()
