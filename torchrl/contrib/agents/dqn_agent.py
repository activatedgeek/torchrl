from copy import deepcopy
import torch
from torch.optim import Adam
from torchrl.agents import BaseAgent
from torchrl.policies import epsilon_greedy
from torchrl.utils import ExpDecaySchedule

from ..models import QNet


class BaseDQNAgent(BaseAgent):
  def __init__(self, observation_space, action_space,
               double_dqn=False,
               gamma=0.99,
               lr=1e-3,
               eps_max=1.0,
               eps_min=0.01,
               num_eps_steps=1000,
               target_update_interval=5):
    super(BaseDQNAgent, self).__init__(observation_space, action_space)

    self.q_net = QNet(observation_space.shape[0], action_space.n)
    self.target_q_net = deepcopy(self.q_net)
    self.q_net_optim = Adam(self.q_net.parameters(), lr=lr)

    self.double_dqn = double_dqn
    self.gamma = gamma
    self.target_update_interval = target_update_interval

    self._steps = 0
    self.eps = ExpDecaySchedule(start=eps_max, end=eps_min,
                                num_steps=num_eps_steps)

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
    obs_tensor = self.obs_to_tensor(obs)

    actions = self.q_net(obs_tensor)
    actions = actions.max(dim=1)[1].cpu().numpy()
    actions = epsilon_greedy(self.action_space.n, actions, self.eps.value)
    return actions

  def compute_q_values(self, obs, action, reward, next_obs, done):
    current_q_values = self.q_net(obs).gather(1, action.long())

    with torch.no_grad():
      if self.double_dqn:
        _, next_actions = self.q_net(next_obs).max(1, keepdim=True)
        next_q_values = self.target_q_net(next_obs).gather(1, next_actions)
      else:
        next_q_values = self.target_q_net(next_obs)
        next_q_values = next_q_values.max(1)[0].unsqueeze(1)

      expected_q_values = reward + \
                          self.gamma * next_q_values * (1.0 - done.float())

    return current_q_values, expected_q_values

  def learn(self, obs, action, reward, next_obs, done,  # pylint: disable=unused-argument
            td_error):
    loss = td_error.pow(2).mean()

    self.q_net_optim.zero_grad()
    loss.backward()
    self.q_net_optim.step()

    self._steps += 1
    if self._steps % self.target_update_interval == 0:
      self.target_q_net.load_state_dict(self.q_net.state_dict())

    return loss.detach().cpu().item()
