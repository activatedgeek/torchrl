import torch
import torch.nn.functional as F
from torch.optim import Adam

from .base_agent import BaseAgent
from ..models import A2CNet


class BaseA2CAgent(BaseAgent):
  def __init__(self, observation_space, action_space,
               lr=1e-3,
               gamma=0.99,
               lmbda=1.0,
               alpha=0.5,
               beta=1.0):
    super(BaseA2CAgent, self).__init__(observation_space, action_space)

    self.ac_net = A2CNet(observation_space.shape[0], action_space.n, 256)
    self.ac_net_optim = Adam(self.ac_net.parameters(), lr=lr)

    self.gamma = gamma
    self.lmbda = lmbda
    self.alpha = alpha
    self.beta = beta

  @property
  def models(self):
    return [self.ac_net]

  @property
  def checkpoint(self):
    return {
        'ac_net': self.ac_net.state_dict()
    }

  @checkpoint.setter
  def checkpoint(self, cp):
    self.ac_net.load_state_dict(cp['ac_net'])

  def act(self, obs):
    _, dist = self.ac_net(obs)
    action = dist.sample()
    return action.unsqueeze(1).cpu().numpy()

  def compute_returns(self, obs, action, reward, next_obs, done):  # pylint: disable=unused-argument
    with torch.no_grad():
      values, _ = self.ac_net(obs)
      if not done[-1]:
        next_value, _ = self.ac_net(next_obs[-1:])
        values = torch.cat([values, next_value], dim=0)
      else:
        values = torch.cat([values, torch.zeros(1, 1)], dim=0)

      returns = torch.zeros(len(reward), 1)
      gae = 0.0
      for step in reversed(range(len(reward))):
        delta = reward[step] + self.gamma * values[step + 1] - values[step]
        gae = delta + self.gamma * self.lmbda * gae
        returns[step] = gae + values[step]

      return returns

  def learn(self, obs, action, reward, next_obs, done, returns):  # pylint: disable=unused-argument
    values, dist = self.ac_net(obs)

    advantages = returns - values

    action_log_probs = dist.log_prob(action.squeeze(-1)).unsqueeze(1)
    actor_loss = - (advantages.detach() * action_log_probs).mean()

    critic_loss = F.mse_loss(values, returns)

    entropy_loss = dist.entropy().mean()

    loss = actor_loss + self.alpha * critic_loss - self.beta * entropy_loss

    self.ac_net_optim.zero_grad()
    loss.backward()
    self.ac_net_optim.step()

    return actor_loss.detach().cpu().item(), \
        critic_loss.detach().cpu().item(), \
        entropy_loss.detach().cpu().item()
