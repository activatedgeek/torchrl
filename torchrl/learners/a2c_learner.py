import numpy as np
import torch
from torch.optim import Adam
from torch.distributions import Categorical

from torchrl.learners import BaseLearner
from torchrl.models import A2CNet


class BaseA2CLearner(BaseLearner):
  def __init__(self, observation_space, action_space,
               lr=1e-3,
               gamma=0.99,
               lmbda=1.0,
               alpha=0.5,
               beta=1.0):
    super(BaseA2CLearner, self).__init__(observation_space, action_space)

    self.ac_net = A2CNet(observation_space.shape[0], action_space.n, 256)
    self.ac_net_optim = Adam(self.ac_net.parameters(), lr=lr)

    self.gamma = gamma
    self.lmbda = lmbda
    self.alpha = alpha
    self.beta = beta

    self.train()

  def act(self, obs):
    _, prob = self.ac_net(obs)
    dist = Categorical(prob)
    action = dist.sample()
    return action.unsqueeze(1).cpu().data.numpy()

  def compute_returns(self, obs, action, reward, next_obs, done):  # pylint: disable=unused-argument
    with torch.no_grad():
      obs_tensor = torch.from_numpy(obs).float()

      if self.is_cuda:
        obs_tensor = obs_tensor.cuda()

      values, _ = self.ac_net(obs_tensor)
      values = values.cpu().data.numpy()
      if not done[-1]:
        next_obs_tensor = torch.from_numpy(next_obs[-1]).float().unsqueeze(0)
        if self.is_cuda:
          next_obs_tensor = next_obs_tensor.cuda()

        next_value, _ = self.ac_net(next_obs_tensor)
        next_value = next_value.cpu().data.numpy()
        values = np.append(values, next_value, axis=0)
      else:
        values = np.append(values, np.array([[0.0]]), axis=0)

      returns = [0.0] * len(reward)
      gae = 0.0
      for step in reversed(range(len(reward))):
        delta = reward[step] + self.gamma * values[step + 1] - values[step]
        gae = delta + self.gamma * self.lmbda * gae
        returns[step] = gae + values[step]

      returns = np.array(returns)
      returns = returns[::-1]

      return returns

  def learn(self, obs, action, reward, next_obs, done, returns):  # pylint: disable=unused-argument
    obs_tensor = torch.from_numpy(obs).float()
    action_tensor = torch.from_numpy(action).long()
    return_tensor = torch.from_numpy(returns).float()

    if self.is_cuda:
      obs_tensor = obs_tensor.cuda()
      action_tensor = action_tensor.cuda()
      return_tensor = return_tensor.cuda()

    values, prob = self.ac_net(obs_tensor)
    dist = Categorical(prob)

    advantages = return_tensor - values

    action_log_probs = dist.log_prob(action_tensor.squeeze(-1)).unsqueeze(1)
    actor_loss = - (advantages.detach() * action_log_probs).mean()

    critic_loss = advantages.pow(2).mean()

    entropy_loss = dist.entropy().mean()

    loss = actor_loss + self.alpha * critic_loss + self.beta * entropy_loss

    self.ac_net_optim.zero_grad()
    loss.backward()
    self.ac_net_optim.step()

    return actor_loss.detach().cpu().data.numpy(), \
        critic_loss.detach().cpu().data.numpy(), \
        entropy_loss.detach().cpu().data.numpy()

  def cuda(self):
    self.ac_net.cuda()
    self.is_cuda = True

  def train(self):
    self.ac_net.train()
    self.training = True

  def eval(self):
    self.ac_net.eval()
    self.training = False
